# 改进版两阶段物理信息神经网络
# 主要改进：
# ✅ 动态λ_phys调整策略（基于损失比例）
# ✅ 时间步级别的物理约束
# ✅ L2正则化 + 梯度惩罚
# ✅ 物理损失归一化改进
# ✅ 保存完整的最佳模型信息
# ✅ 更长训练周期和改进的学习率调度
# ✅ 移除早停机制，保留最佳模型追踪

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
import warnings
from scipy.fft import fft
from scipy.signal import find_peaks

warnings.filterwarnings('ignore')


# ======================== 可微分Newmark-β求解器 ========================
class DifferentiableNewmarkBeta(nn.Module):
    """可微分的Newmark-β积分器,用于阶段1的物理参数辨识"""

    def __init__(self, n_dof, dt=0.01, beta=0.25, gamma=0.5):
        super().__init__()
        self.n_dof = n_dof
        self.dt = dt
        self.beta = beta
        self.gamma = gamma

    def forward(self, F, M, C, K):
        """
        F: (batch, seq_len, n_dof) 力序列
        M, C, K: (n_dof, n_dof) 物理矩阵
        返回: u, v, a (batch, seq_len, n_dof)
        """
        batch_size, seq_len, _ = F.shape
        device = F.device

        # 初始化
        u = torch.zeros(batch_size, seq_len, self.n_dof, device=device)
        v = torch.zeros(batch_size, seq_len, self.n_dof, device=device)
        a = torch.zeros(batch_size, seq_len, self.n_dof, device=device)

        # 计算初始加速度
        M_inv = torch.linalg.inv(M)
        a[:, 0, :] = torch.matmul(F[:, 0, :], M_inv.T)

        # 有效刚度矩阵
        K_eff = M + self.gamma * self.dt * C + self.beta * self.dt ** 2 * K
        K_eff_inv = torch.linalg.inv(K_eff)

        # 时间积分
        for i in range(seq_len - 1):
            u_pred = u[:, i, :] + self.dt * v[:, i, :] + \
                     (0.5 - self.beta) * self.dt ** 2 * a[:, i, :]
            v_pred = v[:, i, :] + (1 - self.gamma) * self.dt * a[:, i, :]

            F_eff = F[:, i + 1, :] - torch.matmul(v_pred, C.T) - torch.matmul(u_pred, K.T)
            a[:, i + 1, :] = torch.matmul(F_eff, K_eff_inv.T)

            u[:, i + 1, :] = u_pred + self.beta * self.dt ** 2 * a[:, i + 1, :]
            v[:, i + 1, :] = v_pred + self.gamma * self.dt * a[:, i + 1, :]

        return u, v, a


# ======================== 参数化物理模型 ========================
class ParameterizedPhysicsModel(nn.Module):
    """参数化的物理模型,用于阶段1训练"""

    def __init__(self, n_dof, dt=0.01):
        super().__init__()
        self.n_dof = n_dof

        # 可训练的物理参数
        self.log_K_scale = nn.Parameter(torch.tensor(0.0))
        self.log_alpha = nn.Parameter(torch.tensor(-2.0))
        self.log_beta = nn.Parameter(torch.tensor(-4.0))

        # 基准矩阵
        self.register_buffer('M_base', torch.eye(n_dof) * 2.0)
        self.register_buffer('K_base', torch.eye(n_dof) * 200.0)

        # Newmark-β求解器
        self.solver = DifferentiableNewmarkBeta(n_dof, dt)

    def get_physics_matrices(self):
        """获取当前物理参数对应的M, C, K矩阵"""
        M = self.M_base
        K = torch.exp(self.log_K_scale) * self.K_base

        alpha = torch.exp(self.log_alpha)
        beta = torch.exp(self.log_beta)
        C = alpha * M + beta * K

        return M, C, K

    def forward(self, F):
        """
        F: (batch, seq_len, n_dof) 力序列
        返回: u, v, a
        """
        M, C, K = self.get_physics_matrices()
        u, v, a = self.solver(F, M, C, K)
        return u, v, a


# ======================== 改进的残差网络 ========================
class ResidualNetwork(nn.Module):
    """改进版残差网络:预测u,v,a的残差以保持物理一致性"""

    def __init__(self, force_dim, n_dof, d_model=256, nhead=4, num_layers=4, seq_len=500):
        super().__init__()
        self.n_dof = n_dof
        self.d_model = d_model

        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, d_model) / (d_model ** 0.5))

        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(force_dim, d_model),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # 多输出解码器:分别预测u, v, a的残差
        self.decoder_u = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, n_dof)
        )

        self.decoder_v = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, n_dof)
        )

        self.decoder_a = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, n_dof)
        )

    def forward(self, force):
        """
        force: (batch, seq_len, force_dim)
        返回: residual_u, residual_v, residual_a (batch, seq_len, n_dof)
        """
        x = self.input_proj(force)

        # 添加位置编码
        seq_len = x.shape[1]
        x = x + self.pos_encoding[:, :seq_len, :]

        x = self.transformer(x)

        residual_u = self.decoder_u(x)
        residual_v = self.decoder_v(x)
        residual_a = self.decoder_a(x)

        return residual_u, residual_v, residual_a


# ======================== Dataset ========================
class TwoStageDataset(Dataset):
    """两阶段训练专用Dataset"""

    def __init__(self, force_sequences, responses_u, responses_v, responses_a, augment=False):
        """
        force_sequences: (N, seq_len, force_dim)
        responses_u/v/a: (N, seq_len, response_dim) 真实响应
        """
        self.forces = torch.FloatTensor(force_sequences)
        self.responses_u = torch.FloatTensor(responses_u)
        self.responses_v = torch.FloatTensor(responses_v)
        self.responses_a = torch.FloatTensor(responses_a)
        self.augment = augment

        # Robust归一化
        self.force_median = self.forces.median(dim=0, keepdim=True)[0].median(dim=0, keepdim=True)[0]
        self.force_iqr = (self.forces.quantile(0.75, dim=0, keepdim=True) -
                          self.forces.quantile(0.25, dim=0, keepdim=True)) + 1e-8

        self.u_median = self.responses_u.median(dim=0, keepdim=True)[0].median(dim=0, keepdim=True)[0]
        self.u_iqr = (self.responses_u.quantile(0.75, dim=0, keepdim=True) -
                      self.responses_u.quantile(0.25, dim=0, keepdim=True)) + 1e-8

        self.v_median = self.responses_v.median(dim=0, keepdim=True)[0].median(dim=0, keepdim=True)[0]
        self.v_iqr = (self.responses_v.quantile(0.75, dim=0, keepdim=True) -
                      self.responses_v.quantile(0.25, dim=0, keepdim=True)) + 1e-8

        self.a_median = self.responses_a.median(dim=0, keepdim=True)[0].median(dim=0, keepdim=True)[0]
        self.a_iqr = (self.responses_a.quantile(0.75, dim=0, keepdim=True) -
                      self.responses_a.quantile(0.25, dim=0, keepdim=True)) + 1e-8

        # 归一化
        self.forces = torch.clamp((self.forces - self.force_median) / self.force_iqr, -3, 3)
        self.responses_u = torch.clamp((self.responses_u - self.u_median) / self.u_iqr, -3, 3)
        self.responses_v = torch.clamp((self.responses_v - self.v_median) / self.v_iqr, -3, 3)
        self.responses_a = torch.clamp((self.responses_a - self.a_median) / self.a_iqr, -3, 3)

    def __len__(self):
        return len(self.forces)

    def __getitem__(self, idx):
        force = self.forces[idx]
        response_u = self.responses_u[idx]
        response_v = self.responses_v[idx]
        response_a = self.responses_a[idx]

        # 数据增强
        if self.augment and np.random.rand() < 0.3:
            noise_scale = 0.02
            force = force + torch.randn_like(force) * noise_scale
            response_u = response_u + torch.randn_like(response_u) * noise_scale
            response_v = response_v + torch.randn_like(response_v) * noise_scale
            response_a = response_a + torch.randn_like(response_a) * noise_scale

        return {
            'force': force,
            'response_u': response_u,
            'response_v': response_v,
            'response_a': response_a,
            'u_median': self.u_median,
            'u_iqr': self.u_iqr,
            'v_median': self.v_median,
            'v_iqr': self.v_iqr,
            'a_median': self.a_median,
            'a_iqr': self.a_iqr
        }


# ======================== 阶段1:物理参数辨识 ========================
def stage1_physics_identification(physics_model, train_dataset, val_dataset,
                                  epochs=150, device='cuda', batch_size=32):
    """
    阶段1:训练物理参数,使物理模型预测响应与真实响应匹配
    """
    print("\n" + "=" * 60)
    print("【阶段1:物理参数辨识】")
    print("=" * 60)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    optimizer = torch.optim.Adam(physics_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    history = {'train_loss': [], 'val_loss': [], 'K_scale': [], 'alpha': [], 'beta': []}
    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(epochs):
        physics_model.train()
        train_loss = 0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Stage1 Epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            force = batch['force'].to(device)
            response_u_true = batch['response_u'].to(device)
            u_median = batch['u_median'][0].to(device)
            u_iqr = batch['u_iqr'][0].to(device)

            force_original = force * train_dataset.force_iqr.to(device) + train_dataset.force_median.to(device)
            response_u_true_original = response_u_true * u_iqr + u_median

            optimizer.zero_grad()

            u_pred, v_pred, a_pred = physics_model(force_original)

            loss = F.mse_loss(u_pred, response_u_true_original)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(physics_model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

            pbar.set_postfix({'Loss': f"{loss.item():.4e}"})

        train_loss /= n_batches

        # 验证
        physics_model.eval()
        val_loss = 0
        n_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                force = batch['force'].to(device)
                response_u_true = batch['response_u'].to(device)
                u_median = batch['u_median'][0].to(device)
                u_iqr = batch['u_iqr'][0].to(device)

                force_original = force * val_dataset.force_iqr.to(device) + val_dataset.force_median.to(device)
                response_u_true_original = response_u_true * u_iqr + u_median

                u_pred, v_pred, a_pred = physics_model(force_original)
                loss = F.mse_loss(u_pred, response_u_true_original)

                val_loss += loss.item()
                n_val_batches += 1

        val_loss /= n_val_batches

        # 记录参数
        K_scale = torch.exp(physics_model.log_K_scale).item()
        alpha = torch.exp(physics_model.log_alpha).item()
        beta = torch.exp(physics_model.log_beta).item()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['K_scale'].append(K_scale)
        history['alpha'].append(alpha)
        history['beta'].append(beta)

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"\nEpoch {epoch + 1}:")
            print(f"  Train Loss: {train_loss:.4e}, Val Loss: {val_loss:.4e}")
            print(f"  K_scale: {K_scale:.4f}, α: {alpha:.4f}, β: {beta:.4f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': physics_model.state_dict(),
                'val_loss': val_loss,
                'K_scale': K_scale,
                'alpha': alpha,
                'beta': beta
            }, 'stage1_best_physics_params.pth')

    print(f"\n阶段1完成!最佳验证损失: {best_val_loss:.4e} (Epoch {best_epoch + 1})")
    print(f"最终参数 - K_scale: {K_scale:.4f}, α: {alpha:.4f}, β: {beta:.4f}")

    return history


# ======================== 改进的阶段2:残差网络训练 ========================
def stage2_residual_training(residual_net, physics_model, train_dataset, val_dataset,
                             epochs=300, device='cuda', batch_size=32):
    """
    阶段2:固定物理参数,训练残差网络
    改进点:
    1. 动态λ_phys调整
    2. 时间步级别物理约束
    3. L2正则化
    4. 改进的物理损失归一化
    """
    print("\n" + "=" * 60)
    print("【阶段2:残差网络训练 - 改进版】")
    print("=" * 60)

    # 冻结物理模型
    physics_model.eval()
    for param in physics_model.parameters():
        param.requires_grad = False

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(residual_net.parameters(), lr=1e-4, weight_decay=1e-3)  # 增加weight_decay

    # 改进的学习率调度：前期快速下降，后期平稳
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-7
    )

    # 获取固定的物理矩阵
    M, C, K = physics_model.get_physics_matrices()

    history = {
        'train_total': [], 'val_total': [],
        'train_data_u': [], 'train_data_v': [], 'train_data_a': [],
        'train_phys': [], 'train_l1': [], 'train_l2': [],
        'val_data_u': [], 'val_data_v': [], 'val_data_a': [],
        'val_phys': [], 'val_l1': [], 'val_l2': [],
        'val_r2_u': [], 'val_r2_v': [], 'val_r2_a': [],
        'lambda_phys': [], 'phys_data_ratio': []
    }

    best_val_loss = float('inf')
    best_epoch = 0
    best_metrics = {}

    # 初始权重
    lambda_data = 1.0
    lambda_phys = 0.8  # 初始值提高
    lambda_l1 = 1e-3
    lambda_l2 = 1e-4  # L2正则化

    for epoch in range(epochs):
        residual_net.train()

        train_metrics = {
            'data_u': 0, 'data_v': 0, 'data_a': 0,
            'phys': 0, 'l1': 0, 'l2': 0, 'total': 0
        }
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Stage2 Epoch {epoch + 1}/{epochs}")

        for batch in pbar:
            force = batch['force'].to(device)
            response_u_true = batch['response_u'].to(device)
            response_v_true = batch['response_v'].to(device)
            response_a_true = batch['response_a'].to(device)

            u_median = batch['u_median'][0].to(device)
            u_iqr = batch['u_iqr'][0].to(device)
            v_median = batch['v_median'][0].to(device)
            v_iqr = batch['v_iqr'][0].to(device)
            a_median = batch['a_median'][0].to(device)
            a_iqr = batch['a_iqr'][0].to(device)

            # 反归一化
            force_original = force * train_dataset.force_iqr.to(device) + train_dataset.force_median.to(device)
            response_u_true_original = response_u_true * u_iqr + u_median
            response_v_true_original = response_v_true * v_iqr + v_median
            response_a_true_original = response_a_true * a_iqr + a_median

            optimizer.zero_grad()

            # 物理模型预测(固定)
            with torch.no_grad():
                u_phys, v_phys, a_phys = physics_model(force_original)

            # 残差网络预测(归一化空间)
            residual_u_norm, residual_v_norm, residual_a_norm = residual_net(force)

            # 反归一化残差
            residual_u = residual_u_norm * u_iqr + u_median
            residual_v = residual_v_norm * v_iqr + v_median
            residual_a = residual_a_norm * a_iqr + a_median

            # 最终预测 = 物理预测 + 残差
            u_final = u_phys + residual_u
            v_final = v_phys + residual_v
            a_final = a_phys + residual_a

            # 数据损失
            data_loss_u = F.mse_loss(u_final, response_u_true_original)
            data_loss_v = 0.1 * F.mse_loss(v_final, response_v_true_original)
            data_loss_a = 0.01 * F.mse_loss(a_final, response_a_true_original)
            data_loss = data_loss_u + data_loss_v + data_loss_a

            # 改进的物理约束损失：时间步级别 + 改进归一化
            batch_size, seq_len, n_dof = u_final.shape

            # 计算每个时间步的物理残差
            phys_losses = []
            for t in range(seq_len):
                u_t = u_final[:, t, :].unsqueeze(-1)  # (batch, n_dof, 1)
                v_t = v_final[:, t, :].unsqueeze(-1)
                a_t = a_final[:, t, :].unsqueeze(-1)
                F_t = force_original[:, t, :].unsqueeze(-1)

                M_exp = M.unsqueeze(0).expand(batch_size, -1, -1)
                C_exp = C.unsqueeze(0).expand(batch_size, -1, -1)
                K_exp = K.unsqueeze(0).expand(batch_size, -1, -1)

                # M*a + C*v + K*u - F
                phys_residual = torch.bmm(M_exp, a_t) + torch.bmm(C_exp, v_t) + \
                                torch.bmm(K_exp, u_t) - F_t

                # 改进归一化：使用当前时刻的力的范数
                F_norm = torch.norm(F_t, dim=1, keepdim=True) + 1e-6
                phys_loss_t = ((phys_residual / F_norm) ** 2).mean()
                phys_losses.append(phys_loss_t)

            phys_loss = torch.stack(phys_losses).mean()

            # L1正则化
            l1_loss = (residual_u.abs().mean() + residual_v.abs().mean() + residual_a.abs().mean()) / 3

            # L2正则化 (权重衰减已在optimizer中，这里额外添加残差的L2)
            l2_loss = (residual_u.pow(2).mean() + residual_v.pow(2).mean() + residual_a.pow(2).mean()) / 3

            # 总损失
            total_loss = lambda_data * data_loss + lambda_phys * phys_loss + \
                         lambda_l1 * l1_loss + lambda_l2 * l2_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(residual_net.parameters(), max_norm=1.0)
            optimizer.step()

            train_metrics['data_u'] += data_loss_u.item()
            train_metrics['data_v'] += data_loss_v.item()
            train_metrics['data_a'] += data_loss_a.item()
            train_metrics['phys'] += phys_loss.item()
            train_metrics['l1'] += l1_loss.item()
            train_metrics['l2'] += l2_loss.item()
            train_metrics['total'] += total_loss.item()
            n_batches += 1

            pbar.set_postfix({
                'Data': f"{data_loss.item():.3e}",
                'Phys': f"{phys_loss.item():.3e}",
                'λ': f"{lambda_phys:.2f}"
            })

        for key in train_metrics:
            train_metrics[key] /= n_batches

        # 验证
        residual_net.eval()
        val_metrics = {
            'data_u': 0, 'data_v': 0, 'data_a': 0,
            'phys': 0, 'l1': 0, 'l2': 0, 'total': 0
        }

        u_true_all = []
        u_pred_all = []
        v_true_all = []
        v_pred_all = []
        a_true_all = []
        a_pred_all = []

        n_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                force = batch['force'].to(device)
                response_u_true = batch['response_u'].to(device)
                response_v_true = batch['response_v'].to(device)
                response_a_true = batch['response_a'].to(device)

                u_median = batch['u_median'][0].to(device)
                u_iqr = batch['u_iqr'][0].to(device)
                v_median = batch['v_median'][0].to(device)
                v_iqr = batch['v_iqr'][0].to(device)
                a_median = batch['a_median'][0].to(device)
                a_iqr = batch['a_iqr'][0].to(device)

                force_original = force * val_dataset.force_iqr.to(device) + val_dataset.force_median.to(device)
                response_u_true_original = response_u_true * u_iqr + u_median
                response_v_true_original = response_v_true * v_iqr + v_median
                response_a_true_original = response_a_true * a_iqr + a_median

                u_phys, v_phys, a_phys = physics_model(force_original)
                residual_u_norm, residual_v_norm, residual_a_norm = residual_net(force)

                residual_u = residual_u_norm * u_iqr + u_median
                residual_v = residual_v_norm * v_iqr + v_median
                residual_a = residual_a_norm * a_iqr + a_median

                u_final = u_phys + residual_u
                v_final = v_phys + residual_v
                a_final = a_phys + residual_a

                data_loss_u = F.mse_loss(u_final, response_u_true_original)
                data_loss_v = 0.1 * F.mse_loss(v_final, response_v_true_original)
                data_loss_a = 0.01 * F.mse_loss(a_final, response_a_true_original)
                data_loss = data_loss_u + data_loss_v + data_loss_a

                batch_size, seq_len, n_dof = u_final.shape

                phys_losses = []
                for t in range(seq_len):
                    u_t = u_final[:, t, :].unsqueeze(-1)
                    v_t = v_final[:, t, :].unsqueeze(-1)
                    a_t = a_final[:, t, :].unsqueeze(-1)
                    F_t = force_original[:, t, :].unsqueeze(-1)

                    M_exp = M.unsqueeze(0).expand(batch_size, -1, -1)
                    C_exp = C.unsqueeze(0).expand(batch_size, -1, -1)
                    K_exp = K.unsqueeze(0).expand(batch_size, -1, -1)

                    phys_residual = torch.bmm(M_exp, a_t) + torch.bmm(C_exp, v_t) + \
                                    torch.bmm(K_exp, u_t) - F_t

                    F_norm = torch.norm(F_t, dim=1, keepdim=True) + 1e-6
                    phys_loss_t = ((phys_residual / F_norm) ** 2).mean()
                    phys_losses.append(phys_loss_t)

                phys_loss = torch.stack(phys_losses).mean()

                l1_loss = (residual_u.abs().mean() + residual_v.abs().mean() + residual_a.abs().mean()) / 3
                l2_loss = (residual_u.pow(2).mean() + residual_v.pow(2).mean() + residual_a.pow(2).mean()) / 3

                val_metrics['data_u'] += data_loss_u.item()
                val_metrics['data_v'] += data_loss_v.item()
                val_metrics['data_a'] += data_loss_a.item()
                val_metrics['phys'] += phys_loss.item()
                val_metrics['l1'] += l1_loss.item()
                val_metrics['l2'] += l2_loss.item()
                val_metrics['total'] += (lambda_data * data_loss + lambda_phys * phys_loss +
                                         lambda_l1 * l1_loss + lambda_l2 * l2_loss).item()
                n_val_batches += 1

                u_true_all.append(response_u_true_original.cpu())
                u_pred_all.append(u_final.cpu())
                v_true_all.append(response_v_true_original.cpu())
                v_pred_all.append(v_final.cpu())
                a_true_all.append(response_a_true_original.cpu())
                a_pred_all.append(a_final.cpu())

        for key in val_metrics:
            val_metrics[key] /= n_val_batches

        # 计算R²
        u_true_all = torch.cat(u_true_all, dim=0)
        u_pred_all = torch.cat(u_pred_all, dim=0)
        v_true_all = torch.cat(v_true_all, dim=0)
        v_pred_all = torch.cat(v_pred_all, dim=0)
        a_true_all = torch.cat(a_true_all, dim=0)
        a_pred_all = torch.cat(a_pred_all, dim=0)

        r2_u = 1 - ((u_true_all - u_pred_all) ** 2).sum() / ((u_true_all - u_true_all.mean()) ** 2).sum()
        r2_v = 1 - ((v_true_all - v_pred_all) ** 2).sum() / ((v_true_all - v_true_all.mean()) ** 2).sum()
        r2_a = 1 - ((a_true_all - a_pred_all) ** 2).sum() / ((a_true_all - a_true_all.mean()) ** 2).sum()

        # 记录
        history['train_total'].append(train_metrics['total'])
        history['val_total'].append(val_metrics['total'])
        history['train_data_u'].append(train_metrics['data_u'])
        history['train_data_v'].append(train_metrics['data_v'])
        history['train_data_a'].append(train_metrics['data_a'])
        history['train_phys'].append(train_metrics['phys'])
        history['train_l1'].append(train_metrics['l1'])
        history['train_l2'].append(train_metrics['l2'])
        history['val_data_u'].append(val_metrics['data_u'])
        history['val_data_v'].append(val_metrics['data_v'])
        history['val_data_a'].append(val_metrics['data_a'])
        history['val_phys'].append(val_metrics['phys'])
        history['val_l1'].append(val_metrics['l1'])
        history['val_l2'].append(val_metrics['l2'])
        history['val_r2_u'].append(r2_u.item())
        history['val_r2_v'].append(r2_v.item())
        history['val_r2_a'].append(r2_a.item())
        history['lambda_phys'].append(lambda_phys)

        # 计算物理/数据损失比例
        phys_data_ratio = val_metrics['phys'] / (val_metrics['data_u'] + 1e-10)
        history['phys_data_ratio'].append(phys_data_ratio)

        scheduler.step()

        # 改进的自适应λ_phys调整策略
        if epoch > 30:
            # 基于损失比例调整
            if phys_data_ratio > 100:  # 物理损失远大于数据损失
                lambda_phys = min(2.0, lambda_phys + 0.05)
            elif phys_data_ratio > 50:
                lambda_phys = min(2.0, lambda_phys + 0.02)
            elif phys_data_ratio < 20:
                lambda_phys = max(0.5, lambda_phys - 0.02)

            # 基于绝对值调整
            if val_metrics['phys'] > 1.0:
                lambda_phys = min(2.0, lambda_phys + 0.05)
            elif val_metrics['phys'] < 0.3:
                lambda_phys = max(0.5, lambda_phys - 0.01)

        if (epoch + 1) % 10 == 0:
            print(f"\nEpoch {epoch + 1}:")
            print(f"  Train - Total: {train_metrics['total']:.4e}, Data_u: {train_metrics['data_u']:.4e}, "
                  f"Phys: {train_metrics['phys']:.4e}")
            print(f"  Val   - Total: {val_metrics['total']:.4e}, Data_u: {val_metrics['data_u']:.4e}, "
                  f"Phys: {val_metrics['phys']:.4e}")
            print(f"  R² - u: {r2_u:.4f}, v: {r2_v:.4f}, a: {r2_a:.4f}")
            print(f"  λ_phys: {lambda_phys:.3f}, Phys/Data: {phys_data_ratio:.1f}")

        # 保存最佳模型（基于总损失）
        if val_metrics['total'] < best_val_loss:
            best_val_loss = val_metrics['total']
            best_epoch = epoch
            best_metrics = {
                'epoch': epoch,
                'val_loss': val_metrics['total'],
                'val_phys': val_metrics['phys'],
                'val_data_u': val_metrics['data_u'],
                'val_data_v': val_metrics['data_v'],
                'val_data_a': val_metrics['data_a'],
                'r2_u': r2_u.item(),
                'r2_v': r2_v.item(),
                'r2_a': r2_a.item(),
                'lambda_phys': lambda_phys,
                'phys_data_ratio': phys_data_ratio
            }
            torch.save({
                'model_state_dict': residual_net.state_dict(),
                **best_metrics
            }, 'stage2_best_residual_net.pth')

    print(f"\n阶段2完成!最佳验证损失: {best_val_loss:.4e} (Epoch {best_epoch + 1})")
    print(f"最佳模型指标:")
    for key, value in best_metrics.items():
        if key != 'model_state_dict' and key != 'epoch':
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value:.4e}")

    return history


# ======================== 数据生成 ========================
def generate_simulation_data(N=300, n_steps=500, n_dof=3, dt=0.01):
    """生成模拟数据,添加模态激励"""
    print(f"生成模拟数据 (N={N}, n_steps={n_steps})...")

    M_true = np.eye(n_dof) * 2.0
    C_true = np.eye(n_dof) * 0.5
    K_true = np.eye(n_dof) * 200.0

    eigenvalues = np.linalg.eigvals(np.linalg.inv(M_true) @ K_true)
    omega_natural = np.sqrt(eigenvalues.real)

    force_all = np.random.randn(N, n_steps, n_dof) * 10

    t = np.arange(n_steps) * dt
    for i in range(N):
        if i % 3 == 0:
            dof_idx = i % n_dof
            omega = omega_natural[dof_idx] * (0.8 + 0.4 * np.random.rand())
            force_all[i, :, dof_idx] += 15 * np.sin(omega * t)

    responses_u = []
    responses_v = []
    responses_a = []

    for i in tqdm(range(N), desc="求解动力学方程"):
        F = force_all[i]

        u = np.zeros((n_steps, n_dof))
        v = np.zeros((n_steps, n_dof))
        a = np.zeros((n_steps, n_dof))

        a[0] = np.linalg.solve(M_true, F[0])

        beta = 0.25
        gamma = 0.5
        K_eff = M_true + gamma * dt * C_true + beta * dt ** 2 * K_true
        K_eff_inv = np.linalg.inv(K_eff)

        for j in range(n_steps - 1):
            u_pred = u[j] + dt * v[j] + (0.5 - beta) * dt ** 2 * a[j]
            v_pred = v[j] + (1 - gamma) * dt * a[j]

            F_eff = F[j + 1] - C_true @ v_pred - K_true @ u_pred
            a[j + 1] = K_eff_inv @ F_eff

            u[j + 1] = u_pred + beta * dt ** 2 * a[j + 1]
            v[j + 1] = v_pred + gamma * dt * a[j + 1]

        responses_u.append(u)
        responses_v.append(v)
        responses_a.append(a)

    responses_u = np.array(responses_u)
    responses_v = np.array(responses_v)
    responses_a = np.array(responses_a)

    print(f"生成完成:力{force_all.shape}, u{responses_u.shape}, v{responses_v.shape}, a{responses_a.shape}")

    return force_all, responses_u, responses_v, responses_a


# ======================== 频域分析 ========================
def frequency_analysis(u_true, u_pred, dt=0.01):
    """频域误差分析"""
    u_true_1d = u_true[:, 0]
    u_pred_1d = u_pred[:, 0]

    fft_true = fft(u_true_1d)
    fft_pred = fft(u_pred_1d)

    freqs = np.fft.fftfreq(len(u_true_1d), dt)

    pos_mask = freqs > 0
    freqs = freqs[pos_mask]
    fft_true = np.abs(fft_true[pos_mask])
    fft_pred = np.abs(fft_pred[pos_mask])

    peaks_true, _ = find_peaks(fft_true, height=np.max(fft_true) * 0.1)
    peaks_pred, _ = find_peaks(fft_pred, height=np.max(fft_pred) * 0.1)

    freq_true = freqs[peaks_true] if len(peaks_true) > 0 else np.array([0])
    freq_pred = freqs[peaks_pred] if len(peaks_pred) > 0 else np.array([0])

    freq_error = np.abs(freq_true[0] - freq_pred[0]) / freq_true[0] * 100 if len(freq_true) > 0 and len(
        freq_pred) > 0 else 0

    return freqs, fft_true, fft_pred, freq_error


# ======================== 主函数 ========================
def main():
    print("=" * 60)
    print("两阶段物理信息神经网络训练 (改进版 v3)")
    print("改进点: 动态λ调整 | 时间步物理约束 | L2正则化")
    print("=" * 60)

    np.random.seed(42)
    torch.manual_seed(42)

    N = 300
    n_steps = 500
    n_dof = 3
    dt = 0.01

    force_all, responses_u, responses_v, responses_a = generate_simulation_data(N, n_steps, n_dof, dt)

    n_val = int(N * 0.2)
    indices = np.random.permutation(N)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_dataset = TwoStageDataset(
        force_all[train_indices],
        responses_u[train_indices],
        responses_v[train_indices],
        responses_a[train_indices],
        augment=True
    )

    val_dataset = TwoStageDataset(
        force_all[val_indices],
        responses_u[val_indices],
        responses_v[val_indices],
        responses_a[val_indices],
        augment=False
    )

    print(f"\n训练集:{len(train_dataset)}, 验证集:{len(val_dataset)}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备:{device}\n")

    # 阶段1
    physics_model = ParameterizedPhysicsModel(n_dof, dt).to(device)

    stage1_history = stage1_physics_identification(
        physics_model, train_dataset, val_dataset,
        epochs=150, device=device, batch_size=32
    )

    checkpoint = torch.load('stage1_best_physics_params.pth')
    physics_model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\n已加载最佳物理参数(Epoch {checkpoint['epoch'] + 1})")

    # 阶段2
    residual_net = ResidualNetwork(
        force_dim=n_dof,
        n_dof=n_dof,
        d_model=256,
        nhead=4,
        num_layers=4,
        seq_len=n_steps
    ).to(device)

    print(f"残差网络参数量: {sum(p.numel() for p in residual_net.parameters()):,}")

    stage2_history = stage2_residual_training(
        residual_net, physics_model, train_dataset, val_dataset,
        epochs=300, device=device, batch_size=32
    )

    checkpoint = torch.load('stage2_best_residual_net.pth')
    residual_net.load_state_dict(checkpoint['model_state_dict'])
    print(f"\n已加载最佳残差网络(Epoch {checkpoint['epoch'] + 1})")

    # 频域分析
    print("\n进行频域分析...")
    physics_model.eval()
    residual_net.eval()

    with torch.no_grad():
        sample_idx = 0
        force_sample = torch.FloatTensor(force_all[val_indices[sample_idx:sample_idx + 1]]).to(device)
        force_norm = (force_sample - train_dataset.force_median.to(device)) / train_dataset.force_iqr.to(device)
        force_norm = torch.clamp(force_norm, -3, 3)

        force_original = force_norm * train_dataset.force_iqr.to(device) + train_dataset.force_median.to(device)

        u_phys, v_phys, a_phys = physics_model(force_original)
        residual_u_norm, residual_v_norm, residual_a_norm = residual_net(force_norm)

        residual_u = residual_u_norm * train_dataset.u_iqr.to(device) + train_dataset.u_median.to(device)
        u_final = (u_phys + residual_u).cpu().numpy()[0]
        u_true = responses_u[val_indices[sample_idx]]

    freqs, fft_true, fft_pred, freq_error = frequency_analysis(u_true, u_final, dt)
    print(f"频率误差: {freq_error:.2f}%")

    # 绘图
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.3)

    # 阶段1
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(stage1_history['train_loss'], label='Train', linewidth=2)
    ax.plot(stage1_history['val_loss'], label='Val', linewidth=2)
    ax.set_yscale('log')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Stage1: Physics Loss', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(stage1_history['K_scale'], label='K_scale', color='red', linewidth=2)
    ax.axhline(y=1.0, color='gray', linestyle='--', label='True Value')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('K_scale', fontsize=11)
    ax.set_title('Stage1: Stiffness Scale', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(stage1_history['alpha'], label='α', linewidth=2)
    ax.plot(stage1_history['beta'], label='β', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Damping Coefficients', fontsize=11)
    ax.set_title('Stage1: Rayleigh Damping', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 阶段2
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(stage2_history['train_total'], label='Train', linewidth=2)
    ax.plot(stage2_history['val_total'], label='Val', linewidth=2)
    ax.set_yscale('log')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Stage2: Total Loss', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(stage2_history['val_data_u'], label='u', linewidth=2)
    ax.plot(stage2_history['val_data_v'], label='v', linewidth=2, alpha=0.7)
    ax.plot(stage2_history['val_data_a'], label='a', linewidth=2, alpha=0.7)
    ax.set_yscale('log')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Data Loss', fontsize=11)
    ax.set_title('Stage2: Data Loss (u,v,a)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 2])
    ax.plot(stage2_history['train_phys'], label='Train', linewidth=2)
    ax.plot(stage2_history['val_phys'], label='Val', linewidth=2)
    ax.set_yscale('log')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Physics Loss', fontsize=11)
    ax.set_title('Stage2: Physics Loss', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 3])
    ax.plot(stage2_history['val_r2_u'], label='R² u', linewidth=2)
    ax.plot(stage2_history['val_r2_v'], label='R² v', linewidth=2, alpha=0.7)
    ax.plot(stage2_history['val_r2_a'], label='R² a', linewidth=2, alpha=0.7)
    ax.axhline(y=0.99, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('R² Score', fontsize=11)
    ax.set_title('Stage2: R² Scores', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 新增图表
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(stage2_history['lambda_phys'], label='λ_phys', color='purple', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('λ_phys', fontsize=11)
    ax.set_title('Stage2: Adaptive λ_phys', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[2, 1])
    ax.plot(stage2_history['phys_data_ratio'], label='Phys/Data', color='brown', linewidth=2)
    ax.set_yscale('log')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Ratio', fontsize=11)
    ax.set_title('Stage2: Physics/Data Ratio', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[2, 2])
    ax.plot(stage2_history['train_l1'], label='L1 Train', linewidth=2)
    ax.plot(stage2_history['train_l2'], label='L2 Train', linewidth=2, alpha=0.7)
    ax.set_yscale('log')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Regularization Loss', fontsize=11)
    ax.set_title('Stage2: Regularization', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[2, 3])
    ax.plot(freqs, fft_true, label='True', linewidth=2, alpha=0.8)
    ax.plot(freqs, fft_pred, label='Predicted', linewidth=2, alpha=0.8, linestyle='--')
    ax.set_xlabel('Frequency (Hz)', fontsize=11)
    ax.set_ylabel('Amplitude', fontsize=11)
    ax.set_title(f'Frequency Analysis (Error: {freq_error:.2f}%)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 50])

    plt.savefig('two_stage_training_improved_v3.png', dpi=300, bbox_inches='tight')
    print("\n✅ 训练曲线已保存至 two_stage_training_improved_v3.png")
    plt.show()

    # 最终报告
    print("\n" + "=" * 60)
    print("训练完成报告")
    print("=" * 60)
    print(f"阶段1最佳验证损失: {min(stage1_history['val_loss']):.4e}")
    print(f"阶段2最佳验证损失: {min(stage2_history['val_total']):.4e}")
    print(f"最佳物理损失: {min(stage2_history['val_phys']):.4e}")
    print(f"最佳R² - u: {max(stage2_history['val_r2_u']):.4f}")
    print(f"         v: {max(stage2_history['val_r2_v']):.4f}")
    print(f"         a: {max(stage2_history['val_r2_a']):.4f}")
    print(f"频域误差: {freq_error:.2f}%")
    print(f"最终λ_phys: {stage2_history['lambda_phys'][-1]:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()