# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import math
from dataclasses import dataclass
from typing import Optional, Tuple
from torch_radon import Radon
import numpy as np
import torch
import torch.nn.functional as F
from .vit_seg_modeling import VisionTransformer as ViT_seg
from .vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from torch import nn
from einops import rearrange
from time import time
import numbers
from wavelet2 import osem_reconstruct_safe_angdet
from wavelet import osem_reconstruct
import os
import torch_radon


# ==============================================================================
# 1. [New] 실시간 MI 계산 프로세서 (PyTorch GPU 버전)
# ==============================================================================
# class RealTimeMIProcessor(nn.Module):
#     def __init__(self, input_len=36, bins=64):
#         super().__init__()
#         self.input_len = input_len # 최종 출력 개수 (36)
#         self.bins = bins
        
#         # 352 / 11 = 32
#         self.patch_h = 32 
#         self.patch_w = 32 

#     def compute_mi_tensor(self, x_binned, y_binned):
#         """
#         x_binned, y_binned: [Batch, N_selected, P_pixels]
#         """
#         B, N, P = x_binned.shape
        
#         # Joint Histogram Indexing
#         joint_idx = x_binned * self.bins + y_binned
#         joint_idx_flat = joint_idx.view(B * N, P)
        
#         # Histogram Calculation
#         joint_hist = torch.zeros(B * N, self.bins * self.bins, device=x_binned.device)
#         ones = torch.ones_like(joint_idx_flat, dtype=torch.float32)
#         joint_hist.scatter_add_(1, joint_idx_flat.long(), ones)
        
#         # Probability Distribution
#         joint_prob = joint_hist / P
#         joint_prob_2d = joint_prob.view(B * N, self.bins, self.bins)
        
#         p_x = joint_prob_2d.sum(dim=2)
#         p_y = joint_prob_2d.sum(dim=1)
        
#         # Entropy & MI
#         eps = 1e-8
#         h_x = -torch.sum(p_x * torch.log(p_x + eps), dim=1)
#         h_y = -torch.sum(p_y * torch.log(p_y + eps), dim=1)
#         h_xy = -torch.sum(joint_prob * torch.log(joint_prob + eps), dim=1)
        
#         mi = h_x + h_y - h_xy
#         return mi.view(B, N)

#     def forward(self, sinogram):
#         # 1. 입력 확인 [Batch, 1, 352, 352]
#         if sinogram.dim() == 3: sinogram = sinogram.unsqueeze(1)
#         B, C, H, W = sinogram.shape 
        
#         # 2. 전처리 (Min-Max Windowing & Binning)
#         s_min = sinogram.amin(dim=(2, 3), keepdim=True)
#         s_max = sinogram.amax(dim=(2, 3), keepdim=True)
#         sino_norm = (sinogram - s_min) / (s_max - s_min + 1e-6)
        
#         sino_binned = (sino_norm * (self.bins - 1)).long()
#         sino_binned = torch.clamp(sino_binned, 0, self.bins - 1)
        
#         # 3. 전체 패치 추출 (11x11 = 121개)
#         # patches_all: [B, P_pixels, 121]
#         patches_all_raw = F.unfold(sino_binned.float(), kernel_size=(self.patch_h, self.patch_w), stride=(self.patch_h, self.patch_w))
#         patches_all = patches_all_raw.transpose(1, 2) # [B, 121, 1024]
        
#         # 4. 서브샘플링 (Target Patches 선정)
#         # 1D(121) -> 2D(11, 11)로 뷰 변경 후 슬라이싱
#         n_rows = H // self.patch_h # 11
#         n_cols = W // self.patch_w # 11
        
#         # [B, 11, 11, 1024]
#         patches_grid = patches_all.view(B, n_rows, n_cols, -1)
        
#         # 행/열 2칸 간격으로 선택 (::2) -> 6x6 = 36개
#         # selected_patches: [B, 6, 6, 1024] -> [B, 36, 1024]
#         patches_selected = patches_grid[:, ::2, ::2, :].reshape(B, -1, self.patch_h * self.patch_w)
        
#         # 5. Best Match 검색 (Selected 36개 vs All 121개)
#         # Target(36개)과 전체(121개) 사이의 거리 계산
#         # p_target: [B, 36, 1, 1024]
#         # p_all:    [B, 1, 121, 1024]
#         p_target = patches_selected.unsqueeze(2)
#         p_all = patches_all.unsqueeze(1)
        
#         # L1 Distance Matrix: [B, 36, 121]
#         dist_matrix = torch.abs(p_target - p_all).sum(dim=-1)
        
#         # 자기 자신(또는 겹치는 위치)이 선택되는 것을 방지하기 위해 마스킹
#         # 36개의 선택된 인덱스 계산
#         indices_1d = torch.arange(121, device=dist_matrix.device).view(11, 11)
#         selected_indices = indices_1d[::2, ::2].reshape(-1) # [0, 2, 4, ..., 120] (36개)
        
#         # dist_matrix[b, i, selected_indices[i]] = 무한대
#         # (Batch 차원 처리를 위해 expand 사용)
#         batch_idx = torch.arange(B, device=dist_matrix.device).view(-1, 1)
#         i_idx = torch.arange(36, device=dist_matrix.device).view(1, -1)
        
#         # 자기 자신의 위치에 큰 값 더하기
#         # dist_matrix의 (b, i, selected_indices[i]) 위치
#         mask = torch.zeros_like(dist_matrix)
#         mask.scatter_(2, selected_indices.view(1, 36, 1).expand(B, -1, -1), 1e9)
#         dist_matrix = dist_matrix + mask
        
#         # 가장 유사한 패치 인덱스 (Best Match)
#         best_match_idx = torch.argmin(dist_matrix, dim=2) # [B, 36]
        
#         # Best Match 패치 수집
#         # [B, 36, 1024]
#         # gather를 쓰거나 팬시 인덱싱 사용
#         flat_best_indices = best_match_idx.view(-1) # [B*36]
#         flat_patches_all = patches_all.reshape(B*121, -1)
        
#         # 배치를 고려한 인덱스 보정
#         batch_offsets = (torch.arange(B, device=dist_matrix.device) * 121).view(-1, 1)
#         final_indices = (batch_offsets + best_match_idx).view(-1)
        
#         best_match_patches = flat_patches_all[final_indices].view(B, 36, -1)
        
#         # 6. MI 계산 (Selected <-> Best Match)
#         # [B, 36]
#         mi_values = self.compute_mi_tensor(patches_selected, best_match_patches)
        
#         # 7. 후처리 (NaN 제거 및 로그 변환)
#         mi_values = torch.nan_to_num(mi_values, nan=0.0)
        
#         return torch.log1p(mi_values) # [B, 36]

class RealTimeMIProcessor(nn.Module):
    def __init__(
        self,
        input_len=36,
        bins=64,
        patch_h=32,
        patch_w=32,
        window_size=5,                 # ✅ 추가: patch-grid window size (e.g., 5x5)
        # windowing options (intensity clipping)
        window_mode="minmax",          # {"minmax", "fixed", "percentile"}
        window_min=None,              # used if window_mode="fixed"
        window_max=None,              # used if window_mode="fixed"
        window_percentiles=(0.01, 0.99) # used if window_mode="percentile"
    ):
        super().__init__()
        self.input_len = input_len
        self.bins = bins
        self.patch_h = patch_h
        self.patch_w = patch_w

        self.window_mode = window_mode
        self.window_min = window_min
        self.window_max = window_max
        self.window_percentiles = window_percentiles

        self.window_size = window_size  # ✅ OK

    def compute_mi_tensor(self, x_binned, y_binned):
        B, N, P = x_binned.shape
        joint_idx = x_binned * self.bins + y_binned
        joint_idx_flat = joint_idx.view(B * N, P)

        joint_hist = torch.zeros(B * N, self.bins * self.bins, device=x_binned.device)
        ones = torch.ones_like(joint_idx_flat, dtype=torch.float32)
        joint_hist.scatter_add_(1, joint_idx_flat.long(), ones)

        joint_prob = joint_hist / P
        joint_prob_2d = joint_prob.view(B * N, self.bins, self.bins)

        p_x = joint_prob_2d.sum(dim=2)
        p_y = joint_prob_2d.sum(dim=1)

        eps = 1e-8
        h_x  = -torch.sum(p_x * torch.log(p_x + eps), dim=1)
        h_y  = -torch.sum(p_y * torch.log(p_y + eps), dim=1)
        h_xy = -torch.sum(joint_prob * torch.log(joint_prob + eps), dim=1)

        mi = h_x + h_y - h_xy
        return mi.view(B, N)

    def _window_and_normalize(self, sinogram):
        eps = 1e-6
        if self.window_mode == "fixed":
            if self.window_min is None or self.window_max is None:
                raise ValueError("window_min/window_max must be set for fixed windowing.")
            low = torch.as_tensor(self.window_min, device=sinogram.device, dtype=sinogram.dtype).view(1, 1, 1, 1)
            high = torch.as_tensor(self.window_max, device=sinogram.device, dtype=sinogram.dtype).view(1, 1, 1, 1)
            sinogram = torch.clamp(sinogram, low, high)
            sino_norm = (sinogram - low) / (high - low + eps)
            return sino_norm.clamp(0.0, 1.0)

        elif self.window_mode == "percentile":
            ql, qh = self.window_percentiles
            flat = sinogram.view(sinogram.size(0), -1)
            low = torch.quantile(flat, ql, dim=1).view(-1, 1, 1, 1)
            high = torch.quantile(flat, qh, dim=1).view(-1, 1, 1, 1)
            sinogram = torch.clamp(sinogram, low, high)
            sino_norm = (sinogram - low) / (high - low + eps)
            return sino_norm.clamp(0.0, 1.0)

        else:  # "minmax"
            s_min = sinogram.amin(dim=(2, 3), keepdim=True)
            s_max = sinogram.amax(dim=(2, 3), keepdim=True)
            sino_norm = (sinogram - s_min) / (s_max - s_min + eps)
            return sino_norm.clamp(0.0, 1.0)

    def forward(self, sinogram):
        if sinogram.dim() == 3:
            sinogram = sinogram.unsqueeze(1)
        B, C, H, W = sinogram.shape

        # 1) windowing + normalize
        sino_norm = self._window_and_normalize(sinogram)

        # 2) binning
        sino_binned = (sino_norm * (self.bins - 1)).long()
        sino_binned = torch.clamp(sino_binned, 0, self.bins - 1)

        # 3) extract all 32x32 non-overlapping patches (11x11 = 121)
        patches_all_raw = F.unfold(
            sino_binned.float(),
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w)
        )
        patches_all = patches_all_raw.transpose(1, 2)  # [B, 121, 1024]

        # 4) select 36 target patches by subsampling the 11x11 patch grid with stride 2
        n_rows = H // self.patch_h  # 11
        n_cols = W // self.patch_w  # 11
        patches_grid = patches_all.view(B, n_rows, n_cols, -1)  # [B,11,11,1024]
        patches_selected = patches_grid[:, ::2, ::2, :].reshape(B, -1, self.patch_h * self.patch_w)  # [B,36,1024]

        # 5) best-match search within a local window (e.g., 5x5) on the 11x11 patch grid
        win = self.window_size
        r = win // 2

        indices_1d = torch.arange(n_rows * n_cols, device=patches_all.device).view(n_rows, n_cols)
        selected_indices = indices_1d[::2, ::2].reshape(-1)  # [36] indices in [0..120]

        t_row = selected_indices // n_cols
        t_col = selected_indices % n_cols

        dr = torch.arange(-r, r + 1, device=patches_all.device)
        dc = torch.arange(-r, r + 1, device=patches_all.device)
        rr = (t_row[:, None, None] + dr[None, :, None]).clamp(0, n_rows - 1)  # [36,win,1]
        cc = (t_col[:, None, None] + dc[None, None, :]).clamp(0, n_cols - 1)  # [36,1,win]
        cand_idx = (rr * n_cols + cc).view(-1, win * win)  # [36, win^2]

        cand_patches = patches_all[:, cand_idx, :]  # [B,36,win^2,1024]
        p_target = patches_selected.unsqueeze(2)    # [B,36,1,1024]
        dist_local = torch.abs(p_target - cand_patches).sum(dim=-1)  # [B,36,win^2]

        # mask self-match
        self_mask = (cand_idx == selected_indices.view(-1, 1))  # [36,win^2]
        dist_local = dist_local + self_mask.view(1, 36, -1).float() * 1e9

        best_local = torch.argmin(dist_local, dim=2)  # [B,36]
        best_match_idx = cand_idx.unsqueeze(0).expand(B, -1, -1).gather(2, best_local.unsqueeze(2)).squeeze(2)  # [B,36]

        # gather best-match patches
        flat_patches_all = patches_all.reshape(B * (n_rows * n_cols), -1)
        batch_offsets = (torch.arange(B, device=patches_all.device) * (n_rows * n_cols)).view(-1, 1)
        final_indices = (batch_offsets + best_match_idx).view(-1)
        best_match_patches = flat_patches_all[final_indices].view(B, 36, -1)  # [B,36,1024]

        # 6) MI
        patches_selected_i = patches_selected.long()
        best_match_patches_i = best_match_patches.long()
        mi_values = self.compute_mi_tensor(patches_selected_i, best_match_patches_i)
        # mi_values = self.compute_mi_tensor(patches_selected, best_match_patches)  # [B,36]
        #mi_values = torch.nan_to_num(mi_values, nan=0.0)
        mi_values[torch.isnan(mi_values)] = 0.0
        # mi_values = torch.where(torch.isnan(mi_values), torch.zeros_like(mi_values), mi_values)

        return torch.log1p(mi_values)



# ==============================================================================
# 2. 검증된 MI 모델 (DoseResNet) 구조
# ==============================================================================
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.block(x))

class DoseResNet(nn.Module):
    def __init__(self, input_dim=36): 
        super(DoseResNet, self).__init__()
        self.stem = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU()
        )
        self.res_blocks = nn.Sequential(
            ResidualBlock(512), ResidualBlock(512), ResidualBlock(512)
        )
        self.feature_extractor = nn.Sequential(
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.1)
        )
        self.head = nn.Linear(128, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.res_blocks(x)
        feat = self.feature_extractor(x)
        return feat

# ==============================================================================
# 3. 하이브리드 Dose Estimator (MI + Attention)
# ==============================================================================
class HybridDoseEstimator(nn.Module):
    def __init__(self, mi_dim=36):
        super().__init__()
        
        # [A] MI Branch (Pre-trained Weight Load용)
        self.mi_model = DoseResNet(input_dim=mi_dim)
        
        # [B] Attention Branch (400x400 처리)
        self.attn_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), 
            nn.Flatten() # [B, 128]
        )
        
        # [C] Fusion
        self.fusion = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.head_dose = nn.Linear(128, 1)

    def forward(self, mi_vec, attn_score):
        # 1. MI Feature
        # attn_score: [Batch, 4, 352, 352] (4D Tensor)
        feat_mi = self.mi_model(mi_vec)
        #print("aa",feat_mi.shape) # [16, 128]
        # 2. Attn Feature
        if attn_score.dim() == 3: 
            attn_score = attn_score.unsqueeze(1)
        feat_attn = self.attn_encoder(attn_score)
        #print("bb",feat_attn.shape) # [16, 128]
        # 3. Fusion
        combined = torch.cat([feat_mi, feat_attn], dim=-1)
        dose_context = self.fusion(combined)
        #print("cc",dose_context.shape) # [16, 128]
        # 4. Predict
        pred_dose = self.head_dose(dose_context)
        
        return dose_context, pred_dose

# ==============================================================================
# 4. FiLM & TransSino Components
# ==============================================================================
class DoseFiLM(nn.Module):
    def __init__(self, feature_dim, context_dim=129):
        super().__init__()
        self.film_gen = nn.Sequential(
            nn.Linear(context_dim, feature_dim * 2),
            nn.GELU()
        )
    def forward(self, x, context, pred_dose):
        if pred_dose.dim() == 1: pred_dose = pred_dose.unsqueeze(-1)
        inp = torch.cat([context, pred_dose], dim=-1)
        params = self.film_gen(inp)
        gamma, beta = torch.chunk(params, 2, dim=-1)
        return (1.0 + gamma.unsqueeze(1)) * x + beta.unsqueeze(1)

# ==============================================================================
# 4. DRC-FiLM (paper-aligned)
#   DRC-FiLM_l(A_l | d_hat, c) = (1 + Δγ_l(d_hat)) ⊙ A_l + β_l(c)
#   - Δγ: depends ONLY on dose
#   - β : depends on c = [dose_context; dose]
# ==============================================================================

class DRCFiLM(nn.Module):
    def __init__(self, feature_dim: int, context_dim: int = 128, hidden: int = 256):
        super().__init__()
        # Δγ_l(d_hat): R^1 -> R^{C_l}
        self.gamma_net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, feature_dim),
        )
        # β_l(c): R^{128+1} -> R^{C_l}
        self.beta_net = nn.Sequential(
            nn.Linear(context_dim + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, feature_dim),
        )

    def forward(self, A_l: torch.Tensor, dose_context: torch.Tensor, pred_dose: torch.Tensor):
        """
        A_l: [B, T, C]  (prior-attention output to be modulated)
        dose_context: [B, 128]
        pred_dose: [B, 1] (we treat this as log-dose or log1p-dose depending on your loss)
        """
        if pred_dose.dim() == 1:
            pred_dose = pred_dose.unsqueeze(-1)

        # Δγ(d_hat) : [B, C]
        delta_gamma = self.gamma_net(pred_dose) # [B, C]

        # β(c=[z; d_hat]) : [B, C]
        c = torch.cat([dose_context, pred_dose], dim=-1) # [B, 129]
        beta = self.beta_net(c) # [B, C]
        # A_l = h_restored: [B, T, C]
        # pred_dose: [B, 1]
        # dose_context: [B, 128]
        # delta_gamma: [B, C]
        # beta: [B, C]
        # broadcast over token dimension T
        return (1.0 + delta_gamma.unsqueeze(1)) * A_l + beta.unsqueeze(1)

@dataclass
class ModelArgs:
    dim: int = 4096 # 실제로는 352 사용 trans_train_limit.py 에서 덮어씀
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256 
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000
    num_priors: int = 400
    max_batch_size: int = 32
    max_seq_len: int = 2048


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs) 
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


# class Attention(nn.Module):
#     def __init__(self, args: ModelArgs):
#         super().__init__()
#         self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
#         model_parallel_size = 1
#         self.n_local_heads = args.n_heads // model_parallel_size
#         self.head_dim = args.dim // args.n_heads

#         self.wq = nn.Linear(
#             args.dim,
#             args.n_heads * self.head_dim,
#             bias=False,
#         )
#         self.wk = nn.Linear(
#             args.dim,
#             args.n_heads * self.head_dim,
#             bias=False,
#         )
#         self.wv = nn.Linear(
#             args.dim,
#             args.n_heads * self.head_dim,
#             bias=False,
#         )
#         self.wo = nn.Linear(
#             args.n_heads * self.head_dim,
#             args.dim,
#             bias=False,
#         )


#     def forward(
#         self,
#         x: torch.Tensor,
#         start_pos: int,
#         freqs_cis: torch.Tensor,
#     ):
#         bsz, seqlen, _ = x.shape
#         xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        
#         xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        
#         xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
#         xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim) 
        
#         xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
#         keys = xk
#         values = xv
#         xq = xq.transpose(1, 2) 
#         keys = keys.transpose(1, 2) 
#         values = values.transpose(
#             1, 2
#         )
#         scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
#         scores = F.softmax(scores.float(), dim=-1).type_as(xq)
#         output = torch.matmul(scores, values) 
#         output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
#         return self.wo(output)

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wo = nn.Linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        k: torch.Tensor = None, # [추가] 외부 Key (Prior/Cross Attention용)
        v: torch.Tensor = None, # [추가] 외부 Value (Prior/Cross Attention용)
        start_pos: int = 0,     # [유지] 기존 인자 호환성 유지
        freqs_cis: torch.Tensor = None,
    ):
        bsz, seqlen, _ = x.shape
        
        # 1. Q Projection (항상 x 사용)
        xq = self.wq(x)
        
        # 2. K, V Projection (Cross Attention 지원)
        # k, v 인자가 들어오면 그것을 변환(Cross), 안 들어오면 x를 변환(Self)
        if k is not None and v is not None:
            xk = self.wk(k)
            xv = self.wv(v)
            # Cross Attention인 경우 Key/Value의 시퀀스 길이가 Query와 다를 수 있음
            seqlen_k = xk.shape[1]  # Top-K * 352
        else:
            xk = self.wk(x)
            xv = self.wv(x)
            seqlen_k = seqlen

        # 3. Reshape for Multi-head
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        # Reshape 시 가변적인 seqlen_k를 수용하도록 -1 사용
        xk = xk.view(bsz, -1, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, -1, self.n_local_heads, self.head_dim)
        # xk = xk.view(bsz, seqlen_k, self.n_local_heads, self.head_dim)
        # xv = xv.view(bsz, seqlen_k, self.n_local_heads, self.head_dim) 

        # 4. RoPE (Optional)
        # Cross Attention 상황에서는 보통 Positional Encoding을 Key/Value에 적용하지 않거나
        # 적용하더라도 차원이 맞아야 함. 여기서는 freqs_cis가 있고 Self-Attn일 때만 적용하도록 안전장치
        if freqs_cis is not None and seqlen == seqlen_k:
             xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # 5. Attention Score Calculation
        keys = xk.transpose(1, 2)   # [B, H, Seq_K, D]
        values = xv.transpose(1, 2) # [B, H, Seq_K, D]
        query = xq.transpose(1, 2)  # [B, H, Seq_Q, D]

        # Score 계산
        scores = torch.matmul(query, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        # Softmax
        attn_weights = F.softmax(scores.float(), dim=-1).type_as(query)
        
        # 6. Output Calculation
        output = torch.matmul(attn_weights, values) # [B, H, Seq_Q, D]
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        
        # [중요] Output과 함께 attn_weights(점수)도 반환하도록 변경
        # 기존 TransformerBlock에서는 output[0]만 쓰면 되고, 
        # posTransformerBlock에서는 output[1](attn_weights)을 Dose 예측에 사용함
        return self.wo(output), attn_weights

class Positional_Attention(nn.Module):
    def __init__(self, args: ModelArgs, num_priors: int):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.temperature = nn.Parameter(torch.ones(1, 1, 1))
        self.wq = nn.Conv2d(
            in_channels=num_priors, 
            out_channels=num_priors,
            kernel_size=8,   
            stride=8,        
            padding=0,
            groups=num_priors
        )
        self.pos_wk = nn.Conv2d(
            in_channels=1,
            out_channels=num_priors,
            kernel_size=8,
            stride=8,
            padding=0,
        )
        
        self.pos_wv = nn.Conv2d(
            in_channels=1,
            out_channels=num_priors,
            kernel_size=8,
            stride=8,
            padding=0,
        )
        
        self.upsample_layer = nn.Upsample(size=(args.dim, args.dim), mode='bilinear', align_corners=False)
        self.project_out = nn.Conv2d(
            in_channels=num_priors,
            out_channels=1,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1)
        )

    def forward(
        self,
        x: torch.Tensor,
        pos_kv: torch.Tensor,
    ):
        bsz, seqlen, _ = x.shape
        
        x = x.unsqueeze(1)

        b, c, h, w = pos_kv.shape
        h = h // 8
        w = w // 8
        
        xq = self.wq(pos_kv)
        
        xk = self.pos_wk(x)
        xv = self.pos_wv(x)
        
        xq = rearrange(xq, 'b c h w -> b c (h w)')
        xk = rearrange(xk, 'b c h w -> b c (h w)')
        xv = rearrange(xv, 'b c h w -> b c (h w)')

        xq = torch.nn.functional.normalize(xq, dim=-1)
        xk = torch.nn.functional.normalize(xk, dim=-1)

        attn_score = (xq @ xk.transpose(-2, -1)) * self.temperature
        attn = attn_score.softmax(dim=-1)

        out = (attn @ xv)
        
        out = rearrange(out, 'b c (h w) -> b c h w', h=h, w=w)
        # 여기 out shape 찍어보기
        # print(out.shape)
        out = self.project_out(self.upsample_layer(out)).squeeze(1)
        # print(out.shape)
        return out, attn_score









class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)

        self.w1 = nn.Linear(
            dim, hidden_dim, bias=False
        )
        self.w2 = nn.Linear(
            hidden_dim, dim, bias=False
        )
        self.w3 = nn.Linear(
            dim, hidden_dim, bias=False
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        #self.pos_attn = Positional_Attention(args, num_priors)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    # def forward(
    #     self,
    #     x: torch.Tensor,
    #     start_pos: int,
    #     freqs_cis: torch.Tensor,
    # ):
    #     # print(torch.isnan(x).any())
    #     h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis)
    #     out = h + self.feed_forward(self.ffn_norm(h))
    #     return out
    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
    ):
        # attention이 이제 (output, score) 튜플을 반환하므로 [0] 인덱싱 필요
        attn_out, _ = self.attention(self.attention_norm(x), start_pos=start_pos, freqs_cis=freqs_cis)
        
        h = x + attn_out
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class posTransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs, num_priors: int):
        super().__init__()
        self.layer_id = layer_id
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.num_priors = num_priors
        self.attention = Attention(args)
        self.pos_attn = Positional_Attention(args, num_priors)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # ------------------------------------------------------------------
        # [추가 1] Hybrid Estimator 초기화
        self.dose_estimator = HybridDoseEstimator(mi_dim=36)
        #self.film_layer = DoseFiLM(feature_dim=args.dim)
        self.film_layer = DRCFiLM(feature_dim=args.dim, context_dim=128)
        # # [2] [추가] 사전 학습된 MI 가중치 로드 (Partial Loading)
        # # 파일명이 정확한지 확인하세요 (같은 폴더에 있어야 함)
        # pth_path = 'best_dose_regressor_log(36mi)_512_.pth' 
        
        # if os.path.exists(pth_path):
        #     try:
        #         # 1. 체크포인트 로드
        #         pretrained_dict = torch.load(pth_path)
        #         model_dict = self.dose_estimator.state_dict()
                
        #         # 2. 키 매핑 (Key Mapping)
        #         # 저장된 가중치 키(예: 'stem.0.weight')를 
        #         # 현재 모델의 서브모듈 키(예: 'mi_model.stem.0.weight')로 변경
        #         new_dict = {}
        #         for k, v in pretrained_dict.items():
        #             new_key = 'mi_model.' + k 
        #             if new_key in model_dict:
        #                 new_dict[new_key] = v
                
        #         # 3. 매칭된 가중치만 업데이트 (strict=False 필수)
        #         self.dose_estimator.load_state_dict(new_dict, strict=False)
        #         print(f"[Layer {layer_id}] ✅ Loaded pre-trained MI weights! ({len(new_dict)} keys matched)")
                
        #         # (선택) MI Branch를 고정(Freeze)하고 싶다면 아래 주석 해제
        #         # for param in self.dose_estimator.mi_model.parameters():
        #         #     param.requires_grad = False
                
        #     except Exception as e:
        #         print(f"[Layer {layer_id}] ⚠️ Error loading weights: {e}")
        # else:
        #     print(f"[Layer {layer_id}] ⚠️ Pre-trained weight file not found: {pth_path}")

        # [추가 3] FiLM 레이어
        # ------------------------------------------------------------------

    # def forward(
    #     self,
    #     x: torch.Tensor,
    #     pos_kv: torch.Tensor,
    #     mi_vec,
    # ):   
    #     h_restored, attn_score = self.pos_attn(self.attention_norm(x), pos_kv)
        
    #     # [Step 2] Dose 추정 및 FiLM 준비
    #     dose_context, pred_dose = self.dose_estimator(mi_vec, attn_score)
        
    #     # [Step 3] 단순 잔차 연결 (알파 제거)
    #     h = x + h_restored
    #     h_norm = self.ffn_norm(h)
    #     h_modulated = self.film_layer(h_norm, dose_context, pred_dose)
    #     out = h + self.feed_forward(h_modulated)
    #     # [Step 4] FFN 및 FiLM 보정
    #     # out = h + self.feed_forward(self.ffn_norm(h))
    #     # out = self.film_layer(out, dose_context, pred_dose)
    #     return out, attn_score, pred_dose
    def forward(self, x: torch.Tensor, pos_kv: torch.Tensor, mi_vec):
        # (1) prior-attn:  Ã_l = Attn_prior(LN(F_{l-1}), L)
        h_restored, attn_score = self.pos_attn(self.attention_norm(x), pos_kv)

        # (2) SRDE: dose_context z and predicted dose d_hat
        dose_context, pred_dose = self.dose_estimator(mi_vec, attn_score)

        # (3) DRC-FiLM on prior-attn output BEFORE residual fusion:
        #     Ã_l <- DRC-FiLM_l(Ã_l | d_hat, c)
        h_restored = self.film_layer(h_restored, dose_context, pred_dose)

        # (4) residual fusion:
        #     F_l = F_{l-1} + Ã_l
        h = x + h_restored

        # (5) FFN:
        #     F_l <- F_l + FFN(LN(F_l))
        out = h + self.feed_forward(self.ffn_norm(h))

        return out, attn_score, pred_dose


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs, configs=None):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.tok_embeddings1 = nn.Linear(in_features=params.dim, out_features=params.dim, bias=True)
        # (pos_data_loader.py에 따르면 shape는 [1, 400, 352, 352] 입니다)
        #prior_sino_data = np.load("pos_data/prior_sino.npy")
        # [기존 코드 복구] 400개 패턴이 담긴 prior_sino.npy를 직접 로드합니다.
        
        prior_sino_data = np.load("pos_data/prior_sino_bank.npy") # [1, 400, 352, 352]
        self.num_priors = prior_sino_data.shape[1] # 400개
        self.learnable_prior_patterns = nn.Parameter(torch.from_numpy(prior_sino_data).float())
        print(f"[Init] Loaded original Prior Sino with {self.num_priors} patterns.")


        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
            # posTransformerBlock에 num_priors 인자 추가
            n_priors = 400 if layer_id < 4 else 625
            self.layers.append(posTransformerBlock(layer_id, params, num_priors=n_priors))

        self.pos_embeddings = nn.Linear(in_features=params.dim, out_features=params.dim, bias=True)
        self.configs = configs
        # self.layers = torch.nn.ModuleList()
        # for layer_id in range(params.n_layers):
        #     self.layers.append(TransformerBlock(layer_id, params))
        #     self.layers.append(posTransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        )
        self.output1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=3,
            padding=1,
        )  
        self.output2 = nn.Conv2d(
            in_channels=64,
            out_channels=1,
            kernel_size=3,
            padding=1,
        )  
        
        self.seg_net = ViT_seg(self.configs, img_size=self.configs.img_size, num_classes=self.configs.n_classes).cuda()
        
        # [New] MI Processor (Initialize once)
        self.mi_processor = RealTimeMIProcessor()
        self.init_dose_net = DoseResNet(input_dim=36)

        # [중요] '정확도 8%' 성능을 내려면 사전 학습된 가중치(pth)를 반드시 로드해야 합니다.
        pth_path = 'best_dose_regressor_log(36mi)_512_.pth'
        if os.path.exists(pth_path):
            try:
                # 저장된 가중치 로드
                pretrained_dict = torch.load(pth_path)
                model_dict = self.init_dose_net.state_dict()
                
                # 키 매칭 (저장된 키가 'stem...', 현재 키도 'stem...' 이므로 그대로 매칭 가능)
                # 만약 저장된 키 앞에 'module.' 등이 붙어있다면 제거하는 로직 필요
                new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                
                self.init_dose_net.load_state_dict(new_dict, strict=False)
                print(f"[Transformer Init] ✅ Loaded pre-trained DoseResNet weights! ({len(new_dict)} keys)")
                
            except Exception as e:
                print(f"[Transformer Init] ⚠️ Error loading DoseResNet: {e}")
        else:
            print(f"[Transformer Init] ⚠️ Pre-trained weight not found. Initializing randomly.")

    def radon(self, sinogram, num_view=352):
        image_size = 300
        detector_count = 352
        angles = torch.linspace(0, np.pi, num_view, device='cuda')
        
        radon = torch_radon.Radon(
            image_size,
            det_count=detector_count,
            angles=angles,
        )
        
        filterd_sino = radon.filter_sinogram(sinogram)
        recon_img = radon.backprojection(filterd_sino)
        
        return recon_img
    
    #def forward(self, tokens: torch.Tensor, start_pos: int, pos_kv: torch.Tensor, min, max):
    def forward(self, tokens: torch.Tensor, start_pos: int, min, max):
        _bsz, seqlen, _ = tokens.shape
        # 1. MI Calculation (Input Sinogram -> MI Vector)
        # tokens는 [B, 720, 768] 형태의 Sinogram 데이터
        #print("tokens:", tokens.shape)
        mi_vec = self.mi_processor(tokens) # [B, 36]
        #print("mi_vec:", mi_vec.shape)

        # pos_kv_all shape: [Batch, 1300, 352, 352]
        # pos_kv_all = self.learnable_prior_patterns.expand(_bsz, -1, -1, -1)
        # pos_kv_all = self.pos_embeddings(pos_kv_all)
        pos_kv_all = self.pos_embeddings(self.learnable_prior_patterns)
        pos_kv_all = pos_kv_all.expand(_bsz, -1, -1, -1) # [16, 1300, 352, 352]
        # [수정] DoseResNet을 사용하여 정교한 초기값(curr_dose) 생성
        # 1) DoseResNet 피처 추출 (128차원)
        init_feat = self.init_dose_net(mi_vec) 
        
        # 2) Head를 통과시켜 최종 Dose 값(1차원) 예측
        curr_dose = self.init_dose_net.head(init_feat)

        h = self.tok_embeddings1(tokens)
        #print("h shape", h.shape)
        # 2. 외부 입력 대신, 내부의 '학습 가능한 파라미터'를 사용합니다.
        #    현재 배치 사이즈(_bsz)에 맞게 expand 합니다.
        # pos_kv = self.learnable_prior_patterns.expand(_bsz, -1, -1, -1)
        # pos_kv = self.pos_embeddings(pos_kv)
        
        # 예측된 dose를 저장할 리스트 (Layer가 여러 개일 경우 대비)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        pred_doses = []
        # for layer in self.layers:
        #     h = layer(h, pos_kv, start_pos, freqs_cis)


        # for layer in self.layers:
        #     if isinstance(layer, TransformerBlock):
        #         h = layer(h, start_pos, freqs_cis)
        #     elif isinstance(layer, posTransformerBlock):
        #         h, attn_score, pred_dose = layer(h, pos_kv, mi_vec)
        #         pred_doses.append(pred_dose)
        #     else:
        #         raise TypeError(f"Unexpected layer type: {type(layer)}")

        for i, layer in enumerate(self.layers):
            if isinstance(layer, TransformerBlock):
                h = layer(h, start_pos, freqs_cis)
            else:
                # 계층적 슬라이싱 (i는 루프 index이므로 layer_id 환산 필요)
                layer_id = i // 2
                if layer_id < 4:
                    # 400개만 따로 복제하여 새로운 메모리에 할당
                    current_kv = pos_kv_all[:, :400].clone() 
                else:
                    # 625개만 따로 복제 (원본과의 메모리 참조를 끊음)
                    current_kv = pos_kv_all[:, 400:400+625].clone()
                # if layer_id < 4:
                #     current_kv = pos_kv_all[:, :400]              # 15-patch bank
                # else:
                #     current_kv = pos_kv_all[:, 400:400+625]       # 12-patch bank (625개)
                h, attn_score, layer_dose = layer(h, current_kv, mi_vec)
                #print("attn_score:", attn_score.shape)
                pred_doses.append(layer_dose)
        h_r = self.norm(h)
        raw_output = self.output2(self.output1(h_r.unsqueeze(1))).squeeze(1) # 사이노그램
        
        # 1) h를 이미지 형태(4D)로 변환: [Batch, 1, 720, 352]
        h_img = h_r.unsqueeze(1)     
        # 2) seg_net 입력 크기(352, 352)로 리사이징
        #    (ResNet 백본은 352x352 입력을 기대함)
        h_resized = F.interpolate(h_img, size=(352, 352), mode='bilinear', align_corners=False)
        # 3) Segmentation 수행 (결과: [Batch, Num_Classes, 352, 352])
        seg_out_small = self.seg_net(h_resized)
        # 4) 결과를 원래 사이노그램 크기(720, 768)로 복원
        #    그래야 학습 코드에서 radon() 함수를 통해 이미지 마스크로 변환 가능
        target_H, target_W = tokens.shape[1], tokens.shape[2]
        # 만약 352가 아니라면 자동으로 맞추기 위해 변수 사용
        seg_output = F.interpolate(seg_out_small, size=(target_H, target_W), mode='bilinear', align_corners=False)
        #print("raw_output max:",raw_output.max(),"  raw_output min", raw_output.min()) 정규화 x 0~16, 0~4 등등 다양
        #seg_output = self.seg_net(h.unsqueeze(1))  

        # Output Normalization
        op_min = raw_output.amin(dim=(-2, -1), keepdim=True)
        op_max = raw_output.amax(dim=(-2, -1), keepdim=True)
        epsilon = 1e-8

        output = (raw_output - op_min) / (op_max - op_min + epsilon) # 사이노그램
        output1 = self.radon((output * (max - min) + min)) # 이미지
        #print("output1 max:",output1.max(),"  output1 min", output1.min()) #-0.1~0.2, -0.1~0.8 등등 다양
        
        # 마지막 Layer의 dose 예측값 반환 (또는 평균)
        final_pred_dose = pred_doses[-1] if pred_doses else None
        return output1, output, seg_output, h, attn_score, final_pred_dose