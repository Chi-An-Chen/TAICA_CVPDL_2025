"""
Author: Chi-An Chen
Date: 2025-11-10
Description: utils for diffusion model
"""
import math
import torch
import torch.nn as nn
from typing import Dict

def ddpm_schedules(beta1: float, beta2: float, T: int) -> Dict[str, torch.Tensor]:
    """
    返回 DDPM 採樣和訓練過程所需的預計算排程
    """
    assert beta1 < beta2 < 1.0, "beta1 和 beta2 必須在 (0, 1) 之間"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,
        "oneover_sqrta": oneover_sqrta,
        "sqrt_beta_t": sqrt_beta_t,
        "alphabar_t": alphabar_t,
        "sqrtab": sqrtab,
        "sqrtmab": sqrtmab,
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,
    }

def ddpm_schedules_cosine(T: int, s: float = 0.008) -> Dict[str, torch.Tensor]:
    """
    使用 cosine schedule (Improved DDPM, Nichol & Dhariwal 2021)
    """
    steps = T + 1
    t = torch.linspace(0, T, steps, dtype=torch.float32)
    
    # Cosine schedule for alpha_bar
    f_t = torch.cos(((t / T + s) / (1 + s)) * torch.pi / 2) ** 2
    alphabar_t = f_t / f_t[0]
    
    # 計算其他相關量
    alpha_t = torch.cat([torch.tensor([1.0]), alphabar_t[1:] / alphabar_t[:-1]])
    alpha_t = torch.clamp(alpha_t, 0.0001, 0.9999)  # 避免極端值
    
    beta_t = 1 - alpha_t
    sqrt_beta_t = torch.sqrt(beta_t)
    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)
    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab = (1 - alpha_t) / sqrtmab
    
    return {
        "alpha_t": alpha_t,
        "oneover_sqrta": oneover_sqrta,
        "sqrt_beta_t": sqrt_beta_t,
        "alphabar_t": alphabar_t,
        "sqrtab": sqrtab,
        "sqrtmab": sqrtmab,
        "mab_over_sqrtmab": mab_over_sqrtmab,
    }

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    創建正弦時間步嵌入
    :param timesteps: 1-D 張量，包含時間步索引
    :param dim: 輸出的嵌入維度
    :param max_period: 控制嵌入的最小頻率
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

def normalization(channels):
    """
    創建標準化層（GroupNorm）
    """
    return nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)

def zero_module(module):
    """
    將模組的參數初始化為零
    """
    for p in module.parameters():
        p.detach().zero_()
    return module



