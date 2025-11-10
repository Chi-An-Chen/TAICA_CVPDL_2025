"""
Author: Chi-An Chen
Date: 2025-11-10
Description: UNet, DDPM model with consine schedules
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple
from model.utils import ddpm_schedules, timestep_embedding, normalization, zero_module


class Upsample(nn.Module):
    """
    上採樣層，可選擇使用卷積
    """
    def __init__(self, channels, use_conv=True, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    下採樣層，可選擇使用卷積
    """
    def __init__(self, channels, use_conv=True, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if use_conv:
            self.op = nn.Conv2d(self.channels, self.out_channels, 3, stride=2, padding=1)
        else:
            self.op = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(nn.Module):
    """
    殘差塊，可以選擇性地改變通道數量
    支援時間步嵌入的條件化
    """
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_scale_shift_norm=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False)
            self.x_upd = Upsample(channels, False)
        elif down:
            self.h_upd = Downsample(channels, False)
            self.x_upd = Downsample(channels, False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        對張量應用區塊，條件化於時間步嵌入
        :param x: [N x C x H x W] 特徵張量
        :param emb: [N x emb_channels] 時間步嵌入張量
        """
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        
        return self.skip_connection(x) + h


class UNetModel(nn.Module):
    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        channel_mult=(1, 2, 4, 8),
        dropout=0,
        use_scale_shift_norm=False,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.channel_mult = channel_mult
        self.dropout = dropout

        # 時間步嵌入
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # 輸入區塊（下採樣路徑）
        self.input_blocks = nn.ModuleList([
            nn.ModuleList([nn.Conv2d(in_channels, model_channels, 3, padding=1)])
        ])
        
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                self.input_blocks.append(nn.ModuleList(layers))
                input_block_chans.append(ch)
            
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    nn.ModuleList([
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                    ])
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2

        # 中間區塊
        self.middle_block = nn.ModuleList([
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        ])

        # 輸出區塊（上採樣路徑）
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                    )
                    ds //= 2
                
                self.output_blocks.append(nn.ModuleList(layers))

        # 輸出層
        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(nn.Conv2d(model_channels, out_channels, 3, padding=1)),
        )

    def forward(self, x, timesteps):
        """
        對輸入批次應用模型
        :param x: [N x C x H x W] 輸入張量
        :param timesteps: [N] 時間步張量
        """
        hs = []
        
        # 時間步嵌入
        t_emb = timestep_embedding(timesteps, self.model_channels)
        emb = self.time_embed(t_emb)

        # 下採樣路徑
        h = x
        for module_list in self.input_blocks:
            for layer in module_list:
                if isinstance(layer, ResBlock):
                    h = layer(h, emb)
                else:
                    h = layer(h)
            hs.append(h)

        # 中間區塊
        for layer in self.middle_block:
            h = layer(h, emb)

        # 上採樣路徑
        for module_list in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            for layer in module_list:
                h = layer(h, emb)

        # 輸出
        return self.out(h)


class DDPM(nn.Module):
    def __init__(
        self,
        eps_model: nn.Module,
        betas: Tuple[float, float],
        n_T: int,
        criterion: nn.Module = nn.SmoothL1Loss(beta=0.03),
        schedule_type: str = "cosine",  # 新增參數: "linear" 或 "cosine"
    ) -> None:
        super(DDPM, self).__init__()
        self.eps_model = eps_model
        
        # 根據類型選擇 schedule
        if schedule_type == "cosine":
            schedules = ddpm_schedules_cosine(n_T)
        else:  # linear
            schedules = ddpm_schedules(betas[0], betas[1], n_T)
        
        for k, v in schedules.items():
            self.register_buffer(k, v)
        
        self.n_T = n_T
        self.criterion = criterion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        進行前向擴散得到 x_t，並使用 eps_model 預測噪聲
        實現 Algorithm 1
        """
        _ts = torch.randint(1, self.n_T + 1, (x.shape[0],)).to(x.device)
        eps = torch.randn_like(x)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * eps
        )
        
        # 預測噪聲
        predicted_eps = self.eps_model(x_t, _ts)
        
        return self.criterion(eps, predicted_eps)

    def sample(self, n_sample: int, size: Tuple, device: str) -> torch.Tensor:
        """
        根據 Algorithm 2 進行採樣
        """
        x_i = torch.randn(n_sample, *size).to(device)

        for i in range(self.n_T, 0, -1):
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            
            t = torch.tensor([i]).to(device).repeat(n_sample)
            eps = self.eps_model(x_i, t)
            
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )

        return x_i


