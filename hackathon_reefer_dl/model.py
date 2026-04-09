from __future__ import annotations

import torch
from torch import nn


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.pad,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.pad:
            out = out[..., :-self.pad]
        return out


class ResidualTCNBlock(nn.Module):
    def __init__(self, hidden_dim: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, dilation=dilation)
        self.conv2 = CausalConv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.norm2 = nn.BatchNorm1d(hidden_dim)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.activation(out)
        out = self.dropout(out)
        return residual + out


class ReeferTCNForecaster(nn.Module):
    def __init__(
        self,
        input_dim: int,
        target_calendar_dim: int,
        hidden_dim: int = 128,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.tcn = nn.ModuleList(
            [
                ResidualTCNBlock(
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
                for dilation in dilations
            ]
        )
        self.attention = nn.Linear(hidden_dim, 1)
        self.target_embedding = nn.Sequential(
            nn.Linear(target_calendar_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim // 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, sequence_features: torch.Tensor, target_calendar: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(sequence_features)
        x = self.input_norm(x)
        x = x.transpose(1, 2)
        for block in self.tcn:
            x = block(x)
        x = x.transpose(1, 2)
        attention_weights = torch.softmax(self.attention(x).squeeze(-1), dim=1)
        pooled = torch.sum(x * attention_weights.unsqueeze(-1), dim=1)
        last_step = x[:, -1, :]
        target_emb = self.target_embedding(target_calendar)
        features = torch.cat([pooled, last_step, target_emb], dim=1)
        return self.head(features).squeeze(-1)

