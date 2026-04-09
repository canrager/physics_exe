from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


LONG_FEATURES = [
    "label_power_kw",
    "active_rows",
    "unique_containers",
    "unique_visits",
    "distinct_customers",
    "mean_power_per_active_reefer_kw",
]
SHORT_FEATURES = [
    "mean_stack_tier",
    "share_stack_tier_1",
    "share_stack_tier_2",
    "share_stack_tier_3",
    "share_hardware_scc6",
    "share_hardware_decosvb",
    "share_hardware_ml5",
    "share_hardware_decosiiih",
    "share_hardware_decosiiij",
    "share_hardware_decosva",
    "share_hardware_mp4000",
    "share_hardware_ml3",
    "share_hardware_other",
    "mean_temperature_setpoint_c",
    "mean_temperature_ambient_c",
    "mean_temperature_return_c",
    "mean_temperature_supply_c",
    "mean_ambient_minus_setpoint_c",
    "mean_return_minus_supply_c",
]
WEATHER_FEATURES = [
    "weather_temperature_vc_halle3_c",
    "weather_temperature_mean_c",
    "weather_temperature_zentralgate_c",
    "weather_temperature_spread_c",
    "weather_wind_speed_vc_halle3",
    "weather_wind_speed_zentralgate",
    "weather_wind_speed_mean",
    "weather_wind_speed_spread",
    "weather_wind_direction_vc_halle3_cos",
    "weather_wind_direction_mean_cos",
    "weather_wind_direction_zentralgate_cos",
    "weather_wind_direction_zentralgate_sin",
    "weather_wind_direction_vc_halle3_sin",
    "weather_wind_direction_mean_sin",
    "weather_wind_direction_zentralgate_consistency",
    "weather_wind_direction_vc_halle3_consistency",
    "weather_wind_direction_mean_consistency",
]
STATIC_CONTINUOUS_COLUMNS = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "month_sin",
    "month_cos",
    "weather_history_expected_feature_count",
    "weather_history_available_feature_count",
    "weather_history_available_fraction",
]
STATIC_BINARY_COLUMNS = [
    "is_weekend",
    "was_gap_shifted",
    "was_dst_adjusted",
    "weather_history_expected",
    "weather_history_complete",
]
CATEGORICAL_COLUMNS = ["hour_of_day", "day_of_week", "month", "season"]
TARGET_COLUMN = "label_power_kw"
TIMESTAMP_COLUMN = "source_timestamp_utc"
LONG_LAGS = list(range(191, 23, -1))
SHORT_LAGS = list(range(71, 23, -1))
WEATHER_LAGS = list(range(86, 23, -1))
PINBALL_QUANTILE = 0.9
LONG_KEYS = [[f"{feature}_tminus{lag}h" for lag in LONG_LAGS] for feature in LONG_FEATURES]
SHORT_KEYS = [[f"{feature}_tminus{lag}h" for lag in SHORT_LAGS] for feature in SHORT_FEATURES]
WEATHER_KEYS = [[f"{feature}_tminus{lag}h" for lag in WEATHER_LAGS] for feature in WEATHER_FEATURES]


@dataclass(frozen=True)
class FoldWindow:
    name: str
    start: str
    end: str
    is_holdout: bool = False


FORWARD_WINDOWS = [
    FoldWindow("fold1", "2025-11-15T00:00:00Z", "2025-11-24T23:00:00Z"),
    FoldWindow("fold2", "2025-12-01T00:00:00Z", "2025-12-10T23:00:00Z"),
    FoldWindow("fold3", "2025-12-11T00:00:00Z", "2025-12-20T23:00:00Z"),
    FoldWindow("holdout", "2025-12-22T00:00:00Z", "2025-12-31T23:00:00Z", is_holdout=True),
]


@dataclass
class ForecastTable:
    source_timestamps: np.ndarray
    sequence_index: np.ndarray
    targets: np.ndarray
    long_history: np.ndarray
    short_history: np.ndarray
    weather_history: np.ndarray
    weather_mask: np.ndarray
    categorical: np.ndarray
    static_continuous: np.ndarray
    static_binary: np.ndarray
    weather_complete: np.ndarray

    @property
    def num_rows(self) -> int:
        return int(self.targets.shape[0])


@dataclass
class PreparedData:
    source_timestamps: np.ndarray
    targets: np.ndarray
    long_history: np.ndarray
    short_history: np.ndarray
    weather_history: np.ndarray
    weather_mask: np.ndarray
    categorical: np.ndarray
    static_continuous: np.ndarray
    static_binary: np.ndarray
    weather_complete: np.ndarray

    @property
    def num_rows(self) -> int:
        return int(self.targets.shape[0])


@dataclass
class StructuredScalers:
    long_center: np.ndarray
    long_scale: np.ndarray
    short_center: np.ndarray
    short_scale: np.ndarray
    weather_center: np.ndarray
    weather_scale: np.ndarray
    static_center: np.ndarray
    static_scale: np.ndarray


@dataclass
class FeatureScaler:
    center: np.ndarray
    scale: np.ndarray


@dataclass
class FoldPrediction:
    model_name: str
    fold_name: str
    source_timestamps: np.ndarray
    targets: np.ndarray
    point_pred: np.ndarray
    raw_uplift: np.ndarray
    p90_pred: np.ndarray
    peak_threshold: float
    stage1_best_epoch: int
    stage2_best_epoch: int


@dataclass
class TrainArtifacts:
    model_state: dict[str, torch.Tensor]
    stage1_best_epoch: int
    stage2_best_epoch: int


@dataclass
class AffineCalibrator:
    alpha: float
    beta: float

    def apply(self, point_pred: np.ndarray, raw_uplift: np.ndarray) -> np.ndarray:
        uplift = np.maximum(self.alpha * np.maximum(raw_uplift, 0.0) + self.beta, 0.0)
        return point_pred + uplift


class TensorForecastDataset(Dataset):
    def __init__(self, data: PreparedData) -> None:
        self.long_history = torch.from_numpy(data.long_history)
        self.short_history = torch.from_numpy(data.short_history)
        self.weather_history = torch.from_numpy(data.weather_history)
        self.weather_mask = torch.from_numpy(data.weather_mask)
        self.categorical = torch.from_numpy(data.categorical)
        self.static_continuous = torch.from_numpy(data.static_continuous)
        self.static_binary = torch.from_numpy(data.static_binary)
        self.targets = torch.from_numpy(data.targets)

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "long_history": self.long_history[index],
            "short_history": self.short_history[index],
            "weather_history": self.weather_history[index],
            "weather_mask": self.weather_mask[index],
            "categorical": self.categorical[index],
            "static_continuous": self.static_continuous[index],
            "static_binary": self.static_binary[index],
            "target": self.targets[index],
        }


class SqueezeExcitation1d(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=-1)
        weights = torch.sigmoid(self.fc2(F.gelu(self.fc1(pooled))))
        return x * weights.unsqueeze(-1)


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.se = SqueezeExcitation1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.se(x)
        x = self.dropout(x)
        return F.gelu(x + residual)


class AttentionPool1d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.score = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.score(x).squeeze(1)
        if mask is not None:
            masked_value = torch.finfo(logits.dtype).min
            logits = logits.masked_fill(~mask, masked_value)
        weights = torch.softmax(logits, dim=-1)
        return torch.sum(x * weights.unsqueeze(1), dim=-1)


class TCNBranch(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilations: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.output_dim = hidden_channels
        self.input_proj = nn.Conv1d(input_channels, hidden_channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [ResidualTCNBlock(hidden_channels, kernel_size, dilation, dropout) for dilation in dilations]
        )
        self.pool = AttentionPool1d(hidden_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.pool(x)


class MaskAwareTemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(channels + 1, channels, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(channels + 1, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, time_mask: torch.Tensor) -> torch.Tensor:
        residual = x
        mask_channel = time_mask.unsqueeze(1).to(dtype=x.dtype)
        x = x * mask_channel
        x = torch.cat([x, mask_channel], dim=1)
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = x * mask_channel
        x = torch.cat([x, mask_channel], dim=1)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.dropout(x)
        return F.gelu((x + residual) * mask_channel)


class TemporalAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        residual = x
        normed = self.norm1(x)
        attn_out, _ = self.attention(normed, normed, normed, key_padding_mask=key_padding_mask, need_weights=False)
        x = residual + attn_out
        x = x + self.feed_forward(self.norm2(x))
        return x


class WeatherBranch(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, dropout: float, channel_dropout: float) -> None:
        super().__init__()
        self.output_dim = hidden_channels
        self.channel_dropout = channel_dropout
        self.input_proj = nn.Conv1d(input_channels * 2, hidden_channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                MaskAwareTemporalBlock(hidden_channels, kernel_size=3, dilation=1, dropout=dropout),
                MaskAwareTemporalBlock(hidden_channels, kernel_size=3, dilation=2, dropout=dropout),
            ]
        )
        self.attention_blocks = nn.ModuleList(
            [
                TemporalAttentionBlock(hidden_channels, num_heads=4, dropout=dropout),
                TemporalAttentionBlock(hidden_channels, num_heads=4, dropout=dropout),
            ]
        )
        self.pool = AttentionPool1d(hidden_channels)

    def forward(self, weather: torch.Tensor, weather_mask: torch.Tensor) -> torch.Tensor:
        if self.training and self.channel_dropout > 0.0:
            keep = (
                torch.rand(weather.shape[0], weather.shape[1], 1, device=weather.device) >= self.channel_dropout
            ).to(dtype=weather.dtype)
            weather = weather * keep
            weather_mask = weather_mask * keep
        weather = weather * weather_mask
        x = torch.cat([weather, weather_mask], dim=1)
        x = self.input_proj(x)
        time_mask = weather_mask.amax(dim=1) > 0.0
        empty_rows = ~time_mask.any(dim=1)
        if empty_rows.any():
            time_mask = time_mask.clone()
            time_mask[empty_rows, -1] = True
        for block in self.blocks:
            x = block(x, time_mask)
        x = x.transpose(1, 2)
        padding_mask = ~time_mask
        for block in self.attention_blocks:
            x = block(x, padding_mask)
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        x = x.transpose(1, 2)
        return self.pool(x, mask=time_mask)


class StaticBranch(nn.Module):
    def __init__(self, num_continuous: int, num_binary: int) -> None:
        super().__init__()
        self.hour_embedding = nn.Embedding(24, 8)
        self.weekday_embedding = nn.Embedding(7, 4)
        self.month_embedding = nn.Embedding(12, 4)
        self.season_embedding = nn.Embedding(4, 3)
        input_dim = 8 + 4 + 4 + 3 + num_continuous + num_binary
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 32),
        )
        self.output_dim = 32

    def forward(
        self,
        categorical: torch.Tensor,
        static_continuous: torch.Tensor,
        static_binary: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = [
            self.hour_embedding(categorical[:, 0]),
            self.weekday_embedding(categorical[:, 1]),
            self.month_embedding(categorical[:, 2]),
            self.season_embedding(categorical[:, 3]),
        ]
        x = torch.cat(embeddings + [static_continuous, static_binary], dim=-1)
        return self.mlp(x)


class GatedResidualFusion(nn.Module):
    def __init__(self, input_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.residual = nn.Linear(input_dim, 64)
        self.gate = nn.Sequential(nn.Linear(input_dim, 64), nn.Sigmoid())
        self.main = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return self.residual(x) + self.gate(x) * self.main(x)


class ReeferDemandForecaster(nn.Module):
    def __init__(
        self,
        input_noise_std: float,
        weather_channel_dropout: float,
    ) -> None:
        super().__init__()
        self.input_noise_std = input_noise_std
        self.long_branch = TCNBranch(len(LONG_FEATURES), 64, kernel_size=5, dilations=[1, 2, 4, 8], dropout=0.1)
        self.short_branch = TCNBranch(len(SHORT_FEATURES), 48, kernel_size=3, dilations=[1, 2, 4], dropout=0.1)
        self.weather_branch = WeatherBranch(
            input_channels=len(WEATHER_FEATURES),
            hidden_channels=48,
            dropout=0.1,
            channel_dropout=weather_channel_dropout,
        )
        self.static_branch = StaticBranch(
            num_continuous=len(STATIC_CONTINUOUS_COLUMNS),
            num_binary=len(STATIC_BINARY_COLUMNS),
        )
        fused_dim = self.long_branch.output_dim + self.short_branch.output_dim + self.weather_branch.output_dim + self.static_branch.output_dim
        self.fusion = GatedResidualFusion(fused_dim, dropout=0.15)
        self.point_head = nn.Linear(64, 1)
        self.uplift_head = nn.Linear(64, 1)

    def _maybe_add_noise(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.input_noise_std > 0.0:
            x = x + torch.randn_like(x) * self.input_noise_std
        return x

    def forward(
        self,
        long_history: torch.Tensor,
        short_history: torch.Tensor,
        weather_history: torch.Tensor,
        weather_mask: torch.Tensor,
        categorical: torch.Tensor,
        static_continuous: torch.Tensor,
        static_binary: torch.Tensor,
        use_weather: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        long_repr = self.long_branch(self._maybe_add_noise(long_history))
        short_repr = self.short_branch(self._maybe_add_noise(short_history))
        static_repr = self.static_branch(
            categorical,
            self._maybe_add_noise(static_continuous),
            static_binary,
        )
        if use_weather:
            weather_repr = self.weather_branch(self._maybe_add_noise(weather_history), weather_mask)
        else:
            weather_repr = torch.zeros(
                long_repr.shape[0],
                self.weather_branch.output_dim,
                device=long_repr.device,
                dtype=long_repr.dtype,
            )
        fused = self.fusion(torch.cat([long_repr, short_repr, weather_repr, static_repr], dim=-1))
        point_pred = self.point_head(fused).squeeze(-1)
        raw_uplift = F.softplus(self.uplift_head(fused).squeeze(-1))
        return point_pred, raw_uplift


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train and evaluate the deep-learning reefer forecaster.")
    parser.add_argument(
        "--trainval-csv",
        type=Path,
        default=repo_root / "outputs" / "preprocessed_dataset" / "trainval_hourly.csv",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=repo_root / "outputs" / "preprocessed_dataset" / "test_hourly.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "outputs" / "deep_learning_forecast",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--stage1-epochs", type=int, default=80)
    parser.add_argument("--stage2-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--stage1-batch-size", type=int, default=256)
    parser.add_argument("--stage2-batch-size", type=int, default=128)
    parser.add_argument("--stage1-lr", type=float, default=1e-3)
    parser.add_argument("--stage2-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ridge-alpha", type=float, default=20.0)
    parser.add_argument("--input-noise-std", type=float, default=0.01)
    parser.add_argument("--weather-channel-dropout", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 23, 41])
    parser.add_argument("--quick-run", action="store_true")
    return parser.parse_args()


def parse_float(value: str | None) -> float:
    if value is None:
        return float("nan")
    stripped = value.strip()
    if not stripped:
        return float("nan")
    if stripped.upper() == "NULL":
        return float("nan")
    if stripped.lower() == "nan":
        return float("nan")
    return float(stripped)


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate_cuda_runtime(device: torch.device, requested: str) -> torch.device:
    if device.type != "cuda":
        return device
    try:
        probe = torch.randn(8, device=device)
        _ = probe + torch.randn_like(probe)
        torch.cuda.synchronize(device)
        return device
    except Exception as exc:
        message = str(exc)
        if requested == "auto":
            print(
                "CUDA preflight failed; falling back to CPU.\n"
                f"python={sys.executable}\n"
                f"torch={torch.__version__}\n"
                f"cuda={torch.version.cuda}\n"
                f"error={message}",
                flush=True,
            )
            return torch.device("cpu")
        device_name = "unknown"
        capability = "unknown"
        try:
            device_name = torch.cuda.get_device_name(0)
            capability = ".".join(str(part) for part in torch.cuda.get_device_capability(0))
        except Exception:
            pass
        raise RuntimeError(
            "CUDA is visible, but this PyTorch build cannot execute kernels on the current GPU.\n"
            f"python={sys.executable}\n"
            f"torch={torch.__version__}\n"
            f"cuda={torch.version.cuda}\n"
            f"gpu={device_name}\n"
            f"compute_capability={capability}\n"
            "If you are on a V100 / sm_70 node, do not use the repo .venv for CUDA.\n"
            "Use the cluster system Python instead, for example:\n"
            "deactivate\n"
            "PYTHONUNBUFFERED=1 python forecast_power.py --device cuda --output-dir outputs/deep_learning_forecast\n"
            f"Original CUDA error: {message}"
        ) from exc


def to_float32_array(values: list[list[float]] | list[np.ndarray], shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if shape is not None:
        array = array.reshape(shape)
    return array


def load_preprocessed_table(path: Path) -> ForecastTable:
    source_timestamps: list[str] = []
    sequence_index: list[int] = []
    targets: list[float] = []
    long_rows: list[np.ndarray] = []
    short_rows: list[np.ndarray] = []
    weather_rows: list[np.ndarray] = []
    weather_mask_rows: list[np.ndarray] = []
    categorical_rows: list[list[int]] = []
    static_continuous_rows: list[list[float]] = []
    static_binary_rows: list[list[float]] = []
    weather_complete: list[float] = []

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source_timestamps.append(row[TIMESTAMP_COLUMN])
            sequence_index.append(int(float(row["sequence_index"])))
            targets.append(parse_float(row[TARGET_COLUMN]))

            long_array = np.empty((len(LONG_FEATURES), len(LONG_LAGS)), dtype=np.float32)
            for feature_index, keys in enumerate(LONG_KEYS):
                for lag_index, key in enumerate(keys):
                    long_array[feature_index, lag_index] = parse_float(row[key])
            long_rows.append(long_array)

            short_array = np.empty((len(SHORT_FEATURES), len(SHORT_LAGS)), dtype=np.float32)
            for feature_index, keys in enumerate(SHORT_KEYS):
                for lag_index, key in enumerate(keys):
                    short_array[feature_index, lag_index] = parse_float(row[key])
            short_rows.append(short_array)

            weather_array = np.empty((len(WEATHER_FEATURES), len(WEATHER_LAGS)), dtype=np.float32)
            weather_mask = np.empty((len(WEATHER_FEATURES), len(WEATHER_LAGS)), dtype=np.float32)
            for feature_index, keys in enumerate(WEATHER_KEYS):
                for lag_index, key in enumerate(keys):
                    value = parse_float(row.get(key))
                    weather_array[feature_index, lag_index] = value
                    weather_mask[feature_index, lag_index] = 0.0 if math.isnan(value) else 1.0
            weather_rows.append(weather_array)
            weather_mask_rows.append(weather_mask)

            categorical_rows.append(
                [
                    int(float(row["hour_of_day"])),
                    int(float(row["day_of_week"])),
                    int(float(row["month"])) - 1,
                    int(float(row["season"])) - 1,
                ]
            )
            static_continuous_rows.append([parse_float(row[name]) for name in STATIC_CONTINUOUS_COLUMNS])
            binary_values: list[float] = []
            for name in STATIC_BINARY_COLUMNS:
                value = parse_float(row[name])
                binary_values.append(0.0 if math.isnan(value) else value)
            static_binary_rows.append(binary_values)
            weather_complete_value = parse_float(row["weather_history_complete"])
            weather_complete.append(0.0 if math.isnan(weather_complete_value) else weather_complete_value)

    return ForecastTable(
        source_timestamps=np.asarray(source_timestamps, dtype=object),
        sequence_index=np.asarray(sequence_index, dtype=np.int64),
        targets=np.asarray(targets, dtype=np.float32),
        long_history=to_float32_array(long_rows),
        short_history=to_float32_array(short_rows),
        weather_history=to_float32_array(weather_rows),
        weather_mask=to_float32_array(weather_mask_rows),
        categorical=np.asarray(categorical_rows, dtype=np.int64),
        static_continuous=np.asarray(static_continuous_rows, dtype=np.float32),
        static_binary=np.asarray(static_binary_rows, dtype=np.float32),
        weather_complete=np.asarray(weather_complete, dtype=np.float32),
    )


def fit_history_scaler(history: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flattened = np.transpose(history, (0, 2, 1)).reshape(-1, history.shape[1])
    center = np.zeros(flattened.shape[1], dtype=np.float32)
    scale = np.ones(flattened.shape[1], dtype=np.float32)
    for column_index in range(flattened.shape[1]):
        values = flattened[:, column_index]
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        q25, median, q75 = np.percentile(values, [25.0, 50.0, 75.0])
        center[column_index] = np.float32(median)
        scale[column_index] = np.float32(max(q75 - q25, 1e-3))
    return center, scale


def transform_history(history: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    transformed = (history - center[None, :, None]) / scale[None, :, None]
    transformed = np.clip(transformed, -6.0, 6.0)
    transformed = np.nan_to_num(transformed, nan=0.0, posinf=6.0, neginf=-6.0)
    return transformed.astype(np.float32)


def fit_feature_scaler(features: np.ndarray) -> FeatureScaler:
    center = np.zeros(features.shape[1], dtype=np.float32)
    scale = np.ones(features.shape[1], dtype=np.float32)
    for column_index in range(features.shape[1]):
        values = features[:, column_index]
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        q25, median, q75 = np.percentile(values, [25.0, 50.0, 75.0])
        center[column_index] = np.float32(median)
        scale[column_index] = np.float32(max(q75 - q25, 1e-3))
    return FeatureScaler(center=center, scale=scale)


def transform_features(features: np.ndarray, scaler: FeatureScaler) -> np.ndarray:
    transformed = (features - scaler.center[None, :]) / scaler.scale[None, :]
    transformed = np.clip(transformed, -6.0, 6.0)
    transformed = np.nan_to_num(transformed, nan=0.0, posinf=6.0, neginf=-6.0)
    return transformed.astype(np.float32)


def fit_structured_scalers(table: ForecastTable, train_indices: np.ndarray) -> StructuredScalers:
    long_center, long_scale = fit_history_scaler(table.long_history[train_indices])
    short_center, short_scale = fit_history_scaler(table.short_history[train_indices])
    weather_center, weather_scale = fit_history_scaler(table.weather_history[train_indices])
    static_scaler = fit_feature_scaler(table.static_continuous[train_indices])
    return StructuredScalers(
        long_center=long_center,
        long_scale=long_scale,
        short_center=short_center,
        short_scale=short_scale,
        weather_center=weather_center,
        weather_scale=weather_scale,
        static_center=static_scaler.center,
        static_scale=static_scaler.scale,
    )


def prepare_data(table: ForecastTable, indices: np.ndarray, scalers: StructuredScalers) -> PreparedData:
    static_scaler = FeatureScaler(center=scalers.static_center, scale=scalers.static_scale)
    return PreparedData(
        source_timestamps=table.source_timestamps[indices],
        targets=table.targets[indices].astype(np.float32),
        long_history=transform_history(table.long_history[indices], scalers.long_center, scalers.long_scale),
        short_history=transform_history(table.short_history[indices], scalers.short_center, scalers.short_scale),
        weather_history=transform_history(table.weather_history[indices], scalers.weather_center, scalers.weather_scale),
        weather_mask=table.weather_mask[indices].astype(np.float32),
        categorical=table.categorical[indices].astype(np.int64),
        static_continuous=transform_features(table.static_continuous[indices], static_scaler),
        static_binary=np.nan_to_num(table.static_binary[indices], nan=0.0).astype(np.float32),
        weather_complete=table.weather_complete[indices].astype(np.float32),
    )


def subset_prepared(data: PreparedData, mask: np.ndarray) -> PreparedData:
    return PreparedData(
        source_timestamps=data.source_timestamps[mask],
        targets=data.targets[mask],
        long_history=data.long_history[mask],
        short_history=data.short_history[mask],
        weather_history=data.weather_history[mask],
        weather_mask=data.weather_mask[mask],
        categorical=data.categorical[mask],
        static_continuous=data.static_continuous[mask],
        static_binary=data.static_binary[mask],
        weather_complete=data.weather_complete[mask],
    )


def build_flat_matrix(table: ForecastTable, indices: np.ndarray) -> tuple[np.ndarray, list[str]]:
    matrix = np.concatenate(
        [
            table.long_history[indices].reshape(len(indices), -1),
            table.short_history[indices].reshape(len(indices), -1),
            table.weather_history[indices].reshape(len(indices), -1),
            table.static_continuous[indices],
            table.static_binary[indices],
            table.categorical[indices].astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    feature_names: list[str] = []
    for feature, lags in zip(LONG_FEATURES, LONG_KEYS):
        feature_names.extend(lags)
    for feature, lags in zip(SHORT_FEATURES, SHORT_KEYS):
        feature_names.extend(lags)
    for feature, lags in zip(WEATHER_FEATURES, WEATHER_KEYS):
        feature_names.extend(lags)
    feature_names.extend(STATIC_CONTINUOUS_COLUMNS)
    feature_names.extend(STATIC_BINARY_COLUMNS)
    feature_names.extend(CATEGORICAL_COLUMNS)
    return matrix, feature_names


def pinball_loss_numpy(y_true: np.ndarray, y_pred: np.ndarray, quantile: float = PINBALL_QUANTILE) -> float:
    residual = y_true - y_pred
    loss = np.maximum(quantile * residual, (quantile - 1.0) * residual)
    return float(np.mean(loss))


def score_predictions(
    y_true: np.ndarray,
    point_pred: np.ndarray,
    p90_pred: np.ndarray,
    peak_threshold: float,
) -> dict[str, float]:
    absolute_error = np.abs(point_pred - y_true)
    peak_mask = y_true >= peak_threshold
    if peak_mask.any():
        mae_peak = float(np.mean(absolute_error[peak_mask]))
    else:
        mae_peak = float(np.mean(absolute_error))
    mae_all = float(np.mean(absolute_error))
    pinball = pinball_loss_numpy(y_true, p90_pred, quantile=PINBALL_QUANTILE)
    score = 0.5 * mae_all + 0.3 * mae_peak + 0.2 * pinball
    return {
        "mae_all": mae_all,
        "mae_peak": mae_peak,
        "pinball_p90": pinball,
        "score": score,
    }


def challenge_loss_torch(
    y_true: torch.Tensor,
    point_pred: torch.Tensor,
    raw_uplift: torch.Tensor,
    peak_threshold: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    p90_pred = point_pred + raw_uplift
    absolute_error = (point_pred - y_true).abs()
    mae_all = absolute_error.mean()
    peak_mask = y_true >= peak_threshold
    if torch.any(peak_mask):
        mae_peak = absolute_error[peak_mask].mean()
    else:
        mae_peak = mae_all
    residual = y_true - p90_pred
    pinball = torch.maximum(PINBALL_QUANTILE * residual, (PINBALL_QUANTILE - 1.0) * residual).mean()
    blended = 0.5 * mae_all + 0.3 * mae_peak + 0.2 * pinball
    metrics = {
        "mae_all": float(mae_all.detach().cpu()),
        "mae_peak": float(mae_peak.detach().cpu()),
        "pinball_p90": float(pinball.detach().cpu()),
        "score": float(blended.detach().cpu()),
    }
    return blended, metrics


def make_loader(data: PreparedData, batch_size: int, shuffle: bool, num_workers: int, device: torch.device) -> DataLoader:
    return DataLoader(
        TensorForecastDataset(data),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    moved: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=device.type == "cuda")
    return moved


def evaluate_model(
    model: ReeferDemandForecaster,
    data: PreparedData,
    batch_size: int,
    device: torch.device,
    peak_threshold: float,
    use_weather: bool,
    use_amp: bool,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    loader = make_loader(data, batch_size=batch_size, shuffle=False, num_workers=num_workers, device=device)
    model.eval()
    point_predictions: list[np.ndarray] = []
    uplift_predictions: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            with (torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp) if use_amp else nullcontext()):
                point_pred, raw_uplift = model(
                    long_history=batch["long_history"],
                    short_history=batch["short_history"],
                    weather_history=batch["weather_history"],
                    weather_mask=batch["weather_mask"],
                    categorical=batch["categorical"],
                    static_continuous=batch["static_continuous"],
                    static_binary=batch["static_binary"],
                    use_weather=use_weather,
                )
            point_predictions.append(point_pred.detach().cpu().numpy())
            uplift_predictions.append(raw_uplift.detach().cpu().numpy())
    point_pred_np = np.concatenate(point_predictions, axis=0).astype(np.float32)
    raw_uplift_np = np.concatenate(uplift_predictions, axis=0).astype(np.float32)
    p90_pred_np = point_pred_np + np.maximum(raw_uplift_np, 0.0)
    metrics = score_predictions(data.targets, point_pred_np, p90_pred_np, peak_threshold)
    return point_pred_np, raw_uplift_np, metrics


def train_one_stage(
    model: ReeferDemandForecaster,
    train_data: PreparedData,
    val_data: PreparedData | None,
    batch_size: int,
    max_epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    grad_clip: float,
    peak_threshold: float,
    device: torch.device,
    use_weather: bool,
    use_amp: bool,
    num_workers: int,
) -> tuple[dict[str, torch.Tensor], int]:
    if train_data.num_rows == 0:
        raise ValueError("Cannot train on an empty split.")
    train_loader = make_loader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_epoch = 1
    best_score = float("inf")
    bad_epochs = 0
    best_state = copy.deepcopy({name: tensor.detach().cpu() for name, tensor in model.state_dict().items()})

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        total_rows = 0
        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with (torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp) if use_amp else nullcontext()):
                point_pred, raw_uplift = model(
                    long_history=batch["long_history"],
                    short_history=batch["short_history"],
                    weather_history=batch["weather_history"],
                    weather_mask=batch["weather_mask"],
                    categorical=batch["categorical"],
                    static_continuous=batch["static_continuous"],
                    static_binary=batch["static_binary"],
                    use_weather=use_weather,
                )
                loss, _ = challenge_loss_torch(batch["target"], point_pred, raw_uplift, peak_threshold)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            batch_rows = int(batch["target"].shape[0])
            epoch_loss += float(loss.detach().cpu()) * batch_rows
            total_rows += batch_rows
        scheduler.step()

        if val_data is None:
            train_score = epoch_loss / max(total_rows, 1)
            print(
                f"  epoch {epoch:03d} use_weather={int(use_weather)} "
                f"train_score={train_score:.4f}"
            , flush=True)
            best_epoch = epoch
            best_state = copy.deepcopy({name: tensor.detach().cpu() for name, tensor in model.state_dict().items()})
            continue

        _, _, val_metrics = evaluate_model(
            model=model,
            data=val_data,
            batch_size=batch_size,
            device=device,
            peak_threshold=peak_threshold,
            use_weather=use_weather,
            use_amp=use_amp,
            num_workers=num_workers,
        )
        print(
            f"  epoch {epoch:03d} use_weather={int(use_weather)} "
            f"val_score={val_metrics['score']:.4f} "
            f"mae_all={val_metrics['mae_all']:.4f} "
            f"mae_peak={val_metrics['mae_peak']:.4f} "
            f"pinball={val_metrics['pinball_p90']:.4f}"
        , flush=True)
        if val_metrics["score"] + 1e-6 < best_score:
            best_score = val_metrics["score"]
            best_epoch = epoch
            bad_epochs = 0
            best_state = copy.deepcopy({name: tensor.detach().cpu() for name, tensor in model.state_dict().items()})
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    return best_state, best_epoch


def fit_fold_model(
    table: ForecastTable,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> FoldPrediction:
    set_seed(seed)
    scalers = fit_structured_scalers(table, train_indices)
    prepared_train = prepare_data(table, train_indices, scalers)
    prepared_val = prepare_data(table, val_indices, scalers)
    peak_threshold = float(np.quantile(table.targets[train_indices], 0.9))
    use_amp = device.type == "cuda"

    model = ReeferDemandForecaster(
        input_noise_std=args.input_noise_std,
        weather_channel_dropout=args.weather_channel_dropout,
    ).to(device)
    print(f"seed={seed} stage1 train_rows={prepared_train.num_rows} val_rows={prepared_val.num_rows}", flush=True)
    stage1_state, stage1_epoch = train_one_stage(
        model=model,
        train_data=prepared_train,
        val_data=prepared_val,
        batch_size=args.stage1_batch_size,
        max_epochs=args.stage1_epochs,
        learning_rate=args.stage1_lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        grad_clip=args.grad_clip,
        peak_threshold=peak_threshold,
        device=device,
        use_weather=False,
        use_amp=use_amp,
        num_workers=args.num_workers,
    )
    model.load_state_dict(stage1_state)

    complete_train_mask = prepared_train.weather_complete > 0.5
    complete_val_mask = prepared_val.weather_complete > 0.5
    stage2_epoch = 0
    if complete_train_mask.any() and complete_val_mask.any():
        stage2_train = subset_prepared(prepared_train, complete_train_mask)
        stage2_val = subset_prepared(prepared_val, complete_val_mask)
        print(f"seed={seed} stage2 train_rows={stage2_train.num_rows} val_rows={stage2_val.num_rows}", flush=True)
        stage2_state, stage2_epoch = train_one_stage(
            model=model,
            train_data=stage2_train,
            val_data=stage2_val,
            batch_size=args.stage2_batch_size,
            max_epochs=args.stage2_epochs,
            learning_rate=args.stage2_lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            grad_clip=args.grad_clip,
            peak_threshold=peak_threshold,
            device=device,
            use_weather=True,
            use_amp=use_amp,
            num_workers=args.num_workers,
        )
        model.load_state_dict(stage2_state)

    point_pred, raw_uplift, _ = evaluate_model(
        model=model,
        data=prepared_val,
        batch_size=args.stage2_batch_size,
        device=device,
        peak_threshold=peak_threshold,
        use_weather=True,
        use_amp=use_amp,
        num_workers=args.num_workers,
    )
    return FoldPrediction(
        model_name="deep_model",
        fold_name="",
        source_timestamps=prepared_val.source_timestamps,
        targets=prepared_val.targets,
        point_pred=point_pred,
        raw_uplift=raw_uplift,
        p90_pred=point_pred + np.maximum(raw_uplift, 0.0),
        peak_threshold=peak_threshold,
        stage1_best_epoch=stage1_epoch,
        stage2_best_epoch=stage2_epoch,
    )


def fit_final_model(
    table: ForecastTable,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
    stage1_epochs: int,
    stage2_epochs: int,
) -> tuple[TrainArtifacts, StructuredScalers]:
    set_seed(seed)
    train_indices = np.arange(table.num_rows, dtype=np.int64)
    scalers = fit_structured_scalers(table, train_indices)
    prepared_train = prepare_data(table, train_indices, scalers)
    peak_threshold = float(np.quantile(table.targets, 0.9))
    use_amp = device.type == "cuda"

    model = ReeferDemandForecaster(
        input_noise_std=args.input_noise_std,
        weather_channel_dropout=args.weather_channel_dropout,
    ).to(device)
    stage1_state, _ = train_one_stage(
        model=model,
        train_data=prepared_train,
        val_data=None,
        batch_size=args.stage1_batch_size,
        max_epochs=stage1_epochs,
        learning_rate=args.stage1_lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        grad_clip=args.grad_clip,
        peak_threshold=peak_threshold,
        device=device,
        use_weather=False,
        use_amp=use_amp,
        num_workers=args.num_workers,
    )
    model.load_state_dict(stage1_state)

    complete_train_mask = prepared_train.weather_complete > 0.5
    stage2_state = stage1_state
    if complete_train_mask.any() and stage2_epochs > 0:
        stage2_train = subset_prepared(prepared_train, complete_train_mask)
        stage2_state, _ = train_one_stage(
            model=model,
            train_data=stage2_train,
            val_data=None,
            batch_size=args.stage2_batch_size,
            max_epochs=stage2_epochs,
            learning_rate=args.stage2_lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            grad_clip=args.grad_clip,
            peak_threshold=peak_threshold,
            device=device,
            use_weather=True,
            use_amp=use_amp,
            num_workers=args.num_workers,
        )
    return (
        TrainArtifacts(
            model_state=stage2_state,
            stage1_best_epoch=stage1_epochs,
            stage2_best_epoch=stage2_epochs,
        ),
        scalers,
    )


def predict_with_artifact(
    artifact: TrainArtifacts,
    scalers: StructuredScalers,
    table: ForecastTable,
    indices: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> FoldPrediction:
    prepared = prepare_data(table, indices, scalers)
    peak_threshold = float(np.quantile(table.targets, 0.9))
    model = ReeferDemandForecaster(
        input_noise_std=args.input_noise_std,
        weather_channel_dropout=args.weather_channel_dropout,
    ).to(device)
    model.load_state_dict(artifact.model_state)
    point_pred, raw_uplift, _ = evaluate_model(
        model=model,
        data=prepared,
        batch_size=args.stage2_batch_size,
        device=device,
        peak_threshold=peak_threshold,
        use_weather=True,
        use_amp=device.type == "cuda",
        num_workers=args.num_workers,
    )
    return FoldPrediction(
        model_name="deep_model",
        fold_name="final",
        source_timestamps=prepared.source_timestamps,
        targets=prepared.targets,
        point_pred=point_pred,
        raw_uplift=raw_uplift,
        p90_pred=point_pred + np.maximum(raw_uplift, 0.0),
        peak_threshold=peak_threshold,
        stage1_best_epoch=artifact.stage1_best_epoch,
        stage2_best_epoch=artifact.stage2_best_epoch,
    )


def fit_ridge_regression(train_x: np.ndarray, train_y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, float]:
    x_mean = train_x.mean(axis=0)
    y_mean = float(train_y.mean())
    centered_x = train_x - x_mean[None, :]
    centered_y = train_y - y_mean
    gram = centered_x.T @ centered_x
    gram.flat[:: gram.shape[0] + 1] += alpha
    rhs = centered_x.T @ centered_y
    weights = np.linalg.solve(gram, rhs)
    return weights.astype(np.float32), x_mean.astype(np.float32), y_mean


def predict_ridge_regression(weights: np.ndarray, x_mean: np.ndarray, y_mean: float, features: np.ndarray) -> np.ndarray:
    return ((features - x_mean[None, :]) @ weights + y_mean).astype(np.float32)


def fit_constant_uplift(y_true: np.ndarray, point_pred: np.ndarray) -> float:
    residuals = y_true - point_pred
    return max(float(np.quantile(residuals, PINBALL_QUANTILE)), 0.0)


def fit_affine_calibrator(y_true: np.ndarray, point_pred: np.ndarray, raw_uplift: np.ndarray) -> AffineCalibrator:
    positive_uplift = np.maximum(raw_uplift, 0.0)
    residuals = y_true - point_pred
    beta_max = max(float(np.quantile(np.maximum(residuals, 0.0), 0.95)), 5.0)
    alpha_grid = np.linspace(0.0, 2.5, 26)
    beta_grid = np.linspace(0.0, beta_max, 31)
    best = AffineCalibrator(alpha=1.0, beta=0.0)
    best_pinball = float("inf")
    for alpha in alpha_grid:
        for beta in beta_grid:
            candidate = point_pred + np.maximum(alpha * positive_uplift + beta, 0.0)
            pinball = pinball_loss_numpy(y_true, candidate, quantile=PINBALL_QUANTILE)
            if pinball < best_pinball:
                best_pinball = pinball
                best = AffineCalibrator(alpha=float(alpha), beta=float(beta))
    return best


def evaluate_yesterday_baseline(
    train_table: ForecastTable,
    train_indices: np.ndarray,
    eval_table: ForecastTable,
    eval_indices: np.ndarray,
) -> FoldPrediction:
    train_point = train_table.long_history[train_indices, 0, -1]
    val_point = eval_table.long_history[eval_indices, 0, -1]
    uplift = fit_constant_uplift(train_table.targets[train_indices], train_point)
    val_p90 = val_point + uplift
    peak_threshold = float(np.quantile(train_table.targets[train_indices], 0.9))
    return FoldPrediction(
        model_name="yesterday_same_hour",
        fold_name="",
        source_timestamps=eval_table.source_timestamps[eval_indices],
        targets=eval_table.targets[eval_indices],
        point_pred=val_point.astype(np.float32),
        raw_uplift=np.full_like(val_point, uplift, dtype=np.float32),
        p90_pred=val_p90.astype(np.float32),
        peak_threshold=peak_threshold,
        stage1_best_epoch=0,
        stage2_best_epoch=0,
    )


def evaluate_ridge_baseline(
    train_table: ForecastTable,
    train_indices: np.ndarray,
    eval_table: ForecastTable,
    eval_indices: np.ndarray,
    alpha: float,
) -> FoldPrediction:
    train_x, _ = build_flat_matrix(train_table, train_indices)
    val_x, _ = build_flat_matrix(eval_table, eval_indices)
    scaler = fit_feature_scaler(train_x)
    train_x = transform_features(train_x, scaler).astype(np.float64)
    val_x = transform_features(val_x, scaler).astype(np.float64)
    train_y = train_table.targets[train_indices].astype(np.float64)
    weights, x_mean, y_mean = fit_ridge_regression(train_x, train_y, alpha=alpha)
    train_point = predict_ridge_regression(weights, x_mean, y_mean, train_x)
    val_point = predict_ridge_regression(weights, x_mean, y_mean, val_x)
    uplift = fit_constant_uplift(train_table.targets[train_indices], train_point)
    val_p90 = val_point + uplift
    peak_threshold = float(np.quantile(train_table.targets[train_indices], 0.9))
    return FoldPrediction(
        model_name="ridge_flattened",
        fold_name="",
        source_timestamps=eval_table.source_timestamps[eval_indices],
        targets=eval_table.targets[eval_indices],
        point_pred=val_point.astype(np.float32),
        raw_uplift=np.full_like(val_point, uplift, dtype=np.float32),
        p90_pred=val_p90.astype(np.float32),
        peak_threshold=peak_threshold,
        stage1_best_epoch=0,
        stage2_best_epoch=0,
    )


def run_forward_windows(
    trainval_table: ForecastTable,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, object]], list[dict[str, object]], AffineCalibrator]:
    fold_metrics: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    tuning_point: list[np.ndarray] = []
    tuning_uplift: list[np.ndarray] = []
    tuning_targets: list[np.ndarray] = []
    holdout_deep_predictions: FoldPrediction | None = None
    stage1_epochs: list[int] = []
    stage2_epochs: list[int] = []

    for window in FORWARD_WINDOWS:
        train_indices = np.where(trainval_table.source_timestamps < window.start)[0]
        val_mask = (trainval_table.source_timestamps >= window.start) & (trainval_table.source_timestamps <= window.end)
        val_indices = np.where(val_mask)[0]
        print(
            f"window={window.name} train_rows={len(train_indices)} val_rows={len(val_indices)} "
            f"holdout={int(window.is_holdout)}"
        , flush=True)
        baseline_predictions = [
            evaluate_yesterday_baseline(trainval_table, train_indices, trainval_table, val_indices),
            evaluate_ridge_baseline(trainval_table, train_indices, trainval_table, val_indices, alpha=args.ridge_alpha),
        ]
        for prediction in baseline_predictions:
            prediction.fold_name = window.name
            metrics = score_predictions(
                prediction.targets,
                prediction.point_pred,
                prediction.p90_pred,
                prediction.peak_threshold,
            )
            metrics_row = {
                "model": prediction.model_name,
                "fold": window.name,
                "split_type": "holdout" if window.is_holdout else "tuning",
                "seed": "",
                **metrics,
            }
            fold_metrics.append(metrics_row)
            for index, timestamp in enumerate(prediction.source_timestamps):
                prediction_rows.append(
                    {
                        "model": prediction.model_name,
                        "fold": window.name,
                        "timestamp_utc": timestamp,
                        "label_power_kw": float(prediction.targets[index]),
                        "pred_power_kw": float(prediction.point_pred[index]),
                        "pred_p90_kw": float(prediction.p90_pred[index]),
                        "raw_uplift_kw": float(prediction.raw_uplift[index]),
                    }
                )

        deep_seed_predictions: list[FoldPrediction] = []
        for seed in args.seeds:
            fold_prediction = fit_fold_model(
                table=trainval_table,
                train_indices=train_indices,
                val_indices=val_indices,
                seed=seed,
                args=args,
                device=device,
            )
            fold_prediction.fold_name = window.name
            deep_seed_predictions.append(fold_prediction)
            stage1_epochs.append(fold_prediction.stage1_best_epoch)
            if fold_prediction.stage2_best_epoch > 0:
                stage2_epochs.append(fold_prediction.stage2_best_epoch)
            metrics = score_predictions(
                fold_prediction.targets,
                fold_prediction.point_pred,
                fold_prediction.p90_pred,
                fold_prediction.peak_threshold,
            )
            fold_metrics.append(
                {
                    "model": "deep_model_seed",
                    "fold": window.name,
                    "split_type": "holdout" if window.is_holdout else "tuning",
                    "seed": seed,
                    **metrics,
                }
            )

        ensemble_point = np.mean([prediction.point_pred for prediction in deep_seed_predictions], axis=0)
        ensemble_uplift = np.mean([prediction.raw_uplift for prediction in deep_seed_predictions], axis=0)
        ensemble_targets = deep_seed_predictions[0].targets
        ensemble_threshold = deep_seed_predictions[0].peak_threshold
        ensemble_p90 = ensemble_point + np.maximum(ensemble_uplift, 0.0)
        ensemble_prediction = FoldPrediction(
            model_name="deep_model_ensemble",
            fold_name=window.name,
            source_timestamps=deep_seed_predictions[0].source_timestamps,
            targets=ensemble_targets,
            point_pred=ensemble_point.astype(np.float32),
            raw_uplift=ensemble_uplift.astype(np.float32),
            p90_pred=ensemble_p90.astype(np.float32),
            peak_threshold=ensemble_threshold,
            stage1_best_epoch=int(round(np.mean([prediction.stage1_best_epoch for prediction in deep_seed_predictions]))),
            stage2_best_epoch=int(round(np.mean([prediction.stage2_best_epoch for prediction in deep_seed_predictions]))),
        )
        raw_metrics = score_predictions(
            ensemble_prediction.targets,
            ensemble_prediction.point_pred,
            ensemble_prediction.p90_pred,
            ensemble_prediction.peak_threshold,
        )
        fold_metrics.append(
            {
                "model": "deep_model_ensemble_raw",
                "fold": window.name,
                "split_type": "holdout" if window.is_holdout else "tuning",
                "seed": "",
                **raw_metrics,
            }
        )
        if window.is_holdout:
            holdout_deep_predictions = ensemble_prediction
        else:
            tuning_point.append(ensemble_prediction.point_pred)
            tuning_uplift.append(ensemble_prediction.raw_uplift)
            tuning_targets.append(ensemble_prediction.targets)

        for index, timestamp in enumerate(ensemble_prediction.source_timestamps):
            prediction_rows.append(
                {
                    "model": "deep_model_ensemble_raw",
                    "fold": window.name,
                    "timestamp_utc": timestamp,
                    "label_power_kw": float(ensemble_prediction.targets[index]),
                    "pred_power_kw": float(ensemble_prediction.point_pred[index]),
                    "pred_p90_kw": float(ensemble_prediction.p90_pred[index]),
                    "raw_uplift_kw": float(ensemble_prediction.raw_uplift[index]),
                }
            )

    if holdout_deep_predictions is None:
        raise RuntimeError("Expected to produce holdout predictions for the deep ensemble.")

    tuning_point_concat = np.concatenate(tuning_point, axis=0)
    tuning_uplift_concat = np.concatenate(tuning_uplift, axis=0)
    tuning_targets_concat = np.concatenate(tuning_targets, axis=0)
    holdout_calibrator = fit_affine_calibrator(
        y_true=tuning_targets_concat,
        point_pred=tuning_point_concat,
        raw_uplift=tuning_uplift_concat,
    )
    holdout_calibrated_p90 = holdout_calibrator.apply(
        holdout_deep_predictions.point_pred,
        holdout_deep_predictions.raw_uplift,
    )
    holdout_calibrated_metrics = score_predictions(
        holdout_deep_predictions.targets,
        holdout_deep_predictions.point_pred,
        holdout_calibrated_p90,
        holdout_deep_predictions.peak_threshold,
    )
    fold_metrics.append(
        {
            "model": "deep_model_ensemble_calibrated",
            "fold": "holdout",
            "split_type": "holdout",
            "seed": "",
            **holdout_calibrated_metrics,
        }
    )
    for index, timestamp in enumerate(holdout_deep_predictions.source_timestamps):
        prediction_rows.append(
            {
                "model": "deep_model_ensemble_calibrated",
                "fold": "holdout",
                "timestamp_utc": timestamp,
                "label_power_kw": float(holdout_deep_predictions.targets[index]),
                "pred_power_kw": float(holdout_deep_predictions.point_pred[index]),
                "pred_p90_kw": float(holdout_calibrated_p90[index]),
                "raw_uplift_kw": float(holdout_deep_predictions.raw_uplift[index]),
            }
        )

    final_calibrator = fit_affine_calibrator(
        y_true=np.concatenate([tuning_targets_concat, holdout_deep_predictions.targets], axis=0),
        point_pred=np.concatenate([tuning_point_concat, holdout_deep_predictions.point_pred], axis=0),
        raw_uplift=np.concatenate([tuning_uplift_concat, holdout_deep_predictions.raw_uplift], axis=0),
    )
    args.final_stage1_epochs = max(1, int(round(float(np.mean(stage1_epochs)))))
    args.final_stage2_epochs = max(1, int(round(float(np.mean(stage2_epochs))))) if stage2_epochs else 1
    return fold_metrics, prediction_rows, final_calibrator


def evaluate_final_test(
    trainval_table: ForecastTable,
    test_table: ForecastTable,
    args: argparse.Namespace,
    device: torch.device,
    final_calibrator: AffineCalibrator,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    test_metrics_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    checkpoint_rows: list[dict[str, object]] = []
    train_indices = np.arange(trainval_table.num_rows, dtype=np.int64)
    test_indices = np.arange(test_table.num_rows, dtype=np.int64)

    yesterday_prediction = evaluate_yesterday_baseline(trainval_table, train_indices, test_table, test_indices)
    ridge_prediction = evaluate_ridge_baseline(trainval_table, train_indices, test_table, test_indices, alpha=args.ridge_alpha)

    for prediction in [yesterday_prediction, ridge_prediction]:
        metrics = score_predictions(prediction.targets, prediction.point_pred, prediction.p90_pred, prediction.peak_threshold)
        test_metrics_rows.append({"model": prediction.model_name, **metrics})
        for index, timestamp in enumerate(prediction.source_timestamps):
            prediction_rows.append(
                {
                    "model": prediction.model_name,
                    "timestamp_utc": timestamp,
                    "label_power_kw": float(prediction.targets[index]),
                    "pred_power_kw": float(prediction.point_pred[index]),
                    "pred_p90_kw": float(prediction.p90_pred[index]),
                    "raw_uplift_kw": float(prediction.raw_uplift[index]),
                }
            )

    deep_predictions: list[FoldPrediction] = []
    for seed in args.seeds:
        print(
            f"final_train seed={seed} stage1_epochs={args.final_stage1_epochs} stage2_epochs={args.final_stage2_epochs}",
            flush=True,
        )
        artifact, scalers = fit_final_model(
            table=trainval_table,
            seed=seed,
            args=args,
            device=device,
            stage1_epochs=args.final_stage1_epochs,
            stage2_epochs=args.final_stage2_epochs,
        )
        checkpoint_path = args.output_dir / f"deep_model_seed_{seed}.pt"
        torch.save(
            {
                "seed": seed,
                "state_dict": artifact.model_state,
                "scalers": asdict(scalers),
            },
            checkpoint_path,
        )
        checkpoint_rows.append({"seed": seed, "checkpoint_path": str(checkpoint_path)})
        deep_prediction = predict_with_artifact(
            artifact=artifact,
            scalers=scalers,
            table=test_table,
            indices=test_indices,
            args=args,
            device=device,
        )
        deep_prediction.source_timestamps = test_table.source_timestamps
        deep_prediction.targets = test_table.targets
        deep_predictions.append(deep_prediction)

    ensemble_point = np.mean([prediction.point_pred for prediction in deep_predictions], axis=0)
    ensemble_uplift = np.mean([prediction.raw_uplift for prediction in deep_predictions], axis=0)
    ensemble_threshold = float(np.quantile(trainval_table.targets, 0.9))
    ensemble_p90 = final_calibrator.apply(ensemble_point, ensemble_uplift)
    ensemble_metrics = score_predictions(test_table.targets, ensemble_point, ensemble_p90, ensemble_threshold)
    test_metrics_rows.append({"model": "deep_model_ensemble_calibrated", **ensemble_metrics})

    for index, timestamp in enumerate(test_table.source_timestamps):
        prediction_rows.append(
            {
                "model": "deep_model_ensemble_calibrated",
                "timestamp_utc": timestamp,
                "label_power_kw": float(test_table.targets[index]),
                "pred_power_kw": float(ensemble_point[index]),
                "pred_p90_kw": float(ensemble_p90[index]),
                "raw_uplift_kw": float(ensemble_uplift[index]),
            }
        )
    return test_metrics_rows, prediction_rows, checkpoint_rows


def aggregate_metric_rows(rows: list[dict[str, object]], model: str, fold_names: list[str]) -> dict[str, float]:
    filtered = [row for row in rows if row["model"] == model and row["fold"] in fold_names]
    if not filtered:
        return {}
    return {
        "mae_all": float(np.mean([float(row["mae_all"]) for row in filtered])),
        "mae_peak": float(np.mean([float(row["mae_peak"]) for row in filtered])),
        "pinball_p90": float(np.mean([float(row["pinball_p90"]) for row in filtered])),
        "score": float(np.mean([float(row["score"]) for row in filtered])),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def build_summary(
    fold_metrics: list[dict[str, object]],
    test_metrics: list[dict[str, object]],
    calibrator: AffineCalibrator,
) -> dict[str, object]:
    tuning_folds = [window.name for window in FORWARD_WINDOWS if not window.is_holdout]
    holdout_model_rows = [row for row in fold_metrics if row["fold"] == "holdout"]
    best_baseline_holdout = min(
        float(row["score"])
        for row in holdout_model_rows
        if row["model"] in {"yesterday_same_hour", "ridge_flattened"}
    )
    deep_holdout_score = next(
        float(row["score"])
        for row in holdout_model_rows
        if row["model"] == "deep_model_ensemble_calibrated"
    )
    deep_test_score = next(
        float(row["score"])
        for row in test_metrics
        if row["model"] == "deep_model_ensemble_calibrated"
    )
    return {
        "tuning_summary": {
            "yesterday_same_hour": aggregate_metric_rows(fold_metrics, "yesterday_same_hour", tuning_folds),
            "ridge_flattened": aggregate_metric_rows(fold_metrics, "ridge_flattened", tuning_folds),
            "deep_model_ensemble_raw": aggregate_metric_rows(fold_metrics, "deep_model_ensemble_raw", tuning_folds),
        },
        "holdout_summary": {
            "yesterday_same_hour": aggregate_metric_rows(fold_metrics, "yesterday_same_hour", ["holdout"]),
            "ridge_flattened": aggregate_metric_rows(fold_metrics, "ridge_flattened", ["holdout"]),
            "deep_model_ensemble_raw": aggregate_metric_rows(fold_metrics, "deep_model_ensemble_raw", ["holdout"]),
            "deep_model_ensemble_calibrated": aggregate_metric_rows(
                fold_metrics, "deep_model_ensemble_calibrated", ["holdout"]
            ),
        },
        "test_summary": {row["model"]: {k: row[k] for k in row if k != "model"} for row in test_metrics},
        "p90_calibrator": {"alpha": calibrator.alpha, "beta": calibrator.beta},
        "warnings": {
            "deep_model_beats_holdout_baselines": deep_holdout_score < best_baseline_holdout,
            "message": (
                "Deep model beat the best simple baseline on the final holdout."
                if deep_holdout_score < best_baseline_holdout
                else "Deep model did not beat the best simple baseline on the final holdout."
            ),
            "deep_model_holdout_score": deep_holdout_score,
            "best_simple_holdout_score": best_baseline_holdout,
            "deep_model_test_score": deep_test_score,
        },
    }


def maybe_shorten_run(args: argparse.Namespace) -> None:
    if not args.quick_run:
        return
    args.stage1_epochs = min(args.stage1_epochs, 2)
    args.stage2_epochs = min(args.stage2_epochs, 2)
    args.patience = min(args.patience, 2)
    args.seeds = args.seeds[:1]


def main() -> None:
    args = parse_args()
    maybe_shorten_run(args)
    ensure_output_dir(args.output_dir)
    device = choose_device(args.device)
    device = validate_cuda_runtime(device, requested=args.device)
    print(f"using_device={device}", flush=True)
    print("loading datasets", flush=True)
    trainval_table = load_preprocessed_table(args.trainval_csv)
    test_table = load_preprocessed_table(args.test_csv)
    print(
        f"trainval_rows={trainval_table.num_rows} test_rows={test_table.num_rows} "
        f"weather_complete_trainval={int(trainval_table.weather_complete.sum())}"
    , flush=True)

    write_json(
        args.output_dir / "config.json",
        {
            **vars(args),
            "trainval_csv": str(args.trainval_csv),
            "test_csv": str(args.test_csv),
            "output_dir": str(args.output_dir),
            "resolved_device": str(device),
        },
    )

    fold_metrics, validation_predictions, final_calibrator = run_forward_windows(
        trainval_table=trainval_table,
        args=args,
        device=device,
    )
    write_csv(args.output_dir / "fold_metrics.csv", fold_metrics)
    write_csv(args.output_dir / "validation_predictions.csv", validation_predictions)

    test_metrics, test_predictions, checkpoint_rows = evaluate_final_test(
        trainval_table=trainval_table,
        test_table=test_table,
        args=args,
        device=device,
        final_calibrator=final_calibrator,
    )
    write_csv(args.output_dir / "test_metrics.csv", test_metrics)
    write_csv(args.output_dir / "test_predictions.csv", test_predictions)
    write_csv(args.output_dir / "checkpoints.csv", checkpoint_rows)

    summary = build_summary(fold_metrics=fold_metrics, test_metrics=test_metrics, calibrator=final_calibrator)
    summary["final_training_epochs"] = {
        "stage1": args.final_stage1_epochs,
        "stage2": args.final_stage2_epochs,
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary["warnings"], indent=2), flush=True)


if __name__ == "__main__":
    main()
