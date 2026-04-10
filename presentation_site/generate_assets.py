from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

REPO_ROOT = Path(__file__).resolve().parents[1]
SITE_ROOT = Path(__file__).resolve().parent
PLOTS_DIR = SITE_ROOT / "assets" / "plots"
DATA_DIR = SITE_ROOT / "assets" / "data"
PARTICIPANT_DIR = REPO_ROOT.parent / "participant_package" / "participant_package"

for directory in (PLOTS_DIR, DATA_DIR):
    directory.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_ROOT))

from hackathon_reefer_dl.baselines import aggregate_hourly_load, load_target_hours  # noqa: E402
from hackathon_reefer_dl.metrics import composite_metrics  # noqa: E402

COLORS = {
    "ink": "#10243F",
    "slate": "#5F6F86",
    "teal": "#178C82",
    "teal_soft": "#8FD3C8",
    "orange": "#EB6A3E",
    "gold": "#D8A34A",
    "sand": "#F2D8A7",
    "rose": "#F29D85",
    "red": "#B7412E",
    "grid": "#D8DEEA",
    "panel": "#FFFDF8",
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "none",
            "axes.facecolor": "none",
            "savefig.facecolor": "none",
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 17,
            "axes.labelsize": 11,
            "axes.edgecolor": COLORS["grid"],
            "axes.linewidth": 1.0,
            "axes.labelcolor": COLORS["ink"],
            "xtick.color": COLORS["slate"],
            "ytick.color": COLORS["slate"],
            "text.color": COLORS["ink"],
            "axes.titlecolor": COLORS["ink"],
            "grid.color": COLORS["grid"],
            "grid.alpha": 0.7,
            "grid.linestyle": "-",
            "legend.frameon": False,
        }
    )


def save_svg(fig: plt.Figure, filename: str) -> None:
    fig.savefig(PLOTS_DIR / filename, format="svg", bbox_inches="tight")
    plt.close(fig)


def metric_formatter(_: float, __: int) -> str:
    return ""


def kw_formatter(value: float, _: int) -> str:
    return f"{value:,.0f} kW"


def pretty_label(raw: str) -> str:
    return (
        raw.replace("weather_", "")
        .replace("_vc_halle3", " Halle 3")
        .replace("_zentralgate", " Zentralgate")
        .replace("_mean", " Mean")
        .replace("_temperature", "Temperature")
        .replace("_wind_direction", "Wind Dir.")
        .replace("_wind_speed", "Wind Speed")
        .replace("_consistency", " Consistency")
        .replace("_c", " (C)")
        .replace("_", " ")
        .title()
    )


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_prediction_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp_utc"], utc=True).dt.tz_convert(None)
    return frame[["timestamp", "pred_power_kw", "pred_p90_kw"]].rename(
        columns={"pred_power_kw": "point", "pred_p90_kw": "p90"}
    )


def build_public_window_frame() -> tuple[pd.DataFrame, pd.Series]:
    hourly_load = aggregate_hourly_load(PARTICIPANT_DIR)
    history = pd.Series(hourly_load).sort_index()
    history.index = pd.to_datetime(history.index)
    history = history.rename("actual_kw")

    target_times = pd.to_datetime(load_target_hours(PARTICIPANT_DIR / "target_timestamps.csv"))
    frame = pd.DataFrame(index=target_times)
    frame.index.name = "timestamp"
    frame["actual_kw"] = [history.loc[timestamp] for timestamp in target_times]
    frame["lag24_kw"] = [hourly_load[timestamp.to_pydatetime() - timedelta(hours=24)] for timestamp in target_times]

    anchor_df = read_prediction_frame(REPO_ROOT / "outputs" / "results" / "predictions.csv")
    tail_df = read_prediction_frame(REPO_ROOT / "hackathon_reefer_dl" / "outputs" / "predictions_tabular_emergency.csv")
    old_winner_df = read_prediction_frame(REPO_ROOT / "hackathon_reefer_dl" / "outputs" / "predictions_final_blend.csv")
    hybrid_df = read_prediction_frame(REPO_ROOT / "hackathon_reefer_dl" / "outputs" / "predictions_anchor_tail_v1.csv")

    frame = frame.join(anchor_df.set_index("timestamp").rename(columns={"point": "anchor_kw", "p90": "anchor_p90_kw"}))
    frame = frame.join(tail_df.set_index("timestamp").rename(columns={"point": "tail_kw", "p90": "tail_p90_kw"}))
    frame = frame.join(
        old_winner_df.set_index("timestamp").rename(columns={"point": "old_winner_kw", "p90": "old_winner_p90_kw"})
    )
    frame = frame.join(hybrid_df.set_index("timestamp").rename(columns={"point": "hybrid_kw", "p90": "hybrid_p90_kw"}))
    frame["hybrid_swapped"] = (frame["anchor_kw"] - frame["hybrid_kw"]).abs() > 1e-9
    frame = frame.reset_index()
    return frame, history


def metric_bundle_for_frame(public_frame: pd.DataFrame, point_key: str, p90_key: str) -> dict[str, float]:
    metrics = composite_metrics(
        public_frame["actual_kw"].to_numpy(dtype=float),
        public_frame[point_key].to_numpy(dtype=float),
        public_frame[p90_key].to_numpy(dtype=float),
    )
    return metrics.to_dict()


def plot_history_context(history: pd.Series, public_frame: pd.DataFrame) -> None:
    rolling = history.rolling(24, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(12.5, 4.6), constrained_layout=True)
    ax.plot(history.index, history.values, color=COLORS["slate"], alpha=0.28, linewidth=0.9, label="Hourly load")
    ax.plot(rolling.index, rolling.values, color=COLORS["teal"], linewidth=1.9, label="24h rolling mean")

    start = public_frame["timestamp"].min()
    end = public_frame["timestamp"].max()
    ax.axvspan(start, end, color=COLORS["orange"], alpha=0.13)
    ax.annotate(
        "Public target window\nJanuary 1-10, 2026",
        xy=(start, float(history.loc[start])),
        xytext=(18, 28),
        textcoords="offset points",
        fontsize=10,
        color=COLORS["orange"],
        fontweight="bold",
    )

    ax.set_title("Year-Long Load History With The Public Forecast Window Highlighted", loc="left", pad=14)
    ax.set_ylabel("Aggregate reefer load")
    ax.yaxis.set_major_formatter(FuncFormatter(kw_formatter))
    ax.grid(axis="y")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", ncol=2)
    save_svg(fig, "history_context.svg")


def plot_january_forecast(public_frame: pd.DataFrame, peak_threshold: float) -> None:
    routed = public_frame[public_frame["hybrid_swapped"]]
    fig, ax = plt.subplots(figsize=(12.5, 5.8), constrained_layout=True)

    ax.fill_between(
        public_frame["timestamp"],
        public_frame["hybrid_kw"],
        public_frame["hybrid_p90_kw"],
        color=COLORS["sand"],
        alpha=0.55,
        label="Anchor-tail hybrid P90 band",
    )
    ax.plot(public_frame["timestamp"], public_frame["actual_kw"], color=COLORS["ink"], linewidth=2.6, label="Observed load")
    ax.plot(
        public_frame["timestamp"],
        public_frame["anchor_kw"],
        color=COLORS["slate"],
        linewidth=1.6,
        linestyle=(0, (4, 3)),
        label="Strong anchor forecast",
    )
    ax.plot(
        public_frame["timestamp"],
        public_frame["hybrid_kw"],
        color=COLORS["teal"],
        linewidth=2.4,
        label="Winning anchor-tail hybrid",
    )
    ax.scatter(
        routed["timestamp"],
        routed["hybrid_kw"],
        s=48,
        color=COLORS["orange"],
        edgecolor="white",
        linewidth=0.5,
        zorder=4,
        label="Tail-routed hours",
    )
    ax.axhline(
        peak_threshold,
        color=COLORS["red"],
        linewidth=1.2,
        linestyle=(0, (2, 3)),
        label="Peak-hour cutoff",
    )

    ax.set_title("January 2026 Public Window: Anchor Forecast, Tail Handoff, And The Winning Hybrid", loc="left", pad=14)
    ax.set_ylabel("Aggregate reefer load")
    ax.yaxis.set_major_formatter(FuncFormatter(kw_formatter))
    ax.grid(axis="y")
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", ncol=2)
    save_svg(fig, "january_forecast.svg")


def plot_daily_mae(public_frame: pd.DataFrame) -> pd.DataFrame:
    working = public_frame.copy()
    working["day"] = working["timestamp"].dt.floor("D")
    daily = (
        working.groupby("day")
        .apply(
            lambda group: pd.Series(
                {
                    "anchor_mae": float(np.mean(np.abs(group["actual_kw"] - group["anchor_kw"]))),
                    "old_winner_mae": float(np.mean(np.abs(group["actual_kw"] - group["old_winner_kw"]))),
                    "hybrid_mae": float(np.mean(np.abs(group["actual_kw"] - group["hybrid_kw"]))),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )

    x = np.arange(len(daily))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10.8, 4.8), constrained_layout=True)
    ax.bar(x - width, daily["anchor_mae"], width=width, color=COLORS["slate"], alpha=0.78, label="Anchor")
    ax.bar(x, daily["old_winner_mae"], width=width, color=COLORS["gold"], alpha=0.9, label="Previous winner")
    ax.bar(x + width, daily["hybrid_mae"], width=width, color=COLORS["teal"], label="Anchor-tail hybrid")

    ax.set_title("Daily MAE: Strong Anchor vs Previous Winner vs New Hybrid", loc="left", pad=14)
    ax.set_ylabel("Mean absolute error")
    ax.yaxis.set_major_formatter(FuncFormatter(kw_formatter))
    ax.set_xticks(x)
    ax.set_xticklabels([day.strftime("%b %d") for day in daily["day"]])
    ax.grid(axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right")
    save_svg(fig, "daily_mae.svg")
    return daily


def plot_peak_scatter(public_frame: pd.DataFrame, peak_threshold: float) -> None:
    routed_mask = public_frame["hybrid_swapped"]
    other_mask = ~routed_mask

    fig, ax = plt.subplots(figsize=(5.8, 5.2), constrained_layout=True)
    ax.scatter(
        public_frame.loc[other_mask, "actual_kw"],
        public_frame.loc[other_mask, "hybrid_kw"],
        s=34,
        alpha=0.7,
        color=COLORS["teal_soft"],
        edgecolor="none",
        label="Hours kept on anchor",
    )
    ax.scatter(
        public_frame.loc[routed_mask, "actual_kw"],
        public_frame.loc[routed_mask, "hybrid_kw"],
        s=54,
        alpha=0.9,
        color=COLORS["orange"],
        edgecolor="white",
        linewidth=0.4,
        label="Tail-routed hours",
    )

    lower = float(min(public_frame["actual_kw"].min(), public_frame["hybrid_kw"].min()) - 15.0)
    upper = float(max(public_frame["actual_kw"].max(), public_frame["hybrid_kw"].max()) + 15.0)
    ax.plot([lower, upper], [lower, upper], color=COLORS["ink"], linestyle=(0, (3, 3)), linewidth=1.1)

    ax.axvline(peak_threshold, color=COLORS["red"], linewidth=1.1, linestyle=(0, (2, 3)))
    ax.axhline(peak_threshold, color=COLORS["red"], linewidth=1.1, linestyle=(0, (2, 3)))

    ax.set_title("Selective Tail Routing Tightens The Highest-Load Hours", loc="left", pad=14)
    ax.set_xlabel("Observed load")
    ax.set_ylabel("Predicted load")
    ax.xaxis.set_major_formatter(FuncFormatter(kw_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(kw_formatter))
    ax.grid(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left")
    save_svg(fig, "peak_scatter.svg")


def plot_model_ladder(public_frame: pd.DataFrame) -> list[dict[str, float | str]]:
    runs = [
        {
            "label": "Organizer baseline",
            "slug": "organizer_baseline",
            **read_json(REPO_ROOT / "hackathon_reefer_dl" / "artifacts" / "public_baseline_metrics.json"),
        },
        {
            "label": "Tail specialist",
            "slug": "tail_specialist",
            **read_json(REPO_ROOT / "hackathon_reefer_dl" / "outputs" / "predictions_tabular_emergency.metrics.json"),
        },
        {
            "label": "Previous winner",
            "slug": "previous_winner",
            **read_json(REPO_ROOT / "hackathon_reefer_dl" / "outputs" / "predictions_final_blend.metrics.json"),
        },
        {
            "label": "Strong anchor forecast",
            "slug": "anchor_forecast",
            **metric_bundle_for_frame(public_frame, "anchor_kw", "anchor_p90_kw"),
        },
        {
            "label": "Winning anchor-tail hybrid",
            "slug": "anchor_tail_hybrid",
            **read_json(REPO_ROOT / "hackathon_reefer_dl" / "outputs" / "predictions_anchor_tail_v1.metrics.json"),
        },
    ]

    ordered = sorted(runs, key=lambda row: row["composite"], reverse=True)
    color_map = {
        "organizer_baseline": COLORS["slate"],
        "tail_specialist": COLORS["orange"],
        "previous_winner": COLORS["gold"],
        "anchor_forecast": "#4F86C6",
        "anchor_tail_hybrid": COLORS["teal"],
    }

    fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    ax.barh(
        [row["label"] for row in ordered],
        [row["composite"] for row in ordered],
        color=[color_map[row["slug"]] for row in ordered],
    )
    ax.invert_yaxis()

    for idx, row in enumerate(ordered):
        ax.text(row["composite"] + 0.8, idx, f"{row['composite']:.2f}", va="center", ha="left", color=COLORS["ink"])

    ax.set_title("Public-Window Composite Score By Forecast Role", loc="left", pad=14)
    ax.set_xlabel("Lower is better")
    ax.grid(axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_svg(fig, "model_ladder.svg")
    return ordered


def plot_active_container_scatter() -> None:
    frame = pd.read_csv(REPO_ROOT / "outputs" / "dataset_analysis" / "hourly_features.csv")
    correlation = float(frame["label_power_kw"].corr(frame["active_container_count"]))
    density_map = LinearSegmentedColormap.from_list(
        "container_density",
        ["#FFF7E5", COLORS["sand"], COLORS["orange"], COLORS["ink"]],
    )

    fig, ax = plt.subplots(figsize=(5.8, 5.2), constrained_layout=True)
    hb = ax.hexbin(
        frame["active_container_count"],
        frame["label_power_kw"],
        gridsize=34,
        mincnt=1,
        cmap=density_map,
        linewidths=0.0,
    )

    x_values = frame["active_container_count"].to_numpy()
    y_values = frame["label_power_kw"].to_numpy()
    coeffs = np.polyfit(x_values, y_values, deg=1)
    x_line = np.linspace(x_values.min(), x_values.max(), 200)
    ax.plot(x_line, coeffs[0] * x_line + coeffs[1], color=COLORS["teal"], linewidth=2.0)

    ax.set_title("The Strongest Driver: Active Reefer Count vs Aggregate Load", loc="left", pad=14)
    ax.set_xlabel("Active container count")
    ax.set_ylabel("Aggregate reefer load")
    ax.yaxis.set_major_formatter(FuncFormatter(kw_formatter))
    ax.grid(True)
    ax.text(
        0.03,
        0.96,
        f"Pearson r = {correlation:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=COLORS["ink"],
        fontweight="bold",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    colorbar = fig.colorbar(hb, ax=ax, shrink=0.86, pad=0.03)
    colorbar.outline.set_visible(False)
    colorbar.ax.yaxis.set_major_formatter(FuncFormatter(metric_formatter))
    colorbar.ax.set_ylabel("Higher density", rotation=270, labelpad=12, color=COLORS["slate"])
    save_svg(fig, "active_count_scatter.svg")


def plot_feature_importance() -> None:
    frame = pd.read_csv(REPO_ROOT / "outputs" / "dataset_analysis" / "feature_importance.csv").head(8)
    frame = frame.sort_values("mae_increase_kw")
    labels = [pretty_label(value) for value in frame["feature"]]

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    ax.barh(labels, frame["mae_increase_kw"], color=COLORS["orange"], alpha=0.88)
    ax.set_title("Top Forward-Safe Features By MAE Impact", loc="left", pad=14)
    ax.set_xlabel("MAE increase when permuted")
    ax.xaxis.set_major_formatter(FuncFormatter(kw_formatter))
    ax.grid(axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_svg(fig, "feature_importance.svg")


def plot_weather_windows() -> None:
    frame = pd.read_csv(REPO_ROOT / "outputs" / "weather_analysis" / "weather_history_window_recommendations.csv")
    frame = frame.head(6).sort_values("efficient_history_window_hours")
    labels = [pretty_label(value) for value in frame["feature"]]
    positions = np.arange(len(frame))

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    ax.barh(positions, frame["efficient_history_window_hours"], color=COLORS["teal"], alpha=0.92, label="Efficient window")
    ax.scatter(
        frame["peak_history_window_hours"],
        positions,
        s=64,
        color=COLORS["orange"],
        zorder=3,
        label="Peak-scoring window",
    )

    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_title("Weather Context Needed For The Next 24 Hours", loc="left", pad=14)
    ax.set_xlabel("Hours of history")
    ax.grid(axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right")
    save_svg(fig, "weather_windows.svg")


def build_weather_proxy_analysis() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int | str]]:
    frame = pd.read_csv(REPO_ROOT / "outputs" / "weather_analysis" / "weather_label_hourly.csv")[
        ["timestamp_utc", "label_power_kw", "weather_temperature_vc_halle3_c"]
    ].dropna()
    frame["timestamp"] = pd.to_datetime(frame["timestamp_utc"], utc=True).dt.tz_convert(None)
    frame = frame.sort_values("timestamp").reset_index(drop=True)

    proxy_sigma = 2.0
    rng = np.random.default_rng(42)
    frame["proxy_temp_c"] = frame["weather_temperature_vc_halle3_c"] + rng.normal(0.0, proxy_sigma, size=len(frame))

    actual_corr = float(frame["weather_temperature_vc_halle3_c"].corr(frame["label_power_kw"]))
    proxy_corr = float(frame["proxy_temp_c"].corr(frame["label_power_kw"]))
    retention = float(proxy_corr / actual_corr) if actual_corr else 0.0

    sweep_rows = []
    for sigma in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        correlations = []
        for seed in range(200):
            trial_rng = np.random.default_rng(1000 + seed)
            proxy = frame["weather_temperature_vc_halle3_c"].to_numpy() + trial_rng.normal(0.0, sigma, size=len(frame))
            correlations.append(float(np.corrcoef(proxy, frame["label_power_kw"].to_numpy())[0, 1]))
        sweep_rows.append(
            {
                "sigma_c": float(sigma),
                "mean_corr": float(np.mean(correlations)),
                "p10_corr": float(np.quantile(correlations, 0.1)),
                "p90_corr": float(np.quantile(correlations, 0.9)),
            }
        )

    sweep = pd.DataFrame(sweep_rows)
    summary = {
        "proxy_sigma_c": proxy_sigma,
        "actual_corr": actual_corr,
        "proxy_corr": proxy_corr,
        "retention_pct": retention * 100.0,
        "rows": int(len(frame)),
        "range_start": frame["timestamp"].min().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "range_end": frame["timestamp"].max().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    return frame, sweep, summary


def plot_weather_proxy_scatter(proxy_frame: pd.DataFrame, summary: dict[str, float | int | str]) -> None:
    sample = proxy_frame.sample(n=min(700, len(proxy_frame)), random_state=42)
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), constrained_layout=True, sharey=True)

    panels = [
        {
            "axis": axes[0],
            "x": "weather_temperature_vc_halle3_c",
            "title": "Realized temperature vs load",
            "color": COLORS["orange"],
            "corr": float(summary["actual_corr"]),
        },
        {
            "axis": axes[1],
            "x": "proxy_temp_c",
            "title": f"Noisy 24h forecast proxy vs load\n($\\sigma$ = {float(summary['proxy_sigma_c']):.1f}°C)",
            "color": COLORS["teal"],
            "corr": float(summary["proxy_corr"]),
        },
    ]

    for panel in panels:
        ax = panel["axis"]
        x_key = panel["x"]
        ax.scatter(
            sample[x_key],
            sample["label_power_kw"],
            s=20,
            alpha=0.16,
            color=panel["color"],
            edgecolor="none",
        )
        coeffs = np.polyfit(proxy_frame[x_key], proxy_frame["label_power_kw"], deg=1)
        x_line = np.linspace(proxy_frame[x_key].min(), proxy_frame[x_key].max(), 120)
        ax.plot(x_line, coeffs[0] * x_line + coeffs[1], color=COLORS["ink"], linewidth=2.0)
        ax.set_title(panel["title"], loc="left", pad=10)
        ax.set_xlabel("Temperature")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}°C"))
        ax.grid(True)
        ax.text(
            0.04,
            0.95,
            f"r = {panel['corr']:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            color=COLORS["ink"],
            fontweight="bold",
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Aggregate reefer load")
    axes[0].yaxis.set_major_formatter(FuncFormatter(kw_formatter))
    fig.suptitle("Proxy Weather Forecasts Still Track Reefer Load", x=0.02, ha="left", fontsize=17, color=COLORS["ink"])
    save_svg(fig, "forecast_weather_proxy_scatter.svg")


def plot_weather_proxy_noise_sweep(sweep: pd.DataFrame, summary: dict[str, float | int | str]) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)

    ax.fill_between(
        sweep["sigma_c"],
        sweep["p10_corr"],
        sweep["p90_corr"],
        color=COLORS["teal_soft"],
        alpha=0.55,
        label="10th-90th percentile over 200 proxy runs",
    )
    ax.plot(
        sweep["sigma_c"],
        sweep["mean_corr"],
        color=COLORS["teal"],
        linewidth=2.4,
        marker="o",
        label="Mean proxy correlation",
    )
    ax.axhline(
        float(summary["actual_corr"]),
        color=COLORS["orange"],
        linewidth=1.4,
        linestyle=(0, (4, 3)),
        label="Realized same-hour temperature correlation",
    )
    ax.scatter(
        [float(summary["proxy_sigma_c"])],
        [float(summary["proxy_corr"])],
        color=COLORS["ink"],
        s=52,
        zorder=4,
    )
    ax.annotate(
        f"{float(summary['proxy_corr']):.3f} at {float(summary['proxy_sigma_c']):.1f}°C noise",
        xy=(float(summary["proxy_sigma_c"]), float(summary["proxy_corr"])),
        xytext=(12, -24),
        textcoords="offset points",
        color=COLORS["ink"],
        fontsize=10,
        fontweight="bold",
    )

    ax.set_title("Temperature Signal Retention Under Forecast Noise", loc="left", pad=14)
    ax.set_xlabel("Injected 24h forecast noise")
    ax.set_ylabel("Correlation with load")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1f}°C"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
    ax.grid(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower left")
    save_svg(fig, "forecast_weather_proxy_noise_sweep.svg")


def write_presentation_data(
    public_frame: pd.DataFrame,
    daily_mae: pd.DataFrame,
    ordered_runs: list[dict[str, float | str]],
    peak_threshold: float,
    weather_proxy_summary: dict[str, float | int | str],
) -> None:
    winning = next(row for row in ordered_runs if row["slug"] == "anchor_tail_hybrid")
    baseline = next(row for row in ordered_runs if row["slug"] == "organizer_baseline")
    anchor = next(row for row in ordered_runs if row["slug"] == "anchor_forecast")
    tail = next(row for row in ordered_runs if row["slug"] == "tail_specialist")
    previous_winner = next(row for row in ordered_runs if row["slug"] == "previous_winner")
    emergency_summary = read_json(REPO_ROOT / "hackathon_reefer_dl" / "outputs" / "model_tabular_emergency" / "training_summary.json")
    routed_rows = public_frame[public_frame["hybrid_swapped"]].copy()

    best_day = daily_mae.loc[daily_mae["hybrid_mae"].idxmin()]
    toughest_day = daily_mae.loc[daily_mae["hybrid_mae"].idxmax()]
    improvement_vs_baseline = (baseline["composite"] - winning["composite"]) / baseline["composite"] * 100.0
    improvement_vs_anchor = (anchor["composite"] - winning["composite"]) / anchor["composite"] * 100.0
    improvement_vs_previous = (previous_winner["composite"] - winning["composite"]) / previous_winner["composite"] * 100.0

    data = {
        "headline": {
            "windowLabel": "January 1-10, 2026",
            "windowHours": int(len(public_frame)),
            "historyHours": int(emergency_summary["history_hours"]),
            "forecastHorizonHours": int(emergency_summary["forecast_horizon_hours"]),
            "datasetRows": 8983,
            "peakThresholdKw": round(float(peak_threshold), 1),
            "bestComposite": round(float(winning["composite"]), 2),
            "baselineComposite": round(float(baseline["composite"]), 2),
            "anchorComposite": round(float(anchor["composite"]), 2),
            "previousWinnerComposite": round(float(previous_winner["composite"]), 2),
            "improvementVsBaselinePct": round(float(improvement_vs_baseline), 1),
            "improvementVsAnchorPct": round(float(improvement_vs_anchor), 1),
            "tailHours": int(routed_rows.shape[0]),
        },
        "heroMetrics": [
            {
                "label": "Public Composite",
                "value": f"{winning['composite']:.2f}",
                "detail": "Measured on the released public target window",
            },
            {
                "label": "Gain vs Anchor",
                "value": f"{improvement_vs_anchor:.1f}%",
                "detail": "Improvement over the already-strong anchor forecast",
            },
            {
                "label": "Gain vs Old Winner",
                "value": f"{improvement_vs_previous:.1f}%",
                "detail": "Improvement over the previous best public result",
            },
            {
                "label": "Tail-Routed Hours",
                "value": f"{int(routed_rows.shape[0])}",
                "detail": "Only the highest anchor hours are handed to the peak specialist",
            },
        ],
        "challengeFacts": [
            "Target window: January 1, 2026 to January 10, 2026 (223 hours).",
            "Scoring: 0.5 * MAE_all + 0.3 * MAE_peak + 0.2 * pinball_p90.",
            "Peak-hour cutoff on the public window: "
            + f"{peak_threshold:.1f} kW.",
            "The routed tail hours are concentrated at the end of the window, from "
            + f"{routed_rows['timestamp'].min().strftime('%b %d %H:%M')} to {routed_rows['timestamp'].max().strftime('%b %d %H:%M')} UTC.",
            "Best day in the public window: "
            + f"{pd.Timestamp(best_day['day']).strftime('%b %d')} with {best_day['hybrid_mae']:.1f} kW MAE.",
            "Hardest day in the public window: "
            + f"{pd.Timestamp(toughest_day['day']).strftime('%b %d')} with {toughest_day['hybrid_mae']:.1f} kW MAE.",
        ],
        "blendChips": [
            "Anchor forecast = outputs/results/predictions.csv",
            "Tail specialist = predictions_tabular_emergency.csv",
            "Top 15 highest anchor hours are rerouted",
            "Tail point blend = 27.5% anchor + 72.5% tail",
            "Tail p90 blend = 95% anchor + 5% tail",
            "Final score = 32.43 on the public window",
        ],
        "experimentRows": [
            {
                "label": row["label"],
                "composite": round(float(row["composite"]), 2),
                "maeAll": round(float(row["mae_all"]), 2),
                "maePeak": round(float(row["mae_peak"]), 2),
                "pinball": round(float(row["pinball_p90"]), 2),
                "highlight": row["slug"] == "anchor_tail_hybrid",
            }
            for row in ordered_runs
        ],
        "featureNotes": [
            "The tail specialist is not the best overall model, but it is much better on the hardest high-load hours.",
            "Strong anchor forecast: "
            + f"{anchor['composite']:.2f} composite, {anchor['mae_peak']:.2f} peak MAE.",
            "Tail specialist: "
            + f"{tail['composite']:.2f} composite, {tail['mae_peak']:.2f} peak MAE.",
            "The hybrid keeps the anchor on 208 hours and only hands 15 hours to the peak specialist.",
            "The tabular tail specialist still uses the strict T-24 setup with "
            + f"{int(emergency_summary['history_hours'])}h history and {int(emergency_summary['forecast_horizon_hours'])}h horizon.",
        ],
        "outlookMetrics": [
            {
                "label": "Realized Temp Correlation",
                "value": f"{float(weather_proxy_summary['actual_corr']):.3f}",
                "detail": "VC Halle 3 temperature vs load on the full weather-overlap window",
            },
            {
                "label": "Noisy Forecast Correlation",
                "value": f"{float(weather_proxy_summary['proxy_corr']):.3f}",
                "detail": f"Synthetic 24h forecast proxy with {float(weather_proxy_summary['proxy_sigma_c']):.1f}°C noise",
            },
            {
                "label": "Signal Retained",
                "value": f"{float(weather_proxy_summary['retention_pct']):.1f}%",
                "detail": "Share of the realized weather signal that survives the forecast-noise stress test",
            },
        ],
        "outlookPoints": [
            "Temperature is already the strongest same-hour weather signal in the existing analysis, led by VC Halle 3.",
            "Because archived forecast files are not part of the participant package, we build a proxy forecast by taking realized temperature and adding Gaussian noise.",
            "We then treat that noisy value as if it were a 24-hour-ahead forecast available at prediction time and measure how much correlation to load survives.",
            f"Across {int(weather_proxy_summary['rows'])} weather-overlap hours from {str(weather_proxy_summary['range_start'])[:10]} to {str(weather_proxy_summary['range_end'])[:10]}, the proxy still keeps a strong signal.",
            "This is a sensitivity analysis, not a true historical forecast backtest, so it should be read as evidence of likely usefulness rather than proof of final leaderboard gain.",
        ],
        "outlookSteps": [
            "Ingest 24h site weather forecasts for temperature, wind, and spread as first-class T-24 features.",
            "Feed forecast temperatures into both the anchor model and the tail specialist, especially for the late-window peaks that trigger rerouting.",
            "Add forecast-minus-recent-history features so the model can react to tomorrow being warmer or colder than the recent reefer regime.",
            "Re-run the anchor-tail routing experiment with weather-aware features and compare whether fewer hours need specialist handoff.",
        ],
        "takeaways": [
            "The new winner is a routed hybrid: a strong anchor forecast for almost every hour, plus a peak specialist only where the anchor itself predicts the heaviest load.",
            "This is why the score jumps so sharply: the anchor protects the broad shape of the curve, while the tail specialist only touches the handful of hours where peak sensitivity matters most.",
            "Container count remains the clearest operational driver, and temperature history still explains why the hardest hours cluster late in the January window.",
            "The most natural next feature family is a 24h weather forecast, because even a noisy proxy forecast still preserves most of the temperature-to-load relationship.",
            "The site is generated from repo artifacts, so future improvements in `hackathon_reefer_dl` can be pushed into the presentation by rerunning one script.",
        ],
        "generatedAt": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
    }

    output = "window.presentationData = " + json.dumps(data, indent=2) + ";\n"
    (DATA_DIR / "presentation-data.js").write_text(output, encoding="utf-8")


def main() -> None:
    apply_style()
    public_frame, history = build_public_window_frame()
    peak_threshold = float(np.quantile(public_frame["actual_kw"], 0.9))
    weather_proxy_frame, weather_proxy_sweep, weather_proxy_summary = build_weather_proxy_analysis()

    plot_history_context(history, public_frame)
    plot_january_forecast(public_frame, peak_threshold)
    daily_mae = plot_daily_mae(public_frame)
    plot_peak_scatter(public_frame, peak_threshold)
    ordered_runs = plot_model_ladder(public_frame)
    plot_active_container_scatter()
    plot_feature_importance()
    plot_weather_windows()
    plot_weather_proxy_scatter(weather_proxy_frame, weather_proxy_summary)
    plot_weather_proxy_noise_sweep(weather_proxy_sweep, weather_proxy_summary)
    write_presentation_data(public_frame, daily_mae, ordered_runs, peak_threshold, weather_proxy_summary)
    print(f"Presentation assets written to {SITE_ROOT}")


if __name__ == "__main__":
    main()
