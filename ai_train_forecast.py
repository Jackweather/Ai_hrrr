from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.spatial import cKDTree

from HRRR_prate import (
    CONUS_EXTENT,
    DEFAULT_DOWNLOAD_WORKERS,
    DEFAULT_LOOKBACK_HOURS,
    DEFAULT_MAX_REQUEST_RETRIES,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_REQUEST_MIN_INTERVAL,
    NomadsClient,
    build_url,
    download_file,
    load_prate_mmhr,
    local_grib_path,
    resolve_latest_available_cycle,
    setup_logging,
    valid_time,
)
from mrms_realtime import (
    MIN_REFLECTIVITY_DBZ,
    REFLECTIVITY_COLORS,
    REFLECTIVITY_LEVELS_DBZ,
    ensure_utc,
    iter_mrms_grib_files,
    load_mrms_reflectivity,
    parse_time_from_mrms_filename,
    smooth_field_for_contours,
)

UTC = dt.timezone.utc
DEFAULT_HRRR_STRIDE = 8
DEFAULT_MRMS_STRIDE = 24
DEFAULT_MATCH_TOLERANCE_MINUTES = 10
DEFAULT_MAX_SAMPLES_PER_PAIR = 20000
DEFAULT_REQUEST_TIMEOUT = 180
DEFAULT_TRAINING_DAYS = 7
DEFAULT_TEST_FRACTION = 0.2
DEFAULT_NEIGHBORHOOD_RADIUS = 2


def relative_path_string(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


@dataclass(frozen=True)
class LocalRun:
    tag: str
    init_time: dt.datetime

    @property
    def cycle_hour(self) -> int:
        return self.init_time.hour

    def max_forecast_hour(self) -> int:
        return 48 if self.cycle_hour in {0, 6, 12, 18} else 18


@dataclass(frozen=True)
class TrainingPairSamples:
    features: np.ndarray
    targets: np.ndarray
    valid_time: np.datetime64
    summary: dict[str, str | int | float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a simple reflectivity forecast model from HRRR precipitation rate and "
            "local MRMS truth snapshots, then generate forecast PNGs and verification outputs."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Base directory used for HRRR data, MRMS history, and AI forecast outputs.",
    )
    parser.add_argument(
        "--max-forecast-hour",
        type=int,
        default=None,
        help="Highest HRRR forecast hour to download and turn into AI forecast PNGs. Defaults to the full run length.",
    )
    parser.add_argument(
        "--cycle-lookback-hours",
        type=int,
        default=DEFAULT_LOOKBACK_HOURS,
        help="How many recent hourly HRRR cycles to inspect.",
    )
    parser.add_argument(
        "--match-tolerance-minutes",
        type=int,
        default=DEFAULT_MATCH_TOLERANCE_MINUTES,
        help="Maximum allowed timing gap between an HRRR valid time and an MRMS snapshot.",
    )
    parser.add_argument(
        "--training-days",
        type=int,
        default=DEFAULT_TRAINING_DAYS,
        help="How many days of local HRRR and MRMS history to use for model training.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=DEFAULT_TEST_FRACTION,
        help="Fraction of the most recent matched examples held out for verification.",
    )
    parser.add_argument(
        "--neighborhood-radius",
        type=int,
        default=DEFAULT_NEIGHBORHOOD_RADIUS,
        help="Radius in grid cells used to build local HRRR neighborhood features.",
    )
    parser.add_argument(
        "--hrrr-stride",
        type=int,
        default=DEFAULT_HRRR_STRIDE,
        help="Grid decimation factor used to sample HRRR fields for training and plotting.",
    )
    parser.add_argument(
        "--mrms-stride",
        type=int,
        default=DEFAULT_MRMS_STRIDE,
        help="Grid decimation factor used to sample MRMS fields before aligning to HRRR.",
    )
    parser.add_argument(
        "--max-samples-per-pair",
        type=int,
        default=DEFAULT_MAX_SAMPLES_PER_PAIR,
        help="Maximum training samples to keep from each matched HRRR/MRMS pair.",
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=DEFAULT_DOWNLOAD_WORKERS,
        help="Reserved for compatibility with the HRRR downloader; files are fetched serially here.",
    )
    parser.add_argument(
        "--request-min-interval",
        type=float,
        default=DEFAULT_REQUEST_MIN_INTERVAL,
        help="Minimum seconds between outbound NOMADS requests.",
    )
    parser.add_argument(
        "--max-request-retries",
        type=int,
        default=DEFAULT_MAX_REQUEST_RETRIES,
        help="How many times to retry transient NOMADS errors.",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=DEFAULT_REQUEST_TIMEOUT,
        help="Seconds allowed for each HRRR request.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload HRRR files and rebuild forecast PNGs.",
    )
    parser.add_argument(
        "--skip-hrrr-download",
        action="store_true",
        help="Use already-downloaded HRRR files in DATA/grib.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_forecast_hour is not None and args.max_forecast_hour < 0:
        raise ValueError("--max-forecast-hour must be zero or greater.")
    if args.cycle_lookback_hours < 0:
        raise ValueError("--cycle-lookback-hours must be zero or greater.")
    if args.match_tolerance_minutes < 0:
        raise ValueError("--match-tolerance-minutes must be zero or greater.")
    if args.training_days < 1:
        raise ValueError("--training-days must be at least 1.")
    if not 0.05 <= args.test_fraction <= 0.5:
        raise ValueError("--test-fraction must be between 0.05 and 0.5.")
    if args.neighborhood_radius < 0:
        raise ValueError("--neighborhood-radius must be zero or greater.")
    if args.hrrr_stride < 1:
        raise ValueError("--hrrr-stride must be at least 1.")
    if args.mrms_stride < 1:
        raise ValueError("--mrms-stride must be at least 1.")
    if args.max_samples_per_pair < 100:
        raise ValueError("--max-samples-per-pair must be at least 100.")
    if args.request_timeout < 1:
        raise ValueError("--request-timeout must be at least 1.")


def forecast_png_path(root: Path, run_tag: str, forecast_hour: int) -> Path:
    return root / "ai_forecast" / "png" / run_tag / f"ai_reflectivity_f{forecast_hour:02d}.png"


def verification_dir(root: Path, run_tag: str) -> Path:
    return root / "ai_forecast" / "verification" / run_tag


def model_path(root: Path, run_tag: str) -> Path:
    return root / "ai_forecast" / "models" / f"reflectivity_model_{run_tag}.pkl"


def metrics_history_path(root: Path) -> Path:
    return root / "ai_forecast" / "verification" / "metrics_history.jsonl"


def reset_png_output(root: Path) -> None:
    for png_root in (root / "ai_forecast" / "png", root / "ai_forecast" / "verification"):
        if png_root.exists():
            shutil.rmtree(png_root)
        png_root.mkdir(parents=True, exist_ok=True)


def reflectivity_cmap():
    from matplotlib.colors import BoundaryNorm, ListedColormap

    boundaries = np.asarray(REFLECTIVITY_LEVELS_DBZ, dtype=np.float32)
    cmap = ListedColormap(REFLECTIVITY_COLORS, name="ai_reflectivity")
    norm = BoundaryNorm(boundaries, cmap.N, clip=False)
    return cmap, norm


def mask_reflectivity_for_plot(field: np.ndarray) -> np.ma.MaskedArray:
    smoothed = smooth_field_for_contours(field)
    return np.ma.masked_less_equal(np.ma.masked_invalid(smoothed), MIN_REFLECTIVITY_DBZ)


def ensure_hrrr_files(
    root: Path,
    client: NomadsClient,
    run_cycle,
    max_forecast_hour: int,
    overwrite: bool,
) -> None:
    for forecast_hour in range(max_forecast_hour + 1):
        destination = local_grib_path(root, run_cycle, forecast_hour)
        if destination.exists() and not overwrite:
            continue
        url = build_url(run_cycle, forecast_hour)
        download_file(client, url, destination, overwrite=overwrite)
        logging.info("Downloaded %s", destination)


def parse_run_tag(tag: str) -> dt.datetime:
    return dt.datetime.strptime(tag, "%Y%m%d_%HZ").replace(tzinfo=UTC)


def iter_local_hrrr_runs(root: Path) -> list[LocalRun]:
    grib_root = root / "grib"
    if not grib_root.exists():
        return []

    runs: list[LocalRun] = []
    for child in grib_root.iterdir():
        if not child.is_dir():
            continue
        try:
            init_time = parse_run_tag(child.name)
        except ValueError:
            continue
        runs.append(LocalRun(tag=child.name, init_time=init_time))
    return sorted(runs, key=lambda run: run.init_time)


def local_grib_file_for_run(root: Path, run: LocalRun, forecast_hour: int) -> Path:
    return root / "grib" / run.tag / f"hrrr.t{run.cycle_hour:02d}z.wrfsfcf{forecast_hour:02d}.grib2"


def nearest_mrms_file(root: Path, target_time: dt.datetime, tolerance_minutes: int) -> Path | None:
    candidates = iter_mrms_grib_files(root)
    if not candidates:
        return None

    target_time = ensure_utc(target_time)
    tolerance = dt.timedelta(minutes=tolerance_minutes)
    best_path: Path | None = None
    best_delta: dt.timedelta | None = None

    for path in candidates:
        candidate_time = parse_time_from_mrms_filename(path)
        delta = abs(candidate_time - target_time)
        if delta > tolerance:
            continue
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_path = path
    return best_path


def align_mrms_to_hrrr(
    hrrr_lats: np.ndarray,
    hrrr_lons: np.ndarray,
    mrms_values: np.ndarray,
    mrms_lats: np.ndarray,
    mrms_lons: np.ndarray,
    mrms_stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    mrms_slice = (slice(None, None, mrms_stride), slice(None, None, mrms_stride))
    mrms_values_coarse = mrms_values[mrms_slice]
    mrms_lats_coarse = mrms_lats[mrms_slice]
    mrms_lons_coarse = mrms_lons[mrms_slice]
    finite_mask = np.isfinite(mrms_values_coarse)

    if not np.any(finite_mask):
        aligned = np.full(hrrr_lats.shape, np.nan, dtype=np.float32)
        distances = np.full(hrrr_lats.shape, np.nan, dtype=np.float32)
        return aligned, distances

    source_points = np.column_stack((mrms_lats_coarse[finite_mask], mrms_lons_coarse[finite_mask])).astype(np.float32)
    tree = cKDTree(source_points)
    query_points = np.column_stack((hrrr_lats.ravel(), hrrr_lons.ravel())).astype(np.float32)
    distances, indices = tree.query(query_points, k=1)
    aligned_flat = mrms_values_coarse[finite_mask][indices]
    return aligned_flat.reshape(hrrr_lats.shape).astype(np.float32), distances.reshape(hrrr_lats.shape).astype(np.float32)


def rolling_statistic(field: np.ndarray, radius: int, reducer: str) -> np.ndarray:
    if radius <= 0:
        return field.astype(np.float32, copy=True)

    window = radius * 2 + 1
    padded = np.pad(field, radius, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (window, window))
    if reducer == "mean":
        return np.mean(windows, axis=(-1, -2), dtype=np.float32)
    if reducer == "max":
        return np.max(windows, axis=(-1, -2))
    if reducer == "std":
        return np.std(windows, axis=(-1, -2), dtype=np.float32)
    raise ValueError(f"Unsupported reducer: {reducer}")


def build_feature_matrix(
    prate_mmhr: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    forecast_hour: int,
    valid_time_utc: dt.datetime,
    neighborhood_radius: int,
) -> np.ndarray:
    prate_nonnegative = np.maximum(prate_mmhr, 0.0).astype(np.float32, copy=False)
    log_prate = np.log1p(prate_nonnegative, dtype=np.float32)
    local_mean = rolling_statistic(prate_nonnegative, neighborhood_radius, "mean")
    local_max = rolling_statistic(prate_nonnegative, neighborhood_radius, "max")
    local_std = rolling_statistic(prate_nonnegative, neighborhood_radius, "std")
    gradient_y, gradient_x = np.gradient(prate_nonnegative)
    valid_time_utc = ensure_utc(valid_time_utc)
    day_of_year = valid_time_utc.timetuple().tm_yday
    hour_fraction = valid_time_utc.hour + (valid_time_utc.minute / 60.0)
    diurnal_angle = 2.0 * np.pi * (hour_fraction / 24.0)
    annual_angle = 2.0 * np.pi * (day_of_year / 366.0)

    feature_count = 15
    features = np.empty((prate_mmhr.size, feature_count), dtype=np.float32)
    features[:, 0] = log_prate.ravel()
    features[:, 1] = np.sqrt(prate_nonnegative.ravel(), dtype=np.float32)
    features[:, 2] = prate_nonnegative.ravel()
    features[:, 3] = local_mean.ravel()
    features[:, 4] = local_max.ravel()
    features[:, 5] = local_std.ravel()
    features[:, 6] = gradient_x.ravel()
    features[:, 7] = gradient_y.ravel()
    features[:, 8] = np.float32(forecast_hour)
    features[:, 9] = np.float32(np.sin(diurnal_angle))
    features[:, 10] = np.float32(np.cos(diurnal_angle))
    features[:, 11] = np.float32(np.sin(annual_angle))
    features[:, 12] = np.float32(np.cos(annual_angle))
    features[:, 13] = lats.ravel().astype(np.float32, copy=False)
    features[:, 14] = lons.ravel().astype(np.float32, copy=False)
    return features


def split_training_pairs(
    samples: list[TrainingPairSamples],
    test_fraction: float,
) -> tuple[list[TrainingPairSamples], list[TrainingPairSamples]]:
    ordered = sorted(samples, key=lambda item: item.valid_time)
    split_index = max(1, int(np.floor(len(ordered) * (1.0 - test_fraction))))
    split_index = min(split_index, len(ordered) - 1)
    return ordered[:split_index], ordered[split_index:]


def stack_training_pairs(
    samples: list[TrainingPairSamples],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.concatenate([sample.features for sample in samples], axis=0)
    targets = np.concatenate([sample.targets for sample in samples], axis=0)
    valid_times = np.array([sample.valid_time for sample in samples], dtype="datetime64[ns]")
    return features, targets, valid_times


def training_examples_for_pair(
    root: Path,
    run_cycle,
    forecast_hour: int,
    match_tolerance_minutes: int,
    hrrr_stride: int,
    mrms_stride: int,
    max_samples: int,
    neighborhood_radius: int,
    rng: np.random.Generator,
) -> TrainingPairSamples | None:
    target_time = valid_time(run_cycle, forecast_hour)
    mrms_path = nearest_mrms_file(root, target_time, tolerance_minutes=match_tolerance_minutes)
    if mrms_path is None:
        return None

    if isinstance(run_cycle, LocalRun):
        grib_path = local_grib_file_for_run(root, run_cycle, forecast_hour)
    else:
        grib_path = local_grib_path(root, run_cycle, forecast_hour)

    prate_mmhr, hrrr_lats, hrrr_lons = load_prate_mmhr(grib_path)
    mrms_values, mrms_lats, mrms_lons, mrms_valid_time = load_mrms_reflectivity(mrms_path)

    prate_coarse = prate_mmhr[::hrrr_stride, ::hrrr_stride]
    hrrr_lats_coarse = hrrr_lats[::hrrr_stride, ::hrrr_stride]
    hrrr_lons_coarse = hrrr_lons[::hrrr_stride, ::hrrr_stride]
    target_grid, distance_grid = align_mrms_to_hrrr(
        hrrr_lats=hrrr_lats_coarse,
        hrrr_lons=hrrr_lons_coarse,
        mrms_values=mrms_values,
        mrms_lats=mrms_lats,
        mrms_lons=mrms_lons,
        mrms_stride=mrms_stride,
    )

    valid_mask = np.isfinite(target_grid) & np.isfinite(prate_coarse) & (distance_grid <= 0.25)
    if not np.any(valid_mask):
        return None

    features = build_feature_matrix(
        prate_coarse,
        hrrr_lats_coarse,
        hrrr_lons_coarse,
        forecast_hour,
        target_time,
        neighborhood_radius,
    )
    targets = target_grid.ravel()
    valid_indices = np.flatnonzero(valid_mask.ravel())
    if valid_indices.size > max_samples:
        valid_indices = rng.choice(valid_indices, size=max_samples, replace=False)

    feature_subset = features[valid_indices].astype(np.float32, copy=False)
    target_subset = targets[valid_indices].astype(np.float32, copy=False)
    pair_valid_time = np.datetime64(ensure_utc(target_time).replace(tzinfo=None), "ns")
    summary = {
        "run_tag": run_cycle.tag,
        "forecast_hour": forecast_hour,
        "mrms_path": relative_path_string(mrms_path, root),
        "target_valid_time_utc": ensure_utc(target_time).isoformat(),
        "mrms_valid_time_utc": ensure_utc(mrms_valid_time).isoformat(),
        "sample_count": int(valid_indices.size),
    }
    return TrainingPairSamples(
        features=feature_subset,
        targets=target_subset,
        valid_time=pair_valid_time,
        summary=summary,
    )


def collect_training_pairs(
    root: Path,
    run_limit_time: dt.datetime,
    max_forecast_hour: int,
    training_days: int,
    match_tolerance_minutes: int,
    hrrr_stride: int,
    mrms_stride: int,
    max_samples: int,
    neighborhood_radius: int,
    rng: np.random.Generator,
) -> list[TrainingPairSamples]:
    earliest_time = ensure_utc(run_limit_time) - dt.timedelta(days=training_days)
    training_pairs: list[TrainingPairSamples] = []

    for run in iter_local_hrrr_runs(root):
        if run.init_time < earliest_time or run.init_time > ensure_utc(run_limit_time):
            continue
        usable_forecast_hour = min(max_forecast_hour, run.max_forecast_hour())
        for forecast_hour in range(usable_forecast_hour + 1):
            grib_path = local_grib_file_for_run(root, run, forecast_hour)
            if not grib_path.exists():
                continue
            pair = training_examples_for_pair(
                root=root,
                run_cycle=run,
                forecast_hour=forecast_hour,
                match_tolerance_minutes=match_tolerance_minutes,
                hrrr_stride=hrrr_stride,
                mrms_stride=mrms_stride,
                max_samples=max_samples,
                neighborhood_radius=neighborhood_radius,
                rng=rng,
            )
            if pair is None:
                continue
            training_pairs.append(pair)

    return training_pairs


def append_metrics_history(root: Path, payload: dict[str, object]) -> None:
    history_path = metrics_history_path(root)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as file_handle:
        file_handle.write(json.dumps(payload) + "\n")


def plot_forecast_map(
    save_path: Path,
    run_tag: str,
    forecast_hour: int,
    valid_time_utc: dt.datetime,
    lats: np.ndarray,
    lons: np.ndarray,
    predicted_reflectivity: np.ndarray,
) -> None:
    cmap, norm = reflectivity_cmap()
    masked = mask_reflectivity_for_plot(predicted_reflectivity)

    figure = plt.figure(figsize=(15, 9))
    axis = plt.axes(projection=ccrs.PlateCarree())
    axis.set_extent(CONUS_EXTENT, crs=ccrs.PlateCarree())
    axis.coastlines(linewidth=0.7)
    axis.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5)
    axis.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.3)
    axis.add_feature(cfeature.LAKES.with_scale("50m"), linewidth=0.2, edgecolor="#6b7c93", facecolor="none")

    mesh = axis.contourf(
        lons,
        lats,
        masked,
        levels=REFLECTIVITY_LEVELS_DBZ,
        cmap=cmap,
        norm=norm,
        extend="max",
        antialiased=True,
        transform=ccrs.PlateCarree(),
    )

    axis.set_title(
        (
            f"AI Reflectivity Forecast F{forecast_hour:02d}\n"
            f"Run {run_tag} | Valid {ensure_utc(valid_time_utc).strftime('%Y-%m-%d %H:%M UTC')}"
        ),
        fontsize=15,
        loc="left",
        pad=10,
    )

    colorbar = plt.colorbar(mesh, ax=axis, shrink=0.84, pad=0.02)
    colorbar.set_label("Predicted reflectivity (dBZ)")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_verification_scatter(save_path: Path, truth: np.ndarray, prediction: np.ndarray, metrics: dict[str, float]) -> None:
    figure, axis = plt.subplots(figsize=(8, 8))
    axis.scatter(truth, prediction, s=6, alpha=0.25, color="#0f6cbd", edgecolors="none")
    bounds = [0.0, max(float(np.nanmax(truth)), float(np.nanmax(prediction)), 5.0)]
    axis.plot(bounds, bounds, linestyle="--", color="#333333", linewidth=1.0)
    axis.set_xlim(bounds)
    axis.set_ylim(bounds)
    axis.set_xlabel("Observed MRMS reflectivity (dBZ)")
    axis.set_ylabel("Predicted reflectivity (dBZ)")
    axis.set_title(
        (
            "AI Forecast Verification\n"
            f"RMSE {metrics['rmse_dbz']:.2f} | MAE {metrics['mae_dbz']:.2f} | R2 {metrics['r2']:.3f}"
        )
    )
    axis.grid(True, alpha=0.2)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_latest_comparison(
    save_path: Path,
    lats: np.ndarray,
    lons: np.ndarray,
    prediction: np.ndarray,
    truth: np.ndarray,
    title: str,
) -> None:
    cmap, norm = reflectivity_cmap()
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(18, 8),
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True,
    )

    for axis, field, panel_title in (
        (axes[0], prediction, "Predicted"),
        (axes[1], truth, "Observed MRMS"),
    ):
        axis.set_extent(CONUS_EXTENT, crs=ccrs.PlateCarree())
        axis.coastlines(linewidth=0.7)
        axis.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5)
        axis.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.3)
        mesh = axis.contourf(
            lons,
            lats,
            mask_reflectivity_for_plot(field),
            levels=REFLECTIVITY_LEVELS_DBZ,
            cmap=cmap,
            norm=norm,
            extend="max",
            antialiased=True,
            transform=ccrs.PlateCarree(),
        )
        axis.set_title(panel_title)

    figure.suptitle(title, fontsize=14)
    colorbar = figure.colorbar(mesh, ax=axes, shrink=0.85, pad=0.03)
    colorbar.set_label("Reflectivity (dBZ)")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    validate_args(args)
    setup_logging(args.log_level)

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    reset_png_output(output_root)

    mrms_files = iter_mrms_grib_files(output_root)
    if not mrms_files:
        raise RuntimeError(
            "No MRMS files found under DATA/mrms/raw. Run mrms_realtime.py first so the AI script has truth data to learn from."
        )

    client = NomadsClient(
        timeout=args.request_timeout,
        min_interval_seconds=args.request_min_interval,
        max_retries=args.max_request_retries,
    )
    run_cycle = resolve_latest_available_cycle(
        now_utc=dt.datetime.now(tz=UTC),
        lookback_hours=args.cycle_lookback_hours,
        client=client,
    )
    if args.max_forecast_hour is None:
        max_forecast_hour = run_cycle.max_forecast_hour
    else:
        max_forecast_hour = min(args.max_forecast_hour, run_cycle.max_forecast_hour)

    if not args.skip_hrrr_download:
        ensure_hrrr_files(output_root, client, run_cycle, max_forecast_hour=max_forecast_hour, overwrite=args.overwrite)

    rng = np.random.default_rng(42)
    training_pairs = collect_training_pairs(
        root=output_root,
        run_limit_time=run_cycle.init_time,
        max_forecast_hour=max_forecast_hour,
        training_days=args.training_days,
        match_tolerance_minutes=args.match_tolerance_minutes,
        hrrr_stride=args.hrrr_stride,
        mrms_stride=args.mrms_stride,
        max_samples=args.max_samples_per_pair,
        neighborhood_radius=args.neighborhood_radius,
        rng=rng,
    )

    if not training_pairs:
        raise RuntimeError(
            "Could not build any HRRR/MRMS training pairs. Let the MRMS collector run for a while or relax --match-tolerance-minutes."
        )
    if len(training_pairs) < 2:
        raise RuntimeError(
            "Built only one HRRR/MRMS training pair. Collect more MRMS history or increase --match-tolerance-minutes so the model has both train and test data."
        )

    train_pairs, test_pairs = split_training_pairs(training_pairs, test_fraction=args.test_fraction)
    x_train, y_train, t_train = stack_training_pairs(train_pairs)
    x_test, y_test, t_test = stack_training_pairs(test_pairs)

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_depth=6,
        max_iter=300,
        min_samples_leaf=25,
        l2_regularization=0.1,
        random_state=42,
    )
    model.fit(x_train, y_train)

    y_pred = np.clip(model.predict(x_test).astype(np.float32), 0.0, 75.0)
    metrics = {
        "rmse_dbz": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae_dbz": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
        "sample_count": int(y_train.size + y_test.size),
        "train_sample_count": int(y_train.size),
        "test_sample_count": int(y_test.size),
        "training_pair_count": len(training_pairs),
        "training_days": args.training_days,
        "test_fraction": args.test_fraction,
        "training_start_utc": str(t_train.min()) if t_train.size else None,
        "training_end_utc": str(t_train.max()) if t_train.size else None,
        "test_start_utc": str(t_test.min()) if t_test.size else None,
        "test_end_utc": str(t_test.max()) if t_test.size else None,
    }

    model_save_path = model_path(output_root, run_cycle.tag)
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    with model_save_path.open("wb") as file_handle:
        pickle.dump(model, file_handle)

    verify_root = verification_dir(output_root, run_cycle.tag)
    verify_root.mkdir(parents=True, exist_ok=True)
    (verify_root / "metrics.json").write_text(
        json.dumps(
            {
                "run_tag": run_cycle.tag,
                "metrics": metrics,
                "training_pairs": [pair.summary for pair in training_pairs],
                "model_path": relative_path_string(model_save_path, output_root),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    append_metrics_history(
        output_root,
        {
            "run_tag": run_cycle.tag,
            "created_at_utc": dt.datetime.now(tz=UTC).isoformat(),
            "metrics": metrics,
            "model_path": relative_path_string(model_save_path, output_root),
        },
    )
    plot_verification_scatter(verify_root / "verification_scatter.png", y_test, y_pred, metrics)

    latest_comparison_written = False
    for forecast_hour in range(max_forecast_hour + 1):
        prate_mmhr, hrrr_lats, hrrr_lons = load_prate_mmhr(local_grib_path(output_root, run_cycle, forecast_hour))
        prate_coarse = prate_mmhr[::args.hrrr_stride, ::args.hrrr_stride]
        lats_coarse = hrrr_lats[::args.hrrr_stride, ::args.hrrr_stride]
        lons_coarse = hrrr_lons[::args.hrrr_stride, ::args.hrrr_stride]

        feature_grid = build_feature_matrix(
            prate_coarse,
            lats_coarse,
            lons_coarse,
            forecast_hour,
            valid_time(run_cycle, forecast_hour),
            args.neighborhood_radius,
        )
        predicted = np.clip(model.predict(feature_grid).reshape(prate_coarse.shape), 0.0, 75.0).astype(np.float32)

        save_path = forecast_png_path(output_root, run_cycle.tag, forecast_hour)
        plot_forecast_map(
            save_path=save_path,
            run_tag=run_cycle.tag,
            forecast_hour=forecast_hour,
            valid_time_utc=valid_time(run_cycle, forecast_hour),
            lats=lats_coarse,
            lons=lons_coarse,
            predicted_reflectivity=predicted,
        )
        logging.info("Saved AI forecast PNG to %s", save_path)

        if latest_comparison_written:
            continue

        mrms_match = nearest_mrms_file(output_root, valid_time(run_cycle, forecast_hour), args.match_tolerance_minutes)
        if mrms_match is None:
            continue

        observed, mrms_lats, mrms_lons, matched_valid_time = load_mrms_reflectivity(mrms_match)
        observed_aligned, _ = align_mrms_to_hrrr(
            hrrr_lats=lats_coarse,
            hrrr_lons=lons_coarse,
            mrms_values=observed,
            mrms_lats=mrms_lats,
            mrms_lons=mrms_lons,
            mrms_stride=args.mrms_stride,
        )
        comparison_title = (
            f"AI Forecast vs MRMS | Run {run_cycle.tag} F{forecast_hour:02d} | "
            f"MRMS {ensure_utc(matched_valid_time).strftime('%Y-%m-%d %H:%M UTC')}"
        )
        plot_latest_comparison(
            verify_root / f"latest_comparison_f{forecast_hour:02d}.png",
            lats_coarse,
            lons_coarse,
            predicted,
            observed_aligned,
            comparison_title,
        )
        latest_comparison_written = True

    logging.info("Finished AI forecast training for %s", run_cycle.tag)
    logging.info(
        "Verification metrics on later timestamps: RMSE %.2f dBZ | MAE %.2f dBZ | R2 %.3f",
        metrics["rmse_dbz"],
        metrics["mae_dbz"],
        metrics["r2"],
    )


if __name__ == "__main__":
    main()
