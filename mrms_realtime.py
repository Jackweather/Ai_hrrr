from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pygrib
import requests
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.ndimage import gaussian_filter

APP_ROOT = Path(__file__).resolve().parent
UTC = dt.timezone.utc
DEFAULT_OUTPUT_ROOT = APP_ROOT / "DATA"
MRMS_URL = (
    "https://mrms.ncep.noaa.gov/2D/ReflectivityAtLowestAltitude/"
    "MRMS_ReflectivityAtLowestAltitude.latest.grib2.gz"
)
CONUS_EXTENT = (-127.0, -66.0, 20.0, 54.0)
MRMS_MISSING_VALUE = -999.0
DEFAULT_TIMEOUT = 120
DEFAULT_PLOT_STRIDE = 8
MIN_REFLECTIVITY_DBZ = 5.0
DEFAULT_SMOOTHING_SIGMA = 0.35
MAX_PNG_HISTORY = 8
REFLECTIVITY_LEVELS_DBZ = [
    0.0,
    5.0,
    10.0,
    15.0,
    20.0,
    25.0,
    30.0,
    35.0,
    40.0,
    45.0,
    50.0,
    55.0,
    60.0,
    65.0,
    70.0,
]
REFLECTIVITY_COLORS = [
    "#04e9e7",
    "#019ff4",
    "#0300f4",
    "#02fd02",
    "#01c501",
    "#008e00",
    "#fdf802",
    "#e5bc00",
    "#fd9500",
    "#fd0000",
    "#d40000",
    "#bc0000",
    "#f800fd",
    "#9854c6",
]


@dataclass(frozen=True)
class MRMSFramePaths:
    valid_time: dt.datetime
    raw_path: Path
    filtered_data_path: Path
    png_path: Path
    metadata_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the latest MRMS reflectivity frame, save the GRIB locally, "
            "and render a PNG. Use --loop-seconds 300 to keep collecting frames every 5 minutes."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Base directory used for MRMS raw files, metadata, and PNGs.",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="Seconds allowed for downloading the MRMS feed.",
    )
    parser.add_argument(
        "--plot-stride",
        type=int,
        default=DEFAULT_PLOT_STRIDE,
        help="Grid decimation factor used for plotting to keep renders fast.",
    )
    parser.add_argument(
        "--loop-seconds",
        type=int,
        default=0,
        help="If greater than zero, keep polling MRMS on this cadence.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild files even if the same MRMS valid time is already on disk.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.request_timeout < 1:
        raise ValueError("--request-timeout must be at least 1.")
    if args.plot_stride < 1:
        raise ValueError("--plot-stride must be at least 1.")
    if args.loop_seconds < 0:
        raise ValueError("--loop-seconds must be zero or greater.")


def ensure_utc(timestamp: dt.datetime) -> dt.datetime:
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC)


def relative_path_string(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def normalize_longitudes(
    values: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if np.nanmax(lons) <= 180:
        return values, lats, lons

    adjusted_lons = np.where(lons > 180, lons - 360, lons)
    sort_order = np.argsort(adjusted_lons[0, :])
    return values[:, sort_order], lats[:, sort_order], adjusted_lons[:, sort_order]


def reflectivity_cmap() -> tuple[ListedColormap, BoundaryNorm]:
    boundaries = np.asarray(REFLECTIVITY_LEVELS_DBZ, dtype=np.float32)
    cmap = ListedColormap(REFLECTIVITY_COLORS, name="mrms_reflectivity")
    norm = BoundaryNorm(boundaries, cmap.N, clip=False)
    return cmap, norm


def smooth_field_for_contours(field: np.ndarray, sigma: float = DEFAULT_SMOOTHING_SIGMA) -> np.ndarray:
    if sigma <= 0:
        return field.astype(np.float32, copy=True)

    finite_mask = np.isfinite(field)
    if not np.any(finite_mask):
        return np.full(field.shape, np.nan, dtype=np.float32)

    filled = np.where(finite_mask, field, 0.0).astype(np.float32)
    weights = finite_mask.astype(np.float32)
    smoothed_values = gaussian_filter(filled, sigma=sigma, mode="nearest")
    smoothed_weights = gaussian_filter(weights, sigma=sigma, mode="nearest")
    smoothed = np.divide(
        smoothed_values,
        smoothed_weights,
        out=np.full(field.shape, np.nan, dtype=np.float32),
        where=smoothed_weights > 1.0e-6,
    )
    return smoothed.astype(np.float32)


def frame_paths_for_time(root: Path, valid_time: dt.datetime) -> MRMSFramePaths:
    valid_time = ensure_utc(valid_time)
    date_dir = valid_time.strftime("%Y%m%d")
    stamp = valid_time.strftime("%Y%m%d_%H%M")
    raw_path = root / "mrms" / "raw" / date_dir / f"mrms_reflectivity_{stamp}.grib2"
    filtered_data_path = root / "mrms" / "processed" / date_dir / f"mrms_reflectivity_{stamp}.npz"
    png_path = root / "mrms" / "png" / date_dir / f"mrms_reflectivity_{stamp}.png"
    metadata_path = root / "mrms" / "metadata" / date_dir / f"mrms_reflectivity_{stamp}.json"
    return MRMSFramePaths(
        valid_time=valid_time,
        raw_path=raw_path,
        filtered_data_path=filtered_data_path,
        png_path=png_path,
        metadata_path=metadata_path,
    )


def parse_time_from_mrms_filename(grib_path: Path) -> dt.datetime:
    stamp = grib_path.stem.rsplit("_", 2)[-2:]
    timestamp = dt.datetime.strptime("_".join(stamp), "%Y%m%d_%H%M")
    return timestamp.replace(tzinfo=UTC)


def iter_mrms_grib_files(root: Path) -> list[Path]:
    return sorted((root / "mrms" / "raw").glob("**/*.grib2"))


def download_latest_mrms_gz(session: requests.Session, timeout: int, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with session.get(MRMS_URL, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with destination.open("wb") as file_handle:
            for chunk in response.iter_content(chunk_size=1_048_576):
                if chunk:
                    file_handle.write(chunk)
    return destination


def decompress_gzip(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(source, "rb") as gz_handle, destination.open("wb") as file_handle:
        shutil.copyfileobj(gz_handle, file_handle)
    return destination


def load_mrms_reflectivity(grib_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dt.datetime]:
    grib_file = pygrib.open(str(grib_path))
    try:
        message = grib_file.message(1)
        values = np.asarray(message.values, dtype=np.float32)
        lats, lons = message.latlons()
        valid_time = ensure_utc(message.validDate)
    finally:
        grib_file.close()

    values = np.where(values <= MRMS_MISSING_VALUE, np.nan, values)
    lats = np.asarray(lats, dtype=np.float32)
    lons = np.asarray(lons, dtype=np.float32)
    return (*normalize_longitudes(values, lats, lons), valid_time)


def filter_reflectivity(reflectivity_dbz: np.ndarray, minimum_dbz: float = MIN_REFLECTIVITY_DBZ) -> np.ndarray:
    return np.where(np.isfinite(reflectivity_dbz) & (reflectivity_dbz >= minimum_dbz), reflectivity_dbz, np.nan).astype(
        np.float32
    )


def plot_reflectivity_map(
    save_path: Path,
    valid_time: dt.datetime,
    reflectivity_dbz: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    stride: int,
    title_prefix: str,
) -> None:
    stride = max(1, int(stride))
    smoothed_reflectivity = smooth_field_for_contours(reflectivity_dbz[::stride, ::stride])
    masked_reflectivity = np.ma.masked_invalid(smoothed_reflectivity)
    plot_lats = lats[::stride, ::stride]
    plot_lons = lons[::stride, ::stride]
    cmap, norm = reflectivity_cmap()

    figure = plt.figure(figsize=(15, 9))
    axis = plt.axes(projection=ccrs.PlateCarree())
    axis.set_extent(CONUS_EXTENT, crs=ccrs.PlateCarree())
    axis.coastlines(linewidth=0.7)
    axis.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5)
    axis.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.3)
    axis.add_feature(cfeature.LAKES.with_scale("50m"), linewidth=0.2, edgecolor="#6b7c93", facecolor="none")

    mesh = axis.contourf(
        plot_lons,
        plot_lats,
        masked_reflectivity,
        levels=REFLECTIVITY_LEVELS_DBZ,
        cmap=cmap,
        norm=norm,
        extend="max",
        antialiased=True,
        transform=ccrs.PlateCarree(),
    )

    axis.set_title(
        f"{title_prefix}\nValid {ensure_utc(valid_time).strftime('%Y-%m-%d %H:%M UTC')}",
        fontsize=15,
        loc="left",
        pad=10,
    )

    colorbar = plt.colorbar(mesh, ax=axis, shrink=0.84, pad=0.02)
    colorbar.set_label("Reflectivity (dBZ)")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def write_metadata(paths: MRMSFramePaths, reflectivity_dbz: np.ndarray) -> None:
    finite_values = reflectivity_dbz[np.isfinite(reflectivity_dbz)]
    root = paths.raw_path.parents[3]
    payload = {
        "valid_time_utc": ensure_utc(paths.valid_time).isoformat(),
        "raw_grib_path": relative_path_string(paths.raw_path, root),
        "filtered_data_path": relative_path_string(paths.filtered_data_path, root),
        "png_path": relative_path_string(paths.png_path, root),
        "min_dbz": float(np.min(finite_values)) if finite_values.size else None,
        "max_dbz": float(np.max(finite_values)) if finite_values.size else None,
        "mean_dbz": float(np.mean(finite_values)) if finite_values.size else None,
        "grid_shape": list(reflectivity_dbz.shape),
        "source_url": MRMS_URL,
        "minimum_retained_dbz": MIN_REFLECTIVITY_DBZ,
    }
    paths.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    paths.metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    latest_path = paths.raw_path.parents[2] / "latest.json"
    latest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_filtered_data(paths: MRMSFramePaths, reflectivity_dbz: np.ndarray, lats: np.ndarray, lons: np.ndarray) -> None:
    paths.filtered_data_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        paths.filtered_data_path,
        reflectivity_dbz=reflectivity_dbz,
        latitude=lats,
        longitude=lons,
        valid_time_utc=ensure_utc(paths.valid_time).isoformat(),
        minimum_retained_dbz=np.float32(MIN_REFLECTIVITY_DBZ),
    )


def prune_old_pngs(root: Path, max_png_history: int = MAX_PNG_HISTORY) -> None:
    if max_png_history < 1:
        return

    png_files = sorted((root / "mrms" / "png").glob("**/*.png"), key=lambda path: path.stat().st_mtime, reverse=True)
    for old_path in png_files[max_png_history:]:
        try:
            old_path.unlink()
        except FileNotFoundError:
            continue


def ingest_once(root: Path, timeout: int, plot_stride: int, overwrite: bool) -> MRMSFramePaths:
    temp_dir = root / "mrms" / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    gz_path = temp_dir / "latest.grib2.gz"
    grib_path = temp_dir / "latest.grib2"

    session = requests.Session()
    session.headers.update({"User-Agent": "mrms-reflectivity-monitor/1.0", "Accept": "*/*"})

    download_latest_mrms_gz(session, timeout=timeout, destination=gz_path)
    decompress_gzip(gz_path, grib_path)
    reflectivity_dbz, lats, lons, valid_time = load_mrms_reflectivity(grib_path)
    reflectivity_dbz = filter_reflectivity(reflectivity_dbz)
    frame_paths = frame_paths_for_time(root, valid_time)

    if (
        frame_paths.raw_path.exists()
        and frame_paths.filtered_data_path.exists()
        and frame_paths.png_path.exists()
        and not overwrite
    ):
        logging.info("MRMS frame %s already exists, skipping rebuild", frame_paths.valid_time.strftime("%Y%m%d_%H%M"))
        write_metadata(frame_paths, reflectivity_dbz)
        return frame_paths

    frame_paths.raw_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(grib_path), str(frame_paths.raw_path))
    write_filtered_data(frame_paths, reflectivity_dbz, lats, lons)

    plot_reflectivity_map(
        save_path=frame_paths.png_path,
        valid_time=frame_paths.valid_time,
        reflectivity_dbz=reflectivity_dbz,
        lats=lats,
        lons=lons,
        stride=plot_stride,
        title_prefix="MRMS Reflectivity At Lowest Altitude",
    )
    prune_old_pngs(root)
    write_metadata(frame_paths, reflectivity_dbz)
    logging.info("Saved MRMS frame to %s", frame_paths.raw_path)
    logging.info("Saved filtered MRMS data to %s", frame_paths.filtered_data_path)
    logging.info("Saved MRMS PNG to %s", frame_paths.png_path)
    return frame_paths


def main() -> None:
    args = parse_args()
    validate_args(args)
    setup_logging(args.log_level)

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.loop_seconds == 0:
        ingest_once(output_root, timeout=args.request_timeout, plot_stride=args.plot_stride, overwrite=args.overwrite)
        return

    while True:
        try:
            ingest_once(output_root, timeout=args.request_timeout, plot_stride=args.plot_stride, overwrite=args.overwrite)
        except Exception:
            logging.exception("MRMS ingest failed")
        logging.info("Sleeping for %d seconds before the next MRMS poll", args.loop_seconds)
        time.sleep(args.loop_seconds)


if __name__ == "__main__":
    main()
