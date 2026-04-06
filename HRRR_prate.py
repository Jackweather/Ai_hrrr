from __future__ import annotations

import argparse
import datetime as dt
import email.utils
import logging
import random
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import numpy as np
import requests
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

matplotlib.use("Agg")

APP_ROOT = Path(__file__).resolve().parent
BASE_URL = "https://nomads.ncep.noaa.gov/cgi-bin/filter_hrrr_2d.pl"
UTC = dt.timezone.utc
CONUS_EXTENT = (-127.0, -66.0, 20.0, 54.0)
SYNOPTIC_CYCLE_HOURS = {0, 6, 12, 18}
SYNOPTIC_MAX_FORECAST_HOUR = 48
NON_SYNOPTIC_MAX_FORECAST_HOUR = 18
DEFAULT_OUTPUT_ROOT = APP_ROOT / "DATA"
DEFAULT_DOWNLOAD_WORKERS = 2
DEFAULT_REQUEST_MIN_INTERVAL = 0.5
DEFAULT_MAX_REQUEST_RETRIES = 6
DEFAULT_LOOKBACK_HOURS = 12
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
PRATE_LEVELS_MMHR = [
    0.01,
    0.05,
    0.10,
    0.25,
    0.50,
    1.0,
    2.0,
    4.0,
    8.0,
    16.0,
    32.0,
]
PRATE_COLORS = [
    "#f4fbff",
    "#dcefff",
    "#b8d8ff",
    "#80b8ff",
    "#4f98ff",
    "#2ec27e",
    "#98c21d",
    "#f5c400",
    "#f08a24",
    "#d64b32",
    "#8a1c7c",
]
CFGRIB_BACKEND_KWARGS = (
    {
        "filter_by_keys": {"shortName": "prate", "typeOfLevel": "surface", "stepType": "instant"},
        "indexpath": "",
    },
    {
        "filter_by_keys": {"shortName": "prate", "typeOfLevel": "surface", "stepType": "avg"},
        "indexpath": "",
    },
    {"filter_by_keys": {"shortName": "prate", "typeOfLevel": "surface"}, "indexpath": ""},
    {"filter_by_keys": {"typeOfLevel": "surface"}, "indexpath": ""},
    {"indexpath": ""},
)


@dataclass(frozen=True)
class RunCycle:
    init_time: dt.datetime

    def __post_init__(self) -> None:
        if self.init_time.tzinfo is None:
            object.__setattr__(self, "init_time", self.init_time.replace(tzinfo=UTC))
        else:
            object.__setattr__(self, "init_time", self.init_time.astimezone(UTC))

    @property
    def cycle_hour(self) -> int:
        return self.init_time.hour

    @property
    def date_token(self) -> str:
        return self.init_time.strftime("%Y%m%d")

    @property
    def tag(self) -> str:
        return self.init_time.strftime("%Y%m%d_%HZ")

    @property
    def nomads_directory(self) -> str:
        return f"/hrrr.{self.date_token}/conus"

    @property
    def max_forecast_hour(self) -> int:
        if self.cycle_hour in SYNOPTIC_CYCLE_HOURS:
            return SYNOPTIC_MAX_FORECAST_HOUR
        return NON_SYNOPTIC_MAX_FORECAST_HOUR

    def file_name(self, forecast_hour: int) -> str:
        return f"hrrr.t{self.cycle_hour:02d}z.wrfsfcf{forecast_hour:02d}.grib2"

    def shifted(self, hours: int) -> "RunCycle":
        return RunCycle(self.init_time + dt.timedelta(hours=hours))


class NomadsClient:
    def __init__(self, timeout: int, min_interval_seconds: float, max_retries: int) -> None:
        self.timeout = timeout
        self.min_interval_seconds = max(0.0, float(min_interval_seconds))
        self.max_retries = max(1, int(max_retries))
        self._last_request_started = 0.0
        self._rate_lock = threading.Lock()
        self._thread_local = threading.local()

    def _get_session(self) -> requests.Session:
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update(
                {
                    "User-Agent": "hrrr-prate-conus/1.0 (+https://nomads.ncep.noaa.gov/)",
                    "Accept": "*/*",
                }
            )
            self._thread_local.session = session
        return session

    def _throttle(self) -> None:
        if self.min_interval_seconds <= 0:
            return

        with self._rate_lock:
            now = time.monotonic()
            wait_seconds = self.min_interval_seconds - (now - self._last_request_started)
            if wait_seconds > 0:
                time.sleep(wait_seconds)
                now = time.monotonic()
            self._last_request_started = now

    def _retry_delay(self, attempt: int, response: requests.Response | None) -> float:
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    return max(float(retry_after), self.min_interval_seconds)
                except ValueError:
                    retry_after_dt = email.utils.parsedate_to_datetime(retry_after)
                    if retry_after_dt is not None:
                        if retry_after_dt.tzinfo is None:
                            retry_after_dt = retry_after_dt.replace(tzinfo=UTC)
                        delay = (retry_after_dt.astimezone(UTC) - dt.datetime.now(tz=UTC)).total_seconds()
                        return max(delay, self.min_interval_seconds)

        base_delay = max(self.min_interval_seconds, 1.0)
        backoff = min(90.0, base_delay * (2 ** (attempt - 1)))
        jitter = random.uniform(0.0, max(0.25, self.min_interval_seconds))
        return backoff + jitter

    def request(self, method: str, url: str, *, stream: bool = False) -> requests.Response:
        last_error: Exception | None = None
        response: requests.Response | None = None

        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            try:
                response = self._get_session().request(
                    method=method,
                    url=url,
                    stream=stream,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
                delay = self._retry_delay(attempt, None)
                logging.warning(
                    "Request error for %s on attempt %d/%d: %s. Retrying in %.1fs",
                    url,
                    attempt,
                    self.max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)
                continue

            if response.status_code not in RETRYABLE_STATUS_CODES:
                return response

            if attempt == self.max_retries:
                return response

            delay = self._retry_delay(attempt, response)
            logging.warning(
                "Received HTTP %s for %s on attempt %d/%d. Retrying in %.1fs",
                response.status_code,
                url,
                attempt,
                self.max_retries,
                delay,
            )
            response.close()
            time.sleep(delay)

        raise RuntimeError(f"Request failed for {url}: {last_error}") from last_error


def response_contains_grib_payload(response: requests.Response) -> bool:
    content_type = response.headers.get("Content-Type", "")
    disposition = response.headers.get("Content-Disposition", "")
    if response.status_code != 200:
        return False
    if "html" in content_type.lower() and ".grib2" not in disposition.lower():
        return False
    return True


def assert_grib_payload(response: requests.Response, file_label: str) -> None:
    if response_contains_grib_payload(response):
        return

    content_type = response.headers.get("Content-Type", "")
    raise RuntimeError(
        f"Unexpected response while fetching {file_label}: HTTP {response.status_code}, "
        f"content type {content_type or 'unknown'}."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the latest available HRRR CONUS run, using forecast hours f00-f48 "
            "for 00z/06z/12z/18z cycles and f00-f18 for all other cycles, then render "
            "CONUS precipitation-rate plots."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory used for downloaded GRIB files and generated plots.",
    )
    parser.add_argument(
        "--cycle-lookback-hours",
        type=int,
        default=DEFAULT_LOOKBACK_HOURS,
        help="How many recent hourly cycles to inspect when looking for the latest available run.",
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=DEFAULT_DOWNLOAD_WORKERS,
        help="Concurrent downloads for a single run.",
    )
    parser.add_argument(
        "--request-min-interval",
        type=float,
        default=DEFAULT_REQUEST_MIN_INTERVAL,
        help="Minimum seconds between outbound NOMADS requests across all worker threads.",
    )
    parser.add_argument(
        "--max-request-retries",
        type=int,
        default=DEFAULT_MAX_REQUEST_RETRIES,
        help="How many times to retry NOMADS requests after rate limiting or transient server errors.",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=180,
        help="Seconds allowed for each NOMADS request.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload files even if they already exist locally.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Assume the required GRIB files already exist locally and only build plots.",
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
    if args.cycle_lookback_hours < 0:
        raise ValueError("--cycle-lookback-hours must be zero or greater.")
    if args.download_workers < 1:
        raise ValueError("--download-workers must be at least 1.")
    if args.request_min_interval < 0:
        raise ValueError("--request-min-interval must be zero or greater.")
    if args.max_request_retries < 1:
        raise ValueError("--max-request-retries must be at least 1.")


def floor_to_hour(timestamp: dt.datetime) -> RunCycle:
    timestamp = timestamp.astimezone(UTC)
    floored = timestamp.replace(minute=0, second=0, microsecond=0)
    return RunCycle(floored)


def build_url(run_cycle: RunCycle, forecast_hour: int) -> str:
    return requests.Request(
        method="GET",
        url=BASE_URL,
        params={
            "dir": run_cycle.nomads_directory,
            "file": run_cycle.file_name(forecast_hour),
            "var_PRATE": "on",
            "lev_surface": "on",
        },
    ).prepare().url


def local_grib_path(root: Path, run_cycle: RunCycle, forecast_hour: int) -> Path:
    return root / "grib" / run_cycle.tag / run_cycle.file_name(forecast_hour)


def plot_path(root: Path, run_cycle: RunCycle, forecast_hour: int) -> Path:
    return root / "plots" / run_cycle.tag / f"hrrr_prate_f{forecast_hour:02d}.png"


def forecast_hours_for_run(run_cycle: RunCycle) -> tuple[int, ...]:
    return tuple(range(0, run_cycle.max_forecast_hour + 1))


def reset_plot_output(root: Path) -> None:
    plots_root = root / "plots"
    if plots_root.exists():
        shutil.rmtree(plots_root)
    plots_root.mkdir(parents=True, exist_ok=True)


def missing_forecast_hours(root: Path, run_cycle: RunCycle, overwrite: bool) -> list[int]:
    forecast_hours = forecast_hours_for_run(run_cycle)
    if overwrite:
        return list(forecast_hours)
    return [
        forecast_hour
        for forecast_hour in forecast_hours
        if not local_grib_path(root, run_cycle, forecast_hour).exists()
    ]


def probe_file_available(client: NomadsClient, run_cycle: RunCycle, forecast_hour: int) -> bool:
    url = build_url(run_cycle, forecast_hour)
    try:
        response = client.request("GET", url, stream=True)
    except RuntimeError as exc:
        logging.debug("Probe failed for %s f%02d: %s", run_cycle.tag, forecast_hour, exc)
        return False

    with response:
        return response_contains_grib_payload(response)


def resolve_latest_available_cycle(now_utc: dt.datetime, lookback_hours: int, client: NomadsClient) -> RunCycle:
    candidate = floor_to_hour(now_utc)
    for offset in range(lookback_hours + 1):
        run_cycle = candidate.shifted(hours=-offset)
        if probe_file_available(client, run_cycle, run_cycle.max_forecast_hour):
            logging.info(
                "Selected run %s with max forecast hour f%02d",
                run_cycle.tag,
                run_cycle.max_forecast_hour,
            )
            return run_cycle
        logging.info(
            "Run %s is not fully available through f%02d yet, falling back one hour",
            run_cycle.tag,
            run_cycle.max_forecast_hour,
        )

    raise RuntimeError("Could not find a recent HRRR run within the configured lookback window.")


def download_file(client: NomadsClient, url: str, destination: Path, overwrite: bool) -> Path:
    if destination.exists() and not overwrite:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".part")
    try:
        with client.request("GET", url, stream=True) as response:
            assert_grib_payload(response, destination.name)
            with temp_path.open("wb") as file_handle:
                for chunk in response.iter_content(chunk_size=1_048_576):
                    if chunk:
                        file_handle.write(chunk)
        temp_path.replace(destination)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise
    return destination


def download_run(client: NomadsClient, root: Path, run_cycle: RunCycle, overwrite: bool, workers: int) -> None:
    hours_to_download = missing_forecast_hours(root, run_cycle, overwrite)
    all_hours = forecast_hours_for_run(run_cycle)
    if not hours_to_download:
        logging.info("Skipping %s because all %d files already exist", run_cycle.tag, len(all_hours))
        return

    logging.info(
        "Downloading %s (%d missing of %d files; f00-f%02d)",
        run_cycle.tag,
        len(hours_to_download),
        len(all_hours),
        run_cycle.max_forecast_hour,
    )
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = []
        for forecast_hour in hours_to_download:
            url = build_url(run_cycle, forecast_hour)
            destination = local_grib_path(root, run_cycle, forecast_hour)
            futures.append(executor.submit(download_file, client, url, destination, overwrite))
        for future in as_completed(futures):
            future.result()


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


def build_coordinate_grids(dataset: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    latitude = dataset.coords.get("latitude")
    if latitude is None:
        latitude = dataset.coords.get("lat")

    longitude = dataset.coords.get("longitude")
    if longitude is None:
        longitude = dataset.coords.get("lon")

    if latitude is None or longitude is None:
        raise ValueError("GRIB dataset is missing latitude/longitude coordinates.")

    lat_values = np.asarray(latitude.values, dtype=np.float32)
    lon_values = np.asarray(longitude.values, dtype=np.float32)
    if lat_values.ndim == 1 and lon_values.ndim == 1:
        lon_grid, lat_grid = np.meshgrid(lon_values, lat_values)
        return lat_grid.astype(np.float32), lon_grid.astype(np.float32)
    if lat_values.ndim == 2 and lon_values.ndim == 2:
        return lat_values, lon_values

    raise ValueError("Unsupported GRIB coordinate layout: expected 1D or 2D latitude/longitude arrays.")


def select_precipitation_rate(dataset: xr.Dataset) -> xr.DataArray:
    if "prate" in dataset.data_vars:
        return dataset["prate"].squeeze(drop=True)

    for variable in dataset.data_vars.values():
        if variable.attrs.get("GRIB_name") == "Precipitation rate":
            return variable.squeeze(drop=True)
        if variable.attrs.get("GRIB_shortName") == "prate":
            return variable.squeeze(drop=True)

    if len(dataset.data_vars) == 1:
        return next(iter(dataset.data_vars.values())).squeeze(drop=True)

    raise ValueError("Could not locate a precipitation-rate field in the GRIB dataset.")


def load_prate_mmhr(grib_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset: xr.Dataset | None = None
    last_error: Exception | None = None

    for backend_kwargs in CFGRIB_BACKEND_KWARGS:
        try:
            dataset = xr.open_dataset(grib_path, engine="cfgrib", backend_kwargs=backend_kwargs)
            data_array = select_precipitation_rate(dataset)
            values = np.asarray(data_array.values, dtype=np.float32)
            if values.ndim != 2:
                raise ValueError(
                    f"Expected a 2D precipitation field in {grib_path.name}, got shape {values.shape}."
                )
            lats, lons = build_coordinate_grids(dataset)
            break
        except Exception as exc:
            last_error = exc
            if dataset is not None:
                dataset.close()
                dataset = None
    else:
        raise RuntimeError(f"Unable to read precipitation rate from {grib_path.name}: {last_error}") from last_error

    assert dataset is not None
    dataset.close()

    values = np.maximum(values, 0.0) * 3600.0
    return normalize_longitudes(values, lats, lons)


def valid_time(run_cycle: RunCycle, forecast_hour: int) -> dt.datetime:
    return run_cycle.init_time + dt.timedelta(hours=forecast_hour)


def prate_cmap() -> tuple[ListedColormap, BoundaryNorm]:
    boundaries = np.asarray(PRATE_LEVELS_MMHR, dtype=np.float32)
    cmap = ListedColormap(PRATE_COLORS, name="hrrr_prate")
    norm = BoundaryNorm(boundaries, cmap.N, clip=False)
    return cmap, norm


def plot_prate_map(
    save_path: Path,
    run_cycle: RunCycle,
    forecast_hour: int,
    lats: np.ndarray,
    lons: np.ndarray,
    prate_mmhr: np.ndarray,
) -> None:
    masked_prate = np.ma.masked_less(prate_mmhr, PRATE_LEVELS_MMHR[0])
    cmap, norm = prate_cmap()

    figure = plt.figure(figsize=(15, 9))
    axis = plt.axes(projection=ccrs.PlateCarree())
    axis.set_extent(CONUS_EXTENT, crs=ccrs.PlateCarree())
    axis.coastlines(linewidth=0.7)
    axis.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5)
    axis.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.3)
    axis.add_feature(cfeature.LAKES.with_scale("50m"), linewidth=0.2, edgecolor="#6b7c93", facecolor="none")

    filled = axis.contourf(
        lons,
        lats,
        masked_prate,
        levels=PRATE_LEVELS_MMHR,
        cmap=cmap,
        norm=norm,
        extend="max",
        transform=ccrs.PlateCarree(),
    )

    axis.set_title(
        (
            f"HRRR CONUS Surface Precipitation Rate F{forecast_hour:02d}\n"
            f"Run {run_cycle.tag} | Valid {valid_time(run_cycle, forecast_hour).strftime('%Y-%m-%d %H:%M UTC')}"
        ),
        fontsize=15,
        loc="left",
        pad=10,
    )

    colorbar = plt.colorbar(filled, ax=axis, shrink=0.84, pad=0.02)
    colorbar.set_label("Precipitation rate (mm/hr)")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def build_plots(root: Path, run_cycle: RunCycle) -> None:
    for forecast_hour in forecast_hours_for_run(run_cycle):
        grib_path = local_grib_path(root, run_cycle, forecast_hour)
        if not grib_path.exists():
            raise RuntimeError(f"Missing GRIB file for {run_cycle.tag} f{forecast_hour:02d}: {grib_path}")
        values, lats, lons = load_prate_mmhr(grib_path)
        save_path = plot_path(root, run_cycle, forecast_hour)
        plot_prate_map(
            save_path=save_path,
            run_cycle=run_cycle,
            forecast_hour=forecast_hour,
            lats=lats,
            lons=lons,
            prate_mmhr=values,
        )
        logging.info("Saved %s", save_path)


def main() -> None:
    args = parse_args()
    validate_args(args)
    setup_logging(args.log_level)

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
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

    logging.info("Output root: %s", output_root)
    logging.info("Selected run: %s", run_cycle.tag)
    logging.info("Forecast range: f00-f%02d", run_cycle.max_forecast_hour)

    reset_plot_output(output_root)

    if not args.skip_download:
        download_run(
            client=client,
            root=output_root,
            run_cycle=run_cycle,
            overwrite=args.overwrite,
            workers=args.download_workers,
        )

    build_plots(output_root, run_cycle)
    logging.info("Finished building HRRR CONUS PRATE plots for %s", run_cycle.tag)


if __name__ == "__main__":
    main()