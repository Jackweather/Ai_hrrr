from __future__ import annotations

import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, abort, render_template, send_from_directory

APP_ROOT = Path(__file__).resolve().parent
DATA_ROOT = APP_ROOT / "DATA"
LOG_ROOT = APP_ROOT / "logs"


@dataclass(frozen=True)
class ImageEntry:
    name: str
    relative_path: str
    modified_timestamp: float


@dataclass(frozen=True)
class GallerySection:
    key: str
    title: str
    description: str
    images: list[ImageEntry]


app = Flask(__name__, template_folder="templates")


def run_single_script(script_path: str | Path, working_directory: str | Path) -> None:
    script_path = Path(script_path)
    working_directory = Path(working_directory)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"{script_path.stem}.log"

    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(f"\n=== Starting {script_path.name} ===\n")
        process = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(working_directory),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        log_handle.write(f"=== Finished {script_path.name} with exit code {process.returncode} ===\n")


def run_scripts(
    scripts: list[tuple[str | Path, str | Path]],
    repeat_count: int = 1,
    *,
    parallel: bool = False,
    max_parallel: int = 3,
) -> None:
    for _ in range(max(1, repeat_count)):
        if parallel:
            with ThreadPoolExecutor(max_workers=max(1, max_parallel)) as executor:
                futures = [executor.submit(run_single_script, script_path, working_directory) for script_path, working_directory in scripts]
                for future in futures:
                    future.result()
            continue

        for script_path, working_directory in scripts:
            run_single_script(script_path, working_directory)


def gather_images(root: Path, pattern: str) -> list[ImageEntry]:
    if not root.exists():
        return []

    images: list[ImageEntry] = []
    for path in sorted(root.glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True):
        if not path.is_file():
            continue
        images.append(
            ImageEntry(
                name=path.stem,
                relative_path=path.relative_to(DATA_ROOT).as_posix(),
                modified_timestamp=path.stat().st_mtime,
            )
        )
    return images


def latest_subset(images: list[ImageEntry], limit: int = 24) -> list[ImageEntry]:
    return images[:limit]


def build_gallery_sections() -> list[GallerySection]:
    return [
        GallerySection(
            key="ai-forecast",
            title="AI Forecast PNGs",
            description="Latest AI-generated reflectivity forecast images.",
            images=latest_subset(gather_images(DATA_ROOT / "ai_forecast" / "png", "**/*.png")),
        ),
        GallerySection(
            key="mrms",
            title="MRMS PNGs",
            description="Latest observed MRMS reflectivity images collected every 5 minutes.",
            images=latest_subset(gather_images(DATA_ROOT / "mrms" / "png", "**/*.png")),
        ),
        GallerySection(
            key="hrrr-prate",
            title="HRRR PRATE PNGs",
            description="Latest HRRR precipitation-rate plots from the base downloader.",
            images=latest_subset(gather_images(DATA_ROOT / "plots", "**/*.png")),
        ),
    ]


@app.route("/")
def index():
    sections = build_gallery_sections()
    total_images = sum(len(section.images) for section in sections)
    return render_template("index.html", sections=sections, total_images=total_images)


@app.route("/data/<path:relative_path>")
def data_file(relative_path: str):
    target = (DATA_ROOT / relative_path).resolve()
    if DATA_ROOT.resolve() not in target.parents and target != DATA_ROOT.resolve():
        abort(404)
    if not target.exists() or not target.is_file():
        abort(404)
    return send_from_directory(DATA_ROOT, relative_path)


@app.route("/run-task1")
def run_task1():
    scripts = [
        ("/opt/render/project/src/mrms_realtime.py", "/opt/render/project/src"),
        
        
    ]
    threading.Thread(
        target=lambda: run_scripts(scripts, 3, parallel=True, max_parallel=1),
        daemon=True,
    ).start()
    return "Task started in background! Check logs folder for output.", 200

@app.route("/run-task2")
def run_task2():
    scripts = [
        ("/opt/render/project/src/HRRR_prate.py", "/opt/render/project/src"),
        ("/opt/render/project/src/ai_forecast.py", "/opt/render/project/src"),
        
        
    ]
    threading.Thread(
        target=lambda: run_scripts(scripts, 3, parallel=True, max_parallel=1),
        daemon=True,
    ).start()
    return "Task started in background! Check logs folder for output.", 200

if __name__ == "__main__":
    app.run(debug=True)