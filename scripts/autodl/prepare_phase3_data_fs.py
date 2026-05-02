#!/usr/bin/env python3
"""Prepare Phase 3 data on AutoDL file storage before renting a GPU instance.

This script downloads selected processed datasets from the project Google Drive
folder into AutoDL file storage and normalizes the directory layout expected by
the Phase 3 runner:

  /root/autodl-fs/dataset/200Hz/CESCA-AODD/{Feature,Label}
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URL = "https://drive.google.com/drive/folders/1pVUmPlsQN9j5HD5YJSeiDrBAUAKBCQA5?usp=drive_link"
DEFAULT_DATA_ROOT = Path("/root/autodl-fs/dataset/200Hz")
DEFAULT_DOWNLOAD_ROOT = Path("/root/autodl-fs/downloads_gdrive")


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


def run(command: list[str]) -> None:
    log("run: " + " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def ensure_gdown(python_bin: str) -> None:
    check = subprocess.run(
        [python_bin, "-c", "import gdown"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if check.returncode == 0:
        return
    run([python_bin, "-m", "pip", "install", "gdown~=5.2.0"])


def dataset_is_ready(dataset_root: Path) -> bool:
    feature_dir = dataset_root / "Feature"
    label_dir = dataset_root / "Label"
    if not feature_dir.is_dir() or not label_dir.is_dir():
        return False
    feature_count = len(list(feature_dir.glob("*.npy")))
    label_count = len(list(label_dir.glob("*.npy")))
    return feature_count > 0 and feature_count == label_count


def dataset_counts(dataset_root: Path) -> tuple[int, int]:
    feature_count = len(list((dataset_root / "Feature").glob("*.npy")))
    label_count = len(list((dataset_root / "Label").glob("*.npy")))
    return feature_count, label_count


def find_dataset(download_root: Path, dataset: str) -> Path | None:
    candidates = []
    for path in download_root.rglob(dataset):
        if path.is_dir() and dataset_is_ready(path):
            candidates.append(path)
    if not candidates:
        return None
    candidates.sort(key=lambda item: len(item.parts))
    return candidates[0]


def normalize_dataset(found: Path, canonical: Path, force: bool) -> None:
    if dataset_is_ready(canonical):
        log(f"canonical dataset already ready: {canonical}")
        return

    if canonical.exists():
        if not force:
            raise RuntimeError(f"canonical path exists but is not ready: {canonical}")
        if canonical.is_symlink() or canonical.is_file():
            canonical.unlink()
        else:
            shutil.rmtree(canonical)

    canonical.parent.mkdir(parents=True, exist_ok=True)
    try:
        found.rename(canonical)
        log(f"moved dataset to canonical path: {found} -> {canonical}")
    except OSError:
        shutil.copytree(found, canonical)
        log(f"copied dataset to canonical path: {found} -> {canonical}")


def write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ERP processed data on AutoDL file storage.")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--url", default="")
    parser.add_argument("--datasets", default="CESCA-AODD")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--download-root", type=Path, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true", default=True)
    parser.add_argument("--status-file", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env_file(args.env_file)

    url = args.url or os.environ.get("PHASE3_DATA_ARCHIVE_URL", DEFAULT_URL)
    data_root = args.data_root or Path(os.environ.get("PHASE3_DATA_ROOT", str(DEFAULT_DATA_ROOT)))
    download_root = args.download_root or Path(os.environ.get("PHASE3_DATA_DOWNLOAD_ROOT", str(DEFAULT_DOWNLOAD_ROOT)))
    status_file = args.status_file or data_root.parent / "phase3_data_prepare_status.json"
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    if not datasets:
        raise SystemExit("No datasets selected.")

    log(f"data_root={data_root}")
    log(f"download_root={download_root}")
    log(f"datasets={','.join(datasets)}")
    log(f"python={args.python}")

    if not Path("/root/autodl-fs").exists() and str(data_root).startswith("/root/autodl-fs"):
        raise RuntimeError("/root/autodl-fs is not mounted. Check AutoDL file storage initialization and region.")

    ready_before = [dataset for dataset in datasets if dataset_is_ready(data_root / dataset)]
    missing = [dataset for dataset in datasets if dataset not in ready_before]

    if missing and not args.skip_download:
        ensure_gdown(args.python)
        download_root.mkdir(parents=True, exist_ok=True)
        command = [
            args.python,
            str(ROOT / "scripts" / "download_processed_datasets.py"),
            "--url",
            url,
            "--output",
            str(download_root),
            "--datasets",
            ",".join(missing),
        ]
        if args.stop_on_error:
            command.append("--stop-on-error")
        run(command)

    final = {}
    for dataset in datasets:
        canonical = data_root / dataset
        if not dataset_is_ready(canonical):
            found = find_dataset(download_root, dataset)
            if found is None:
                raise RuntimeError(f"dataset not found after download: {dataset}")
            normalize_dataset(found, canonical, args.force)

        feature_count, label_count = dataset_counts(canonical)
        if feature_count == 0 or feature_count != label_count:
            raise RuntimeError(
                f"dataset is not valid: {canonical}, Feature={feature_count}, Label={label_count}"
            )
        final[dataset] = {
            "path": str(canonical),
            "feature_count": feature_count,
            "label_count": label_count,
        }
        log(f"ready: {dataset} Feature={feature_count} Label={label_count} path={canonical}")

    write_status(
        status_file,
        {
            "status": "completed",
            "data_root": str(data_root),
            "download_root": str(download_root),
            "datasets": final,
        },
    )
    log(f"status written to {status_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
