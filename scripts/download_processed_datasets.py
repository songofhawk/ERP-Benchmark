#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import gdown


DEFAULT_URL = "https://drive.google.com/drive/folders/1pVUmPlsQN9j5HD5YJSeiDrBAUAKBCQA5?usp=drive_link"


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def sanitize_output_root(output: str) -> str:
    return output if output.endswith(os.sep) else output + os.sep


def path_matches_dataset_filters(path: str, datasets: list[str] | None) -> bool:
    if not datasets:
        return True
    normalized = path.replace("\\", "/")
    return any(f"/{name}/" in f"/{normalized}" for name in datasets)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def is_valid_download(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False

    if path.suffix == ".npy":
        try:
            with path.open("rb") as handle:
                magic = handle.read(6)
            return magic == b"\x93NUMPY"
        except OSError:
            return False

    return True


def remove_if_exists(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


def run_command(command: list[str]) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        return completed.returncode, completed.stdout
    except FileNotFoundError as error:
        return 127, str(error)


def try_wget(file_id: str, local_path: Path) -> tuple[bool, str]:
    wget_bin = shutil.which("wget")
    if not wget_bin:
        return False, "wget not found"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    code, output = run_command([wget_bin, "-c", "-O", str(local_path), url])
    return code == 0 and is_valid_download(local_path), output


def try_curl(file_id: str, local_path: Path) -> tuple[bool, str]:
    curl_bin = shutil.which("curl")
    if not curl_bin:
        return False, "curl not found"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    code, output = run_command([curl_bin, "-L", "-C", "-", "--fail", "--output", str(local_path), url])
    return code == 0 and is_valid_download(local_path), output


def try_gdown_file(file_id: str, local_path: Path) -> tuple[bool, str]:
    try:
        result = gdown.download(
            url=f"https://drive.google.com/uc?id={file_id}",
            output=str(local_path),
            quiet=False,
            resume=True,
        )
        return result is not None and is_valid_download(local_path), "gdown single-file fallback"
    except Exception as error:
        return False, repr(error)


def download_one(file_id: str, local_path: Path) -> tuple[bool, str]:
    ensure_parent(local_path)

    if is_valid_download(local_path):
        return True, "already valid"

    remove_if_exists(local_path)
    ok, output = try_wget(file_id, local_path)
    if ok:
        return True, "wget"
    remove_if_exists(local_path)

    ok, output = try_curl(file_id, local_path)
    if ok:
        return True, "curl"
    remove_if_exists(local_path)

    ok, output = try_gdown_file(file_id, local_path)
    if ok:
        return True, "gdown-single"
    remove_if_exists(local_path)
    return False, output


def main() -> int:
    parser = argparse.ArgumentParser(description="Reliable Google Drive folder downloader for ERP processed datasets")
    parser.add_argument("--url", default=DEFAULT_URL, help="Google Drive folder URL")
    parser.add_argument("--output", default="downloads_gdrive/", help="Output root directory")
    parser.add_argument("--datasets", default="", help="Comma-separated dataset names to include")
    parser.add_argument("--manifest", default="", help="Optional manifest JSON path")
    parser.add_argument("--start-index", type=int, default=0, help="Start from a given manifest index")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop immediately when a file cannot be downloaded")
    args = parser.parse_args()

    output = sanitize_output_root(args.output)
    dataset_filters = [item.strip() for item in args.datasets.split(",") if item.strip()]

    log("retrieving file manifest from Google Drive folder")
    files = gdown.download_folder(
        url=args.url,
        output=output,
        quiet=True,
        remaining_ok=True,
        skip_download=True,
        resume=True,
    )
    if files is None:
        log("failed to retrieve manifest")
        return 1

    selected = [item for item in files if path_matches_dataset_filters(item.path, dataset_filters)]
    log(f"manifest size: total={len(files)} selected={len(selected)}")

    if args.manifest:
        manifest_path = Path(args.manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                [{"id": item.id, "path": item.path, "local_path": item.local_path} for item in selected],
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        log(f"manifest written to {manifest_path}")

    failures: list[dict[str, str]] = []
    completed = 0
    skipped = 0

    for index, item in enumerate(selected[args.start_index:], start=args.start_index):
        local_path = Path(item.local_path)
        if is_valid_download(local_path):
            skipped += 1
            log(f"[{index + 1}/{len(selected)}] skip valid {item.path}")
            continue

        log(f"[{index + 1}/{len(selected)}] downloading {item.path} (id={item.id})")
        ok, method = download_one(item.id, local_path)
        if ok:
            completed += 1
            log(f"[{index + 1}/{len(selected)}] done via {method}: {item.path}")
            continue

        failures.append({"index": str(index), "id": item.id, "path": item.path, "local_path": str(local_path)})
        log(f"[{index + 1}/{len(selected)}] FAILED: {item.path} (id={item.id})")
        if args.stop_on_error:
            break

    log(f"summary: downloaded={completed} skipped={skipped} failed={len(failures)}")
    if failures:
        failure_path = Path(output) / "download_failures.json"
        failure_path.write_text(json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8")
        log(f"failures written to {failure_path}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
