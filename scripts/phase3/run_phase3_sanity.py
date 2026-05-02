#!/usr/bin/env python3
"""Run the Phase 3 cloud sanity matrix.

Default matrix:
  CESCA-AODD x multi-variate / uni-variate / whole-variate x 1 epoch
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "results" / "phase3_logs"
STATUS_FILE = LOG_DIR / "phase3_sanity.status"
CURRENT_FILE = LOG_DIR / "phase3_sanity.current"

PATCH_CONFIGS = {
    "multi-variate": {
        "label": "Multi-Variate",
        "patch_len": "25",
    },
    "uni-variate": {
        "label": "Uni-Variate",
        "patch_len": "100",
    },
    "whole-variate": {
        "label": "Whole-Variate",
        "patch_len": None,
    },
}


def log_line(handle, message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    handle.write(line + "\n")
    handle.flush()


def result_file(model_id: str) -> Path:
    return ROOT / "results" / "TestFormer" / "supervised" / "TestFormer" / model_id / "results.txt"


def result_is_complete(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    return "Average and std of validation and testing results" in text and "Test results" in text


def precheck_dataset(dataset: str, data_root: Path) -> int:
    dataset_dir = data_root / dataset
    feature_dir = dataset_dir / "Feature"
    label_dir = dataset_dir / "Label"
    if not feature_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(
            f"Missing dataset folders: expected {feature_dir} and {label_dir}"
        )

    feature_count = len(list(feature_dir.glob("*.npy")))
    label_count = len(list(label_dir.glob("*.npy")))
    if feature_count == 0 or feature_count != label_count:
        raise RuntimeError(
            f"Dataset file count is not ready: Feature={feature_count}, Label={label_count}"
        )

    from data_provider.data_factory import data_provider

    args = SimpleNamespace(
        data="MultiDatasets",
        task_name="supervised",
        root_path=str(data_root) + "/",
        pretraining_datasets="TDBRAIN-19",
        training_datasets=dataset,
        testing_datasets=dataset,
        cross_val="mccv",
        seed=41,
        no_normalize=False,
        segment_length=128,
        overlapping=0.5,
        sampling_rate=200,
        low_cut=0.5,
        high_cut=45,
        batch_size=64,
        num_workers=0,
        seq_len=96,
    )
    train_data, _ = data_provider(args, "TRAIN")
    unique_labels = np.unique(train_data.y[:, 0])
    if len(unique_labels) < 2:
        raise RuntimeError(
            f"{dataset} training split has only one class: {unique_labels.tolist()}"
        )
    return len(unique_labels)


def build_command(args: argparse.Namespace, patch_type: str) -> tuple[str, list[str]]:
    config = PATCH_CONFIGS[patch_type]
    label = config["label"]
    model_id = f"S-{args.dataset}-phase3-sanity-{label}"

    command = [
        args.python,
        "-u",
        str(ROOT / "run.py"),
        "--method",
        "TestFormer",
        "--task_name",
        "supervised",
        "--is_training",
        "1",
        "--root_path",
        str(args.data_root) + "/",
        "--model_id",
        model_id,
        "--model",
        "TestFormer",
        "--data",
        "MultiDatasets",
        "--training_datasets",
        args.dataset,
        "--testing_datasets",
        args.dataset,
        "--e_layers",
        str(args.e_layers),
        "--batch_size",
        str(args.batch_size),
        "--n_heads",
        str(args.n_heads),
        "--d_model",
        str(args.d_model),
        "--d_ff",
        str(args.d_ff),
        "--patch_type",
        patch_type,
        "--des",
        args.des,
        "--itr",
        str(args.itr),
        "--learning_rate",
        str(args.learning_rate),
        "--train_epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--devices",
        args.devices,
    ]

    if config["patch_len"] is not None:
        command.extend(["--patch_len", config["patch_len"]])
    if args.swa:
        command.append("--swa")
    if args.use_class_weights:
        command.append("--use_class_weights")

    return model_id, command


def run_command(command: list[str], log_path: Path, env: dict[str, str]) -> int:
    with log_path.open("a", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_handle.write(line)
        return process.wait()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 3 sanity experiments.")
    parser.add_argument("--dataset", default="CESCA-AODD")
    parser.add_argument("--data-root", type=Path, default=ROOT / "dataset" / "200Hz")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--patch-types", default="multi-variate,uni-variate,whole-variate")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--itr", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--e-layers", type=int, default=6)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--devices", default="0")
    parser.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--des", default="Phase3-Sanity")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-precheck", action="store_true")
    parser.add_argument("--no-swa", dest="swa", action="store_false", default=True)
    parser.add_argument("--use-class-weights", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.data_root = args.data_root.resolve()
    patch_types = [item.strip() for item in args.patch_types.split(",") if item.strip()]
    unknown = sorted(set(patch_types) - set(PATCH_CONFIGS))
    if unknown:
        raise SystemExit(f"Unknown patch types: {', '.join(unknown)}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    run_log = LOG_DIR / f"phase3_sanity_{args.dataset}.log"
    STATUS_FILE.write_text("running:bootstrap\n", encoding="utf-8")
    CURRENT_FILE.write_text("bootstrap\n", encoding="utf-8")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    with run_log.open("w", encoding="utf-8") as log_handle:
        log_line(log_handle, f"START Phase3 sanity dataset={args.dataset}")
        log_line(log_handle, f"data_root={args.data_root}")
        log_line(log_handle, f"python={args.python}")
        log_line(log_handle, f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")

        if not args.skip_precheck:
            class_count = precheck_dataset(args.dataset, args.data_root)
            log_line(log_handle, f"dataset precheck passed: class_count={class_count}")

        planned = []
        for patch_type in patch_types:
            model_id, command = build_command(args, patch_type)
            planned.append((patch_type, model_id, command))
            log_line(log_handle, f"planned {patch_type}: {' '.join(command)}")

        if args.dry_run:
            STATUS_FILE.write_text("completed:dry_run\n", encoding="utf-8")
            CURRENT_FILE.write_text("dry_run\n", encoding="utf-8")
            log_line(log_handle, "END Phase3 sanity dry-run")
            return 0

    for patch_type, model_id, command in planned:
        target_result = result_file(model_id)
        if result_is_complete(target_result) and not args.force:
            with run_log.open("a", encoding="utf-8") as log_handle:
                log_line(log_handle, f"SKIP {patch_type}: complete result exists at {target_result}")
            continue

        STATUS_FILE.write_text(f"running:{patch_type}\n", encoding="utf-8")
        CURRENT_FILE.write_text(f"{patch_type}\n", encoding="utf-8")
        with run_log.open("a", encoding="utf-8") as log_handle:
            log_line(log_handle, f"START {patch_type} model_id={model_id}")

        exit_code = run_command(command, run_log, env)
        if exit_code != 0:
            STATUS_FILE.write_text(f"failed:{patch_type}:{exit_code}\n", encoding="utf-8")
            CURRENT_FILE.write_text(f"{patch_type}\n", encoding="utf-8")
            with run_log.open("a", encoding="utf-8") as log_handle:
                log_line(log_handle, f"FAIL {patch_type} exit_code={exit_code}")
            return exit_code

        if not result_is_complete(target_result):
            STATUS_FILE.write_text(f"failed:{patch_type}:missing_result\n", encoding="utf-8")
            with run_log.open("a", encoding="utf-8") as log_handle:
                log_line(log_handle, f"FAIL {patch_type}: missing complete result at {target_result}")
            return 2

        with run_log.open("a", encoding="utf-8") as log_handle:
            log_line(log_handle, f"END {patch_type} result={target_result}")

    STATUS_FILE.write_text("completed\n", encoding="utf-8")
    CURRENT_FILE.write_text("completed\n", encoding="utf-8")
    with run_log.open("a", encoding="utf-8") as log_handle:
        log_line(log_handle, "END Phase3 sanity")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
