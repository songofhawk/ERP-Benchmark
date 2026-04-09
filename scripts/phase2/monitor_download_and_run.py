import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = ROOT / "dataset" / "200Hz"
ROOT_CANDIDATES = [
    ROOT / "dataset" / "200Hz",
    ROOT / "dataset" / "200Hz" / "200Hz",
    ROOT / "dataset" / "dataset" / "200Hz",
    ROOT / "downloads_gdrive",
    ROOT / "downloads_gdrive" / "200Hz",
    ROOT / "downloads_gdrive" / "200Hz" / "200Hz",
    ROOT / "downloads_gdrive" / "dataset",
    ROOT / "downloads_gdrive" / "dataset" / "200Hz",
]
LOG_DIR = ROOT / "results" / "phase2_logs"
STATE_FILE = LOG_DIR / "download_monitor_state.json"
LOG_FILE = LOG_DIR / "download_monitor.log"
SELECTED_FILE = LOG_DIR / "selected_second_dataset.txt"
STATUS_FILE = LOG_DIR / "download_monitor.status"
RUNNER = Path(__file__).resolve().with_name("run_phase2_dataset_mps.sh")
POLL_SECONDS = 300
PRIORITY = [
    "CESCA-AODD", "CESCA-VODD", "CESCA-FLANKER", "mTBI-ODD", "NSERP-MSIT", "NSERP-ODD",
    "PD-SIM", "PD-ODD", "SCPD", "RLPD", "AOPD", "ADHD-WMRI",
]
IGNORE_NAMES = {"200Hz", "dataset"}


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"datasets": {}, "completed": [], "failed": None, "current": None}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def count_npy(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for path in folder.iterdir() if path.is_file() and path.suffix == ".npy")


def resolve_dataset_dir(name: str) -> Path | None:
    best_dir = None
    best_score = -1
    for root in ROOT_CANDIDATES:
        dataset_dir = root / name
        feature_count = count_npy(dataset_dir / "Feature")
        label_count = count_npy(dataset_dir / "Label")
        score = feature_count + label_count
        if score > best_score:
            best_score = score
            best_dir = dataset_dir
    return best_dir if best_score >= 0 else None


def ensure_canonical_link(name: str, actual_dir: Path) -> None:
    canonical_dir = DATASET_ROOT / name
    if canonical_dir.exists() or actual_dir == canonical_dir:
        return
    canonical_dir.symlink_to(actual_dir)
    log(f"created symlink: {canonical_dir} -> {actual_dir}")


def scan_dataset(name: str, previous: dict) -> dict:
    dataset_dir = resolve_dataset_dir(name)
    canonical_dir = DATASET_ROOT / name
    feature_dir = dataset_dir / "Feature" if dataset_dir else Path()
    label_dir = dataset_dir / "Label" if dataset_dir else Path()
    feature_count = count_npy(feature_dir)
    label_count = count_npy(label_dir)
    candidate = feature_count > 0 and feature_count == label_count

    last_feature = previous.get("feature_count")
    last_label = previous.get("label_count")
    stable_rounds = previous.get("stable_rounds", 0)

    if candidate and feature_count == last_feature and label_count == last_label:
        stable_rounds += 1
    else:
        stable_rounds = 0

    ready = candidate and stable_rounds >= 1
    if canonical_dir.exists() and candidate:
        ready = True
    if ready and dataset_dir is not None:
        ensure_canonical_link(name, dataset_dir)
    return {
        "path": str(dataset_dir) if dataset_dir else "",
        "feature_count": feature_count,
        "label_count": label_count,
        "stable_rounds": stable_rounds,
        "ready": ready,
    }


def current_dataset_names():
    names = set()
    for root in ROOT_CANDIDATES:
        if not root.exists():
            continue
        for path in root.iterdir():
            if path.is_dir() and path.name not in IGNORE_NAMES:
                names.add(path.name)
    return sorted(names)


def choose_dataset(state: dict):
    for name in PRIORITY:
        dataset_state = state["datasets"].get(name)
        if dataset_state and dataset_state.get("ready") and name not in state.get("completed", []):
            if dataset_is_running_or_completed(name):
                continue
            if dataset_already_has_results(name):
                continue
            return name
    return None


def dataset_already_has_results(name: str) -> bool:
    required = [
        ROOT / "results" / "EEGFeatures" / "supervised" / "EEGFeatures" / f"S-{name}-phase2-long-mps" / "results.txt",
        ROOT / "results" / "ERPFeatures" / "supervised" / "ERPFeatures" / f"S-{name}-phase2-long-mps" / "results.txt",
        ROOT / "results" / "EEGNet" / "supervised" / "EEGNet" / f"S-{name}-canonical-balanced-long-mps" / "results.txt",
        ROOT / "results" / "EEGConformer" / "supervised" / "EEGConformer" / f"S-{name}-phase2-long-mps" / "results.txt",
    ]
    return all(path.exists() for path in required)


def dataset_status_file(name: str) -> Path:
    return LOG_DIR / f"phase2_{name}_mps.status"


def dataset_is_running_or_completed(name: str) -> bool:
    status_path = dataset_status_file(name)
    if not status_path.exists():
        return False
    content = status_path.read_text(encoding="utf-8").strip()
    return content.startswith("running") or content == "completed" or content.startswith("skipped")


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text("running", encoding="utf-8")
    log("download monitor started")
    state = load_state()

    while True:
        dataset_names = current_dataset_names()
        for name in dataset_names:
            previous = state["datasets"].get(name, {})
            state["datasets"][name] = scan_dataset(name, previous)

        save_state(state)

        summary_parts = []
        for name in dataset_names:
            info = state["datasets"][name]
            ready_flag = "ready" if info["ready"] else "pending"
            summary_parts.append(f"{name}:F{info['feature_count']}/L{info['label_count']}:{ready_flag}")
        log("scan => " + (", ".join(summary_parts) if summary_parts else "no datasets"))

        chosen = choose_dataset(state)
        if chosen:
            state["current"] = chosen
            save_state(state)
            SELECTED_FILE.write_text(chosen, encoding="utf-8")
            STATUS_FILE.write_text(f"running:{chosen}", encoding="utf-8")
            log(f"dataset ready for Phase2 compare: {chosen}")
            result = subprocess.run([str(RUNNER), chosen], check=False)
            if result.returncode == 0:
                if chosen not in state["completed"]:
                    state["completed"].append(chosen)
                state["current"] = None
                save_state(state)
                STATUS_FILE.write_text(f"completed:{chosen}", encoding="utf-8")
                log(f"compare on {chosen} completed successfully; continue monitoring")
                continue
            elif result.returncode == 10:
                if chosen not in state["completed"]:
                    state["completed"].append(chosen)
                state["current"] = None
                save_state(state)
                STATUS_FILE.write_text(f"skipped:{chosen}:single_class_train_labels", encoding="utf-8")
                log(f"compare on {chosen} skipped because training split has only one class; continue monitoring")
                continue
            else:
                state["failed"] = {"dataset": chosen, "exit_code": result.returncode}
                state["current"] = None
                save_state(state)
                STATUS_FILE.write_text(f"failed:{chosen}:{result.returncode}", encoding="utf-8")
                log(f"monitor stopped because compare on {chosen} failed with exit_code={result.returncode}")
                return

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
