#!/usr/bin/env python3
"""Small AutoDL container instance Pro API client for Phase 3 sanity runs."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
from pathlib import Path
from typing import Any
from urllib import error, parse, request


API_HOST = "https://api.autodl.com"
DEFAULT_IMAGE_UUID = "base-image-l2t43iu6uk"  # PyTorch cuda11.8-cudnn8-devel-ubuntu20.04-py38-torch2.0.0
DEFAULT_GPU_SPEC_UUID = "v-48g"  # 4090-48G general


def env_value(name: str, default: str = "") -> str:
    value = os.environ.get(name, "").strip()
    return value if value else default


def load_env_file() -> None:
    env_file = Path(os.environ.get("AUTODL_ENV_FILE", ".env"))
    if not env_file.exists():
        return
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
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


def require_token() -> str:
    token = os.environ.get("AUTODL_TOKEN", "").strip()
    if not token:
        raise SystemExit("请先设置环境变量 AUTODL_TOKEN。")
    return token


def autodl_request(method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    token = require_token()
    data = None
    headers = {"Authorization": token}
    if method == "GET" and payload:
        path += "?" + parse.urlencode(payload)
    elif payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = request.Request(
        API_HOST + path,
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with request.urlopen(req, timeout=60) as resp:
            body = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"AutoDL API HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise SystemExit(f"AutoDL API request failed: {exc}") from exc

    result = json.loads(body)
    if result.get("code") != "Success":
        raise SystemExit(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def explain_payload(command: str, payload: dict[str, Any]) -> dict[str, Any]:
    gpu_spec_uuid = payload.get("gpu_spec_uuid", "")
    image_uuid = payload.get("image_uuid", "")
    return {
        "instance": {
            "name": payload.get("instance_name", ""),
            "gpu_spec_uuid": gpu_spec_uuid,
            "gpu_amount": payload.get("req_gpu_amount", ""),
            "image_uuid": image_uuid,
            "cuda_v_from": payload.get("cuda_v_from", ""),
            "data_center_list": payload.get("data_center_list", []),
            "system_disk_expand_gb": payload.get("expand_system_disk_by_gb", 0),
        },
        "known_defaults": {
            "h800": "AutoDL docs: H800-80G general GPU spec",
            "v-48g": "AutoDL docs: 4090-48G general GPU spec",
            "pro6000-p": "AutoDL docs: PRO6000-96G performance GPU spec",
            "v-32g-p": "AutoDL docs: 4080(S)-32G performance GPU spec",
            "v-48g-350w": "AutoDL docs: 3090-48G general GPU spec",
            "5090-p": "AutoDL docs: 5090-32G performance GPU spec",
            "base-image-l2t43iu6uk": "AutoDL docs: public PyTorch CUDA 11.8 image",
        },
        "remote_start_command": payload.get("start_command", ""),
        "remote_steps": remote_steps_for(command),
    }


def remote_steps_for(command: str) -> list[str]:
    if command == "create-data-prep":
        return [
            "AutoDL creates the instance from gpu_spec_uuid/image_uuid/data_center_list.",
            "The instance runs start_command after boot.",
            "start_command clones ERP_REPO_URL into ERP_REPO_DIR if needed.",
            "It runs scripts/autodl/prepare_phase3_data_fs.sh.",
            "The prep script checks Python, installs gdown if missing, downloads CESCA-AODD, validates Feature/Label npy counts, and writes data under PHASE3_DATA_ROOT.",
        ]
    if command == "create-sanity":
        return [
            "AutoDL creates the GPU sanity instance from gpu_spec_uuid/image_uuid/data_center_list.",
            "The instance runs start_command after boot.",
            "start_command clones ERP_REPO_URL into ERP_REPO_DIR if needed.",
            "It runs scripts/autodl/bootstrap_phase3_sanity.sh.",
            "The bootstrap checks data exists under PHASE3_DATA_ROOT, runs verify_env.py, then runs CESCA-AODD x 3 patch types x 1 epoch.",
        ]
    return []


def build_start_command(args: argparse.Namespace) -> str:
    repo_dir = args.repo_dir
    repo_url = args.repo_url or os.environ.get("ERP_REPO_URL", "")
    data_root = args.data_root or os.environ.get("PHASE3_DATA_ROOT", "")

    exports = {
        "ERP_REPO_DIR": repo_dir,
        "ERP_REPO_URL": repo_url,
        "PHASE3_DATA_ROOT": data_root,
        "PHASE3_DATA_ARCHIVE_URL": args.data_archive_url,
        "PHASE3_DATA_ARCHIVE_NAME": args.data_archive_name,
        "PHASE3_DATA_EXTRACT_DIR": args.data_extract_dir,
        "PHASE3_INSTALL_REQUIREMENTS": "1" if args.install_requirements else "0",
        "PHASE3_BATCH_SIZE": str(args.batch_size),
        "PHASE3_DEVICES": args.devices,
        "CUDA_VISIBLE_DEVICES": args.cuda_visible_devices,
    }
    export_cmd = " ".join(
        f"export {key}={shlex.quote(value)};" for key, value in exports.items() if value
    )
    clone_cmd = ""
    if repo_url:
        clone_cmd = (
            f"if [ ! -d {shlex.quote(repo_dir)}/.git ]; "
            f"then git clone {shlex.quote(repo_url)} {shlex.quote(repo_dir)}; fi;"
        )
    return (
        "bash -lc "
        + shlex.quote(
            "set -e; "
            + export_cmd
            + " "
            + clone_cmd
            + f" cd {shlex.quote(repo_dir)}; "
            + "bash scripts/autodl/bootstrap_phase3_sanity.sh"
        )
    )


def build_data_prep_start_command(args: argparse.Namespace) -> str:
    repo_dir = args.repo_dir
    repo_url = args.repo_url or os.environ.get("ERP_REPO_URL", "")
    data_root = args.data_root or os.environ.get("PHASE3_DATA_ROOT", "/root/autodl-fs/dataset/200Hz")
    download_root = args.download_root or os.environ.get("PHASE3_DATA_DOWNLOAD_ROOT", "/root/autodl-fs/downloads_gdrive")

    exports = {
        "ERP_REPO_DIR": repo_dir,
        "ERP_REPO_URL": repo_url,
        "PHASE3_DATA_ROOT": data_root,
        "PHASE3_DATA_ARCHIVE_URL": args.data_archive_url,
        "PHASE3_DATA_DOWNLOAD_ROOT": download_root,
    }
    export_cmd = " ".join(
        f"export {key}={shlex.quote(value)};" for key, value in exports.items() if value
    )
    clone_cmd = ""
    if repo_url:
        clone_cmd = (
            f"if [ ! -d {shlex.quote(repo_dir)}/.git ]; "
            f"then git clone {shlex.quote(repo_url)} {shlex.quote(repo_dir)}; fi;"
        )

    prep_command = [
        "bash",
        "scripts/autodl/prepare_phase3_data_fs.sh",
        "--datasets",
        args.datasets,
        "--data-root",
        data_root,
        "--download-root",
        download_root,
    ]
    if args.force:
        prep_command.append("--force")
    prep_cmd = " ".join(shlex.quote(part) for part in prep_command)

    return (
        "bash -lc "
        + shlex.quote(
            "set -e; "
            + export_cmd
            + " "
            + clone_cmd
            + f" cd {shlex.quote(repo_dir)}; "
            + prep_cmd
        )
    )


def data_centers_from(args: argparse.Namespace, env_key: str) -> list[str]:
    return [
        item.strip()
        for item in (args.data_centers or os.environ.get(env_key, "") or os.environ.get("AUTODL_DATA_CENTERS", "")).split(",")
        if item.strip()
    ]


def create_sanity(args: argparse.Namespace) -> dict[str, Any]:
    data_centers = data_centers_from(args, "AUTODL_SANITY_DATA_CENTERS")
    payload: dict[str, Any] = {
        "req_gpu_amount": args.gpu_amount,
        "expand_system_disk_by_gb": args.expand_system_disk_gb,
        "gpu_spec_uuid": args.gpu_spec_uuid,
        "image_uuid": args.image_uuid,
        "cuda_v_from": args.cuda_v_from,
        "instance_name": args.instance_name,
        "start_command": args.start_command or build_start_command(args),
    }
    if data_centers:
        payload["data_center_list"] = data_centers

    if args.dry_run:
        result: dict[str, Any] = {"code": "DryRun", "data": payload, "msg": "", "request_id": ""}
        if args.explain:
            result["explain"] = explain_payload("create-sanity", payload)
        return result

    result = autodl_request("POST", "/api/v1/dev/instance/pro/create", payload)
    if args.state_file:
        state_path = Path(args.state_file)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def create_data_prep(args: argparse.Namespace) -> dict[str, Any]:
    if not args.gpu_spec_uuid:
        raise SystemExit(
            "create-data-prep requires AUTODL_DATA_PREP_GPU_SPEC_UUID or --gpu-spec-uuid. "
            "AutoDL Pro API does not support CPU-only creation: req_gpu_amount min is 1 and gpu_spec_uuid is required. "
            "Choose the cheapest GPU spec available in the same region as /root/autodl-fs."
        )
    data_centers = data_centers_from(args, "AUTODL_DATA_PREP_DATA_CENTERS")
    payload: dict[str, Any] = {
        "req_gpu_amount": args.gpu_amount,
        "expand_system_disk_by_gb": args.expand_system_disk_gb,
        "gpu_spec_uuid": args.gpu_spec_uuid,
        "image_uuid": args.image_uuid,
        "cuda_v_from": args.cuda_v_from,
        "instance_name": args.instance_name,
        "start_command": args.start_command or build_data_prep_start_command(args),
    }
    if data_centers:
        payload["data_center_list"] = data_centers

    if args.dry_run:
        result: dict[str, Any] = {"code": "DryRun", "data": payload, "msg": "", "request_id": ""}
        if args.explain:
            result["explain"] = explain_payload("create-data-prep", payload)
        return result

    result = autodl_request("POST", "/api/v1/dev/instance/pro/create", payload)
    if args.state_file:
        state_path = Path(args.state_file)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def wait_status(instance_uuid: str, target: str, timeout_seconds: int) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    last = None
    while time.time() < deadline:
        last = autodl_request("GET", "/api/v1/dev/instance/pro/status", {"instance_uuid": instance_uuid})
        status = last.get("data")
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} status={status}", flush=True)
        if status == target:
            return last
        time.sleep(15)
    raise SystemExit(f"Timed out waiting for status={target}; last={last}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AutoDL Pro API helper.")
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create-sanity", help="Create one GPU instance and run Phase 3 sanity.")
    create.add_argument("--image-uuid", default=env_value("AUTODL_IMAGE_UUID", DEFAULT_IMAGE_UUID))
    create.add_argument("--gpu-spec-uuid", default=env_value("AUTODL_GPU_SPEC_UUID", DEFAULT_GPU_SPEC_UUID))
    create.add_argument("--gpu-amount", type=int, default=int(env_value("AUTODL_GPU_AMOUNT", "1")))
    create.add_argument("--cuda-v-from", type=int, default=int(env_value("AUTODL_CUDA_V_FROM", "118")))
    create.add_argument("--expand-system-disk-gb", type=int, default=int(env_value("AUTODL_SYSTEM_DISK_GB", "0")))
    create.add_argument("--data-centers", default="")
    create.add_argument("--instance-name", default=env_value("AUTODL_INSTANCE_NAME", "ERP Phase3 sanity"))
    create.add_argument("--repo-dir", default=env_value("ERP_REPO_DIR", "/root/ERP-Benchmark"))
    create.add_argument("--repo-url", default=env_value("ERP_REPO_URL", ""))
    create.add_argument("--data-root", default=env_value("PHASE3_DATA_ROOT", ""))
    create.add_argument("--data-archive-url", default=env_value("PHASE3_DATA_ARCHIVE_URL", ""))
    create.add_argument("--data-archive-name", default=env_value("PHASE3_DATA_ARCHIVE_NAME", "cesca-aodd-200hz.tar.gz"))
    create.add_argument("--data-extract-dir", default=env_value("PHASE3_DATA_EXTRACT_DIR", "/root/autodl-fs"))
    create.add_argument("--batch-size", type=int, default=int(env_value("PHASE3_BATCH_SIZE", "128")))
    create.add_argument("--devices", default=env_value("PHASE3_DEVICES", "0"))
    create.add_argument("--cuda-visible-devices", default=env_value("CUDA_VISIBLE_DEVICES", "0"))
    create.add_argument("--install-requirements", action="store_true")
    create.add_argument("--start-command", default=env_value("AUTODL_START_COMMAND", ""))
    create.add_argument("--state-file", default="results/phase3_logs/autodl_create_sanity_response.json")
    create.add_argument("--dry-run", action="store_true")
    create.add_argument("--explain", action="store_true")

    prep = sub.add_parser("create-data-prep", help="Create one instance to populate /root/autodl-fs before GPU sanity.")
    prep.add_argument("--image-uuid", default=env_value("AUTODL_DATA_PREP_IMAGE_UUID", env_value("AUTODL_IMAGE_UUID", DEFAULT_IMAGE_UUID)))
    prep.add_argument("--gpu-spec-uuid", default=env_value("AUTODL_DATA_PREP_GPU_SPEC_UUID", ""))
    prep.add_argument("--gpu-amount", type=int, default=int(env_value("AUTODL_DATA_PREP_GPU_AMOUNT", env_value("AUTODL_GPU_AMOUNT", "1"))))
    prep.add_argument("--cuda-v-from", type=int, default=int(env_value("AUTODL_DATA_PREP_CUDA_V_FROM", env_value("AUTODL_CUDA_V_FROM", "118"))))
    prep.add_argument("--expand-system-disk-gb", type=int, default=int(env_value("AUTODL_DATA_PREP_SYSTEM_DISK_GB", env_value("AUTODL_SYSTEM_DISK_GB", "0"))))
    prep.add_argument("--data-centers", default="")
    prep.add_argument("--instance-name", default=env_value("AUTODL_DATA_PREP_INSTANCE_NAME", "ERP Phase3 data prep"))
    prep.add_argument("--repo-dir", default=env_value("ERP_REPO_DIR", "/root/ERP-Benchmark"))
    prep.add_argument("--repo-url", default=env_value("ERP_REPO_URL", ""))
    prep.add_argument("--datasets", default=env_value("PHASE3_DATASETS", "CESCA-AODD"))
    prep.add_argument("--data-root", default=env_value("PHASE3_DATA_ROOT", "/root/autodl-fs/dataset/200Hz"))
    prep.add_argument("--download-root", default=env_value("PHASE3_DATA_DOWNLOAD_ROOT", "/root/autodl-fs/downloads_gdrive"))
    prep.add_argument("--data-archive-url", default=env_value("PHASE3_DATA_ARCHIVE_URL", ""))
    prep.add_argument("--start-command", default=env_value("AUTODL_DATA_PREP_START_COMMAND", ""))
    prep.add_argument("--state-file", default="results/phase3_logs/autodl_create_data_prep_response.json")
    prep.add_argument("--force", action="store_true")
    prep.add_argument("--dry-run", action="store_true")
    prep.add_argument("--explain", action="store_true")

    for name, method, path in [
        ("list", "POST", "/api/v1/dev/instance/pro/list"),
        ("image-list", "POST", "/api/v1/dev/instance/pro/image/private/list"),
    ]:
        command = sub.add_parser(name)
        command.set_defaults(api_method=method, api_path=path)
        command.add_argument("--page-index", type=int, default=1)
        command.add_argument("--page-size", type=int, default=10)

    for name, method, path in [
        ("status", "GET", "/api/v1/dev/instance/pro/status"),
        ("snapshot", "GET", "/api/v1/dev/instance/pro/snapshot"),
        ("power-off", "POST", "/api/v1/dev/instance/pro/power_off"),
        ("release", "POST", "/api/v1/dev/instance/pro/release"),
    ]:
        command = sub.add_parser(name)
        command.set_defaults(api_method=method, api_path=path)
        command.add_argument("instance_uuid")

    wait = sub.add_parser("wait-status")
    wait.add_argument("instance_uuid")
    wait.add_argument("--target", default="running")
    wait.add_argument("--timeout-seconds", type=int, default=900)
    return parser.parse_args()


def main() -> int:
    load_env_file()
    args = parse_args()
    if args.command == "create-sanity":
        print_json(create_sanity(args))
        return 0
    if args.command == "create-data-prep":
        print_json(create_data_prep(args))
        return 0
    if args.command in {"list", "image-list"}:
        payload = {"page_index": args.page_index, "page_size": args.page_size}
        print_json(autodl_request(args.api_method, args.api_path, payload))
        return 0
    if args.command in {"status", "snapshot", "power-off", "release"}:
        payload = {"instance_uuid": args.instance_uuid}
        print_json(autodl_request(args.api_method, args.api_path, payload))
        return 0
    if args.command == "wait-status":
        print_json(wait_status(args.instance_uuid, args.target, args.timeout_seconds))
        return 0
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
