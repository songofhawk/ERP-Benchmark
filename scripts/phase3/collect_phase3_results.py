#!/usr/bin/env python3
"""Collect Phase 3 TestFormer result files into a compact Markdown table."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT_RE = re.compile(r"Test results --- (?P<body>.+)")
METRIC_RE = re.compile(r"([A-Za-z]+): ([0-9.]+)(?:\+-([0-9.]+))?%?")


def parse_metrics(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = RESULT_RE.findall(text)
    if not matches:
        return {}
    body = matches[-1]
    metrics = {}
    for name, value, _std in METRIC_RE.findall(body):
        metrics[name] = float(value)
    return metrics


def infer_patch_type(model_id: str) -> str:
    for label, value in [
        ("Multi-Variate", "multi-variate"),
        ("Uni-Variate", "uni-variate"),
        ("Whole-Variate", "whole-variate"),
    ]:
        if label in model_id:
            return value
    return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect Phase 3 TestFormer results.")
    parser.add_argument("--dataset", default="CESCA-AODD")
    parser.add_argument("--results-root", type=Path, default=ROOT / "results" / "TestFormer" / "supervised" / "TestFormer")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "phase3_logs" / "phase3_sanity_summary.md")
    args = parser.parse_args()

    rows = []
    for result_path in sorted(args.results_root.glob(f"S-{args.dataset}-phase3-sanity-*/results.txt")):
        model_id = result_path.parent.name
        metrics = parse_metrics(result_path)
        rows.append((infer_patch_type(model_id), model_id, metrics, result_path))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Phase 3 Sanity Summary: {args.dataset}",
        "",
        "| Patch Type | Model ID | Accuracy | F1 | AUROC | AUPRC | Result |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for patch_type, model_id, metrics, result_path in rows:
        lines.append(
            "| {patch_type} | `{model_id}` | {acc} | {f1} | {auroc} | {auprc} | `{path}` |".format(
                patch_type=patch_type,
                model_id=model_id,
                acc=metrics.get("Accuracy", ""),
                f1=metrics.get("F1", ""),
                auroc=metrics.get("AUROC", ""),
                auprc=metrics.get("AUPRC", ""),
                path=result_path.relative_to(ROOT),
            )
        )

    if not rows:
        lines.append("| _no results_ | | | | | | |")

    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
