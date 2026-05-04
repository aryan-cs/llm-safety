from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write a concise support report for hidden H200 GPU-context blockers."
    )
    parser.add_argument(
        "--status-json",
        type=Path,
        default=Path("logs/h200/h200_status_latest.json"),
        help="JSON produced by scripts/report_h200_status.py.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("logs/h200/h200_admin_report.md"),
    )
    args = parser.parse_args()

    if not args.status_json.exists():
        raise SystemExit(
            f"Missing status JSON: {args.status_json}. "
            "Run scripts/report_h200_status.py with --output-json first."
        )
    status = json.loads(args.status_json.read_text(encoding="utf-8"))
    report = admin_report(status)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(report, encoding="utf-8")
    print(f"Wrote {args.output_md}")


def admin_report(status: dict[str, Any]) -> str:
    gpu = status.get("gpu", {})
    lines = [
        "# H200 Hidden GPU Context Support Report",
        "",
        "This report contains infrastructure diagnostics only. It is not an experiment result "
        "and contains no model generations, labels, or paper evidence.",
        "",
        "## Summary",
        "",
        f"- Created: `{status.get('created_at_utc', 'unknown')}`",
        f"- Repo: `{status.get('repo_dir', 'unknown')}`",
        f"- Git commit: `{status.get('git', {}).get('commit', 'unknown')}`",
        f"- Experiment running: `{_bool(status.get('experiment_running'))}`",
        f"- Launcher waiting: `{_bool(status.get('launcher_waiting'))}`",
        f"- GPU gate blocked: `{_bool(status.get('gpu_gate_likely_blocked'))}`",
        f"- Hidden GPU context likely: `{_bool(status.get('hidden_gpu_context_likely'))}`",
        "",
        "## GPU Snapshot",
        "",
    ]
    if gpu.get("available"):
        lines.extend(
            [
                f"- GPU: `{gpu.get('name')}`",
                (
                    f"- Memory used: `{gpu.get('memory_used_mib')}/"
                    f"{gpu.get('memory_total_mib')} MiB`"
                ),
                f"- Utilization: `{gpu.get('utilization_pct')}%`",
            ]
        )
    else:
        lines.append(f"- GPU unavailable to status script: `{gpu.get('error', 'unknown')}`")

    lines.extend(
        [
            "",
            "## Evidence Of Hidden Or Stale Context",
            "",
            f"- Visible compute apps: `{_count(gpu.get('compute_apps'))}`",
            f"- Accounted apps: `{_count(gpu.get('accounted_apps'))}`",
            f"- Local `/proc/*/fd` NVIDIA holders: `{_count(gpu.get('device_holders'))}`",
            "",
        ]
    )
    pmon = str(gpu.get("pmon") or "").strip()
    if pmon:
        lines.extend(["### `nvidia-smi pmon -c 1`", "", "```text", pmon, "```", ""])
    pid_query = str(gpu.get("pid_query") or "").strip()
    if pid_query:
        lines.extend(["### `nvidia-smi -q -d PIDS`", "", "```text", pid_query, "```", ""])

    lines.extend(
        [
            "## Waiting Processes",
            "",
        ]
    )
    processes = status.get("processes") or []
    if processes:
        for row in processes:
            lines.append(
                f"- pid `{row.get('pid')}`, elapsed `{row.get('elapsed')}`: "
                f"`{row.get('command')}`"
            )
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Requested Non-Destructive Action",
            "",
            "Please release or restart the notebook allocation, or clear the stale GPU context "
            "from the infrastructure side. The project launcher is intentionally waiting for "
            "low GPU memory and utilization before starting the registered sweep. We have not "
            "run `nvidia-smi --gpu-reset` or killed unknown processes.",
            "",
            "After the allocation is cleared, rerun:",
            "",
            "```bash",
            "cd /home/aryang9/sandbox/llm-safety",
            "uv run python scripts/report_h200_status.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _bool(value: Any) -> str:
    return str(bool(value)).lower()


def _count(value: Any) -> int:
    if isinstance(value, list):
        return len(value)
    return 0


if __name__ == "__main__":
    main()
