from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any

from _path import add_src_to_path

add_src_to_path()

from cache_safety_erasure.utils.io import utc_timestamp, write_json

EXPECTED_PATHS = [
    Path("results/h200_qwen_full_sweep"),
    Path("results/h200_causal_patch_qwen7b"),
    Path("results/h200_qwen32b_public_followup_primary"),
    Path("paper/generated/h200_qwen_full_sweep"),
    Path("paper/generated/h200_causal_patch_qwen7b"),
    Path("paper/generated/claim_assessment"),
    Path("paper/audit/h200_qwen_full_sweep_summary"),
    Path("paper/audit/h200_causal_patch_qwen7b_summary"),
]
PROCESS_PATTERNS = [
    "wait_and_run_h200_sweep",
    "wait_for_h200_gpu",
    "run_h200_sweep",
    "run_experiment.py",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Report H200 launcher, GPU, and artifact status.")
    parser.add_argument("--repo-dir", type=Path, default=Path.cwd())
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--log-lines", type=int, default=40)
    args = parser.parse_args()

    status = h200_status(args.repo_dir, log_lines=args.log_lines)
    if args.output_json is not None:
        write_json(args.output_json, status)
    markdown = render_markdown(status)
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown, encoding="utf-8")
    print(markdown)


def h200_status(repo_dir: Path, *, log_lines: int = 40) -> dict[str, Any]:
    repo_dir = repo_dir.resolve()
    latest_log = _latest_launcher_log(repo_dir)
    gpu = _gpu_status()
    return {
        "schema_version": 1,
        "created_at_utc": utc_timestamp(),
        "repo_dir": str(repo_dir),
        "git": {
            "commit": _command_text(["git", "rev-parse", "--short", "HEAD"], cwd=repo_dir),
            "status_short": _command_text(["git", "status", "--short"], cwd=repo_dir),
        },
        "processes": _process_rows(),
        "launcher_log": {
            "path": str(latest_log) if latest_log else None,
            "tail": _tail(latest_log, log_lines) if latest_log else "",
        },
        "gpu": gpu,
        "expected_artifacts": _artifact_status(repo_dir),
        "experiment_running": any("run_experiment.py" in row["command"] for row in _process_rows()),
        "launcher_waiting": any("wait_for_h200_gpu" in row["command"] for row in _process_rows()),
        "gpu_gate_likely_blocked": _gpu_gate_likely_blocked(gpu),
    }


def render_markdown(status: dict[str, Any]) -> str:
    lines = [
        "# H200 Status",
        "",
        f"Created: `{status['created_at_utc']}`",
        f"Repo: `{status['repo_dir']}`",
        f"Commit: `{status['git']['commit'] or 'unknown'}`",
        f"Experiment running: `{str(status['experiment_running']).lower()}`",
        f"Launcher waiting: `{str(status['launcher_waiting']).lower()}`",
        f"GPU gate likely blocked: `{str(status['gpu_gate_likely_blocked']).lower()}`",
        "",
        "## GPU",
        "",
    ]
    gpu = status["gpu"]
    if gpu.get("available"):
        lines.append(
            f"- `{gpu['name']}` memory `{gpu['memory_used_mib']}/{gpu['memory_total_mib']} MiB`, "
            f"utilization `{gpu['utilization_pct']}%`"
        )
    else:
        lines.append(f"- unavailable: `{gpu.get('error')}`")
    if gpu.get("available"):
        lines.append("")
        lines.append("### Visible Compute Apps")
        lines.append("")
        compute_apps = gpu.get("compute_apps") or []
        if compute_apps:
            for app in compute_apps:
                lines.append(
                    f"- pid `{app['pid']}`, memory `{app['used_memory_mib']} MiB`: "
                    f"`{app['process_name']}`"
                )
        else:
            lines.append("- none reported by `nvidia-smi --query-compute-apps`")
        pmon = str(gpu.get("pmon") or "").strip()
        if pmon:
            lines.extend(["", "### Process Monitor Snapshot", "", "```text", pmon, "```"])
    lines.extend(["", "## Processes", ""])
    if status["processes"]:
        for row in status["processes"]:
            lines.append(f"- pid `{row['pid']}`, elapsed `{row['elapsed']}`: `{row['command']}`")
    else:
        lines.append("- none")
    lines.extend(["", "## Expected Artifacts", ""])
    for row in status["expected_artifacts"]:
        state = "present" if row["exists"] else "missing"
        lines.append(f"- `{row['path']}`: {state}")
    lines.extend(["", "## Latest Launcher Log", ""])
    if status["launcher_log"]["path"]:
        lines.append(f"`{status['launcher_log']['path']}`")
        lines.append("")
        lines.append("```text")
        lines.append(status["launcher_log"]["tail"].rstrip())
        lines.append("```")
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def _gpu_status() -> dict[str, Any]:
    result = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.used,memory.free,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        cwd=None,
    )
    if result.returncode != 0:
        return {"available": False, "error": result.stderr.strip() or result.stdout.strip()}
    line = next((line for line in result.stdout.splitlines() if line.strip()), "")
    parsed = _parse_gpu_query_line(line)
    if parsed is None:
        return {"available": False, "error": f"could not parse nvidia-smi output: {line}"}
    pmon = _run(["nvidia-smi", "pmon", "-c", "1"], cwd=None)
    parsed["pmon"] = pmon.stdout.strip() if pmon.returncode == 0 else pmon.stderr.strip()
    parsed["compute_apps"] = _compute_apps()
    return parsed


def _parse_gpu_query_line(line: str) -> dict[str, Any] | None:
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 5:
        return None
    try:
        return {
            "available": True,
            "name": parts[0],
            "memory_used_mib": int(parts[1]),
            "memory_free_mib": int(parts[2]),
            "memory_total_mib": int(parts[3]),
            "utilization_pct": int(parts[4]),
        }
    except ValueError:
        return None


def _gpu_gate_likely_blocked(gpu: dict[str, Any]) -> bool:
    if not gpu.get("available"):
        return False
    return int(gpu.get("memory_used_mib") or 0) > 20_000 or int(gpu.get("utilization_pct") or 0) > 20


def _compute_apps() -> list[dict[str, Any]]:
    result = _run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        cwd=None,
    )
    if result.returncode != 0:
        return []
    rows = []
    for line in result.stdout.splitlines():
        parsed = _parse_compute_app_line(line)
        if parsed is not None:
            rows.append(parsed)
    return rows


def _parse_compute_app_line(line: str) -> dict[str, Any] | None:
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 3 or not parts[0]:
        return None
    try:
        used_memory_mib = int(parts[2])
    except ValueError:
        return None
    return {"pid": parts[0], "process_name": parts[1], "used_memory_mib": used_memory_mib}


def _process_rows() -> list[dict[str, str]]:
    result = _run(["ps", "-eo", "pid,ppid,stat,etime,cmd"], cwd=None)
    if result.returncode != 0:
        return []
    rows = []
    for line in result.stdout.splitlines()[1:]:
        if not any(pattern in line for pattern in PROCESS_PATTERNS):
            continue
        if _is_status_probe_process(line):
            continue
        parts = line.split(None, 4)
        if len(parts) < 5:
            continue
        rows.append(
            {
                "pid": parts[0],
                "ppid": parts[1],
                "stat": parts[2],
                "elapsed": parts[3],
                "command": parts[4],
            }
        )
    return rows


def _is_status_probe_process(line: str) -> bool:
    probe_markers = [
        "report_h200_status.py",
        "ps -eo",
        "grep -E",
        "nvidia-smi --query",
    ]
    return any(marker in line for marker in probe_markers)


def _artifact_status(repo_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for relative_path in EXPECTED_PATHS:
        path = repo_dir / relative_path
        rows.append({"path": str(relative_path), "exists": path.exists()})
    return rows


def _latest_launcher_log(repo_dir: Path) -> Path | None:
    logs = sorted((repo_dir / "logs" / "h200").glob("wait_and_run_*.log"), reverse=True)
    return logs[0] if logs else None


def _tail(path: Path, line_count: int) -> str:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-line_count:])


def _command_text(command: list[str], *, cwd: Path) -> str:
    result = _run(command, cwd=cwd)
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _run(command: list[str], *, cwd: Path | None) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except FileNotFoundError as exc:
        return subprocess.CompletedProcess(command, 127, "", str(exc))


if __name__ == "__main__":
    main()
