import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))

from report_h200_status import (
    _artifact_status,
    _gpu_gate_likely_blocked,
    _is_status_probe_process,
    _parse_compute_app_line,
    _parse_gpu_query_line,
    _run,
    render_markdown,
)


def test_parse_gpu_query_line() -> None:
    parsed = _parse_gpu_query_line("NVIDIA H200 NVL, 142461, 707, 143771, 100")

    assert parsed == {
        "available": True,
        "name": "NVIDIA H200 NVL",
        "memory_used_mib": 142461,
        "memory_free_mib": 707,
        "memory_total_mib": 143771,
        "utilization_pct": 100,
    }
    assert _gpu_gate_likely_blocked(parsed)


def test_parse_gpu_query_line_rejects_malformed_output() -> None:
    assert _parse_gpu_query_line("not enough fields") is None


def test_parse_compute_app_line() -> None:
    parsed = _parse_compute_app_line("1234, python, 4096")

    assert parsed == {"pid": "1234", "process_name": "python", "used_memory_mib": 4096}
    assert _parse_compute_app_line("not enough fields") is None
    assert _parse_compute_app_line("1234, python, N/A") is None


def test_status_probe_process_filter_skips_monitoring_shells() -> None:
    assert _is_status_probe_process("bash -c ps -eo pid,ppid,stat,etime,cmd | grep -E wait")
    assert _is_status_probe_process("python scripts/report_h200_status.py")
    assert not _is_status_probe_process("bash scripts/wait_for_h200_gpu.sh")


def test_run_reports_missing_executable() -> None:
    result = _run(["definitely_missing_h200_status_binary"], cwd=None)

    assert result.returncode == 127
    assert "definitely_missing_h200_status_binary" in result.stderr


def test_artifact_status_marks_expected_h200_dirs(tmp_path: Path) -> None:
    (tmp_path / "results" / "h200_qwen_full_sweep").mkdir(parents=True)

    rows = _artifact_status(tmp_path)

    lookup = {row["path"]: row["exists"] for row in rows}
    assert lookup["results/h200_qwen_full_sweep"] is True
    assert lookup["results/h200_causal_patch_qwen7b"] is False


def test_render_markdown_summarizes_blocked_launcher() -> None:
    text = render_markdown(
        {
            "created_at_utc": "20260504T000000Z",
            "repo_dir": "/home/aryang9/sandbox/llm-safety",
            "git": {"commit": "abc123"},
            "experiment_running": False,
            "launcher_waiting": True,
            "gpu_gate_likely_blocked": True,
            "gpu": {
                "available": True,
                "name": "NVIDIA H200 NVL",
                "memory_used_mib": 142461,
                "memory_total_mib": 143771,
                "utilization_pct": 100,
                "compute_apps": [],
                "pmon": "# gpu pid type sm mem\n0 - - - -",
            },
            "processes": [
                {
                    "pid": "10",
                    "elapsed": "01:00",
                    "command": "bash scripts/wait_for_h200_gpu.sh",
                }
            ],
            "expected_artifacts": [{"path": "results/h200_qwen_full_sweep", "exists": False}],
            "launcher_log": {"path": "logs/h200/wait.log", "tail": "Waiting for H200 GPU"},
        }
    )

    assert "GPU gate likely blocked: `true`" in text
    assert "none reported by `nvidia-smi --query-compute-apps`" in text
    assert "Process Monitor Snapshot" in text
    assert "`results/h200_qwen_full_sweep`: missing" in text
