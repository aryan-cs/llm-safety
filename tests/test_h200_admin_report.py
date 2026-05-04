import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path("scripts").resolve()))

from write_h200_admin_report import admin_report, main


def test_admin_report_summarizes_hidden_gpu_context_without_results() -> None:
    report = admin_report(
        {
            "created_at_utc": "20260504T000000Z",
            "repo_dir": "/home/aryang9/sandbox/llm-safety",
            "git": {"commit": "abc123"},
            "experiment_running": False,
            "launcher_waiting": True,
            "gpu_gate_likely_blocked": True,
            "hidden_gpu_context_likely": True,
            "gpu": {
                "available": True,
                "name": "NVIDIA H200 NVL",
                "memory_used_mib": 142461,
                "memory_total_mib": 143771,
                "utilization_pct": 100,
                "compute_apps": [],
                "accounted_apps": [],
                "device_holders": [],
                "pmon": "# gpu pid type sm mem\n0 - - - -",
                "pid_query": "Processes                             : None",
            },
            "processes": [
                {
                    "pid": "10",
                    "elapsed": "01:00",
                    "command": "bash scripts/wait_for_h200_gpu.sh",
                }
            ],
        }
    )

    assert "infrastructure diagnostics only" in report
    assert "Hidden GPU context likely: `true`" in report
    assert "Visible compute apps: `0`" in report
    assert "Processes                             : None" in report
    assert "release or restart the notebook allocation" in report
    assert "nvidia-smi --gpu-reset" in report
    assert "model generations" in report


def test_admin_report_cli_reports_missing_status_json() -> None:
    original_argv = sys.argv
    sys.argv = [
        "write_h200_admin_report.py",
        "--status-json",
        "logs/h200/does_not_exist.json",
    ]
    try:
        with pytest.raises(SystemExit) as excinfo:
            main()
    finally:
        sys.argv = original_argv

    assert "Run scripts/report_h200_status.py with --output-json first" in str(excinfo.value)
