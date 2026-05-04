from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _path import add_src_to_path

add_src_to_path()

from cache_safety_erasure.utils.io import file_sha256

SOURCE_ARTIFACTS = ["manifest.json", "metrics.json", "figures/manifest.json"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail if generated paper tables/macros are stale relative to result artifacts."
    )
    parser.add_argument(
        "--pair",
        action="append",
        required=True,
        help="Paper/results pair in the form paper/generated/run=results/run.",
    )
    args = parser.parse_args()

    failures: list[str] = []
    for value in args.pair:
        paper_dir, results_dir = _parse_pair(value)
        failures.extend(check_paper_asset_freshness(paper_dir, results_dir))
    if failures:
        print("PAPER ASSET FRESHNESS CHECK FAILED")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("PAPER ASSET FRESHNESS CHECK PASSED")


def check_paper_asset_freshness(paper_dir: Path, results_dir: Path) -> list[str]:
    failures: list[str] = []
    manifest_path = paper_dir / "artifact_manifest.json"
    if not manifest_path.exists():
        return [f"missing paper artifact manifest: {manifest_path}"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"invalid paper artifact manifest `{manifest_path}`: {exc}"]
    failures.extend(_table_failures(manifest, paper_dir))
    failures.extend(_source_failures(manifest, results_dir))
    return failures


def _table_failures(manifest: dict[str, Any], paper_dir: Path) -> list[str]:
    failures: list[str] = []
    tables = manifest.get("tables")
    if not isinstance(tables, dict) or not tables:
        return [f"paper artifact manifest lacks table entries: {paper_dir}"]
    for name, table in tables.items():
        if not isinstance(table, dict):
            failures.append(f"paper artifact table entry `{name}` is malformed")
            continue
        path = Path(str(table.get("path", "")))
        if not path.exists():
            failures.append(f"paper artifact table `{name}` is missing")
            continue
        if table.get("sha256") != file_sha256(path):
            failures.append(f"paper artifact table `{name}` hash is stale")
    return failures


def _source_failures(manifest: dict[str, Any], results_dir: Path) -> list[str]:
    failures: list[str] = []
    sources = manifest.get("source_artifacts")
    if not isinstance(sources, dict):
        return [f"paper artifact manifest lacks source entries for {results_dir}"]
    for name in SOURCE_ARTIFACTS:
        source = sources.get(name)
        if not isinstance(source, dict):
            failures.append(f"paper artifact manifest lacks source `{name}`")
            continue
        path = results_dir / name
        if not path.exists():
            failures.append(f"paper artifact source `{name}` is missing")
            continue
        if source.get("sha256") != file_sha256(path):
            failures.append(f"paper artifact source `{name}` hash is stale")
    return failures


def _parse_pair(value: str) -> tuple[Path, Path]:
    if "=" not in value:
        raise SystemExit(f"Expected --pair value like paper/generated/run=results/run, got `{value}`")
    paper_dir, results_dir = value.split("=", 1)
    return Path(paper_dir), Path(results_dir)


if __name__ == "__main__":
    main()
