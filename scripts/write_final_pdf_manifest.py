from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from _path import add_src_to_path

add_src_to_path()

from cache_safety_erasure.utils.io import file_sha256, write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write a provenance manifest for the rendered final paper PDF."
    )
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="KIND=PATH",
        help="Source artifact included in the paper build; may be repeated.",
    )
    args = parser.parse_args()

    manifest = final_pdf_manifest(args.pdf, _parse_sources(args.source))
    write_json(args.output, manifest)
    print(f"Wrote {args.output}")


def final_pdf_manifest(pdf: Path, sources: list[tuple[str, Path]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "pdf": {
            "path": str(pdf),
            "sha256": file_sha256(pdf),
            "bytes": pdf.stat().st_size if pdf.exists() else None,
        },
        "source_artifacts": [
            {
                "kind": kind,
                "path": str(path),
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size if path.exists() else None,
            }
            for kind, path in sources
        ],
    }


def _parse_sources(raw_sources: list[str]) -> list[tuple[str, Path]]:
    sources = []
    for raw_source in raw_sources:
        if "=" not in raw_source:
            raise SystemExit(f"Expected --source KIND=PATH, got: {raw_source}")
        kind, raw_path = raw_source.split("=", 1)
        kind = kind.strip()
        if not kind:
            raise SystemExit(f"Missing source kind in: {raw_source}")
        sources.append((kind, Path(raw_path)))
    return sources


if __name__ == "__main__":
    main()
