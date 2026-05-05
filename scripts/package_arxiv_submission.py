from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path

from _path import add_src_to_path

add_src_to_path()

from check_final_pdf_text import forbidden_final_prose_failures
from check_latex_citations import citation_failures
from check_latex_placeholders import (
    PLACEHOLDER_TEXT_MARKERS,
    _semantic_tex_failures,
    _strip_tex_comments,
)
from check_publication_readiness import _pdf_page_failure

from cache_safety_erasure.utils.io import file_sha256, write_json

DEFAULT_PRIMARY_RESULTS_DIR = Path("results/h200_qwen_full_sweep")
DEFAULT_CAUSAL_RESULTS_DIR = Path("results/h200_causal_patch_qwen7b")
DEFAULT_PRIMARY_GENERATED_DIR = Path("paper/generated/h200_qwen_full_sweep")
DEFAULT_CAUSAL_GENERATED_DIR = Path("paper/generated/h200_causal_patch_qwen7b")
DEFAULT_ACTIVE_PRIMARY_GENERATED_DIR = Path("paper/generated/active_primary")
DEFAULT_ACTIVE_CAUSAL_GENERATED_DIR = Path("paper/generated/active_causal")
DEFAULT_CLAIM_GENERATED_DIR = Path("paper/generated/claim_assessment")
DEFAULT_QWEN32_GENERATED_DIR = Path("paper/generated/h200_qwen32b_public_followup")
DEFAULT_PRIMARY_AUDIT_DIR = Path("paper/audit/h200_qwen_full_sweep_summary")
DEFAULT_CAUSAL_AUDIT_DIR = Path("paper/audit/h200_causal_patch_qwen7b_summary")
DEFAULT_ACTIVE_PRIMARY_AUDIT_DIR = Path("paper/audit/active_primary_summary")
DEFAULT_ACTIVE_CAUSAL_AUDIT_DIR = Path("paper/audit/active_causal_summary")


def build_figure_sources(
    primary_results_dir: Path = DEFAULT_PRIMARY_RESULTS_DIR,
    causal_results_dir: Path = DEFAULT_CAUSAL_RESULTS_DIR,
) -> dict[str, Path]:
    return {
        "safety_capability_phase_portrait.pdf": (
            primary_results_dir / "figures" / "safety_capability_phase_portrait.pdf"
        ),
        "selective_safety_erasure_heatmap.pdf": (
            primary_results_dir / "figures" / "selective_safety_erasure_heatmap.pdf"
        ),
        "prompt_effect_constellation.pdf": (
            primary_results_dir / "figures" / "prompt_effect_constellation.pdf"
        ),
        "cache_state_fingerprint.pdf": (
            primary_results_dir / "figures" / "cache_state_fingerprint.pdf"
        ),
        "safety_state_atlas.pdf": primary_results_dir / "figures" / "safety_state_atlas.pdf",
        "policy_uncertainty_braid.pdf": (
            primary_results_dir / "figures" / "policy_uncertainty_braid.pdf"
        ),
        "causal_restoration_fraction.pdf": (
            causal_results_dir / "figures" / "causal_restoration_fraction.pdf"
        ),
        "causal_restoration_flow.pdf": (
            causal_results_dir / "figures" / "causal_restoration_flow.pdf"
        ),
    }


FIGURE_SOURCES = {
    name: path for name, path in build_figure_sources().items()
}
REQUIRED_GENERATED_DIRS = [
    DEFAULT_PRIMARY_GENERATED_DIR,
    DEFAULT_CAUSAL_GENERATED_DIR,
    DEFAULT_ACTIVE_PRIMARY_GENERATED_DIR,
    DEFAULT_ACTIVE_CAUSAL_GENERATED_DIR,
    DEFAULT_CLAIM_GENERATED_DIR,
]
OPTIONAL_GENERATED_DIRS = [DEFAULT_QWEN32_GENERATED_DIR]
GENERATED_DIRS = REQUIRED_GENERATED_DIRS + OPTIONAL_GENERATED_DIRS
AUDIT_DIRS = [
    DEFAULT_PRIMARY_AUDIT_DIR,
    DEFAULT_CAUSAL_AUDIT_DIR,
    DEFAULT_ACTIVE_PRIMARY_AUDIT_DIR,
    DEFAULT_ACTIVE_CAUSAL_AUDIT_DIR,
]
ARXIV_SAFE_SUPPORT_SUFFIXES = {".tex"}
FINAL_SOURCE_TEXT_MARKERS = [
    *PLACEHOLDER_TEXT_MARKERS,
    "Figure unavailable",
    "registered analysis protocol",
    "reports no empirical claims",
]
STRICT_PUBLICATION_COMMAND_BLOCK = r"""\newcommand{\todoresult}[1]{%
  \PackageError{cache-paper}{Missing required publication artifact}{#1}%
}
\newcommand{\maybeincludegraphic}[3]{%
  \IfFileExists{#1}{%
    \includegraphics[width=#2]{#1}%
  }{%
    \PackageError{cache-paper}{Missing required publication figure}{#1}%
  }%
}
\newcommand{\maybeinputtable}[2]{%
  \IfFileExists{#1}{%
    \input{#1}%
  }{%
    \PackageError{cache-paper}{Missing required publication table}{#1}%
  }%
}
\newcommand{\requiredartifact}[1]{%
  \IfFileExists{#1}{}{%
    \PackageError{cache-paper}{Missing required generated artifact}{#1}%
  }%
}
\newcommand{\EmpiricalStatusSentence}{}
\newcommand{\PrimaryRunId}{}
\newcommand{\PrimaryPolicyCount}{}
\newcommand{\PrimaryTopSSEIPolicy}{}
\newcommand{\PrimaryTopSSEI}{}
\newcommand{\PrimaryTopSSEICILow}{}
\newcommand{\PrimaryTopSSEICIHigh}{}
\newcommand{\PrimarySafetyClusterCount}{}
\newcommand{\PrimaryCapabilityClusterCount}{}
\newcommand{\CausalRunId}{}
\newcommand{\CausalPolicyCount}{}
\newcommand{\CausalTopSSEIPolicy}{}
\newcommand{\CausalTopSSEI}{}
\newcommand{\CausalTopSSEICILow}{}
\newcommand{\CausalTopSSEICIHigh}{}
\newcommand{\CausalSafetyClusterCount}{}
\newcommand{\CausalCapabilityClusterCount}{}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an arXiv-friendly LaTeX source bundle.")
    parser.add_argument("--output-dir", type=Path, default=Path("paper/build/arxiv_source"))
    parser.add_argument("--archive", type=Path, default=Path("paper/build/arxiv_source.tar.gz"))
    parser.add_argument("--primary-results-dir", type=Path, default=DEFAULT_PRIMARY_RESULTS_DIR)
    parser.add_argument("--causal-results-dir", type=Path, default=DEFAULT_CAUSAL_RESULTS_DIR)
    parser.add_argument("--primary-generated-dir", type=Path, default=DEFAULT_PRIMARY_GENERATED_DIR)
    parser.add_argument("--causal-generated-dir", type=Path, default=DEFAULT_CAUSAL_GENERATED_DIR)
    parser.add_argument("--claim-generated-dir", type=Path, default=DEFAULT_CLAIM_GENERATED_DIR)
    parser.add_argument(
        "--qwen32-generated-dir",
        type=Path,
        default=None,
        help=(
            "Optional Qwen-32B generated artifact directory. Omit unless that "
            "follow-up was rebuilt and passed readiness in the current publication build."
        ),
    )
    parser.add_argument("--primary-audit-dir", type=Path, default=DEFAULT_PRIMARY_AUDIT_DIR)
    parser.add_argument("--causal-audit-dir", type=Path, default=DEFAULT_CAUSAL_AUDIT_DIR)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Allow missing empirical assets for draft bundles. Publication builds should not use this.",
    )
    args = parser.parse_args()

    source_dir = args.output_dir
    figure_sources = build_figure_sources(args.primary_results_dir, args.causal_results_dir)
    required_generated_dirs = [
        args.primary_generated_dir,
        args.causal_generated_dir,
        DEFAULT_ACTIVE_PRIMARY_GENERATED_DIR,
        DEFAULT_ACTIVE_CAUSAL_GENERATED_DIR,
        args.claim_generated_dir,
    ]
    optional_generated_dirs = [args.qwen32_generated_dir] if args.qwen32_generated_dir else []
    audit_dirs = [
        args.primary_audit_dir,
        args.causal_audit_dir,
        DEFAULT_ACTIVE_PRIMARY_AUDIT_DIR,
        DEFAULT_ACTIVE_CAUSAL_AUDIT_DIR,
    ]
    if source_dir.exists():
        shutil.rmtree(source_dir)
    (source_dir / "figures").mkdir(parents=True)

    main_tex = Path("paper/latex/main.tex").read_text(encoding="utf-8")
    rewritten_main_tex = _rewrite_main_tex_for_arxiv(
        main_tex,
        figure_sources=figure_sources,
        strict_final=not args.allow_missing,
    )
    rewrite_failures = _rewrite_failures(rewritten_main_tex)
    if rewrite_failures:
        for failure in rewrite_failures:
            print(f"Unrewritten LaTeX source path: {failure}")
        raise SystemExit("Refusing to package arXiv source with repo-local paths in main.tex.")
    final_source_failures = [] if args.allow_missing else _final_source_failures(rewritten_main_tex)
    if final_source_failures:
        for failure in final_source_failures:
            print(f"Final LaTeX source failure: {failure}")
        raise SystemExit("Refusing to package arXiv source with draft fallback text in main.tex.")
    (source_dir / "main.tex").write_text(rewritten_main_tex, encoding="utf-8")
    shutil.copyfile("paper/references.bib", source_dir / "references.bib")
    citation_check_failures = citation_failures(
        source_dir / "main.tex",
        source_dir / "references.bib",
        require_all_bib_used=not args.allow_missing,
    )
    if citation_check_failures:
        for failure in citation_check_failures:
            print(f"LaTeX citation failure: {failure}")
        raise SystemExit("Refusing to package arXiv source with inconsistent citations.")
    copied_file_provenance = [
        _file_provenance(
            kind="latex_main",
            source_path=Path("paper/latex/main.tex"),
            bundle_path=source_dir / "main.tex",
            bundle_root=source_dir,
            direct_copy=False,
            transform="rewrite_main_tex_for_arxiv",
        ),
        _file_provenance(
            kind="bibliography",
            source_path=Path("paper/references.bib"),
            bundle_path=source_dir / "references.bib",
            bundle_root=source_dir,
        ),
    ]

    copied_figures = []
    missing_figures = []
    invalid_figures = []
    for output_name, source_path in figure_sources.items():
        target_path = source_dir / "figures" / output_name
        if source_path.exists():
            if not _is_pdf(source_path):
                invalid_figures.append(str(source_path))
                continue
            shutil.copyfile(source_path, target_path)
            copied_figures.append(_bundle_manifest_path(source_dir, target_path))
            copied_file_provenance.append(
                _file_provenance(
                    kind="figure",
                    source_path=source_path,
                    bundle_path=target_path,
                    bundle_root=source_dir,
                )
            )
        else:
            missing_figures.append(str(source_path))
    copied_generated = []
    missing_generated = []
    invalid_generated = []
    for source_path in required_generated_dirs:
        if not source_path.exists():
            missing_generated.append(str(source_path))
            continue
        target_path = source_dir / "generated" / source_path.name
        copied_files = _copy_arxiv_support_tree(source_path, target_path)
        invalid_generated.extend(_invalid_arxiv_support_files(copied_files))
        copied_generated.append(_bundle_manifest_path(source_dir, target_path))
        copied_file_provenance.extend(
            _directory_provenance("generated", source_path, target_path, copied_files)
        )
    skipped_optional_generated = []
    for source_path in optional_generated_dirs:
        if not source_path.exists():
            skipped_optional_generated.append(str(source_path))
            continue
        target_path = source_dir / "generated" / source_path.name
        copied_files = _copy_arxiv_support_tree(source_path, target_path)
        invalid_generated.extend(_invalid_arxiv_support_files(copied_files))
        copied_generated.append(_bundle_manifest_path(source_dir, target_path))
        copied_file_provenance.extend(
            _directory_provenance("generated", source_path, target_path, copied_files)
        )
    copied_audit = []
    missing_audit = []
    invalid_audit = []
    for source_path in audit_dirs:
        if not source_path.exists():
            missing_audit.append(str(source_path))
            continue
        target_path = source_dir / "audit" / source_path.name
        copied_files = _copy_arxiv_support_tree(source_path, target_path)
        invalid_audit.extend(_invalid_arxiv_support_files(copied_files))
        copied_audit.append(_bundle_manifest_path(source_dir, target_path))
        copied_file_provenance.extend(
            _directory_provenance("audit", source_path, target_path, copied_files)
        )

    manifest = {
        "schema_version": 1,
        "main_tex_sha256": file_sha256(source_dir / "main.tex"),
        "references_sha256": file_sha256(source_dir / "references.bib"),
        "copied_figures": copied_figures,
        "missing_figures": missing_figures,
        "invalid_figures": invalid_figures,
        "copied_generated": copied_generated,
        "missing_generated": missing_generated,
        "invalid_generated": invalid_generated,
        "skipped_optional_generated": skipped_optional_generated,
        "copied_audit": copied_audit,
        "missing_audit": missing_audit,
        "invalid_audit": invalid_audit,
        "copied_file_provenance": copied_file_provenance,
        "allow_missing": args.allow_missing,
    }
    write_json(source_dir / "manifest.json", manifest)
    if invalid_figures:
        for path in invalid_figures:
            print(f"Invalid required arXiv figure PDF: {path}")
        raise SystemExit("Refusing to package arXiv bundle with invalid figure PDFs.")
    invalid_support = [*invalid_generated, *invalid_audit]
    if invalid_support:
        for failure in invalid_support:
            print(f"Invalid arXiv support file: {failure}")
        raise SystemExit(
            "Refusing to package arXiv bundle with placeholder generated support files."
        )
    missing_inputs = _missing_inputs(manifest)
    if missing_inputs and not args.allow_missing:
        for path in missing_inputs:
            print(f"Missing required arXiv input: {path}")
        raise SystemExit(
            "Refusing to package publication arXiv bundle with missing empirical inputs. "
            "Use --allow-missing only for draft/pre-results bundles."
        )

    args.archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.archive, "w:gz") as archive:
        for path in sorted(source_dir.rglob("*")):
            archive.add(path, arcname=path.relative_to(source_dir))
    print(f"Wrote {args.archive}")


def _rewrite_main_tex_for_arxiv(
    text: str,
    *,
    figure_sources: dict[str, Path] | None = None,
    strict_final: bool = False,
) -> str:
    text = text.replace(r"\bibliography{../references}", r"\bibliography{references}")
    text = text.replace("../generated/", "generated/")
    text = text.replace("../audit/", "audit/")
    source_items = list(FIGURE_SOURCES.items())
    if figure_sources is not None:
        source_items.extend(figure_sources.items())
    for output_name, source_path in source_items:
        text = text.replace(str(Path("../..") / source_path), f"figures/{output_name}")
    for output_name in build_figure_sources().keys():
        for prefix in ("../generated", "generated"):
            text = text.replace(
                f"{prefix}/active_primary/figures/{output_name}",
                f"figures/{output_name}",
            )
            text = text.replace(
                f"{prefix}/active_causal/figures/{output_name}",
                f"figures/{output_name}",
            )
    if strict_final:
        text = _rewrite_final_publication_fallbacks(text)
    return text


def _rewrite_final_publication_fallbacks(text: str) -> str:
    start_marker = r"\newcommand{\todoresult}"
    end_marker = r"\IfFileExists{generated/active_primary/result_macros.tex}"
    start = text.find(start_marker)
    end = text.find(end_marker)
    if start == -1 or end == -1 or end <= start:
        return text
    return text[:start] + STRICT_PUBLICATION_COMMAND_BLOCK + text[end:]


def _final_source_failures(text: str) -> list[str]:
    text_lower = text.lower()
    failures = [
        f"placeholder_text:{marker}"
        for marker in FINAL_SOURCE_TEXT_MARKERS
        if marker.lower() in text_lower
    ]
    failures.extend(forbidden_final_prose_failures(_strip_tex_comments(text)))
    return failures


def _rewrite_failures(text: str) -> list[str]:
    markers = [
        "../generated/",
        "../audit/",
        "../references",
        "../../results/",
        "/Users/aryan/Desktop/projects/llm-safety",
        "/home/aryang9/sandbox/llm-safety",
    ]
    return [marker for marker in markers if marker in text]


def _missing_inputs(manifest: dict) -> list[str]:
    return [
        *manifest.get("missing_figures", []),
        *manifest.get("missing_generated", []),
        *manifest.get("missing_audit", []),
    ]


def _is_pdf(path: Path) -> bool:
    try:
        content = path.read_bytes()
    except OSError:
        return False
    if not (
        content.startswith(b"%PDF-")
        and len(content) >= 32
        and b"%%EOF" in content[-2048:]
    ):
        return False
    return not _pdf_page_failure(path)


def _copy_arxiv_support_tree(source_dir: Path, bundle_dir: Path) -> list[Path]:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    copied_files = []
    for source_file in sorted(path for path in source_dir.rglob("*") if path.is_file()):
        if source_file.suffix not in ARXIV_SAFE_SUPPORT_SUFFIXES:
            continue
        bundle_file = bundle_dir / source_file.relative_to(source_dir)
        bundle_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_file, bundle_file)
        copied_files.append(source_file)
    return copied_files


def _invalid_arxiv_support_files(paths: list[Path]) -> list[str]:
    failures = []
    for path in paths:
        if path.suffix != ".tex":
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        rendered_text = _strip_tex_comments(text)
        rendered_text_lower = rendered_text.lower()
        for marker in PLACEHOLDER_TEXT_MARKERS:
            if marker.lower() in rendered_text_lower:
                failures.append(f"{path}:placeholder_text:{marker}")
                break
        failures.extend(
            f"{path}:{failure}" for failure in forbidden_final_prose_failures(rendered_text)
        )
        failures.extend(
            f"{path}:{failure}"
            for failure in _semantic_arxiv_tex_failures(path, rendered_text)
        )
    return failures


def _semantic_arxiv_tex_failures(path: Path, rendered_text: str) -> list[str]:
    failures = _semantic_tex_failures(str(path), path.name, rendered_text)
    if path.name != "causal_restoration_table.tex":
        return failures
    causal_generated_names = {
        DEFAULT_CAUSAL_GENERATED_DIR.name,
        DEFAULT_ACTIVE_CAUSAL_GENERATED_DIR.name,
    }
    if path.parent.name in causal_generated_names:
        return failures
    return [failure for failure in failures if not failure.startswith("missing causal control row")]


def _directory_provenance(
    kind: str, source_dir: Path, bundle_dir: Path, copied_files: list[Path] | None = None
) -> list[dict[str, object]]:
    rows = []
    source_files = copied_files
    if source_files is None:
        source_files = sorted(path for path in source_dir.rglob("*") if path.is_file())
    for source_file in source_files:
        bundle_file = bundle_dir / source_file.relative_to(source_dir)
        rows.append(
            _file_provenance(
                kind=kind,
                source_path=source_file,
                bundle_path=bundle_file,
                bundle_root=bundle_dir.parent.parent,
            )
        )
    return rows


def _file_provenance(
    *,
    kind: str,
    source_path: Path,
    bundle_path: Path,
    bundle_root: Path | None = None,
    direct_copy: bool = True,
    transform: str | None = None,
) -> dict[str, object]:
    manifest_bundle_path = (
        _bundle_manifest_path(bundle_root, bundle_path) if bundle_root is not None else str(bundle_path)
    )
    row: dict[str, object] = {
        "kind": kind,
        "source_path": str(source_path),
        "source_sha256": file_sha256(source_path),
        "source_bytes": source_path.stat().st_size if source_path.exists() else None,
        "bundle_path": manifest_bundle_path,
        "bundle_sha256": file_sha256(bundle_path),
        "bundle_bytes": bundle_path.stat().st_size if bundle_path.exists() else None,
        "direct_copy": direct_copy,
    }
    if transform is not None:
        row["transform"] = transform
    return row


def _bundle_manifest_path(source_dir: Path, path: Path) -> str:
    return path.resolve().relative_to(source_dir.resolve()).as_posix()


if __name__ == "__main__":
    main()
