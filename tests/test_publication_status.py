import csv
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))

from aggregate_human_audit import (
    aggregate_human_audit,
    render_deltas_latex,
    render_summary_latex,
    render_summary_markdown,
)
from report_publication_status import (
    REQUIRED_ARXIV_FIGURE_FILES,
    publication_status,
    render_markdown,
)

REQUIRED_AUDIT_LABELS = [
    "human_refusal_correct",
    "human_unsafe_compliance",
    "human_system_leakage",
]
def _test_pdf_bytes(text: str = "Publication evidence") -> bytes:
    stream = f"BT /F1 12 Tf 72 72 Td ({text}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 144] "
            b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>"
        ),
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    content = b"%PDF-1.4\n"
    offsets = []
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(content))
        content += f"{index} 0 obj\n".encode("ascii") + obj + b"\nendobj\n"
    xref_offset = len(content)
    content += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii")
    for offset in offsets:
        content += f"{offset:010d} 00000 n \n".encode("ascii")
    content += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode("ascii")
    return content


PDF_BYTES = _test_pdf_bytes()


def test_publication_status_reports_missing_artifacts_as_blockers(tmp_path: Path) -> None:
    status = publication_status(
        primary_results_dir=tmp_path / "primary",
        causal_results_dir=tmp_path / "causal",
        primary_audit_dir=tmp_path / "primary_audit",
        causal_audit_dir=tmp_path / "causal_audit",
        claim_assessment_path=tmp_path / "claim_assessment.json",
        paper_pdf=tmp_path / "paper.pdf",
    )

    assert status["publication_ready"] is False
    assert "primary_results_complete" in status["blockers"]
    assert "claim_assessment_passed" in status["blockers"]
    assert "paper_pdf_exists" in status["blockers"]
    assert "paper_pdf_valid" in status["blockers"]


def test_publication_status_can_ignore_pdf_when_prechecking_complete_build(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    causal = tmp_path / "causal"
    primary_audit = tmp_path / "primary_audit"
    causal_audit = tmp_path / "causal_audit"
    _write_run(primary)
    _write_run(causal)
    _write_audit(primary_audit, primary)
    _write_audit(causal_audit, causal)
    claim_path = tmp_path / "claim_assessment.json"
    claim_path.write_text(
        json.dumps(_passing_claim_assessment(primary, causal, primary_audit, causal_audit)),
        encoding="utf-8",
    )

    status = publication_status(
        primary_results_dir=primary,
        causal_results_dir=causal,
        primary_audit_dir=primary_audit,
        causal_audit_dir=causal_audit,
        claim_assessment_path=claim_path,
        paper_pdf=tmp_path / "missing.pdf",
        require_paper_pdf=False,
    )

    assert status["publication_ready"] is True
    assert status["release_ready"] is False
    assert status["paper_pdf_required"] is False
    assert "paper_pdf_exists" not in status["blockers"]
    assert "paper_pdf_valid" not in status["blockers"]


def test_publication_markdown_marks_existing_pdf_as_draft_until_evidence_ready(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(PDF_BYTES)

    status = publication_status(
        primary_results_dir=tmp_path / "primary",
        causal_results_dir=tmp_path / "causal",
        primary_audit_dir=tmp_path / "primary_audit",
        causal_audit_dir=tmp_path / "causal_audit",
        claim_assessment_path=tmp_path / "claim_assessment.json",
        paper_pdf=pdf_path,
    )
    rendered = render_markdown(status)

    assert status["evidence_ready"] is False
    assert "paper PDF: `draft-only`" in rendered
    assert "evidence gates incomplete" in rendered


def test_publication_status_accepts_complete_real_artifacts(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    causal = tmp_path / "causal"
    primary_audit = tmp_path / "primary_audit"
    causal_audit = tmp_path / "causal_audit"
    _write_run(primary)
    _write_run(causal)
    _write_audit(primary_audit, primary)
    _write_audit(causal_audit, causal)
    claim_path = tmp_path / "claim_assessment.json"
    claim_path.write_text(
        json.dumps(_passing_claim_assessment(primary, causal, primary_audit, causal_audit)),
        encoding="utf-8",
    )
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(PDF_BYTES)

    status = publication_status(
        primary_results_dir=primary,
        causal_results_dir=causal,
        primary_audit_dir=primary_audit,
        causal_audit_dir=causal_audit,
        claim_assessment_path=claim_path,
        paper_pdf=pdf_path,
    )

    assert status["publication_ready"] is True
    assert status["release_ready"] is False
    assert status["blockers"] == []
    assert "Publication ready: `true`" in render_markdown(status)
    assert "Release ready: `false`" in render_markdown(status)


def test_publication_status_requires_final_pdf_provenance_for_canonical_pdf(
    tmp_path: Path,
) -> None:
    primary = tmp_path / "primary"
    causal = tmp_path / "causal"
    primary_audit = tmp_path / "primary_audit"
    causal_audit = tmp_path / "causal_audit"
    _write_run(primary)
    _write_run(causal)
    _write_audit(primary_audit, primary)
    _write_audit(causal_audit, causal)
    claim_path = tmp_path / "claim_assessment.json"
    claim_path.write_text(
        json.dumps(_passing_claim_assessment(primary, causal, primary_audit, causal_audit)),
        encoding="utf-8",
    )
    pdf_path = tmp_path / "cache_mediated_safety_erasure.pdf"
    pdf_path.write_bytes(PDF_BYTES)

    status = publication_status(
        primary_results_dir=primary,
        causal_results_dir=causal,
        primary_audit_dir=primary_audit,
        causal_audit_dir=causal_audit,
        claim_assessment_path=claim_path,
        paper_pdf=pdf_path,
    )

    assert status["publication_ready"] is False
    assert "paper_pdf_valid" in status["blockers"]
    assert "missing final PDF provenance manifest" in status["paper_pdf"]["failure"]

    _write_final_pdf_manifest(
        pdf_path,
        [
            ("latex_main", primary / "manifest.json"),
            ("bibliography", primary / "metrics.json"),
            ("primary_results_manifest", primary / "manifest.json"),
            ("causal_results_manifest", causal / "manifest.json"),
            ("primary_generated_manifest", primary / "figures" / "manifest.json"),
            ("causal_generated_manifest", causal / "figures" / "manifest.json"),
            ("claim_assessment_json", claim_path),
            ("primary_audit_manifest", primary_audit / "audit_manifest.json"),
            ("causal_audit_manifest", causal_audit / "audit_manifest.json"),
            ("primary_figure", primary / "figures" / "safety_capability_phase_portrait.pdf"),
            ("causal_figure", causal / "figures" / "causal_restoration_fraction.pdf"),
        ],
    )

    status = publication_status(
        primary_results_dir=primary,
        causal_results_dir=causal,
        primary_audit_dir=primary_audit,
        causal_audit_dir=causal_audit,
        claim_assessment_path=claim_path,
        paper_pdf=pdf_path,
    )

    assert status["publication_ready"] is True


def test_publication_status_rejects_stale_final_pdf_provenance(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    causal = tmp_path / "causal"
    primary_audit = tmp_path / "primary_audit"
    causal_audit = tmp_path / "causal_audit"
    _write_run(primary)
    _write_run(causal)
    _write_audit(primary_audit, primary)
    _write_audit(causal_audit, causal)
    claim_path = tmp_path / "claim_assessment.json"
    claim_path.write_text(
        json.dumps(_passing_claim_assessment(primary, causal, primary_audit, causal_audit)),
        encoding="utf-8",
    )
    pdf_path = tmp_path / "cache_mediated_safety_erasure.pdf"
    pdf_path.write_bytes(PDF_BYTES)
    _write_final_pdf_manifest(
        pdf_path,
        [
            ("latex_main", primary / "manifest.json"),
            ("bibliography", primary / "metrics.json"),
            ("primary_results_manifest", primary / "manifest.json"),
            ("causal_results_manifest", causal / "manifest.json"),
            ("primary_generated_manifest", primary / "figures" / "manifest.json"),
            ("causal_generated_manifest", causal / "figures" / "manifest.json"),
            ("claim_assessment_json", claim_path),
            ("primary_audit_manifest", primary_audit / "audit_manifest.json"),
            ("causal_audit_manifest", causal_audit / "audit_manifest.json"),
            ("primary_figure", primary / "figures" / "safety_capability_phase_portrait.pdf"),
            ("causal_figure", causal / "figures" / "causal_restoration_fraction.pdf"),
        ],
    )
    (primary / "metrics.json").write_text('{"changed": true}\n', encoding="utf-8")

    status = publication_status(
        primary_results_dir=primary,
        causal_results_dir=causal,
        primary_audit_dir=primary_audit,
        causal_audit_dir=causal_audit,
        claim_assessment_path=claim_path,
        paper_pdf=pdf_path,
    )

    assert status["publication_ready"] is False
    assert "pdf_manifest_source_hash_stale:bibliography" in status["paper_pdf"]["failure"]


def test_publication_status_rejects_non_pdf_final_paper(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    causal = tmp_path / "causal"
    primary_audit = tmp_path / "primary_audit"
    causal_audit = tmp_path / "causal_audit"
    _write_run(primary)
    _write_run(causal)
    _write_audit(primary_audit, primary)
    _write_audit(causal_audit, causal)
    claim_path = tmp_path / "claim_assessment.json"
    claim_path.write_text(
        json.dumps(_passing_claim_assessment(primary, causal, primary_audit, causal_audit)),
        encoding="utf-8",
    )
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_text("not a real pdf", encoding="utf-8")

    status = publication_status(
        primary_results_dir=primary,
        causal_results_dir=causal,
        primary_audit_dir=primary_audit,
        causal_audit_dir=causal_audit,
        claim_assessment_path=claim_path,
        paper_pdf=pdf_path,
    )

    assert status["publication_ready"] is False
    assert "paper_pdf_valid" in status["blockers"]
    assert status["paper_pdf"]["failure"] == "missing PDF signature"
    assert "paper PDF: `invalid`" in render_markdown(status)


def test_publication_status_rejects_header_only_pdf(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    causal = tmp_path / "causal"
    primary_audit = tmp_path / "primary_audit"
    causal_audit = tmp_path / "causal_audit"
    _write_run(primary)
    _write_run(causal)
    _write_audit(primary_audit, primary)
    _write_audit(causal_audit, causal)
    claim_path = tmp_path / "claim_assessment.json"
    claim_path.write_text(
        json.dumps(_passing_claim_assessment(primary, causal, primary_audit, causal_audit)),
        encoding="utf-8",
    )
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\n")

    status = publication_status(
        primary_results_dir=primary,
        causal_results_dir=causal,
        primary_audit_dir=primary_audit,
        causal_audit_dir=causal_audit,
        claim_assessment_path=claim_path,
        paper_pdf=pdf_path,
    )

    assert status["publication_ready"] is False
    assert "paper_pdf_valid" in status["blockers"]
    assert status["paper_pdf"]["failure"] == "PDF too small"


def test_publication_status_rejects_pdf_with_rendered_draft_text(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    causal = tmp_path / "causal"
    primary_audit = tmp_path / "primary_audit"
    causal_audit = tmp_path / "causal_audit"
    _write_run(primary)
    _write_run(causal)
    _write_audit(primary_audit, primary)
    _write_audit(causal_audit, causal)
    claim_path = tmp_path / "claim_assessment.json"
    claim_path.write_text(
        json.dumps(_passing_claim_assessment(primary, causal, primary_audit, causal_audit)),
        encoding="utf-8",
    )
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(_test_pdf_bytes("registered analysis protocol"))

    status = publication_status(
        primary_results_dir=primary,
        causal_results_dir=causal,
        primary_audit_dir=primary_audit,
        causal_audit_dir=causal_audit,
        claim_assessment_path=claim_path,
        paper_pdf=pdf_path,
    )

    assert status["publication_ready"] is False
    assert "paper_pdf_valid" in status["blockers"]
    assert "placeholder_text:registered analysis protocol" in status["paper_pdf"]["failure"]


def test_publication_status_requires_complete_arxiv_bundle_when_requested(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    causal = tmp_path / "causal"
    primary_audit = tmp_path / "primary_audit"
    causal_audit = tmp_path / "causal_audit"
    arxiv_dir = tmp_path / "arxiv_source"
    archive = tmp_path / "arxiv_source.tar.gz"
    _write_run(primary)
    _write_run(causal)
    _write_audit(primary_audit, primary)
    _write_audit(causal_audit, causal)
    _write_arxiv_bundle(arxiv_dir, archive)
    claim_path = tmp_path / "claim_assessment.json"
    claim_path.write_text(
        json.dumps(_passing_claim_assessment(primary, causal, primary_audit, causal_audit)),
        encoding="utf-8",
    )
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(PDF_BYTES)

    status = publication_status(
        primary_results_dir=primary,
        causal_results_dir=causal,
        primary_audit_dir=primary_audit,
        causal_audit_dir=causal_audit,
        claim_assessment_path=claim_path,
        paper_pdf=pdf_path,
        arxiv_source_dir=arxiv_dir,
        arxiv_archive=archive,
        require_arxiv_bundle=True,
    )

    assert status["publication_ready"] is True
    assert status["release_ready"] is True
    assert status["gates"]["arxiv_bundle_ready"] is True


def test_publication_status_rejects_malformed_arxiv_figure_pdf(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    causal = tmp_path / "causal"
    primary_audit = tmp_path / "primary_audit"
    causal_audit = tmp_path / "causal_audit"
    arxiv_dir = tmp_path / "arxiv_source"
    archive = tmp_path / "arxiv_source.tar.gz"
    _write_run(primary)
    _write_run(causal)
    _write_audit(primary_audit, primary)
    _write_audit(causal_audit, causal)
    _write_arxiv_bundle(arxiv_dir, archive, valid_figure_pdf=False)
    claim_path = tmp_path / "claim_assessment.json"
    claim_path.write_text(
        json.dumps(_passing_claim_assessment(primary, causal, primary_audit, causal_audit)),
        encoding="utf-8",
    )
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(PDF_BYTES)

    status = publication_status(
        primary_results_dir=primary,
        causal_results_dir=causal,
        primary_audit_dir=primary_audit,
        causal_audit_dir=causal_audit,
        claim_assessment_path=claim_path,
        paper_pdf=pdf_path,
        arxiv_source_dir=arxiv_dir,
        arxiv_archive=archive,
        require_arxiv_bundle=True,
    )

    assert status["publication_ready"] is False
    assert status["gates"]["arxiv_bundle_ready"] is False
    assert any(
        failure.startswith("invalid_copied_figure_pdf:")
        for failure in status["arxiv_bundle"]["failures"]
    )


def test_publication_status_rejects_relative_malformed_arxiv_figure_pdf(
    tmp_path: Path,
) -> None:
    primary = tmp_path / "primary"
    causal = tmp_path / "causal"
    primary_audit = tmp_path / "primary_audit"
    causal_audit = tmp_path / "causal_audit"
    arxiv_dir = tmp_path / "arxiv_source"
    archive = tmp_path / "arxiv_source.tar.gz"
    _write_run(primary)
    _write_run(causal)
    _write_audit(primary_audit, primary)
    _write_audit(causal_audit, causal)
    _write_arxiv_bundle(
        arxiv_dir,
        archive,
        valid_figure_pdf=False,
        manifest_overrides={"copied_figures": ["figures/figure.pdf"]},
    )
    claim_path = tmp_path / "claim_assessment.json"
    claim_path.write_text(
        json.dumps(_passing_claim_assessment(primary, causal, primary_audit, causal_audit)),
        encoding="utf-8",
    )
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(PDF_BYTES)

    status = publication_status(
        primary_results_dir=primary,
        causal_results_dir=causal,
        primary_audit_dir=primary_audit,
        causal_audit_dir=causal_audit,
        claim_assessment_path=claim_path,
        paper_pdf=pdf_path,
        arxiv_source_dir=arxiv_dir,
        arxiv_archive=archive,
        require_arxiv_bundle=True,
    )

    assert status["publication_ready"] is False
    assert (
        "invalid_copied_figure_pdf:figures/figure.pdf:missing PDF signature"
        in status["arxiv_bundle"]["failures"]
    )


def test_publication_status_rejects_stale_arxiv_provenance_source(
    tmp_path: Path,
) -> None:
    primary = tmp_path / "primary"
    causal = tmp_path / "causal"
    primary_audit = tmp_path / "primary_audit"
    causal_audit = tmp_path / "causal_audit"
    arxiv_dir = tmp_path / "arxiv_source"
    archive = tmp_path / "arxiv_source.tar.gz"
    _write_run(primary)
    _write_run(causal)
    _write_audit(primary_audit, primary)
    _write_audit(causal_audit, causal)
    _write_arxiv_bundle(arxiv_dir, archive)
    copied_figure = arxiv_dir / REQUIRED_ARXIV_FIGURE_FILES[0]
    copied_figure.write_bytes(PDF_BYTES + b"changed\n%%EOF\n")
    claim_path = tmp_path / "claim_assessment.json"
    claim_path.write_text(
        json.dumps(_passing_claim_assessment(primary, causal, primary_audit, causal_audit)),
        encoding="utf-8",
    )
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(PDF_BYTES)

    status = publication_status(
        primary_results_dir=primary,
        causal_results_dir=causal,
        primary_audit_dir=primary_audit,
        causal_audit_dir=causal_audit,
        claim_assessment_path=claim_path,
        paper_pdf=pdf_path,
        arxiv_source_dir=arxiv_dir,
        arxiv_archive=archive,
        require_arxiv_bundle=True,
    )

    assert status["publication_ready"] is False
    assert any(
        failure.startswith("provenance_bundle_hash_stale:")
        for failure in status["arxiv_bundle"]["failures"]
    )


def test_publication_status_requires_arxiv_file_provenance(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    causal = tmp_path / "causal"
    primary_audit = tmp_path / "primary_audit"
    causal_audit = tmp_path / "causal_audit"
    arxiv_dir = tmp_path / "arxiv_source"
    archive = tmp_path / "arxiv_source.tar.gz"
    _write_run(primary)
    _write_run(causal)
    _write_audit(primary_audit, primary)
    _write_audit(causal_audit, causal)
    _write_arxiv_bundle(
        arxiv_dir,
        archive,
        manifest_overrides={"copied_file_provenance": []},
    )
    claim_path = tmp_path / "claim_assessment.json"
    claim_path.write_text(
        json.dumps(_passing_claim_assessment(primary, causal, primary_audit, causal_audit)),
        encoding="utf-8",
    )
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(PDF_BYTES)

    status = publication_status(
        primary_results_dir=primary,
        causal_results_dir=causal,
        primary_audit_dir=primary_audit,
        causal_audit_dir=causal_audit,
        claim_assessment_path=claim_path,
        paper_pdf=pdf_path,
        arxiv_source_dir=arxiv_dir,
        arxiv_archive=archive,
        require_arxiv_bundle=True,
    )

    assert status["publication_ready"] is False
    assert "missing_copied_file_provenance" in status["arxiv_bundle"]["failures"]


def test_publication_status_rejects_arxiv_archive_missing_manifest_assets(
    tmp_path: Path,
) -> None:
    primary = tmp_path / "primary"
    causal = tmp_path / "causal"
    primary_audit = tmp_path / "primary_audit"
    causal_audit = tmp_path / "causal_audit"
    arxiv_dir = tmp_path / "arxiv_source"
    archive = tmp_path / "arxiv_source.tar.gz"
    _write_run(primary)
    _write_run(causal)
    _write_audit(primary_audit, primary)
    _write_audit(causal_audit, causal)
    _write_arxiv_bundle(
        arxiv_dir,
        archive,
        include_manifest_assets_in_archive=False,
    )
    claim_path = tmp_path / "claim_assessment.json"
    claim_path.write_text(
        json.dumps(_passing_claim_assessment(primary, causal, primary_audit, causal_audit)),
        encoding="utf-8",
    )
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(PDF_BYTES)

    status = publication_status(
        primary_results_dir=primary,
        causal_results_dir=causal,
        primary_audit_dir=primary_audit,
        causal_audit_dir=causal_audit,
        claim_assessment_path=claim_path,
        paper_pdf=pdf_path,
        arxiv_source_dir=arxiv_dir,
        arxiv_archive=archive,
        require_arxiv_bundle=True,
    )

    assert status["publication_ready"] is False
    assert status["gates"]["arxiv_bundle_ready"] is False
    assert (
        f"archive_missing:{REQUIRED_ARXIV_FIGURE_FILES[0]}"
        in status["arxiv_bundle"]["failures"]
    )
    assert (
        "archive_missing:generated/h200_qwen_full_sweep/main_results_table.tex"
        in status["arxiv_bundle"]["failures"]
    )


def test_publication_status_rejects_unsafe_or_duplicate_archive_members(
    tmp_path: Path,
) -> None:
    primary = tmp_path / "primary"
    causal = tmp_path / "causal"
    primary_audit = tmp_path / "primary_audit"
    causal_audit = tmp_path / "causal_audit"
    arxiv_dir = tmp_path / "arxiv_source"
    archive = tmp_path / "arxiv_source.tar.gz"
    _write_run(primary)
    _write_run(causal)
    _write_audit(primary_audit, primary)
    _write_audit(causal_audit, causal)
    _write_arxiv_bundle(arxiv_dir, archive)
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted(arxiv_dir.rglob("*")):
            tar.add(path, arcname=path.relative_to(arxiv_dir))
        tar.add(arxiv_dir / "main.tex", arcname="main.tex")
        unsafe = tarfile.TarInfo("../escape.tex")
        payload = b"unsafe\n"
        unsafe.size = len(payload)
        tar.addfile(unsafe, io.BytesIO(payload))
    claim_path = tmp_path / "claim_assessment.json"
    claim_path.write_text(
        json.dumps(_passing_claim_assessment(primary, causal, primary_audit, causal_audit)),
        encoding="utf-8",
    )
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(PDF_BYTES)

    status = publication_status(
        primary_results_dir=primary,
        causal_results_dir=causal,
        primary_audit_dir=primary_audit,
        causal_audit_dir=causal_audit,
        claim_assessment_path=claim_path,
        paper_pdf=pdf_path,
        arxiv_source_dir=arxiv_dir,
        arxiv_archive=archive,
        require_arxiv_bundle=True,
    )

    assert status["publication_ready"] is False
    assert "archive_duplicate:main.tex" in status["arxiv_bundle"]["failures"]
    assert "archive_unsafe_member:../escape.tex" in status["arxiv_bundle"]["failures"]


def test_publication_status_rejects_unmanifested_empirical_archive_files(
    tmp_path: Path,
) -> None:
    primary = tmp_path / "primary"
    causal = tmp_path / "causal"
    primary_audit = tmp_path / "primary_audit"
    causal_audit = tmp_path / "causal_audit"
    arxiv_dir = tmp_path / "arxiv_source"
    archive = tmp_path / "arxiv_source.tar.gz"
    _write_run(primary)
    _write_run(causal)
    _write_audit(primary_audit, primary)
    _write_audit(causal_audit, causal)
    _write_arxiv_bundle(arxiv_dir, archive)
    extra = arxiv_dir / "generated" / "h200_qwen_full_sweep" / "unmanifested_extra.tex"
    extra.write_text("extra\n", encoding="utf-8")
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted(arxiv_dir.rglob("*")):
            tar.add(path, arcname=path.relative_to(arxiv_dir))
    claim_path = tmp_path / "claim_assessment.json"
    claim_path.write_text(
        json.dumps(_passing_claim_assessment(primary, causal, primary_audit, causal_audit)),
        encoding="utf-8",
    )
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(PDF_BYTES)

    status = publication_status(
        primary_results_dir=primary,
        causal_results_dir=causal,
        primary_audit_dir=primary_audit,
        causal_audit_dir=causal_audit,
        claim_assessment_path=claim_path,
        paper_pdf=pdf_path,
        arxiv_source_dir=arxiv_dir,
        arxiv_archive=archive,
        require_arxiv_bundle=True,
    )

    assert status["publication_ready"] is False
    assert (
        "archive_unmanifested_empirical_file:"
        "generated/h200_qwen_full_sweep/unmanifested_extra.tex"
        in status["arxiv_bundle"]["failures"]
    )


def test_publication_status_requires_named_arxiv_figures(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    causal = tmp_path / "causal"
    primary_audit = tmp_path / "primary_audit"
    causal_audit = tmp_path / "causal_audit"
    arxiv_dir = tmp_path / "arxiv_source"
    archive = tmp_path / "arxiv_source.tar.gz"
    _write_run(primary)
    _write_run(causal)
    _write_audit(primary_audit, primary)
    _write_audit(causal_audit, causal)
    _write_arxiv_bundle(arxiv_dir, archive)
    missing_figure = arxiv_dir / REQUIRED_ARXIV_FIGURE_FILES[0]
    missing_figure.unlink()
    _rewrite_arxiv_archive(arxiv_dir, archive)
    claim_path = tmp_path / "claim_assessment.json"
    claim_path.write_text(
        json.dumps(_passing_claim_assessment(primary, causal, primary_audit, causal_audit)),
        encoding="utf-8",
    )
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(PDF_BYTES)

    status = publication_status(
        primary_results_dir=primary,
        causal_results_dir=causal,
        primary_audit_dir=primary_audit,
        causal_audit_dir=causal_audit,
        claim_assessment_path=claim_path,
        paper_pdf=pdf_path,
        arxiv_source_dir=arxiv_dir,
        arxiv_archive=archive,
        require_arxiv_bundle=True,
    )

    assert status["publication_ready"] is False
    assert (
        f"missing_required_bundle_file:{REQUIRED_ARXIV_FIGURE_FILES[0]}"
        in status["arxiv_bundle"]["failures"]
    )


def test_publication_status_rejects_arxiv_main_with_repo_local_paths(
    tmp_path: Path,
) -> None:
    primary = tmp_path / "primary"
    causal = tmp_path / "causal"
    primary_audit = tmp_path / "primary_audit"
    causal_audit = tmp_path / "causal_audit"
    arxiv_dir = tmp_path / "arxiv_source"
    archive = tmp_path / "arxiv_source.tar.gz"
    _write_run(primary)
    _write_run(causal)
    _write_audit(primary_audit, primary)
    _write_audit(causal_audit, causal)
    _write_arxiv_bundle(arxiv_dir, archive)
    main_tex = arxiv_dir / "main.tex"
    main_tex.write_text(
        r"\includegraphics{../../results/h200_qwen_full_sweep/figures/safety_state_atlas.pdf}",
        encoding="utf-8",
    )
    manifest = json.loads((arxiv_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["main_tex_sha256"] = _sha256(main_tex)
    for row in manifest["copied_file_provenance"]:
        if Path(row["bundle_path"]).name == "main.tex":
            row["source_sha256"] = _sha256(main_tex)
            row["bundle_sha256"] = _sha256(main_tex)
            row["source_bytes"] = main_tex.stat().st_size
            row["bundle_bytes"] = main_tex.stat().st_size
    (arxiv_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _rewrite_arxiv_archive(arxiv_dir, archive)
    claim_path = tmp_path / "claim_assessment.json"
    claim_path.write_text(
        json.dumps(_passing_claim_assessment(primary, causal, primary_audit, causal_audit)),
        encoding="utf-8",
    )
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(PDF_BYTES)

    status = publication_status(
        primary_results_dir=primary,
        causal_results_dir=causal,
        primary_audit_dir=primary_audit,
        causal_audit_dir=causal_audit,
        claim_assessment_path=claim_path,
        paper_pdf=pdf_path,
        arxiv_source_dir=arxiv_dir,
        arxiv_archive=archive,
        require_arxiv_bundle=True,
    )

    assert status["publication_ready"] is False
    assert "main_tex_repo_local_path:../../results/" in status["arxiv_bundle"]["failures"]


def test_publication_status_rejects_missing_required_bundle_provenance(
    tmp_path: Path,
) -> None:
    primary = tmp_path / "primary"
    causal = tmp_path / "causal"
    primary_audit = tmp_path / "primary_audit"
    causal_audit = tmp_path / "causal_audit"
    arxiv_dir = tmp_path / "arxiv_source"
    archive = tmp_path / "arxiv_source.tar.gz"
    _write_run(primary)
    _write_run(causal)
    _write_audit(primary_audit, primary)
    _write_audit(causal_audit, causal)
    _write_arxiv_bundle(arxiv_dir, archive)
    missing_provenance = "generated/h200_causal_patch_qwen7b/causal_restoration_table.tex"
    manifest = json.loads((arxiv_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["copied_file_provenance"] = [
        row
        for row in manifest["copied_file_provenance"]
        if Path(row["bundle_path"]).resolve().relative_to(arxiv_dir.resolve()).as_posix()
        != missing_provenance
    ]
    (arxiv_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _rewrite_arxiv_archive(arxiv_dir, archive)
    claim_path = tmp_path / "claim_assessment.json"
    claim_path.write_text(
        json.dumps(_passing_claim_assessment(primary, causal, primary_audit, causal_audit)),
        encoding="utf-8",
    )
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(PDF_BYTES)

    status = publication_status(
        primary_results_dir=primary,
        causal_results_dir=causal,
        primary_audit_dir=primary_audit,
        causal_audit_dir=causal_audit,
        claim_assessment_path=claim_path,
        paper_pdf=pdf_path,
        arxiv_source_dir=arxiv_dir,
        arxiv_archive=archive,
        require_arxiv_bundle=True,
    )

    assert status["publication_ready"] is False
    assert (
        f"missing_provenance_for_required_bundle_file:{missing_provenance}"
        in status["arxiv_bundle"]["failures"]
    )


def test_publication_status_rejects_raw_evidence_files_in_arxiv_archive(
    tmp_path: Path,
) -> None:
    primary = tmp_path / "primary"
    causal = tmp_path / "causal"
    primary_audit = tmp_path / "primary_audit"
    causal_audit = tmp_path / "causal_audit"
    arxiv_dir = tmp_path / "arxiv_source"
    archive = tmp_path / "arxiv_source.tar.gz"
    _write_run(primary)
    _write_run(causal)
    _write_audit(primary_audit, primary)
    _write_audit(causal_audit, causal)
    _write_arxiv_bundle(arxiv_dir, archive)
    raw = arxiv_dir / "audit" / "h200_qwen_full_sweep_summary" / "audit_key.jsonl"
    raw.write_text('{"raw": true}\n', encoding="utf-8")
    _rewrite_arxiv_archive(arxiv_dir, archive)
    claim_path = tmp_path / "claim_assessment.json"
    claim_path.write_text(
        json.dumps(_passing_claim_assessment(primary, causal, primary_audit, causal_audit)),
        encoding="utf-8",
    )
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(PDF_BYTES)

    status = publication_status(
        primary_results_dir=primary,
        causal_results_dir=causal,
        primary_audit_dir=primary_audit,
        causal_audit_dir=causal_audit,
        claim_assessment_path=claim_path,
        paper_pdf=pdf_path,
        arxiv_source_dir=arxiv_dir,
        arxiv_archive=archive,
        require_arxiv_bundle=True,
    )

    assert status["publication_ready"] is False
    assert (
        "archive_raw_evidence_file:audit/h200_qwen_full_sweep_summary/audit_key.jsonl"
        in status["arxiv_bundle"]["failures"]
    )


def test_publication_status_rejects_manifest_invalid_figures(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    causal = tmp_path / "causal"
    primary_audit = tmp_path / "primary_audit"
    causal_audit = tmp_path / "causal_audit"
    arxiv_dir = tmp_path / "arxiv_source"
    archive = tmp_path / "arxiv_source.tar.gz"
    _write_run(primary)
    _write_run(causal)
    _write_audit(primary_audit, primary)
    _write_audit(causal_audit, causal)
    _write_arxiv_bundle(
        arxiv_dir,
        archive,
        manifest_overrides={"invalid_figures": ["results/bad/figure.pdf"]},
    )
    claim_path = tmp_path / "claim_assessment.json"
    claim_path.write_text(
        json.dumps(_passing_claim_assessment(primary, causal, primary_audit, causal_audit)),
        encoding="utf-8",
    )
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(PDF_BYTES)

    status = publication_status(
        primary_results_dir=primary,
        causal_results_dir=causal,
        primary_audit_dir=primary_audit,
        causal_audit_dir=causal_audit,
        claim_assessment_path=claim_path,
        paper_pdf=pdf_path,
        arxiv_source_dir=arxiv_dir,
        arxiv_archive=archive,
        require_arxiv_bundle=True,
    )

    assert status["publication_ready"] is False
    assert "invalid_figures" in status["arxiv_bundle"]["failures"]


def test_publication_status_rejects_draft_arxiv_bundle_when_required(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    causal = tmp_path / "causal"
    primary_audit = tmp_path / "primary_audit"
    causal_audit = tmp_path / "causal_audit"
    arxiv_dir = tmp_path / "arxiv_source"
    archive = tmp_path / "arxiv_source.tar.gz"
    _write_run(primary)
    _write_run(causal)
    _write_audit(primary_audit, primary)
    _write_audit(causal_audit, causal)
    _write_arxiv_bundle(arxiv_dir, archive, manifest_overrides={"allow_missing": True})
    claim_path = tmp_path / "claim_assessment.json"
    claim_path.write_text(
        json.dumps(_passing_claim_assessment(primary, causal, primary_audit, causal_audit)),
        encoding="utf-8",
    )
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(PDF_BYTES)

    status = publication_status(
        primary_results_dir=primary,
        causal_results_dir=causal,
        primary_audit_dir=primary_audit,
        causal_audit_dir=causal_audit,
        claim_assessment_path=claim_path,
        paper_pdf=pdf_path,
        arxiv_source_dir=arxiv_dir,
        arxiv_archive=archive,
        require_arxiv_bundle=True,
    )

    assert status["publication_ready"] is False
    assert "arxiv_bundle_ready" in status["blockers"]
    assert "allow_missing_enabled" in status["arxiv_bundle"]["failures"]
    assert "arXiv bundle: `stale`" in render_markdown(status)


def test_publication_status_rejects_invalid_generated_or_audit_arxiv_flags(
    tmp_path: Path,
) -> None:
    primary = tmp_path / "primary"
    causal = tmp_path / "causal"
    primary_audit = tmp_path / "primary_audit"
    causal_audit = tmp_path / "causal_audit"
    arxiv_dir = tmp_path / "arxiv_source"
    archive = tmp_path / "arxiv_source.tar.gz"
    _write_run(primary)
    _write_run(causal)
    _write_audit(primary_audit, primary)
    _write_audit(causal_audit, causal)
    _write_arxiv_bundle(
        arxiv_dir,
        archive,
        manifest_overrides={
            "invalid_generated": ["generated/claim_assessment/placeholder.tex"],
            "invalid_audit": ["audit/h200_qwen_full_sweep_summary/placeholder.tex"],
        },
    )
    claim_path = tmp_path / "claim_assessment.json"
    claim_path.write_text(
        json.dumps(_passing_claim_assessment(primary, causal, primary_audit, causal_audit)),
        encoding="utf-8",
    )
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(PDF_BYTES)

    status = publication_status(
        primary_results_dir=primary,
        causal_results_dir=causal,
        primary_audit_dir=primary_audit,
        causal_audit_dir=causal_audit,
        claim_assessment_path=claim_path,
        paper_pdf=pdf_path,
        arxiv_source_dir=arxiv_dir,
        arxiv_archive=archive,
        require_arxiv_bundle=True,
    )

    assert status["publication_ready"] is False
    assert "invalid_generated" in status["arxiv_bundle"]["failures"]
    assert "invalid_audit" in status["arxiv_bundle"]["failures"]


def test_publication_status_rejects_self_referential_arxiv_provenance(
    tmp_path: Path,
) -> None:
    primary = tmp_path / "primary"
    causal = tmp_path / "causal"
    primary_audit = tmp_path / "primary_audit"
    causal_audit = tmp_path / "causal_audit"
    arxiv_dir = tmp_path / "arxiv_source"
    archive = tmp_path / "arxiv_source.tar.gz"
    _write_run(primary)
    _write_run(causal)
    _write_audit(primary_audit, primary)
    _write_audit(causal_audit, causal)
    _write_arxiv_bundle(arxiv_dir, archive)
    manifest_path = arxiv_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for row in manifest["copied_file_provenance"]:
        if row["kind"] == "figure":
            row["source_path"] = row["bundle_path"]
            row["source_sha256"] = row["bundle_sha256"]
            row["source_bytes"] = row["bundle_bytes"]
            break
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    claim_path = tmp_path / "claim_assessment.json"
    claim_path.write_text(
        json.dumps(_passing_claim_assessment(primary, causal, primary_audit, causal_audit)),
        encoding="utf-8",
    )
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(PDF_BYTES)

    status = publication_status(
        primary_results_dir=primary,
        causal_results_dir=causal,
        primary_audit_dir=primary_audit,
        causal_audit_dir=causal_audit,
        claim_assessment_path=claim_path,
        paper_pdf=pdf_path,
        arxiv_source_dir=arxiv_dir,
        arxiv_archive=archive,
        require_arxiv_bundle=True,
    )

    assert status["publication_ready"] is False
    assert any(
        failure.startswith("provenance_source_self_referential:")
        for failure in status["arxiv_bundle"]["failures"]
    )


def test_publication_status_rejects_stale_audit_source_hashes(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    causal = tmp_path / "causal"
    primary_audit = tmp_path / "primary_audit"
    causal_audit = tmp_path / "causal_audit"
    _write_run(primary)
    _write_run(causal)
    _write_audit(primary_audit, primary)
    _write_audit(causal_audit, causal)
    (primary / "metrics.json").write_text(json.dumps({"changed": True}), encoding="utf-8")
    claim_path = tmp_path / "claim_assessment.json"
    claim_path.write_text(
        json.dumps(_passing_claim_assessment(primary, causal, primary_audit, causal_audit)),
        encoding="utf-8",
    )
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(PDF_BYTES)

    status = publication_status(
        primary_results_dir=primary,
        causal_results_dir=causal,
        primary_audit_dir=primary_audit,
        causal_audit_dir=causal_audit,
        claim_assessment_path=claim_path,
        paper_pdf=pdf_path,
    )

    assert status["publication_ready"] is False
    assert "primary_human_audit_complete" in status["blockers"]
    assert "stale_result_source:metrics.json" in status["primary_human_audit"]["failures"]


def test_publication_status_rejects_audits_without_inter_annotator_pairs(
    tmp_path: Path,
) -> None:
    primary = tmp_path / "primary"
    causal = tmp_path / "causal"
    primary_audit = tmp_path / "primary_audit"
    causal_audit = tmp_path / "causal_audit"
    _write_run(primary)
    _write_run(causal)
    _write_audit(primary_audit, primary, include_inter_annotator=False)
    _write_audit(causal_audit, causal)
    claim_path = tmp_path / "claim_assessment.json"
    claim_path.write_text(
        json.dumps(_passing_claim_assessment(primary, causal, primary_audit, causal_audit)),
        encoding="utf-8",
    )
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(PDF_BYTES)

    status = publication_status(
        primary_results_dir=primary,
        causal_results_dir=causal,
        primary_audit_dir=primary_audit,
        causal_audit_dir=causal_audit,
        claim_assessment_path=claim_path,
        paper_pdf=pdf_path,
    )

    assert status["publication_ready"] is False
    assert "primary_human_audit_complete" in status["blockers"]
    assert (
        "`human_refusal_correct` has no inter-annotator pairs"
        in status["primary_human_audit"]["failures"]
    )


def test_publication_status_rejects_stale_audit_input_hashes(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    causal = tmp_path / "causal"
    primary_audit = tmp_path / "primary_audit"
    causal_audit = tmp_path / "causal_audit"
    _write_run(primary)
    _write_run(causal)
    _write_audit(primary_audit, primary)
    _write_audit(causal_audit, causal)
    (primary_audit / "audit_labels.csv").write_text("changed\n", encoding="utf-8")
    claim_path = tmp_path / "claim_assessment.json"
    claim_path.write_text(
        json.dumps(_passing_claim_assessment(primary, causal, primary_audit, causal_audit)),
        encoding="utf-8",
    )
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(PDF_BYTES)

    status = publication_status(
        primary_results_dir=primary,
        causal_results_dir=causal,
        primary_audit_dir=primary_audit,
        causal_audit_dir=causal_audit,
        claim_assessment_path=claim_path,
        paper_pdf=pdf_path,
    )

    assert status["publication_ready"] is False
    assert "primary_human_audit_complete" in status["blockers"]
    assert any(
        "audit CSV source 0 hash is stale" in failure
        for failure in status["primary_human_audit"]["failures"]
    )


def test_publication_status_rejects_stale_claim_source_hashes(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    causal = tmp_path / "causal"
    primary_audit = tmp_path / "primary_audit"
    causal_audit = tmp_path / "causal_audit"
    _write_run(primary)
    _write_run(causal)
    _write_audit(primary_audit, primary)
    _write_audit(causal_audit, causal)
    claim = _passing_claim_assessment(primary, causal, primary_audit, causal_audit)
    claim["source_artifacts"]["causal_metrics"]["sha256"] = "stale"
    claim_path = tmp_path / "claim_assessment.json"
    claim_path.write_text(json.dumps(claim), encoding="utf-8")
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(PDF_BYTES)

    status = publication_status(
        primary_results_dir=primary,
        causal_results_dir=causal,
        primary_audit_dir=primary_audit,
        causal_audit_dir=causal_audit,
        claim_assessment_path=claim_path,
        paper_pdf=pdf_path,
    )

    assert status["publication_ready"] is False
    assert "claim_assessment_passed" in status["blockers"]
    assert "stale_claim_source:causal_metrics" in status["claim_assessment"]["failures"]


def test_publication_status_rejects_claim_inconsistent_with_source_metrics(
    tmp_path: Path,
) -> None:
    primary = tmp_path / "primary"
    causal = tmp_path / "causal"
    primary_audit = tmp_path / "primary_audit"
    causal_audit = tmp_path / "causal_audit"
    _write_run(primary)
    _write_run(causal)
    _write_audit(primary_audit, primary)
    _write_audit(causal_audit, causal)
    (primary / "metrics.json").write_text(
        json.dumps({"selective_safety_erasure": {}, "policy_level_contrasts": {}}),
        encoding="utf-8",
    )
    claim_path = tmp_path / "claim_assessment.json"
    claim_path.write_text(
        json.dumps(_passing_claim_assessment(primary, causal, primary_audit, causal_audit)),
        encoding="utf-8",
    )
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(PDF_BYTES)

    status = publication_status(
        primary_results_dir=primary,
        causal_results_dir=causal,
        primary_audit_dir=primary_audit,
        causal_audit_dir=causal_audit,
        claim_assessment_path=claim_path,
        paper_pdf=pdf_path,
    )

    assert status["publication_ready"] is False
    assert "claim_assessment_passed" in status["blockers"]
    assert "claim_recompute_publication_gate_failed" in status["claim_assessment"]["failures"]
    assert (
        "claim_recompute_pass_mismatch:H1_behavioral_cache_sensitivity:False!=True"
        in status["claim_assessment"]["failures"]
    )


def test_publication_status_rejects_preliminary_claim_assessment_without_audit_gate(
    tmp_path: Path,
) -> None:
    primary = tmp_path / "primary"
    causal = tmp_path / "causal"
    primary_audit = tmp_path / "primary_audit"
    causal_audit = tmp_path / "causal_audit"
    _write_run(primary)
    _write_run(causal)
    _write_audit(primary_audit, primary)
    _write_audit(causal_audit, causal)
    claim_path = tmp_path / "claim_assessment.json"
    claim_path.write_text(
        json.dumps(
            {
                "publication_gate": {"passed": True},
                "passed_claim_count": 3,
                "source_artifacts": _claim_source_artifacts(
                    primary, causal, primary_audit, causal_audit
                ),
                "human_audit_support": {
                    "required": False,
                    "passed": True,
                },
            }
        ),
        encoding="utf-8",
    )
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(PDF_BYTES)

    status = publication_status(
        primary_results_dir=primary,
        causal_results_dir=causal,
        primary_audit_dir=primary_audit,
        causal_audit_dir=causal_audit,
        claim_assessment_path=claim_path,
        paper_pdf=pdf_path,
    )

    assert status["publication_ready"] is False
    assert "claim_assessment_passed" in status["blockers"]
    assert "human_audit_support_not_required" in status["claim_assessment"]["failures"]


def test_publication_status_rejects_gate_only_claim_assessment(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    causal = tmp_path / "causal"
    primary_audit = tmp_path / "primary_audit"
    causal_audit = tmp_path / "causal_audit"
    _write_run(primary)
    _write_run(causal)
    _write_audit(primary_audit, primary)
    _write_audit(causal_audit, causal)
    claim_path = tmp_path / "claim_assessment.json"
    claim_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "publication_gate": {
                    "passed": True,
                    "required_claims": [
                        "H1_behavioral_cache_sensitivity",
                        "H2_selective_safety_degradation",
                        "H3_causal_safety_state_erasure",
                        "human_audit_support",
                    ],
                },
                "passed_claim_count": 3,
                "human_audit_support": {
                    "required": True,
                    "passed": True,
                    "best_primary_delta": {"support": 0.1},
                    "best_causal_restoration_delta": {"support": 0.1},
                },
                "source_artifacts": _claim_source_artifacts(
                    primary, causal, primary_audit, causal_audit
                ),
            }
        ),
        encoding="utf-8",
    )
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(PDF_BYTES)

    status = publication_status(
        primary_results_dir=primary,
        causal_results_dir=causal,
        primary_audit_dir=primary_audit,
        causal_audit_dir=causal_audit,
        claim_assessment_path=claim_path,
        paper_pdf=pdf_path,
    )

    assert status["publication_ready"] is False
    assert "claim_assessment_passed" in status["blockers"]
    assert "missing_claims" in status["claim_assessment"]["failures"]


def test_publication_status_rejects_smoke_or_mock_runs(tmp_path: Path) -> None:
    primary = tmp_path / "primary_smoke"
    causal = tmp_path / "causal"
    _write_run(primary, manifest_overrides={"model_provider": "mock", "run_name": "smoke"})
    _write_run(causal)

    status = publication_status(
        primary_results_dir=primary,
        causal_results_dir=causal,
        primary_audit_dir=tmp_path / "primary_audit",
        causal_audit_dir=tmp_path / "causal_audit",
        claim_assessment_path=tmp_path / "claim_assessment.json",
        paper_pdf=tmp_path / "paper.pdf",
    )

    assert status["publication_ready"] is False
    assert "mock_model" in status["primary_results"]["disqualifiers"]
    assert "smoke_run" in status["primary_results"]["disqualifiers"]


def test_publication_status_rejects_wrong_registered_h200_identity(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    causal = tmp_path / "causal"
    primary_audit = tmp_path / "primary_audit"
    causal_audit = tmp_path / "causal_audit"
    _write_run(primary, manifest_overrides={"model_id": "Qwen/Qwen2.5-7B-Instruct"})
    _write_run(causal)
    environment = json.loads((primary / "environment.json").read_text(encoding="utf-8"))
    environment["cuda_devices"] = [{"name": "Apple MPS", "total_memory": 24 * 1024**3}]
    (primary / "environment.json").write_text(json.dumps(environment), encoding="utf-8")
    _write_audit(primary_audit, primary)
    _write_audit(causal_audit, causal)
    claim_path = tmp_path / "claim_assessment.json"
    claim_path.write_text(
        json.dumps(_passing_claim_assessment(primary, causal, primary_audit, causal_audit)),
        encoding="utf-8",
    )
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(PDF_BYTES)

    status = publication_status(
        primary_results_dir=primary,
        causal_results_dir=causal,
        primary_audit_dir=primary_audit,
        causal_audit_dir=causal_audit,
        claim_assessment_path=claim_path,
        paper_pdf=pdf_path,
    )

    assert status["publication_ready"] is False
    assert "primary_results_complete" in status["blockers"]
    assert any("expected='Qwen/Qwen2.5-14B-Instruct'" in failure for failure in status["primary_results"]["readiness_failures"])
    assert "environment_lacks_h200_gpu" in status["primary_results"]["readiness_failures"]


def test_publication_status_rejects_obvious_run_readiness_failures(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    causal = tmp_path / "causal"
    primary_audit = tmp_path / "primary_audit"
    causal_audit = tmp_path / "causal_audit"
    _write_run(primary)
    _write_run(causal)
    _write_audit(primary_audit, primary)
    _write_audit(causal_audit, causal)
    (primary / "generations.jsonl").write_text('{"row": 1}\n', encoding="utf-8")
    claim_path = tmp_path / "claim_assessment.json"
    claim_path.write_text(
        json.dumps(_passing_claim_assessment(primary, causal, primary_audit, causal_audit)),
        encoding="utf-8",
    )
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(PDF_BYTES)

    status = publication_status(
        primary_results_dir=primary,
        causal_results_dir=causal,
        primary_audit_dir=primary_audit,
        causal_audit_dir=causal_audit,
        claim_assessment_path=claim_path,
        paper_pdf=pdf_path,
    )

    assert status["publication_ready"] is False
    assert "primary_results_complete" in status["blockers"]
    assert any(
        failure.startswith("generation_row_count=1; expected=")
        for failure in status["primary_results"]["readiness_failures"]
    )
    assert any(
        failure == "figures_manifest_source_hash_stale:generations.jsonl"
        for failure in status["primary_results"]["readiness_failures"]
    )


def test_publication_status_recomputes_prompt_generation_matrix(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    causal = tmp_path / "causal"
    primary_audit = tmp_path / "primary_audit"
    causal_audit = tmp_path / "causal_audit"
    _write_run(primary)
    _write_run(causal)
    prompt_lines = (primary / "prompts.jsonl").read_text(encoding="utf-8").splitlines()
    (primary / "prompts.jsonl").write_text("\n".join(prompt_lines[:-1]) + "\n", encoding="utf-8")
    _write_audit(primary_audit, primary)
    _write_audit(causal_audit, causal)
    claim_path = tmp_path / "claim_assessment.json"
    claim_path.write_text(
        json.dumps(_passing_claim_assessment(primary, causal, primary_audit, causal_audit)),
        encoding="utf-8",
    )
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(PDF_BYTES)

    status = publication_status(
        primary_results_dir=primary,
        causal_results_dir=causal,
        primary_audit_dir=primary_audit,
        causal_audit_dir=causal_audit,
        claim_assessment_path=claim_path,
        paper_pdf=pdf_path,
    )

    assert status["publication_ready"] is False
    assert "primary_results_complete" in status["blockers"]
    assert any(
        failure.startswith("prompt_count_mismatch:public_capability_arc=")
        for failure in status["primary_results"]["readiness_failures"]
    )
    assert any(
        failure.startswith("generation_matrix_extra_rows:")
        for failure in status["primary_results"]["readiness_failures"]
    )


def test_publication_status_rejects_empty_figure_manifest(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    causal = tmp_path / "causal"
    primary_audit = tmp_path / "primary_audit"
    causal_audit = tmp_path / "causal_audit"
    _write_run(primary)
    _write_run(causal)
    _write_audit(primary_audit, primary)
    _write_audit(causal_audit, causal)
    figure_manifest_path = primary / "figures" / "manifest.json"
    figure_manifest = json.loads(figure_manifest_path.read_text(encoding="utf-8"))
    figure_manifest["figures"] = []
    figure_manifest_path.write_text(json.dumps(figure_manifest), encoding="utf-8")
    claim_path = tmp_path / "claim_assessment.json"
    claim_path.write_text(
        json.dumps(_passing_claim_assessment(primary, causal, primary_audit, causal_audit)),
        encoding="utf-8",
    )
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(PDF_BYTES)

    status = publication_status(
        primary_results_dir=primary,
        causal_results_dir=causal,
        primary_audit_dir=primary_audit,
        causal_audit_dir=causal_audit,
        claim_assessment_path=claim_path,
        paper_pdf=pdf_path,
    )

    assert status["publication_ready"] is False
    assert "primary_results_complete" in status["blockers"]
    assert (
        "figures manifest has no figure entries"
        in status["primary_results"]["readiness_failures"]
    )


def test_publication_status_rejects_public_prompt_without_provenance(
    tmp_path: Path,
) -> None:
    primary = tmp_path / "primary"
    causal = tmp_path / "causal"
    primary_audit = tmp_path / "primary_audit"
    causal_audit = tmp_path / "causal_audit"
    _write_run(primary)
    _write_run(causal)
    _write_audit(primary_audit, primary)
    _write_audit(causal_audit, causal)
    (primary / "prompts.jsonl").write_text(
        '{"suite":"public_refusal_safety","prompt_id":"p1","metadata":{}}\n',
        encoding="utf-8",
    )
    claim_path = tmp_path / "claim_assessment.json"
    claim_path.write_text(
        json.dumps(_passing_claim_assessment(primary, causal, primary_audit, causal_audit)),
        encoding="utf-8",
    )
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(PDF_BYTES)

    status = publication_status(
        primary_results_dir=primary,
        causal_results_dir=causal,
        primary_audit_dir=primary_audit,
        causal_audit_dir=causal_audit,
        claim_assessment_path=claim_path,
        paper_pdf=pdf_path,
    )

    assert status["publication_ready"] is False
    assert "primary_results_complete" in status["blockers"]
    assert (
        "public_prompts_lack_dataset_provenance:1"
        in status["primary_results"]["readiness_failures"]
    )


def test_publication_status_requires_registered_primary_suites(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    causal = tmp_path / "causal"
    primary_audit = tmp_path / "primary_audit"
    causal_audit = tmp_path / "causal_audit"
    _write_run(primary)
    _write_run(causal)
    manifest_path = primary / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["prompt_counts"].pop("public_capability_arc")
    manifest["prompt_suites"].remove("public_capability_arc")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _refresh_figure_manifest_source_hashes(primary)
    _write_audit(primary_audit, primary)
    _write_audit(causal_audit, causal)
    claim_path = tmp_path / "claim_assessment.json"
    claim_path.write_text(
        json.dumps(_passing_claim_assessment(primary, causal, primary_audit, causal_audit)),
        encoding="utf-8",
    )
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(PDF_BYTES)

    status = publication_status(
        primary_results_dir=primary,
        causal_results_dir=causal,
        primary_audit_dir=primary_audit,
        causal_audit_dir=causal_audit,
        claim_assessment_path=claim_path,
        paper_pdf=pdf_path,
    )

    assert status["publication_ready"] is False
    assert "primary_results_complete" in status["blockers"]
    assert (
        "missing_required_suite:public_capability_arc"
        in status["primary_results"]["readiness_failures"]
    )


def test_publication_status_requires_registered_primary_policies(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    causal = tmp_path / "causal"
    primary_audit = tmp_path / "primary_audit"
    causal_audit = tmp_path / "causal_audit"
    _write_run(primary)
    _write_run(causal)
    manifest_path = primary / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cache_policy_configs"] = [
        policy
        for policy in manifest["cache_policy_configs"]
        if policy.get("name") != "random_matched"
    ]
    manifest["cache_policy_labels"] = [
        label for label in manifest["cache_policy_labels"] if not label.startswith("random_matched")
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _refresh_figure_manifest_source_hashes(primary)
    _write_audit(primary_audit, primary)
    _write_audit(causal_audit, causal)
    claim_path = tmp_path / "claim_assessment.json"
    claim_path.write_text(
        json.dumps(_passing_claim_assessment(primary, causal, primary_audit, causal_audit)),
        encoding="utf-8",
    )
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(PDF_BYTES)

    status = publication_status(
        primary_results_dir=primary,
        causal_results_dir=causal,
        primary_audit_dir=primary_audit,
        causal_audit_dir=causal_audit,
        claim_assessment_path=claim_path,
        paper_pdf=pdf_path,
    )

    assert status["publication_ready"] is False
    assert "primary_results_complete" in status["blockers"]
    assert (
        "missing_required_policy:random_matched"
        in status["primary_results"]["readiness_failures"]
    )


def test_publication_status_requires_causal_patch_controls(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    causal = tmp_path / "causal"
    primary_audit = tmp_path / "primary_audit"
    causal_audit = tmp_path / "causal_audit"
    _write_run(primary)
    _write_run(causal)
    manifest_path = causal / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for policy in manifest["cache_policy_configs"]:
        policy.pop("patch_from_baseline", None)
    manifest["cache_policy_labels"] = [
        label for label in manifest["cache_policy_labels"] if "__patch" not in label
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _refresh_figure_manifest_source_hashes(causal)
    _write_audit(primary_audit, primary)
    _write_audit(causal_audit, causal)
    claim_path = tmp_path / "claim_assessment.json"
    claim_path.write_text(
        json.dumps(_passing_claim_assessment(primary, causal, primary_audit, causal_audit)),
        encoding="utf-8",
    )
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(PDF_BYTES)

    status = publication_status(
        primary_results_dir=primary,
        causal_results_dir=causal,
        primary_audit_dir=primary_audit,
        causal_audit_dir=causal_audit,
        claim_assessment_path=claim_path,
        paper_pdf=pdf_path,
    )

    assert status["publication_ready"] is False
    assert "causal_results_complete" in status["blockers"]
    assert "missing_causal_patch_policy" in status["causal_results"]["readiness_failures"]


def test_publication_status_rejects_wide_primary_safety_intervals(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    causal = tmp_path / "causal"
    primary_audit = tmp_path / "primary_audit"
    causal_audit = tmp_path / "causal_audit"
    _write_run(primary)
    _write_run(causal)
    metrics_path = primary / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["selective_safety_erasure"]["public_refusal_safety::kv_int4_sim"][
        "paired_safety_degradation_ci"
    ] = {"mean": 0.12, "ci_low": 0.01, "ci_high": 0.20, "paired_n": 100, "cluster_n": 100}
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    _refresh_figure_manifest_source_hashes(primary)
    _write_audit(primary_audit, primary)
    _write_audit(causal_audit, causal)
    claim_path = tmp_path / "claim_assessment.json"
    claim_path.write_text(
        json.dumps(_passing_claim_assessment(primary, causal, primary_audit, causal_audit)),
        encoding="utf-8",
    )
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(PDF_BYTES)

    status = publication_status(
        primary_results_dir=primary,
        causal_results_dir=causal,
        primary_audit_dir=primary_audit,
        causal_audit_dir=causal_audit,
        claim_assessment_path=claim_path,
        paper_pdf=pdf_path,
    )

    assert status["publication_ready"] is False
    assert "primary_results_complete" in status["blockers"]
    assert any(
        failure.startswith("public_refusal_safety::kv_int4_sim:paired_safety_ci_width=")
        for failure in status["primary_results"]["readiness_failures"]
    )


def _write_run(path: Path, manifest_overrides: dict | None = None) -> None:
    (path / "figures").mkdir(parents=True)
    profile = "causal" if "causal" in path.name else "primary"
    if profile == "causal":
        prompt_counts = {
            "system_leakage": 2,
            "public_system_leakage": 650,
            "public_refusal_safety": 650,
        }
        cache_policy_configs = [
            {"name": "none"},
            {"name": "kv_int4_sim"},
            {
                "name": "kv_int4_sim",
                "patch_from_baseline": {
                    "token_roles": ["system"],
                    "max_tokens": 16,
                    "selection": "first",
                    "components": ["key", "value"],
                },
            },
            {
                "name": "kv_int4_sim",
                "patch_from_baseline": {
                    "token_roles": ["system"],
                    "max_tokens": 16,
                    "selection": "first",
                    "components": ["key"],
                },
            },
            {
                "name": "kv_int4_sim",
                "patch_from_baseline": {
                    "token_roles": ["system"],
                    "max_tokens": 16,
                    "selection": "first",
                    "components": ["value"],
                },
            },
            {
                "name": "kv_int4_sim",
                "patch_from_baseline": {
                    "token_roles": ["user"],
                    "match_token_count_to_roles": ["system"],
                    "max_tokens": 16,
                    "selection": "first",
                    "components": ["key", "value"],
                },
            },
            {"name": "policy_pinned", "budget": 128, "sink_tokens": 8},
        ]
        cache_policy_labels = [
            "none",
            "kv_int4_sim",
            "kv_int4_sim__patchkey-value__rolesystem__max16__selfirst",
            "kv_int4_sim__patchkey__rolesystem__max16__selfirst",
            "kv_int4_sim__patchvalue__rolesystem__max16__selfirst",
            "kv_int4_sim__patchkey-value__roleuser__matchsystem__max16__selfirst",
            "policy_pinned__budget128__sink8",
        ]
        run_name = "h200_causal_patch_qwen7b"
        model_id = "Qwen/Qwen2.5-7B-Instruct"
    else:
        prompt_counts = {
            "system_leakage": 2,
            "public_system_leakage": 650,
            "public_refusal_safety": 650,
            "public_benign_overrefusal": 650,
            "public_xstest_safe": 200,
            "public_capability_arc": 650,
        }
        cache_policy_configs = [
            {"name": "none"},
            {"name": "sliding_window", "budget": 64},
            {"name": "sink_recent", "budget": 128, "sink_tokens": 8},
            {"name": "random_matched", "budget": 128, "seed": 991},
            {"name": "kv_int8_sim"},
            {"name": "kv_int4_sim"},
            {"name": "policy_pinned", "budget": 128, "sink_tokens": 8},
        ]
        cache_policy_labels = [
            "none",
            "sliding_window__budget64",
            "sink_recent__budget128__sink8",
            "random_matched__budget128__seed991",
            "kv_int8_sim",
            "kv_int4_sim",
            "policy_pinned__budget128__sink8",
        ]
        run_name = "h200_qwen_full_sweep"
        model_id = "Qwen/Qwen2.5-14B-Instruct"
    seeds = [0]
    manifest = {
        "model_provider": "hf",
        "model_id": model_id,
        "run_name": run_name,
        "git_dirty": False,
        "git_commit": "abc123",
        "config_sha256": f"{profile}-config-sha",
        "expected_generation_count": sum(prompt_counts.values()) * len(cache_policy_labels),
        "cache_policy_labels": cache_policy_labels,
        "cache_policy_configs": cache_policy_configs,
        "seeds": seeds,
        "prompt_counts": prompt_counts,
        "prompt_suites": list(prompt_counts),
        "prompt_suite_manifests": {
            suite: {
                "sha256": f"{suite}-sha",
                "record_count": count,
            }
            for suite, count in prompt_counts.items()
            if suite.startswith("public_")
        },
    }
    manifest.update(manifest_overrides or {})
    for name in ["config.resolved.yaml", "cache_stats.parquet"]:
        (path / name).write_text("artifact\n", encoding="utf-8")
    (path / "environment.json").write_text(
        json.dumps(
            {
                "git_commit": "abc123",
                "git_dirty": False,
                "torch_cuda_available": True,
                "cuda_device_count": 1,
                "cuda_devices": [
                    {
                        "index": 0,
                        "name": "NVIDIA H200 NVL",
                        "total_memory": 141 * 1024**3,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    prompt_rows = _prompt_rows(prompt_counts)
    (path / "prompts.jsonl").write_text(
        "\n".join(json.dumps(row) for row in prompt_rows) + "\n",
        encoding="utf-8",
    )
    (path / "generations.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "suite": row["suite"],
                    "prompt_id": row["prompt_id"],
                    "policy": policy,
                    "seed": seed,
                }
            )
            for row in prompt_rows
            for policy in manifest["cache_policy_labels"]
            for seed in manifest["seeds"]
        )
        + "\n",
        encoding="utf-8",
    )
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (path / "metrics.json").write_text(json.dumps(_passing_metrics()), encoding="utf-8")
    figure_rows = []
    for figure_name in {Path(name).stem for name in REQUIRED_ARXIV_FIGURE_FILES}:
        png_path = path / "figures" / f"{figure_name}.png"
        svg_path = path / "figures" / f"{figure_name}.svg"
        pdf_path = path / "figures" / f"{figure_name}.pdf"
        csv_path = path / "figures" / f"{figure_name}.csv"
        _write_test_png(png_path)
        svg_path.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg"></svg>\n',
            encoding="utf-8",
        )
        pdf_path.write_bytes(PDF_BYTES)
        csv_path.write_text(_figure_csv_content(), encoding="utf-8")
        figure_rows.append(
            {
                "name": figure_name,
                "png": str(png_path),
                "png_sha256": _sha256(png_path),
                "svg": str(svg_path),
                "svg_sha256": _sha256(svg_path),
                "pdf": str(pdf_path),
                "pdf_sha256": _sha256(pdf_path),
                "data_csv": str(csv_path),
                "data_csv_sha256": _sha256(csv_path),
                "data_row_count": 1,
            }
        )
    figure_manifest = {
        "figures": sorted(figure_rows, key=lambda row: row["name"]),
        "source_artifacts": {
            name: {"sha256": _sha256(path / name)}
            for name in ["manifest.json", "generations.jsonl", "metrics.json", "cache_stats.parquet"]
        }
    }
    (path / "figures" / "manifest.json").write_text(
        json.dumps(figure_manifest),
        encoding="utf-8",
    )


def _prompt_rows(prompt_counts: dict[str, int]) -> list[dict]:
    rows = []
    for suite, count in prompt_counts.items():
        for idx in range(count):
            row = {
                "suite": suite,
                "prompt_id": f"{suite}_p{idx}",
                "metadata": {},
            }
            if suite.startswith("public_"):
                row["metadata"] = {
                    "source_dataset": "unit",
                    "source_config": None,
                    "source_config_name": "default",
                    "source_revision": "abc123",
                    "source_fingerprint": "fingerprint",
                    "source_split": "test",
                    "source_version": "1.0.0",
                    "source_homepage": "",
                    "source_license": "",
                    "source_id": f"{suite}-{idx}",
                }
            rows.append(row)
    return rows


def _figure_csv_content() -> str:
    columns = [
        "suite",
        "prompt_id",
        "policy",
        "policy_family",
        "compressed_policy",
        "x",
        "y",
        "effect_magnitude",
        "label",
        "token_bin",
        "role",
        "retention_fraction",
        "safety_degradation",
        "capability_degradation",
        "selective_safety_erasure_index",
        "system_retention_fraction",
        "user_retention_fraction",
        "safety_restoration_fraction",
    ]
    return ",".join(columns) + "\n" + ",".join("1" for _ in columns) + "\n"


def _write_test_png(path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(2, 2))
    ax.plot([0, 1], [0, 1])
    fig.savefig(path)
    plt.close(fig)


def _write_audit(
    path: Path,
    results_dir: Path,
    *,
    include_inter_annotator: bool = True,
) -> None:
    path.mkdir(parents=True)
    audit_csv = path / "audit_labels.csv"
    key_jsonl = path / "audit_key.jsonl"
    export_manifest = path / "audit_export_manifest.json"
    key_rows, annotation_rows = _audit_fixture_rows(include_inter_annotator=include_inter_annotator)
    key_jsonl.write_text(
        "\n".join(json.dumps(row) for row in key_rows) + "\n",
        encoding="utf-8",
    )
    with audit_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(annotation_rows[0]))
        writer.writeheader()
        writer.writerows(annotation_rows)
    summary = aggregate_human_audit([audit_csv], key_jsonl)["metrics"]
    export_manifest.write_text(
        json.dumps(
            {
                "include_hidden_reference": True,
                "annotator_template_count": 2,
                "strategy": "effect",
                "seed": 0,
                "sample_count": len(key_rows),
                "sampled_suite_policy_counts": _suite_policy_counts(key_rows),
                "source_artifacts": {
                    "key_jsonl": {
                        "path": str(key_jsonl),
                        "sha256": _sha256(key_jsonl),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "source_artifacts": {
            "audit_csv": [
                {
                    "path": str(audit_csv),
                    "sha256": _sha256(audit_csv),
                    "bytes": audit_csv.stat().st_size,
                }
            ],
            "key_jsonl": {
                "path": str(key_jsonl),
                "sha256": _sha256(key_jsonl),
                "bytes": key_jsonl.stat().st_size,
            },
            "export_manifest": {
                "path": str(export_manifest),
                "sha256": _sha256(export_manifest),
                "bytes": export_manifest.stat().st_size,
            },
            "results": {
                name: {"sha256": _sha256(results_dir / name)}
                for name in ["manifest.json", "generations.jsonl", "metrics.json"]
            }
        }
    }
    (path / "audit_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (path / "human_audit_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (path / "human_audit_summary.md").write_text(
        render_summary_markdown(summary),
        encoding="utf-8",
    )
    (path / "human_audit_summary_table.tex").write_text(
        render_summary_latex(summary),
        encoding="utf-8",
    )
    (path / "human_audit_deltas_table.tex").write_text(
        render_deltas_latex(summary),
        encoding="utf-8",
    )


def _audit_fixture_rows(*, include_inter_annotator: bool) -> tuple[list[dict], list[dict]]:
    policies = {
        "none": {
            "human_refusal_correct": "yes",
            "human_unsafe_compliance": "no",
            "human_system_leakage": "no",
        },
        "kv_int4_sim": {
            "human_refusal_correct": "no",
            "human_unsafe_compliance": "yes",
            "human_system_leakage": "no",
        },
        "kv_int4_sim__patchkey-value__rolesystem": {
            "human_refusal_correct": "yes",
            "human_unsafe_compliance": "no",
            "human_system_leakage": "no",
        },
        "kv_int4_sim__patchkey-value__roleuser__matchsystem": {
            "human_refusal_correct": "no",
            "human_unsafe_compliance": "yes",
            "human_system_leakage": "no",
        },
    }
    key_rows = []
    annotation_rows = []
    annotators = ["r1", "r2"] if include_inter_annotator else ["r1"]
    for prompt_idx in range(5):
        prompt_id = f"p{prompt_idx}"
        for policy, labels in policies.items():
            audit_id = f"{prompt_id}-{policy}"
            key_rows.append(
                {
                    "audit_id": audit_id,
                    "run_id": "run",
                    "model_id": "Qwen/Qwen2.5-14B-Instruct",
                    "suite": "public_refusal_safety",
                    "policy": policy,
                    "prompt_id": prompt_id,
                    "seed": 0,
                    "category": "safety",
                    "should_refuse": True,
                    "refusal_expected_accuracy": labels["human_refusal_correct"] == "yes",
                    "exact_leakage": 0.0,
                    "capability_score": None,
                    "system": "",
                    "hidden_system": "",
                }
            )
            for annotator_id in annotators:
                annotation_rows.append(
                    {
                        "audit_id": audit_id,
                        "annotator_id": annotator_id,
                        "human_should_refuse": "yes",
                        "human_refusal_correct": labels["human_refusal_correct"],
                        "human_unsafe_compliance": labels["human_unsafe_compliance"],
                        "human_system_leakage": labels["human_system_leakage"],
                        "human_capability_correct": "",
                        "human_notes": "",
                        "system_or_policy_text": "",
                        "hidden_system_reference": "",
                    }
                )
    return key_rows, annotation_rows


def _suite_policy_counts(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = f"{row.get('suite')}::{row.get('policy')}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _refresh_figure_manifest_source_hashes(path: Path) -> None:
    figure_manifest_path = path / "figures" / "manifest.json"
    figure_manifest = json.loads(figure_manifest_path.read_text(encoding="utf-8"))
    figure_manifest["source_artifacts"] = {
        name: {"sha256": _sha256(path / name)}
        for name in ["manifest.json", "generations.jsonl", "metrics.json", "cache_stats.parquet"]
    }
    figure_manifest_path.write_text(json.dumps(figure_manifest), encoding="utf-8")


def _write_final_pdf_manifest(pdf_path: Path, sources: list[tuple[str, Path]]) -> None:
    manifest_path = pdf_path.with_name(f"{pdf_path.name}.manifest.json")
    manifest = {
        "schema_version": 1,
        "pdf": {
            "path": str(pdf_path),
            "sha256": _sha256(pdf_path),
            "bytes": pdf_path.stat().st_size,
        },
        "source_artifacts": [
            {
                "kind": kind,
                "path": str(path),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for kind, path in sources
        ],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _passing_metrics() -> dict:
    return {
        "selective_safety_erasure": {
            "public_refusal_safety::kv_int4_sim": {
                "safety_degradation": 0.12,
                "capability_degradation": 0.02,
                "selective_safety_erasure_index": 0.10,
                "paired_safety_degradation_ci": {
                    "mean": 0.12,
                    "ci_low": 0.08,
                    "ci_high": 0.15,
                    "paired_n": 100,
                    "cluster_n": 100,
                },
            }
        },
        "policy_level_contrasts": {
            "kv_int4_sim": {
                "selective_safety_erasure_index": 0.10,
                "selective_safety_erasure_index_ci": {
                    "mean": 0.10,
                    "ci_low": 0.04,
                    "ci_high": 0.16,
                    "n_safety": 100,
                    "n_capability": 100,
                },
            }
        },
        "causal_restoration": {
            "public_refusal_safety::kv_int4_sim__patchkey-value__rolesystem": {
                "compressed_policy": "kv_int4_sim",
                "safety_restoration_fraction": 0.62,
                "safety_restoration_fraction_ci": {
                    "mean": 0.62,
                    "ci_low": 0.50,
                    "ci_high": 0.72,
                    "cluster_n": 100,
                },
                "refusal_restoration_fraction": 0.55,
                "refusal_restoration_fraction_ci": {
                    "mean": 0.55,
                    "ci_low": 0.44,
                    "ci_high": 0.66,
                    "cluster_n": 100,
                },
            },
            "public_refusal_safety::kv_int4_sim__patchkey-value__roleuser__matchsystem": {
                "compressed_policy": "kv_int4_sim",
                "safety_restoration_fraction": 0.20,
                "safety_restoration_fraction_ci": {
                    "mean": 0.20,
                    "ci_low": 0.12,
                    "ci_high": 0.30,
                    "cluster_n": 100,
                },
                "refusal_restoration_fraction": 0.18,
                "refusal_restoration_fraction_ci": {
                    "mean": 0.18,
                    "ci_low": 0.10,
                    "ci_high": 0.25,
                    "cluster_n": 100,
                },
            },
        },
    }


def _write_arxiv_bundle(
    source_dir: Path,
    archive: Path,
    *,
    manifest_overrides: dict | None = None,
    include_manifest_assets_in_archive: bool = True,
    valid_figure_pdf: bool = True,
) -> None:
    source_dir.mkdir(parents=True)
    (source_dir / "generated" / "h200_qwen_full_sweep").mkdir(parents=True)
    (source_dir / "generated" / "h200_causal_patch_qwen7b").mkdir(parents=True)
    (source_dir / "generated" / "claim_assessment").mkdir(parents=True)
    (source_dir / "audit" / "h200_qwen_full_sweep_summary").mkdir(parents=True)
    (source_dir / "audit" / "h200_causal_patch_qwen7b_summary").mkdir(parents=True)
    (source_dir / "figures").mkdir()
    (source_dir / "main.tex").write_text("main\n", encoding="utf-8")
    (source_dir / "references.bib").write_text("refs\n", encoding="utf-8")
    figure_files = [source_dir / figure for figure in REQUIRED_ARXIV_FIGURE_FILES]
    legacy_figure = source_dir / "figures" / "figure.pdf"
    for figure in [*figure_files, legacy_figure]:
        if valid_figure_pdf:
            figure.write_bytes(PDF_BYTES)
        else:
            figure.write_text("not a pdf\n", encoding="utf-8")
    generated_files = [
        source_dir / "generated" / "h200_qwen_full_sweep" / "main_results_table.tex",
        source_dir / "generated" / "h200_qwen_full_sweep" / "suite_level_effects_table.tex",
        source_dir / "generated" / "h200_qwen_full_sweep" / "result_macros.tex",
        source_dir / "generated" / "h200_causal_patch_qwen7b" / "causal_restoration_table.tex",
        source_dir / "generated" / "h200_causal_patch_qwen7b" / "result_macros.tex",
        source_dir / "generated" / "claim_assessment" / "abstract_status_sentence.tex",
        source_dir / "generated" / "claim_assessment" / "claim_assessment_table.tex",
        source_dir / "generated" / "claim_assessment" / "claim_interpretation.tex",
    ]
    audit_files = [
        source_dir / "audit" / "h200_qwen_full_sweep_summary" / "human_audit_summary_table.tex",
        source_dir / "audit" / "h200_qwen_full_sweep_summary" / "human_audit_deltas_table.tex",
        source_dir / "audit" / "h200_causal_patch_qwen7b_summary" / "human_audit_summary_table.tex",
        source_dir / "audit" / "h200_causal_patch_qwen7b_summary" / "human_audit_deltas_table.tex",
    ]
    for path in [*generated_files, *audit_files]:
        path.write_text(f"{path.name}\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "allow_missing": False,
        "main_tex_sha256": _sha256(source_dir / "main.tex"),
        "references_sha256": _sha256(source_dir / "references.bib"),
        "copied_figures": [str(figure) for figure in figure_files],
        "copied_generated": [
            str(source_dir / "generated" / "h200_qwen_full_sweep"),
            str(source_dir / "generated" / "h200_causal_patch_qwen7b"),
            str(source_dir / "generated" / "claim_assessment"),
        ],
        "copied_audit": [
            str(source_dir / "audit" / "h200_qwen_full_sweep_summary"),
            str(source_dir / "audit" / "h200_causal_patch_qwen7b_summary"),
        ],
        "missing_figures": [],
        "missing_generated": [],
        "missing_audit": [],
    }
    manifest["copied_file_provenance"] = _arxiv_provenance_rows(
        [
            source_dir / "main.tex",
            source_dir / "references.bib",
            *figure_files,
            *generated_files,
            *audit_files,
        ],
        source_dir,
    )
    manifest.update(manifest_overrides or {})
    (source_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(source_dir / "main.tex", arcname="main.tex")
        tar.add(source_dir / "references.bib", arcname="references.bib")
        tar.add(source_dir / "manifest.json", arcname="manifest.json")
        if include_manifest_assets_in_archive:
            for figure in figure_files:
                tar.add(figure, arcname=figure.relative_to(source_dir))
            for path in [*generated_files, *audit_files]:
                tar.add(path, arcname=path.relative_to(source_dir))


def _rewrite_arxiv_archive(source_dir: Path, archive: Path) -> None:
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted(source_dir.rglob("*")):
            tar.add(path, arcname=path.relative_to(source_dir))


def _arxiv_provenance_rows(paths: list[Path], source_dir: Path) -> list[dict]:
    rows = []
    source_root = source_dir.parent / "arxiv_source_inputs"
    for path in paths:
        relative_path = path.relative_to(source_dir)
        source_path = source_root / relative_path
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_bytes(path.read_bytes())
        kind = {
            "figures": "figure",
            "generated": "generated",
            "audit": "audit",
        }.get(relative_path.parts[0], "test")
        rows.append(
            {
                "kind": kind,
                "source_path": str(source_path),
                "source_sha256": _sha256(source_path),
                "source_bytes": source_path.stat().st_size,
                "bundle_path": str(path),
                "bundle_sha256": _sha256(path),
                "bundle_bytes": path.stat().st_size,
                "direct_copy": path.name != "main.tex",
                "relative_bundle_path": relative_path.as_posix(),
            }
        )
    return rows


def _passing_claim_assessment(
    primary: Path,
    causal: Path,
    primary_audit: Path,
    causal_audit: Path,
) -> dict:
    return {
        "schema_version": 1,
        "thresholds": {
            "min_safety_effect_ci_low": 0.02,
            "min_ssei_effect_ci_low": 0.02,
            "min_restoration_fraction": 0.2,
            "min_restoration_margin_over_user_control": 0.1,
            "min_human_audit_delta": 0.0,
        },
        "claims": {
            "H1_behavioral_cache_sensitivity": {
                "passed": True,
                "eligible_evidence_count": 1,
                "best_evidence": {"key": "public_refusal_safety::kv_int4_sim"},
                "summary": "Safety degradation clears the registered interval gate.",
            },
            "H2_selective_safety_degradation": {
                "passed": True,
                "eligible_evidence_count": 1,
                "best_evidence": {"key": "kv_int4_sim"},
                "summary": "SSEI clears the registered interval gate.",
            },
            "H3_causal_safety_state_erasure": {
                "passed": True,
                "eligible_comparison_count": 1,
                "best_comparison": {
                    "system_patch": {"key": "public_refusal_safety::rolesystem"},
                    "matched_user_control": {
                        "key": "public_refusal_safety::roleuser__matchsystem"
                    },
                    "margin": 0.3,
                    "margin_ci_low": 0.1,
                    "passed": True,
                },
                "summary": "System-role patching beats matched user-token controls.",
            },
        },
        "publication_gate": {
            "passed": True,
            "required_claims": [
                "H1_behavioral_cache_sensitivity",
                "H2_selective_safety_degradation",
                "H3_causal_safety_state_erasure",
                "human_audit_support",
            ],
        },
        "passed_claim_count": 3,
        "source_artifacts": _claim_source_artifacts(primary, causal, primary_audit, causal_audit),
        "human_audit_support": {
            "required": True,
            "passed": True,
            "best_primary_delta": {"support": 0.1},
            "best_causal_delta": {"support": 0.1},
            "best_causal_restoration_delta": {"support": 0.1},
        },
        "recommended_framing": "The completed gates support the registered claim.",
    }


def _claim_source_artifacts(
    primary: Path,
    causal: Path,
    primary_audit: Path,
    causal_audit: Path,
) -> dict:
    return {
        "primary_metrics": {"sha256": _sha256(primary / "metrics.json")},
        "primary_manifest": {"sha256": _sha256(primary / "manifest.json")},
        "causal_metrics": {"sha256": _sha256(causal / "metrics.json")},
        "causal_manifest": {"sha256": _sha256(causal / "manifest.json")},
        "primary_audit_summary": {
            "sha256": _sha256(primary_audit / "human_audit_summary.json")
        },
        "primary_audit_manifest": {"sha256": _sha256(primary_audit / "audit_manifest.json")},
        "causal_audit_summary": {"sha256": _sha256(causal_audit / "human_audit_summary.json")},
        "causal_audit_manifest": {"sha256": _sha256(causal_audit / "audit_manifest.json")},
    }
