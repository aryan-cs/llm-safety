import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))

from check_latex_placeholders import missing_placeholder_artifacts
from package_arxiv_submission import (
    GENERATED_DIRS,
    OPTIONAL_GENERATED_DIRS,
    REQUIRED_GENERATED_DIRS,
    _is_pdf,
    _missing_inputs,
    _rewrite_main_tex_for_arxiv,
)


def test_latex_manuscript_is_formal_registered_protocol() -> None:
    tex = Path("paper/latex/main.tex").read_text(encoding="utf-8")

    assert r"\documentclass[11pt]{article}" in tex
    assert "Aryan Gupta" in tex
    assert "aryan.cs.app@gmail.com" in tex
    assert "registered analysis protocol" in tex
    assert "reports no empirical claims" in tex
    assert r"\EmpiricalStatusSentence" in tex
    assert r"\requiredartifact{../generated/claim_assessment/abstract_status_sentence.tex}" in tex
    assert "../generated/claim_assessment/abstract_status_sentence.tex" in tex
    assert "Empirical result not yet reported" in tex
    assert r"\maybeinputtable{../generated/h200_qwen_full_sweep/main_results_table.tex}" in tex
    assert r"\maybeinputtable{../generated/claim_assessment/claim_assessment_table.tex}" in tex
    assert r"\maybeinputtable{../generated/claim_assessment/claim_interpretation.tex}" in tex
    assert r"\maybeinputtable{../audit/h200_qwen_full_sweep_summary/human_audit_summary_table.tex}" in tex
    assert "causal_restoration_fraction.pdf" in tex
    assert r"\PrimaryTopSSEIPolicy" in tex
    assert r"\bibliography{../references}" in tex
    assert "neurips" not in tex.lower()
    assert "H200" not in tex
    assert "cgroup" not in tex
    assert "MacBook" not in tex
    assert "free tooling" not in tex
    assert "opaque serving infrastructure" not in tex
    assert "high-value hypothesis" not in tex
    assert "dirty-tree" not in tex
    assert "mock-model" not in tex
    assert "Replace this box" not in tex
    assert "Failure Examples" not in tex


def test_latex_references_cover_primary_model_and_cache_work() -> None:
    bib = Path("paper/references.bib").read_text(encoding="utf-8")

    for key in [
        "qwen2024qwen25",
        "chen2025pitfalls",
        "ananthanarayanan2026physics",
        "kwon2023pagedattention",
        "wang2025cacheprune",
        "arditi2024refusal",
        "zhang2026anydepth",
    ]:
        assert f"{{{key}," in bib


def test_arxiv_rewrite_uses_local_bibliography_and_figures() -> None:
    source = (
        r"\maybeincludegraphic{../../results/h200_qwen_full_sweep/figures/"
        r"safety_capability_phase_portrait.pdf}{0.9\linewidth}{pending}"
        "\n"
        r"\bibliography{../references}"
    )

    rewritten = _rewrite_main_tex_for_arxiv(source)

    assert r"\bibliography{references}" in rewritten
    assert "figures/safety_capability_phase_portrait.pdf" in rewritten
    assert "figures/prompt_effect_constellation.pdf" in _rewrite_main_tex_for_arxiv(
        "../../results/h200_qwen_full_sweep/figures/prompt_effect_constellation.pdf"
    )
    assert "figures/safety_state_atlas.pdf" in _rewrite_main_tex_for_arxiv(
        "../../results/h200_qwen_full_sweep/figures/safety_state_atlas.pdf"
    )
    assert "figures/causal_restoration_fraction.pdf" in _rewrite_main_tex_for_arxiv(
        "../../results/h200_causal_patch_qwen7b/figures/causal_restoration_fraction.pdf"
    )
    assert "generated/h200_qwen_full_sweep" in _rewrite_main_tex_for_arxiv(
        "../generated/h200_qwen_full_sweep/main_results_table.tex"
    )
    assert "generated/claim_assessment" in _rewrite_main_tex_for_arxiv(
        "../generated/claim_assessment/claim_assessment_table.tex"
    )
    assert "generated/claim_assessment" in _rewrite_main_tex_for_arxiv(
        "../generated/claim_assessment/abstract_status_sentence.tex"
    )
    assert Path("paper/generated/claim_assessment") in GENERATED_DIRS
    assert Path("paper/generated/claim_assessment") in REQUIRED_GENERATED_DIRS
    assert Path("paper/generated/h200_qwen32b_public_followup") in OPTIONAL_GENERATED_DIRS
    assert "audit/h200_qwen_full_sweep_summary" in _rewrite_main_tex_for_arxiv(
        "../audit/h200_qwen_full_sweep_summary/human_audit_summary_table.tex"
    )
    assert "../../results" not in rewritten


def test_arxiv_packager_treats_missing_inputs_as_publication_blockers() -> None:
    manifest = {
        "missing_figures": ["missing_figure.pdf"],
        "missing_generated": ["paper/generated/missing"],
        "missing_audit": ["paper/audit/missing"],
    }

    assert _missing_inputs(manifest) == [
        "missing_figure.pdf",
        "paper/generated/missing",
        "paper/audit/missing",
    ]


def test_arxiv_packager_rejects_malformed_figure_pdfs(tmp_path: Path) -> None:
    fake_pdf = tmp_path / "figure.pdf"
    realish_pdf = tmp_path / "realish.pdf"
    fake_pdf.write_text("not a pdf", encoding="utf-8")
    realish_pdf.write_bytes(b"%PDF-1.7\n")

    assert _is_pdf(fake_pdf) is False
    assert _is_pdf(realish_pdf) is True


def test_latex_placeholder_checker_reports_missing_artifacts(tmp_path: Path) -> None:
    tex = tmp_path / "main.tex"
    existing = tmp_path / "figure.pdf"
    existing.write_text("not a real pdf", encoding="utf-8")
    tex.write_text(
        r"\maybeincludegraphic{figure.pdf}{0.9\linewidth}{ok}"
        "\n"
        r"\maybeinputtable{missing/table.tex}{pending}",
        encoding="utf-8",
    )

    assert missing_placeholder_artifacts(tex) == ["missing/table.tex"]


def test_latex_placeholder_checker_requires_generated_text_artifacts(tmp_path: Path) -> None:
    tex = tmp_path / "main.tex"
    existing = tmp_path / "generated" / "ok.tex"
    existing.parent.mkdir()
    existing.write_text(r"\renewcommand{\EmpiricalStatusSentence}{ok}", encoding="utf-8")
    tex.write_text(
        r"\requiredartifact{generated/ok.tex}"
        "\n"
        r"\requiredartifact{generated/missing_status.tex}",
        encoding="utf-8",
    )

    assert missing_placeholder_artifacts(tex) == ["generated/missing_status.tex"]
