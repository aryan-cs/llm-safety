import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))

from package_arxiv_submission import GENERATED_DIRS, _rewrite_main_tex_for_arxiv


def test_latex_manuscript_is_formal_registered_protocol() -> None:
    tex = Path("paper/latex/main.tex").read_text(encoding="utf-8")

    assert r"\documentclass[11pt]{article}" in tex
    assert "Aryan Gupta" in tex
    assert "aryan.cs.app@gmail.com" in tex
    assert "registered analysis protocol" in tex
    assert "reports no empirical claims" in tex
    assert "Empirical result not yet reported" in tex
    assert r"\maybeinputtable{../generated/h200_qwen_full_sweep/main_results_table.tex}" in tex
    assert r"\maybeinputtable{../generated/claim_assessment/claim_assessment_table.tex}" in tex
    assert r"\maybeinputtable{../audit/h200_qwen_full_sweep_summary/human_audit_summary_table.tex}" in tex
    assert r"\PrimaryTopSSEIPolicy" in tex
    assert r"\bibliography{../references}" in tex
    assert "neurips" not in tex.lower()
    assert "H200" not in tex
    assert "cgroup" not in tex
    assert "MacBook" not in tex
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
    assert "generated/h200_qwen_full_sweep" in _rewrite_main_tex_for_arxiv(
        "../generated/h200_qwen_full_sweep/main_results_table.tex"
    )
    assert "generated/claim_assessment" in _rewrite_main_tex_for_arxiv(
        "../generated/claim_assessment/claim_assessment_table.tex"
    )
    assert Path("paper/generated/claim_assessment") in GENERATED_DIRS
    assert "audit/h200_qwen_full_sweep_summary" in _rewrite_main_tex_for_arxiv(
        "../audit/h200_qwen_full_sweep_summary/human_audit_summary_table.tex"
    )
    assert "../../results" not in rewritten
