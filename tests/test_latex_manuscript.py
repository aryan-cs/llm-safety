import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))

from check_final_pdf_text import placeholder_text_failures
from check_latex_placeholders import missing_placeholder_artifacts, placeholder_artifact_failures
from package_arxiv_submission import (
    GENERATED_DIRS,
    OPTIONAL_GENERATED_DIRS,
    REQUIRED_GENERATED_DIRS,
    _copy_arxiv_support_tree,
    _final_source_failures,
    _invalid_arxiv_support_files,
    _is_pdf,
    _missing_inputs,
    _rewrite_failures,
    _rewrite_main_tex_for_arxiv,
    build_figure_sources,
)


def test_latex_manuscript_is_formal_registered_protocol() -> None:
    tex = Path("paper/latex/main.tex").read_text(encoding="utf-8")

    assert r"\documentclass[11pt]{article}" in tex
    assert "Aryan Gupta" in tex
    assert "aryan.cs.app@gmail.com" in tex
    assert "registered analysis protocol" in tex
    assert "reports no empirical claims" in tex
    assert r"\EmpiricalStatusSentence" in tex
    assert r"\requiredartifact{../generated/h200_qwen_full_sweep/result_macros.tex}" in tex
    assert r"\requiredartifact{../generated/h200_causal_patch_qwen7b/result_macros.tex}" in tex
    assert r"\requiredartifact{../generated/claim_assessment/abstract_status_sentence.tex}" in tex
    assert "../generated/h200_qwen_full_sweep/result_macros.tex" in tex
    assert "../generated/h200_causal_patch_qwen7b/result_macros.tex" in tex
    assert "../generated/claim_assessment/abstract_status_sentence.tex" in tex
    assert "Empirical result not yet reported" in tex
    assert r"\maybeinputtable{../generated/h200_qwen_full_sweep/main_results_table.tex}" in tex
    assert r"\maybeinputtable{../generated/claim_assessment/claim_assessment_table.tex}" in tex
    assert r"\maybeinputtable{../generated/claim_assessment/claim_interpretation.tex}" in tex
    assert r"\maybeinputtable{../audit/h200_qwen_full_sweep_summary/human_audit_summary_table.tex}" in tex
    assert r"\maybeinputtable{../audit/h200_causal_patch_qwen7b_summary/human_audit_summary_table.tex}" in tex
    assert r"\maybeinputtable{../audit/h200_causal_patch_qwen7b_summary/human_audit_deltas_table.tex}" in tex
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
    assert "seven interventions" not in tex
    assert "Attention-H2O retention is treated as a diagnostic extension" in tex


def test_latex_manuscript_uses_formal_publication_wording() -> None:
    tex = Path("paper/latex/main.tex").read_text(encoding="utf-8")

    for phrase in [
        "when budget permits",
        "The shape of the trajectory matters",
        "bend upward",
        "restoration stream",
        "because some safety behavior depends on fragile cache-resident routing state",
        "should appear",
        "should trace",
        "should be summarized",
    ]:
        assert phrase not in tex


def test_paper_notes_avoid_internal_planning_language() -> None:
    for path in [Path("paper/related_work.md"), Path("paper/outline.md")]:
        text = path.read_text(encoding="utf-8").lower()
        for phrase in [
            "user's goal",
            "repository should",
            "planned novelty",
            "no cpu/disk offload",
        ]:
            assert phrase not in text


def test_final_pdf_text_checker_rejects_draft_protocol_markers() -> None:
    failures = placeholder_text_failures(
        "This registered analysis protocol reports no empirical claims. "
        "Figure unavailable."
    )

    assert "placeholder_text:registered analysis protocol" in failures
    assert "placeholder_text:reports no empirical claims" in failures
    assert "placeholder_text:Figure unavailable" in failures


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
        "zou2023universal",
        "databricks2023dolly",
        "clark2018arc",
        "cyberec2026promptinjection",
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
    strict = _rewrite_main_tex_for_arxiv(
        Path("paper/latex/main.tex").read_text(encoding="utf-8"),
        strict_final=True,
    )
    assert _final_source_failures(strict) == []
    assert r"\PackageError{cache-paper}{Missing required publication artifact}" in strict
    assert "generated/h200_qwen_full_sweep" in _rewrite_main_tex_for_arxiv(
        "../generated/h200_qwen_full_sweep/main_results_table.tex"
    )
    assert "generated/claim_assessment" in _rewrite_main_tex_for_arxiv(
        "../generated/claim_assessment/claim_assessment_table.tex"
    )
    assert "generated/claim_assessment" in _rewrite_main_tex_for_arxiv(
        "../generated/claim_assessment/abstract_status_sentence.tex"
    )
    assert "generated/h200_qwen_full_sweep" in _rewrite_main_tex_for_arxiv(
        "../generated/h200_qwen_full_sweep/result_macros.tex"
    )
    assert "generated/h200_causal_patch_qwen7b" in _rewrite_main_tex_for_arxiv(
        "../generated/h200_causal_patch_qwen7b/result_macros.tex"
    )
    assert Path("paper/generated/claim_assessment") in GENERATED_DIRS
    assert Path("paper/generated/claim_assessment") in REQUIRED_GENERATED_DIRS
    assert Path("paper/generated/h200_qwen32b_public_followup") in OPTIONAL_GENERATED_DIRS
    assert "audit/h200_qwen_full_sweep_summary" in _rewrite_main_tex_for_arxiv(
        "../audit/h200_qwen_full_sweep_summary/human_audit_summary_table.tex"
    )
    assert "audit/h200_causal_patch_qwen7b_summary" in _rewrite_main_tex_for_arxiv(
        "../audit/h200_causal_patch_qwen7b_summary/human_audit_summary_table.tex"
    )
    assert "../../results" not in rewritten


def test_arxiv_packager_can_target_custom_result_figure_dirs(tmp_path: Path) -> None:
    primary = tmp_path / "primary_run"
    causal = tmp_path / "causal_run"
    figure_sources = build_figure_sources(primary, causal)

    assert figure_sources["safety_state_atlas.pdf"] == (
        primary / "figures" / "safety_state_atlas.pdf"
    )
    assert figure_sources["causal_restoration_flow.pdf"] == (
        causal / "figures" / "causal_restoration_flow.pdf"
    )
    rewritten = _rewrite_main_tex_for_arxiv(
        str(Path("../..") / figure_sources["safety_state_atlas.pdf"]),
        figure_sources=figure_sources,
    )

    assert rewritten == "figures/safety_state_atlas.pdf"
    assert "figures/safety_state_atlas.pdf" in _rewrite_main_tex_for_arxiv(
        "../../results/h200_qwen_full_sweep/figures/safety_state_atlas.pdf",
        figure_sources=figure_sources,
    )


def test_arxiv_rewrite_current_manuscript_has_no_repo_local_paths() -> None:
    tex = Path("paper/latex/main.tex").read_text(encoding="utf-8")

    rewritten = _rewrite_main_tex_for_arxiv(tex)

    assert _rewrite_failures(rewritten) == []
    assert "../../results" not in rewritten
    assert "../generated" not in rewritten
    assert "../audit" not in rewritten
    assert "../references" not in rewritten


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
    realish_pdf.write_bytes(b"%PDF-1.7\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF\n")

    assert _is_pdf(fake_pdf) is False
    assert _is_pdf(realish_pdf) is True


def test_arxiv_packager_records_file_provenance(tmp_path: Path) -> None:
    output_dir = tmp_path / "arxiv_source"
    archive = tmp_path / "arxiv_source.tar.gz"

    subprocess.run(
        [
            sys.executable,
            "scripts/package_arxiv_submission.py",
            "--output-dir",
            str(output_dir),
            "--archive",
            str(archive),
            "--allow-missing",
        ],
        check=True,
    )

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    provenance = manifest["copied_file_provenance"]

    assert archive.exists()
    assert any(row["kind"] == "latex_main" and row["direct_copy"] is False for row in provenance)
    assert any(row["source_path"] == "paper/references.bib" for row in provenance)
    assert all(not Path(row["bundle_path"]).is_absolute() for row in provenance)
    assert all(row.get("source_sha256") for row in provenance)
    assert all(row.get("bundle_sha256") for row in provenance)
    assert all(not Path(path).is_absolute() for path in manifest["copied_figures"])
    assert all(not Path(path).is_absolute() for path in manifest["copied_generated"])
    assert all(not Path(path).is_absolute() for path in manifest["copied_audit"])


def test_arxiv_packager_excludes_raw_evidence_from_support_trees(tmp_path: Path) -> None:
    source = tmp_path / "audit_source"
    bundle = tmp_path / "arxiv_source" / "audit" / "audit_source"
    source.mkdir()
    (source / "human_audit_summary_table.tex").write_text("table\n", encoding="utf-8")
    (source / "audit_labels.csv").write_text("label\n", encoding="utf-8")
    (source / "audit_key.jsonl").write_text('{"prompt": "hidden"}\n', encoding="utf-8")

    copied = _copy_arxiv_support_tree(source, bundle)

    assert copied == [source / "human_audit_summary_table.tex"]
    assert (bundle / "human_audit_summary_table.tex").exists()
    assert not (bundle / "audit_labels.csv").exists()
    assert not (bundle / "audit_key.jsonl").exists()


def test_arxiv_packager_rejects_placeholder_generated_tex(tmp_path: Path) -> None:
    generated = tmp_path / "generated"
    generated.mkdir(parents=True)
    placeholder = generated / "main_results_table.tex"
    valid = generated / "result_macros.tex"
    placeholder.write_text(
        "Results pending; no readiness-passing rows exported.\n",
        encoding="utf-8",
    )
    valid.write_text(r"\renewcommand{\PrimaryRunId}{h200_qwen_full_sweep}", encoding="utf-8")

    failures = _invalid_arxiv_support_files([placeholder, valid])

    assert f"{placeholder}:placeholder_text:Results pending; no readiness-passing rows exported." in failures


def test_arxiv_packager_rejects_semantically_incomplete_generated_tex(tmp_path: Path) -> None:
    generated = tmp_path / "generated" / "h200_qwen_full_sweep"
    generated.mkdir(parents=True)
    macros = generated / "result_macros.tex"
    table = generated / "main_results_table.tex"
    macros.write_text(r"\renewcommand{\PrimaryRunId}{h200_qwen_full_sweep}", encoding="utf-8")
    table.write_text("policy & estimate \\\\\n", encoding="utf-8")

    failures = _invalid_arxiv_support_files([macros, table])

    assert any("PrimaryTopSSEIPolicy" in failure for failure in failures)
    assert any("policy level ssei" in failure for failure in failures)


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
    assert placeholder_artifact_failures(tex) == [
        "invalid PDF artifact: figure.pdf",
        "missing artifact: missing/table.tex",
    ]


def test_latex_placeholder_checker_rejects_placeholder_artifacts(tmp_path: Path) -> None:
    tex = tmp_path / "main.tex"
    generated = tmp_path / "generated"
    generated.mkdir()
    pending = generated / "result_macros.tex"
    empty = generated / "claim_interpretation.tex"
    pending.write_text(
        r"\newcommand{\PrimaryTopSSEIPolicy}{Results pending; no readiness-passing rows exported.}",
        encoding="utf-8",
    )
    empty.write_text("", encoding="utf-8")
    tex.write_text(
        r"\requiredartifact{generated/result_macros.tex}"
        "\n"
        r"\maybeinputtable{generated/claim_interpretation.tex}{pending}",
        encoding="utf-8",
    )

    assert placeholder_artifact_failures(tex) == [
        "empty artifact: generated/claim_interpretation.tex",
        "placeholder text in artifact: generated/result_macros.tex",
    ]


def test_latex_placeholder_checker_requires_result_macro_values(tmp_path: Path) -> None:
    tex = tmp_path / "main.tex"
    generated = tmp_path / "h200_qwen_full_sweep"
    generated.mkdir()
    macros = generated / "result_macros.tex"
    macros.write_text(
        "\\renewcommand{\\PrimaryRunId}{h200_qwen_full_sweep}\n"
        "\\renewcommand{\\PrimaryPolicyCount}{6}\n"
        "\\renewcommand{\\PrimaryTopSSEIPolicy}{}\n",
        encoding="utf-8",
    )
    tex.write_text(
        r"\requiredartifact{h200_qwen_full_sweep/result_macros.tex}",
        encoding="utf-8",
    )

    failures = placeholder_artifact_failures(tex)

    assert (
        "missing required macro in artifact: "
        "h200_qwen_full_sweep/result_macros.tex::PrimaryTopSSEIPolicy"
    ) in failures
    assert (
        "missing required macro in artifact: "
        "h200_qwen_full_sweep/result_macros.tex::PrimaryTopSSEI"
    ) in failures


def test_latex_placeholder_checker_requires_ci_tables_and_causal_controls(
    tmp_path: Path,
) -> None:
    tex = tmp_path / "main.tex"
    suite = tmp_path / "suite_level_effects_table.tex"
    causal = tmp_path / "causal_restoration_table.tex"
    suite.write_text("suite & policy & paired n & cluster n \\\\\n", encoding="utf-8")
    causal.write_text(
        "safety ci low & safety ci high & refusal ci low & refusal ci high \\\\\n"
        "kv\\_int4\\_sim\\_\\_patchkey-value\\_\\_rolesystem \\\\\n",
        encoding="utf-8",
    )
    tex.write_text(
        r"\maybeinputtable{suite_level_effects_table.tex}{pending}"
        "\n"
        r"\maybeinputtable{causal_restoration_table.tex}{pending}",
        encoding="utf-8",
    )

    failures = placeholder_artifact_failures(tex)

    assert (
        "missing required table marker in artifact: suite_level_effects_table.tex::safety ci low"
        in failures
    )
    assert (
        "missing required table marker in artifact: suite_level_effects_table.tex::safety ci high"
        in failures
    )
    assert "missing causal control row in artifact: causal_restoration_table.tex::roleuser" in failures
    assert (
        "missing causal control row in artifact: causal_restoration_table.tex::policy_pinned"
        in failures
    )


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
    assert placeholder_artifact_failures(tex) == [
        "missing artifact: generated/missing_status.tex",
    ]
