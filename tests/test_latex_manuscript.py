import hashlib
import json
import re
import subprocess
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))

from check_final_pdf_text import (
    final_pdf_text_failures,
    forbidden_final_prose_failures,
    placeholder_text_failures,
)
from check_latex_citations import citation_failures
from check_latex_placeholders import missing_placeholder_artifacts, placeholder_artifact_failures
from export_paper_assets import export_paper_assets
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
from sync_active_paper_assets import sync_active_paper_assets


def test_latex_manuscript_is_formal_registered_protocol() -> None:
    tex = Path("paper/latex/main.tex").read_text(encoding="utf-8")

    assert r"\documentclass[11pt]{article}" in tex
    assert "Aryan Gupta" in tex
    assert "aryan.cs.app@gmail.com" in tex
    assert "registered analysis protocol" in tex
    assert "reports no empirical claims" in tex
    assert r"\EmpiricalStatusSentence" in tex
    assert r"\requiredartifact{../generated/active_primary/result_macros.tex}" in tex
    assert r"\requiredartifact{../generated/active_causal/result_macros.tex}" in tex
    assert r"\requiredartifact{../generated/claim_assessment/abstract_status_sentence.tex}" in tex
    assert "../generated/active_primary/result_macros.tex" in tex
    assert "../generated/active_causal/result_macros.tex" in tex
    assert "../generated/claim_assessment/abstract_status_sentence.tex" in tex
    assert "Evidence gate not satisfied" in tex
    assert r"\maybeinputtable{../generated/active_primary/main_results_table.tex}" in tex
    assert r"\maybeinputtable{../generated/claim_assessment/claim_assessment_table.tex}" in tex
    assert r"\maybeinputtable{../generated/claim_assessment/claim_interpretation.tex}" in tex
    assert r"\maybeinputtable{../audit/active_primary_summary/human_audit_summary_table.tex}" in tex
    assert r"\maybeinputtable{../audit/active_causal_summary/human_audit_summary_table.tex}" in tex
    assert r"\maybeinputtable{../audit/active_causal_summary/human_audit_deltas_table.tex}" in tex
    assert "causal_restoration_fraction.pdf" in tex
    assert "policy_uncertainty_braid.pdf" in tex
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
        "This registered analysis protocol reports no empirical claims. This draft "
        "manuscript must replace these placeholders. Result pending. Figure pending. "
        "Figure unavailable. The H200 launcher produced a smoke run on a MacBook GPU."
    )

    assert "placeholder_text:registered analysis protocol" in failures
    assert "placeholder_text:reports no empirical claims" in failures
    assert "placeholder_text:This draft" in failures
    assert "placeholder_text:draft manuscript" in failures
    assert "placeholder_text:must replace these placeholders" in failures
    assert "placeholder_text:Result pending" in failures
    assert "placeholder_text:Figure pending" in failures
    assert "placeholder_text:Figure unavailable" in failures
    assert "forbidden_final_prose:H200" in failures
    assert "forbidden_final_prose:launcher" in failures
    assert "forbidden_final_prose:smoke run" in failures
    assert "forbidden_final_prose:MacBook" in failures


def test_final_pdf_text_checker_rejects_internal_operational_language() -> None:
    failures = forbidden_final_prose_failures(
        "The evidence-gated fallback is draft-only because of a cgroup hardware constraint. "
        "The notebook allocation report included nvidia-smi, CUDA status, VRAM status, "
        "visible compute apps, infrastructure diagnostics, and a support bundle."
    )

    assert "forbidden_final_prose:evidence-gated fallback" in failures
    assert "forbidden_final_prose:draft-only" in failures
    assert "forbidden_final_prose:cgroup" in failures
    assert "forbidden_final_prose:hardware constraint" in failures
    assert "forbidden_final_prose:notebook allocation" in failures
    assert "forbidden_final_prose:nvidia-smi" in failures
    assert "forbidden_final_prose:CUDA operational status" in failures
    assert "forbidden_final_prose:VRAM operational status" in failures
    assert "forbidden_final_prose:visible compute apps" in failures
    assert "forbidden_final_prose:infrastructure diagnostics" in failures
    assert "forbidden_final_prose:support bundle" in failures


def test_final_pdf_text_checker_normalizes_escaped_operational_terms() -> None:
    failures = forbidden_final_prose_failures(
        r"The H\,200 launcher ran on a Mac\-Book. The G P U was busy."
    )

    assert "forbidden_final_prose:H200" in failures
    assert "forbidden_final_prose:launcher" in failures
    assert "forbidden_final_prose:MacBook" in failures
    assert "forbidden_final_prose:GPU operational status" in failures


def test_final_pdf_text_checker_rejects_unresolved_latex_markers() -> None:
    failures = final_pdf_text_failures(
        "Figure ?? summarizes the main effect. See Table ?? and [?] for prior work.",
        "synthetic",
    )

    assert "unresolved_reference:unresolved_latex_reference" in failures
    assert "unresolved_reference:unresolved_latex_citation" in failures


def test_final_pdf_text_checker_allows_relevant_cache_memory_prose() -> None:
    assert (
        forbidden_final_prose_failures(
            "Cache compression methods reduce key-value memory use during long-context decoding."
        )
        == []
    )
    assert (
        forbidden_final_prose_failures(
            "KV-cache compression can reduce GPU memory utilization in long-context decoding."
        )
        == []
    )


def test_final_pdf_text_checker_rejects_dirty_working_tree_language() -> None:
    failures = forbidden_final_prose_failures(
        "The artifact was generated from a dirty working tree and dirty git working tree."
    )

    assert failures.count("forbidden_final_prose:dirty tree") == 1


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


def test_latex_citations_and_bibliography_are_consistent() -> None:
    tex = Path("paper/latex/main.tex").read_text(encoding="utf-8")
    bib = Path("paper/references.bib").read_text(encoding="utf-8")
    bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    cited_keys = {
        key.strip()
        for citation in re.findall(r"\\cite[tp]\{([^}]+)\}", tex)
        for key in citation.split(",")
    }

    assert cited_keys <= bib_keys
    assert bib_keys <= cited_keys


def test_latex_citation_checker_rejects_missing_unused_and_duplicate_keys(
    tmp_path: Path,
) -> None:
    tex = tmp_path / "main.tex"
    bib = tmp_path / "references.bib"
    tex.write_text(
        "% ignored missing citation: \\citep{commented_missing}\n"
        r"\citep*{present,missing}",
        encoding="utf-8",
    )
    bib.write_text(
        "@article{present,\n  title={TBD},\n  year={2024},\n  url={https://example.com}\n}\n"
        "@article{unused,\n  title={Unused},\n  year={2024},\n  url={https://example.com}\n}\n"
        "@article{present,\n  title={TBD},\n  year={2025},\n  url={https://example.com}\n}\n",
        encoding="utf-8",
    )

    failures = citation_failures(tex, bib, require_all_bib_used=True)

    assert "missing_bib_entry:commented_missing" not in failures
    assert "missing_bib_entry:missing" in failures
    assert "unused_bib_entry:unused" in failures
    assert "duplicate_bib_key:present" in failures
    assert "bib_entry_placeholder_field:present:title" in failures


def test_latex_citation_checker_requires_support_locator(tmp_path: Path) -> None:
    tex = tmp_path / "main.tex"
    bib = tmp_path / "references.bib"
    tex.write_text(r"\citet{unsupported}", encoding="utf-8")
    bib.write_text(
        "@article{unsupported,\n  title={Unsupported Entry},\n  year={2024}\n}\n",
        encoding="utf-8",
    )

    assert citation_failures(tex, bib) == [
        "bib_entry_lacks_support_locator:unsupported"
    ]


def test_arxiv_rewrite_uses_local_bibliography_and_figures() -> None:
    source = (
        r"\maybeincludegraphic{../generated/active_primary/figures/"
        r"safety_capability_phase_portrait.pdf}{0.9\linewidth}{pending}"
        "\n"
        r"\bibliography{../references}"
    )

    rewritten = _rewrite_main_tex_for_arxiv(source)

    assert r"\bibliography{references}" in rewritten
    assert "figures/safety_capability_phase_portrait.pdf" in rewritten
    assert "figures/safety_capability_phase_portrait.pdf" in _rewrite_main_tex_for_arxiv(
        "../generated/active_primary/figures/safety_capability_phase_portrait.pdf"
    )
    assert "figures/prompt_effect_constellation.pdf" in _rewrite_main_tex_for_arxiv(
        "../../results/h200_qwen_full_sweep/figures/prompt_effect_constellation.pdf"
    )
    assert "figures/safety_state_atlas.pdf" in _rewrite_main_tex_for_arxiv(
        "../../results/h200_qwen_full_sweep/figures/safety_state_atlas.pdf"
    )
    assert "figures/policy_uncertainty_braid.pdf" in _rewrite_main_tex_for_arxiv(
        "../../results/h200_qwen_full_sweep/figures/policy_uncertainty_braid.pdf"
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
    assert "generated/active_primary" in _rewrite_main_tex_for_arxiv(
        "../generated/active_primary/main_results_table.tex"
    )
    assert "generated/claim_assessment" in _rewrite_main_tex_for_arxiv(
        "../generated/claim_assessment/claim_assessment_table.tex"
    )
    assert "generated/claim_assessment" in _rewrite_main_tex_for_arxiv(
        "../generated/claim_assessment/abstract_status_sentence.tex"
    )
    assert "generated/active_primary" in _rewrite_main_tex_for_arxiv(
        "../generated/active_primary/result_macros.tex"
    )
    assert "generated/active_causal" in _rewrite_main_tex_for_arxiv(
        "../generated/active_causal/result_macros.tex"
    )
    assert Path("paper/generated/active_primary") in REQUIRED_GENERATED_DIRS
    assert Path("paper/generated/active_causal") in REQUIRED_GENERATED_DIRS
    assert Path("paper/generated/claim_assessment") in GENERATED_DIRS
    assert Path("paper/generated/claim_assessment") in REQUIRED_GENERATED_DIRS
    assert Path("paper/generated/h200_qwen32b_public_followup") in OPTIONAL_GENERATED_DIRS
    assert "audit/active_primary_summary" in _rewrite_main_tex_for_arxiv(
        "../audit/active_primary_summary/human_audit_summary_table.tex"
    )
    assert "audit/active_causal_summary" in _rewrite_main_tex_for_arxiv(
        "../audit/active_causal_summary/human_audit_summary_table.tex"
    )
    assert "../../results" not in rewritten


def test_arxiv_packager_can_target_custom_result_figure_dirs(tmp_path: Path) -> None:
    primary = tmp_path / "primary_run"
    causal = tmp_path / "causal_run"
    figure_sources = build_figure_sources(primary, causal)

    assert figure_sources["safety_state_atlas.pdf"] == (
        primary / "figures" / "safety_state_atlas.pdf"
    )
    assert figure_sources["policy_uncertainty_braid.pdf"] == (
        primary / "figures" / "policy_uncertainty_braid.pdf"
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


def test_active_paper_asset_sync_copies_selected_sources(tmp_path: Path) -> None:
    primary_results = tmp_path / "results" / "merged_primary"
    causal_results = tmp_path / "results" / "causal"
    primary_generated = tmp_path / "generated" / "merged_primary"
    causal_generated = tmp_path / "generated" / "causal"
    primary_audit = tmp_path / "audit" / "merged_primary_summary"
    causal_audit = tmp_path / "audit" / "causal_summary"
    active_primary = tmp_path / "generated" / "active_primary"
    active_causal = tmp_path / "generated" / "active_causal"
    active_primary_audit = tmp_path / "audit" / "active_primary_summary"
    active_causal_audit = tmp_path / "audit" / "active_causal_summary"
    for path in [
        primary_generated / "result_macros.tex",
        primary_generated / "main_results_table.tex",
        primary_generated / "suite_level_effects_table.tex",
        causal_generated / "result_macros.tex",
        causal_generated / "causal_restoration_table.tex",
        primary_audit / "audit_manifest.json",
        primary_audit / "human_audit_summary.json",
        primary_audit / "human_audit_summary.md",
        primary_audit / "human_audit_summary_table.tex",
        primary_audit / "human_audit_deltas_table.tex",
        causal_audit / "audit_manifest.json",
        causal_audit / "human_audit_summary.json",
        causal_audit / "human_audit_summary.md",
        causal_audit / "human_audit_summary_table.tex",
        causal_audit / "human_audit_deltas_table.tex",
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{path.name}\n", encoding="utf-8")
    for path in [
        primary_results / "figures" / "safety_capability_phase_portrait.pdf",
        primary_results / "figures" / "selective_safety_erasure_heatmap.pdf",
        primary_results / "figures" / "prompt_effect_constellation.pdf",
        primary_results / "figures" / "cache_state_fingerprint.pdf",
        primary_results / "figures" / "safety_state_atlas.pdf",
        primary_results / "figures" / "policy_uncertainty_braid.pdf",
        causal_results / "figures" / "causal_restoration_fraction.pdf",
        causal_results / "figures" / "causal_restoration_flow.pdf",
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"%PDF-1.7\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF\n")

    missing = sync_active_paper_assets(
        primary_results_dir=primary_results,
        causal_results_dir=causal_results,
        primary_generated_dir=primary_generated,
        causal_generated_dir=causal_generated,
        primary_audit_dir=primary_audit,
        causal_audit_dir=causal_audit,
        active_primary_dir=active_primary,
        active_causal_dir=active_causal,
        active_primary_audit_dir=active_primary_audit,
        active_causal_audit_dir=active_causal_audit,
    )

    assert missing == []
    assert (active_primary / "main_results_table.tex").read_text(encoding="utf-8") == (
        "main_results_table.tex\n"
    )
    assert (active_primary / "figures" / "safety_state_atlas.pdf").exists()
    assert (active_primary / "figures" / "policy_uncertainty_braid.pdf").exists()
    manifest = json.loads(
        (active_primary / "active_asset_manifest.json").read_text(encoding="utf-8")
    )
    copied_main = next(
        row for row in manifest["copied"] if row["target"].endswith("main_results_table.tex")
    )
    assert manifest["schema_version"] == 2
    assert manifest["results_dir"] == str(primary_results)
    assert copied_main["source_sha256"] == copied_main["target_sha256"]
    assert copied_main["sha256"] == copied_main["target_sha256"]
    assert copied_main["source_bytes"] == copied_main["target_bytes"]
    assert copied_main["bytes"] == copied_main["target_bytes"]
    assert (
        active_primary_audit / "human_audit_summary_table.tex"
    ).read_text(encoding="utf-8") == "human_audit_summary_table.tex\n"
    audit_manifest = json.loads(
        (active_primary_audit / "active_audit_manifest.json").read_text(encoding="utf-8")
    )
    audit_table = next(
        row
        for row in audit_manifest["copied"]
        if row["target"].endswith("human_audit_summary_table.tex")
    )
    assert audit_manifest["schema_version"] == 2
    assert audit_manifest["audit_dir"] == str(primary_audit)
    assert audit_table["source_sha256"] == audit_table["target_sha256"]


def test_active_paper_asset_sync_strict_rejects_stale_generated_sources(
    tmp_path: Path,
) -> None:
    primary_results = tmp_path / "results" / "merged_primary"
    causal_results = tmp_path / "results" / "causal"
    primary_generated = tmp_path / "generated" / "merged_primary"
    causal_generated = tmp_path / "generated" / "causal"
    active_primary = tmp_path / "generated" / "active_primary"
    active_causal = tmp_path / "generated" / "active_causal"
    _write_active_sync_fixture(
        primary_results,
        primary_generated,
        generated_names=[
            "result_macros.tex",
            "main_results_table.tex",
            "suite_level_effects_table.tex",
        ],
        figure_names=[
            "safety_capability_phase_portrait.pdf",
            "selective_safety_erasure_heatmap.pdf",
            "prompt_effect_constellation.pdf",
            "cache_state_fingerprint.pdf",
            "safety_state_atlas.pdf",
            "policy_uncertainty_braid.pdf",
        ],
    )
    _write_active_sync_fixture(
        causal_results,
        causal_generated,
        generated_names=["result_macros.tex", "causal_restoration_table.tex"],
        figure_names=["causal_restoration_fraction.pdf", "causal_restoration_flow.pdf"],
    )
    (primary_results / "metrics.json").write_text('{"changed": true}\n', encoding="utf-8")
    active_primary.mkdir(parents=True)
    (active_primary / "old_table.tex").write_text("stale active artifact\n", encoding="utf-8")

    failures = sync_active_paper_assets(
        primary_results_dir=primary_results,
        causal_results_dir=causal_results,
        primary_generated_dir=primary_generated,
        causal_generated_dir=causal_generated,
        active_primary_dir=active_primary,
        active_causal_dir=active_causal,
        validate_sources=True,
    )

    assert any(
        "primary:paper artifact source `metrics.json` hash is stale" in failure
        for failure in failures
    )
    assert (active_primary / "old_table.tex").read_text(encoding="utf-8") == (
        "stale active artifact\n"
    )


def test_active_paper_asset_sync_strict_recomputes_generated_tables(
    tmp_path: Path,
) -> None:
    primary_results = tmp_path / "results" / "merged_primary"
    causal_results = tmp_path / "results" / "causal"
    primary_generated = tmp_path / "generated" / "merged_primary"
    causal_generated = tmp_path / "generated" / "causal"
    active_primary = tmp_path / "generated" / "active_primary"
    active_causal = tmp_path / "generated" / "active_causal"
    _write_active_sync_fixture(
        primary_results,
        primary_generated,
        generated_names=[
            "result_macros.tex",
            "main_results_table.tex",
            "suite_level_effects_table.tex",
        ],
        figure_names=[
            "safety_capability_phase_portrait.pdf",
            "selective_safety_erasure_heatmap.pdf",
            "prompt_effect_constellation.pdf",
            "cache_state_fingerprint.pdf",
            "safety_state_atlas.pdf",
            "policy_uncertainty_braid.pdf",
        ],
    )
    _write_active_sync_fixture(
        causal_results,
        causal_generated,
        generated_names=["result_macros.tex", "causal_restoration_table.tex"],
        figure_names=["causal_restoration_fraction.pdf", "causal_restoration_flow.pdf"],
    )
    stale_table = primary_generated / "main_results_table.tex"
    stale_table.write_text("internally hash-consistent but stale table\n", encoding="utf-8")
    manifest_path = primary_generated / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tables"]["main_results_table.tex"]["sha256"] = _sha256(stale_table)
    manifest["tables"]["main_results_table.tex"]["bytes"] = stale_table.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    failures = sync_active_paper_assets(
        primary_results_dir=primary_results,
        causal_results_dir=causal_results,
        primary_generated_dir=primary_generated,
        causal_generated_dir=causal_generated,
        active_primary_dir=active_primary,
        active_causal_dir=active_causal,
        validate_sources=True,
    )

    assert (
        "primary:paper artifact generated output `main_results_table.tex` differs from metrics export"
        in failures
    )
    assert not active_primary.exists()


def test_active_paper_asset_sync_strict_requires_manifest_pinned_tables_and_figures(
    tmp_path: Path,
) -> None:
    primary_results = tmp_path / "results" / "merged_primary"
    causal_results = tmp_path / "results" / "causal"
    primary_generated = tmp_path / "generated" / "merged_primary"
    causal_generated = tmp_path / "generated" / "causal"
    primary_audit = tmp_path / "audit" / "merged_primary_summary"
    causal_audit = tmp_path / "audit" / "causal_summary"
    active_primary = tmp_path / "generated" / "active_primary"
    active_causal = tmp_path / "generated" / "active_causal"
    _write_active_sync_fixture(
        primary_results,
        primary_generated,
        generated_names=[
            "result_macros.tex",
            "main_results_table.tex",
            "suite_level_effects_table.tex",
        ],
        figure_names=[
            "safety_capability_phase_portrait.pdf",
            "selective_safety_erasure_heatmap.pdf",
            "prompt_effect_constellation.pdf",
            "cache_state_fingerprint.pdf",
            "safety_state_atlas.pdf",
            "policy_uncertainty_braid.pdf",
        ],
    )
    _write_active_sync_fixture(
        causal_results,
        causal_generated,
        generated_names=["result_macros.tex", "causal_restoration_table.tex"],
        figure_names=["causal_restoration_fraction.pdf", "causal_restoration_flow.pdf"],
    )
    _write_active_sync_audit_fixture(primary_audit)
    _write_active_sync_audit_fixture(causal_audit)
    manifest_path = primary_generated / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["tables"]["main_results_table.tex"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (primary_results / "figures" / "safety_state_atlas.pdf").write_bytes(
        b"%PDF-1.7\nchanged\n%%EOF\n"
    )

    failures = sync_active_paper_assets(
        primary_results_dir=primary_results,
        causal_results_dir=causal_results,
        primary_generated_dir=primary_generated,
        causal_generated_dir=causal_generated,
        primary_audit_dir=primary_audit,
        causal_audit_dir=causal_audit,
        active_primary_dir=active_primary,
        active_causal_dir=active_causal,
        validate_sources=True,
    )

    assert "primary:paper artifact manifest lacks synced table:main_results_table.tex" in failures
    assert "primary:figures manifest synced PDF hash stale:safety_state_atlas.pdf" in failures
    assert not active_primary.exists()
    assert not active_causal.exists()


def test_arxiv_packager_rejects_malformed_figure_pdfs(tmp_path: Path) -> None:
    fake_pdf = tmp_path / "figure.pdf"
    realish_pdf = tmp_path / "realish.pdf"
    fake_pdf.write_text("not a pdf", encoding="utf-8")
    realish_pdf.write_bytes(b"%PDF-1.7\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF\n")

    assert _is_pdf(fake_pdf) is False
    assert _is_pdf(realish_pdf) is False


def _write_active_sync_fixture(
    results_dir: Path,
    generated_dir: Path,
    *,
    generated_names: list[str],
    figure_names: list[str],
) -> None:
    (results_dir / "figures").mkdir(parents=True)
    generated_dir.mkdir(parents=True)
    (results_dir / "manifest.json").write_text(
        '{"git_commit": "run-commit", "git_dirty": false}\n',
        encoding="utf-8",
    )
    metrics = _active_sync_metrics(causal="causal" in generated_dir.name)
    (results_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    for name in figure_names:
        (results_dir / "figures" / name).write_bytes(
            b"%PDF-1.7\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF\n"
        )
    figure_manifest = {
        "figures": [
            {
                "name": Path(name).stem,
                "pdf": str(results_dir / "figures" / name),
                "pdf_sha256": _sha256(results_dir / "figures" / name),
            }
            for name in figure_names
        ]
    }
    (results_dir / "figures" / "manifest.json").write_text(
        json.dumps(figure_manifest),
        encoding="utf-8",
    )
    export_paper_assets(
        results_dir,
        generated_dir,
        macro_prefix="Causal" if "causal" in generated_dir.name else "Primary",
    )
    artifact_manifest = json.loads(
        (generated_dir / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    artifact_manifest["analysis_git_commit"] = "analysis-commit"
    artifact_manifest["analysis_git_dirty"] = False
    artifact_manifest["analysis_git_status_short"] = ""
    (generated_dir / "artifact_manifest.json").write_text(
        json.dumps(artifact_manifest),
        encoding="utf-8",
    )


def _active_sync_metrics(*, causal: bool) -> dict:
    if causal:
        return {
            "causal_restoration": {
                "public_refusal_safety::kv_int4_sim__patchkey-value__rolesystem": {
                    "compressed_policy": "kv_int4_sim",
                    "safety_restoration_fraction": 0.62,
                    "safety_restoration_fraction_ci": {"ci_low": 0.56, "ci_high": 0.67},
                    "refusal_restoration_fraction": 0.55,
                    "refusal_restoration_fraction_ci": {"ci_low": 0.50, "ci_high": 0.60},
                }
            }
        }
    return {
        "publication_summary": {
            "policies": {
                "kv_int4_sim": {
                    "mean_safety_score": 0.72,
                    "mean_capability_score": 0.88,
                    "global_safety_degradation": 0.14,
                    "global_capability_degradation": 0.03,
                    "global_selective_safety_erasure_index": 0.11,
                }
            }
        },
        "policy_level_contrasts": {
            "kv_int4_sim": {
                "selective_safety_erasure_index": 0.11,
                "selective_safety_erasure_index_ci": {
                    "ci_low": 0.08,
                    "ci_high": 0.14,
                    "n_safety": 650,
                    "n_capability": 650,
                },
            }
        },
        "selective_safety_erasure": {
            "public_refusal_safety::kv_int4_sim": {
                "safety_degradation": 0.14,
                "capability_degradation": 0.03,
                "selective_safety_erasure_index": 0.11,
                "paired_safety_degradation_ci": {
                    "paired_n": 650,
                    "cluster_n": 650,
                    "ci_low": 0.08,
                    "ci_high": 0.14,
                },
            }
        },
    }


def _write_active_sync_audit_fixture(audit_dir: Path) -> None:
    audit_dir.mkdir(parents=True)
    for name in [
        "audit_manifest.json",
        "human_audit_summary.json",
        "human_audit_summary.md",
        "human_audit_summary_table.tex",
        "human_audit_deltas_table.tex",
    ]:
        (audit_dir / name).write_text(f"{name}\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
            *_isolated_arxiv_missing_args(tmp_path),
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
    with tarfile.open(archive, "r:gz") as tar:
        member_names = [member.name for member in tar.getmembers()]
    assert len(member_names) == len(set(member_names))


def test_arxiv_packager_copies_optional_qwen32_only_when_requested(tmp_path: Path) -> None:
    output_dir = tmp_path / "arxiv_source"
    archive = tmp_path / "arxiv_source.tar.gz"
    qwen32_dir = tmp_path / "h200_qwen32b_public_followup"
    qwen32_dir.mkdir()
    (qwen32_dir / "qwen32_note.tex").write_text("Qwen 32B follow-up note.\n", encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/package_arxiv_submission.py",
            "--output-dir",
            str(output_dir),
            "--archive",
            str(archive),
            *_isolated_arxiv_missing_args(tmp_path),
            "--allow-missing",
        ],
        check=True,
    )
    manifest_without_optional = json.loads(
        (output_dir / "manifest.json").read_text(encoding="utf-8")
    )

    assert all("h200_qwen32b_public_followup" not in path for path in manifest_without_optional["copied_generated"])

    subprocess.run(
        [
            sys.executable,
            "scripts/package_arxiv_submission.py",
            "--output-dir",
            str(output_dir),
            "--archive",
            str(archive),
            *_isolated_arxiv_missing_args(tmp_path),
            "--qwen32-generated-dir",
            str(qwen32_dir),
            "--allow-missing",
        ],
        check=True,
    )
    manifest_with_optional = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

    assert "generated/h200_qwen32b_public_followup" in manifest_with_optional["copied_generated"]
    assert (output_dir / "generated" / "h200_qwen32b_public_followup" / "qwen32_note.tex").exists()


def _isolated_arxiv_missing_args(tmp_path: Path) -> list[str]:
    isolated = tmp_path / "missing_inputs"
    return [
        "--primary-generated-dir",
        str(isolated / "generated_primary"),
        "--causal-generated-dir",
        str(isolated / "generated_causal"),
        "--claim-generated-dir",
        str(isolated / "generated_claim"),
        "--primary-audit-dir",
        str(isolated / "audit_primary"),
        "--causal-audit-dir",
        str(isolated / "audit_causal"),
    ]


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


def test_arxiv_packager_rejects_internal_operational_generated_tex(tmp_path: Path) -> None:
    generated = tmp_path / "generated"
    generated.mkdir(parents=True)
    artifact = generated / "claim_interpretation.tex"
    artifact.write_text("The H200 finalizer generated this draft-only text.", encoding="utf-8")

    failures = _invalid_arxiv_support_files([artifact])

    assert f"{artifact}:forbidden_final_prose:H200" in failures
    assert f"{artifact}:forbidden_final_prose:finalizer" in failures
    assert f"{artifact}:forbidden_final_prose:draft-only" in failures


def test_arxiv_packager_ignores_internal_operational_tex_comments(tmp_path: Path) -> None:
    generated = tmp_path / "generated"
    generated.mkdir(parents=True)
    artifact = generated / "claim_interpretation.tex"
    artifact.write_text(
        "% H200 finalizer draft-only\n"
        "% Results pending; no readiness-passing rows exported.\n"
        "The results support the registered causal restoration claim.",
        encoding="utf-8",
    )

    assert _invalid_arxiv_support_files([artifact]) == []


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


def test_arxiv_packager_ignores_comments_for_semantic_markers(tmp_path: Path) -> None:
    generated = tmp_path / "generated" / "h200_qwen_full_sweep"
    generated.mkdir(parents=True)
    table = generated / "main_results_table.tex"
    table.write_text(
        "% policy level ssei\n"
        "% policy level ssei ci low\n"
        "% policy level ssei ci high\n"
        "policy & estimate \\\\\n",
        encoding="utf-8",
    )

    failures = _invalid_arxiv_support_files([table])

    assert f"{table}:missing required table marker in artifact: {table}::policy level ssei" in failures


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


def test_latex_placeholder_checker_rejects_blank_pdf_figures(tmp_path: Path) -> None:
    from pypdf import PdfWriter

    tex = tmp_path / "main.tex"
    blank = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=144)
    with blank.open("wb") as f:
        writer.write(f)
    tex.write_text(
        r"\maybeincludegraphic{blank.pdf}{0.9\linewidth}{ok}",
        encoding="utf-8",
    )

    assert placeholder_artifact_failures(tex) == ["invalid PDF artifact: blank.pdf"]


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


def test_latex_placeholder_checker_rejects_internal_generated_prose(tmp_path: Path) -> None:
    generated = tmp_path / "generated"
    generated.mkdir()
    artifact = generated / "claim_interpretation.tex"
    artifact.write_text("The H200 finalizer produced a draft-only fallback.", encoding="utf-8")
    tex = tmp_path / "main.tex"
    tex.write_text(
        r"\maybeinputtable{generated/claim_interpretation.tex}",
        encoding="utf-8",
    )

    assert placeholder_artifact_failures(tex) == [
        (
            "forbidden final prose in artifact: generated/claim_interpretation.tex::"
            "forbidden_final_prose:H200"
        ),
        (
            "forbidden final prose in artifact: generated/claim_interpretation.tex::"
            "forbidden_final_prose:draft-only"
        ),
        (
            "forbidden final prose in artifact: generated/claim_interpretation.tex::"
            "forbidden_final_prose:finalizer"
        ),
    ]


def test_latex_placeholder_checker_ignores_nonrendered_internal_comments(
    tmp_path: Path,
) -> None:
    generated = tmp_path / "generated"
    generated.mkdir()
    artifact = generated / "claim_interpretation.tex"
    artifact.write_text(
        "% H200 finalizer draft-only\n"
        "% Results pending; no readiness-passing rows exported.\n"
        "The causal results show restored refusal behavior after targeted cache repair.",
        encoding="utf-8",
    )
    tex = tmp_path / "main.tex"
    tex.write_text(
        r"\maybeinputtable{generated/claim_interpretation.tex}",
        encoding="utf-8",
    )

    assert placeholder_artifact_failures(tex) == []


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


def test_latex_placeholder_checker_ignores_comments_for_required_macros(tmp_path: Path) -> None:
    tex = tmp_path / "main.tex"
    generated = tmp_path / "h200_qwen_full_sweep"
    generated.mkdir()
    macros = generated / "result_macros.tex"
    macros.write_text(
        "% \\renewcommand{\\PrimaryRunId}{h200_qwen_full_sweep}\n"
        "% \\renewcommand{\\PrimaryPolicyCount}{6}\n",
        encoding="utf-8",
    )
    tex.write_text(
        r"\requiredartifact{h200_qwen_full_sweep/result_macros.tex}",
        encoding="utf-8",
    )

    failures = placeholder_artifact_failures(tex)

    assert (
        "missing required macro in artifact: "
        "h200_qwen_full_sweep/result_macros.tex::PrimaryRunId"
    ) in failures
    assert (
        "missing required macro in artifact: "
        "h200_qwen_full_sweep/result_macros.tex::PrimaryPolicyCount"
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
