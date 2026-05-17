from __future__ import annotations

import argparse
import csv
import json
import math
import re
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any

from _path import add_src_to_path

add_src_to_path()

from cache_safety_erasure.utils.io import file_sha256, write_json

ROLE_ORDER = {
    "system": 0,
    "hidden_system": 1,
    "template": 2,
    "user": 3,
    "generated": 4,
    "special": 5,
    "unknown": 6,
}

# Pastel orange color palette
C_PRIMARY = "#FFB347"      # main pastel orange
C_DARK = "#E8943A"         # darker accent
C_LIGHT = "#FFD699"        # lighter fill
C_MUTED = "#F5C28A"        # muted variant
C_ACCENT = "#FF8C42"       # stronger accent for emphasis


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate figures for a run.")
    parser.add_argument("--results-dir", required=True, type=Path)
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise SystemExit("Install dependencies with `uv sync --extra dev` to make figures.") from exc
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["svg.hashsalt"] = "cache-safety-erasure"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    generations_path = args.results_dir / "generations.jsonl"
    if not generations_path.exists():
        raise SystemExit(f"Missing generations file: {generations_path}")
    df = pd.read_json(generations_path, lines=True)
    figures_dir = args.results_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    made: list[dict[str, Any]] = []
    constellation_rows = _prompt_effect_constellation_rows(df)
    if constellation_rows:
        constellation_df = pd.DataFrame(constellation_rows)
        summary_df = (
            constellation_df.groupby(["suite", "policy"], dropna=False)["effect_magnitude"]
            .agg(mean_effect="mean", max_effect="max", prompt_count="count")
            .reset_index()
        )
        top = (
            summary_df.sort_values("mean_effect", ascending=True)
            .tail(18)
            .reset_index(drop=True)
        )
        fig_height = max(5.6, 0.34 * len(top))
        fig, ax = plt.subplots(figsize=(11, fig_height))
        suite_order = list(dict.fromkeys(top["suite"].tolist()))
        from matplotlib.colors import LinearSegmentedColormap
        orange_shades = [C_LIGHT, C_PRIMARY, C_DARK, C_ACCENT, C_MUTED]
        suite_colors = {suite: orange_shades[index % len(orange_shades)] for index, suite in enumerate(suite_order)}
        labels = [
            _wrap_label(
                f"{_clean_suite_label(row.suite)} | {_clean_policy_label(row.policy)}",
                width=42,
            )
            for row in top.itertuples()
        ]
        y = list(range(len(top)))
        ax.barh(
            y,
            top["mean_effect"],
            color=[suite_colors[row.suite] for row in top.itertuples()],
            edgecolor="white",
            linewidth=0.7,
        )
        ax.set_yticks(y, labels=labels)
        ax.set_xlim(0, max(1.0, float(top["mean_effect"].max()) * 1.08))
        ax.set_xlabel("Mean prompt-level behavior change (larger = farther from baseline)")
        ax.set_title("Which suites and policies changed behavior most?")
        handles = [
            plt.Line2D([0], [0], marker="s", linestyle="", color=color, label=_clean_suite_label(suite))
            for suite, color in suite_colors.items()
        ]
        if handles:
            _legend_below(ax, handles=handles, title="Prompt suite", ncol=min(3, len(handles)))
        ax.grid(axis="x", alpha=0.22)
        fig.tight_layout()
        _save_figure(
            fig,
            figures_dir,
            "prompt_effect_constellation",
            made,
            data_rows=summary_df.to_dict(orient="records"),
        )
        plt.close(fig)

    for metric in ["safety_score", "capability_score", "rouge_l_leakage_recall"]:
        if metric not in df or df[metric].dropna().empty:
            continue
        grouped = df.groupby(["suite", "policy"], dropna=False)[metric].mean().reset_index()
        policy_order = list(dict.fromkeys(grouped["policy"].tolist()))
        policy_lookup = {policy: idx for idx, policy in enumerate(policy_order)}
        fig, ax = plt.subplots(figsize=(10, 5))
        for suite, suite_df in grouped.groupby("suite"):
            suite_df = suite_df.assign(policy_x=suite_df["policy"].map(policy_lookup)).sort_values("policy_x")
            ax.plot(
                suite_df["policy_x"],
                suite_df[metric],
                marker="o",
                label=_clean_suite_label(suite),
            )
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylabel(metric)
        ax.set_xlabel("Cache policy")
        ax.set_xticks(
            range(len(policy_order)),
            labels=[_wrap_label(_clean_policy_label(policy), 18) for policy in policy_order],
        )
        ax.tick_params(axis="x", labelrotation=0)
        _legend_below(ax, title="Prompt suite", ncol=3, y_anchor=-0.32)
        fig.tight_layout()
        _save_figure(
            fig,
            figures_dir,
            metric,
            made,
            data_rows=grouped.to_dict(orient="records"),
        )
        plt.close(fig)

    metrics_path = args.results_dir / "metrics.json"
    selective_rows_for_atlas: list[dict[str, Any]] = []
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        selective_rows = _selective_rows_for_figures(metrics)
        selective_rows_for_atlas = selective_rows
        if selective_rows:
            selective_df = pd.DataFrame(selective_rows)
            pivot = selective_df.pivot_table(
                index="suite", columns="policy", values="index", aggfunc="mean"
            )
            if len(pivot.index) <= 1:
                bar_df = selective_df.sort_values("index", ascending=True).reset_index(drop=True)
                fig_height = max(4.8, 0.42 * len(bar_df))
                ssei_x_limits = _one_axis_cluster_limits(bar_df["index"])
                if ssei_x_limits is None:
                    fig, ax = plt.subplots(figsize=(10.5, fig_height))
                    _plot_selective_ssei_bars_panel(
                        ax,
                        bar_df,
                        title="Selective safety loss by cache policy",
                    )
                else:
                    fig, axes = plt.subplots(
                        1,
                        2,
                        figsize=(14, fig_height),
                        sharey=True,
                        gridspec_kw={"width_ratios": [0.95, 1.15]},
                    )
                    ax, zoom_ax = axes
                    _plot_selective_ssei_bars_panel(ax, bar_df, title="All policies")
                    _plot_selective_ssei_bars_panel(
                        zoom_ax,
                        bar_df,
                        title="Zoomed central cluster",
                        xlim=ssei_x_limits,
                        show_yticklabels=False,
                    )
                    fig.suptitle("Selective safety loss by cache policy", y=0.98)
            else:
                fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(pivot.columns)), 4.8))
                max_abs = max(0.02, float(pivot.abs().max().max()))
                im = ax.imshow(
                    pivot.fillna(0.0).values,
                    aspect="auto",
                    cmap="Oranges",
                    vmin=-max_abs,
                    vmax=max_abs,
                )
                ax.set_xticks(
                    range(len(pivot.columns)),
                    labels=[_wrap_label(_clean_policy_label(policy), 16) for policy in pivot.columns],
                    rotation=0,
                    ha="center",
                )
                ax.set_yticks(range(len(pivot.index)), labels=[_clean_suite_label(suite) for suite in pivot.index])
                ax.set_title("Selective safety loss by cache policy")
                fig.colorbar(im, ax=ax, label="SSeI: safety loss - capability loss")
            fig.tight_layout()
            _save_figure(
                fig,
                figures_dir,
                "selective_safety_erasure_heatmap",
                made,
                data_rows=selective_df.to_dict(orient="records"),
            )
            plt.close(fig)

            plot_df = selective_df.dropna(subset=["safety_degradation"]).copy()
            plot_df["capability_degradation"] = plot_df["capability_degradation"].fillna(0.0)
            plot_df = plot_df.reset_index(drop=True)
            plot_df["x_plot"] = plot_df["capability_degradation"] + (
                (plot_df.index % 5) - 2
            ) * 0.002
            plot_df["y_plot"] = plot_df["safety_degradation"] + (
                ((plot_df.index // 5) % 5) - 2
            ) * 0.002
            zoom_limits = _cluster_zoom_limits(plot_df, x_column="x_plot", y_column="y_plot")
            if zoom_limits is None:
                fig, ax = plt.subplots(figsize=(8.5, 6))
                _plot_safety_capability_panel(
                    ax,
                    plot_df,
                    title="Safety vs Capability Degradation",
                    annotate_count=8,
                )
            else:
                fig, axes = plt.subplots(
                    1,
                    2,
                    figsize=(13.2, 6),
                    gridspec_kw={"width_ratios": [0.95, 1.15]},
                )
                ax, zoom_ax = axes
                _plot_safety_capability_panel(
                    ax,
                    plot_df,
                    title="All policies",
                    annotate_count=5,
                )
                _plot_safety_capability_panel(
                    zoom_ax,
                    plot_df,
                    title="Zoomed central cluster",
                    annotate_count=8,
                    xlim=zoom_limits[0],
                    ylim=zoom_limits[1],
                    show_ylabel=False,
                )
                ax.set_title("All policies")
                zoom_ax.set_title("Zoomed central cluster")
                fig.suptitle("Safety vs Capability Degradation", y=0.98)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                _legend_below(ax, handles=handles, labels=labels, title="Prompt suite")
            fig.tight_layout()
            _save_figure(
                fig,
                figures_dir,
                "safety_vs_capability_degradation",
                made,
                data_rows=plot_df.to_dict(orient="records"),
            )
            plt.close(fig)

            phase_df = _phase_portrait_rows(selective_df)
            if not phase_df.empty:
                import numpy as np

                phase_df = phase_df.sort_values("selective_safety_erasure_index", ascending=False)
                labels = [_clean_policy_label(p) for p in phase_df["policy"]]
                safety_vals = phase_df["safety_degradation"].values
                capability_vals = phase_df["capability_degradation"].values

                n = len(labels)
                fig, ax = plt.subplots(figsize=(8, max(4, 0.45 * n)))
                y_pos = np.arange(n)
                bar_h = 0.35

                ax.barh(y_pos - bar_h / 2, safety_vals, bar_h,
                        color=C_ACCENT, label="Safety loss", zorder=2)
                ax.barh(y_pos + bar_h / 2, capability_vals, bar_h,
                        color=C_LIGHT, label="Capability loss", zorder=2)
                ax.axvline(0, color="0.15", linewidth=0.9, zorder=1)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(labels, fontsize=9)
                ax.set_xlabel("Degradation from baseline (higher = worse)")
                ax.set_title("Safety vs Capability Degradation by Policy")
                _legend_below(ax, ncol=2)
                ax.grid(axis="x", alpha=0.25, zorder=0)
                fig.tight_layout()
                _save_figure(
                    fig,
                    figures_dir,
                    "safety_capability_phase_portrait",
                    made,
                    data_rows=phase_df.to_dict(orient="records"),
                )
                plt.close(fig)

            forest_rows = _paired_safety_forest_rows(metrics)
            if forest_rows:
                forest_df = pd.DataFrame(forest_rows).sort_values("mean")
                fig_height = max(4, 0.35 * len(forest_df))
                fig, ax = plt.subplots(figsize=(9, fig_height))
                y = range(len(forest_df))
                xerr = [
                    forest_df["mean"] - forest_df["ci_low"],
                    forest_df["ci_high"] - forest_df["mean"],
                ]
                ax.errorbar(forest_df["mean"], y, xerr=xerr, fmt="o", capsize=3)
                ax.axvline(0, color="black", linewidth=0.8)
                ax.set_yticks(
                    list(y),
                    labels=[label.replace(" / ", "\n") for label in forest_df["label"]],
                )
                ax.set_xlabel("Paired safety degradation")
                ax.set_title("Paired Safety Degradation With Prompt-Clustered CIs")
                fig.tight_layout()
                _save_figure(
                    fig,
                    figures_dir,
                    "paired_safety_degradation_forest",
                    made,
                    data_rows=forest_rows,
                )
                plt.close(fig)

            braid_rows = _policy_uncertainty_braid_rows(metrics)
            if braid_rows:
                braid_df = pd.DataFrame(braid_rows)
                policy_order = (
                    braid_df[braid_df["metric"] == "ssei"]
                    .sort_values("mean")
                    ["policy"]
                    .tolist()
                )
                y_lookup = {policy: idx for idx, policy in enumerate(policy_order)}
                metric_offsets = {"capability": -0.22, "safety": 0.0, "ssei": 0.22}
                metric_colors = {
                    "capability": C_LIGHT,
                    "safety": C_ACCENT,
                    "ssei": C_DARK,
                }
                fig_height = max(4.5, 0.46 * len(policy_order))
                braid_x_limits = _one_axis_cluster_limits(
                    braid_df[["ci_low", "mean", "ci_high"]].melt(value_name="value")["value"]
                )
                if braid_x_limits is None:
                    fig, ax = plt.subplots(figsize=(10, fig_height))
                    _plot_policy_uncertainty_braid_panel(
                        ax,
                        braid_df,
                        policy_order,
                        y_lookup,
                        metric_offsets,
                        metric_colors,
                        title="Do policy effects separate from zero?",
                    )
                else:
                    fig, axes = plt.subplots(
                        1,
                        2,
                        figsize=(14, fig_height),
                        sharey=True,
                        gridspec_kw={"width_ratios": [0.95, 1.15]},
                    )
                    ax, zoom_ax = axes
                    _plot_policy_uncertainty_braid_panel(
                        ax,
                        braid_df,
                        policy_order,
                        y_lookup,
                        metric_offsets,
                        metric_colors,
                        title="All estimates",
                    )
                    _plot_policy_uncertainty_braid_panel(
                        zoom_ax,
                        braid_df,
                        policy_order,
                        y_lookup,
                        metric_offsets,
                        metric_colors,
                        title="Zoomed central cluster",
                        xlim=braid_x_limits,
                        show_yticklabels=False,
                    )
                    fig.suptitle("Do policy effects separate from zero?", y=0.98)
                _legend_below(ax, title="Quantity", ncol=3)
                fig.tight_layout()
                _save_figure(
                    fig,
                    figures_dir,
                    "policy_uncertainty_braid",
                    made,
                    data_rows=braid_rows,
                )
                plt.close(fig)

            top = (
                selective_df.assign(abs_index=selective_df["index"].abs())
                .sort_values("abs_index", ascending=False)
                .head(12)
                .sort_values("index", ascending=True)
                .reset_index(drop=True)
            )
            fig_height = max(4.8, 0.42 * len(top))
            top_x_limits = _one_axis_cluster_limits(top["index"])
            if top_x_limits is None:
                fig, ax = plt.subplots(figsize=(10.5, fig_height))
                _plot_selective_ssei_bars_panel(
                    ax,
                    top,
                    title="Largest absolute selective-safety effects",
                    label_suite=True,
                )
            else:
                fig, axes = plt.subplots(
                    1,
                    2,
                    figsize=(14, fig_height),
                    sharey=True,
                    gridspec_kw={"width_ratios": [0.95, 1.15]},
                )
                ax, zoom_ax = axes
                _plot_selective_ssei_bars_panel(
                    ax,
                    top,
                    title="All selected effects",
                    label_suite=True,
                )
                _plot_selective_ssei_bars_panel(
                    zoom_ax,
                    top,
                    title="Zoomed central cluster",
                    xlim=top_x_limits,
                    show_yticklabels=False,
                    label_suite=True,
                )
                fig.suptitle("Largest absolute selective-safety effects", y=0.98)
            fig.tight_layout()
            _save_figure(
                fig,
                figures_dir,
                "top_selective_effects",
                made,
                data_rows=top.drop(columns=["abs_index"]).to_dict(orient="records"),
            )
            plt.close(fig)

        # Use the CI's own mean (mean-of-per-prompt-ratios) as the point
        # estimate so that the figure bar and its CI describe the same
        # quantity.  Fall back to the ratio-of-means value only when the
        # CI dict is absent.
        def _ci_mean_or_fallback(value: dict, metric: str) -> object:
            ci = value.get(f"{metric}_ci") or {}
            if ci.get("mean") is not None:
                return ci["mean"]
            return value.get(metric)

        restoration_rows = [
            {
                "suite_policy": key,
                "suite": key.split("::", 1)[0],
                "policy": key.split("::", 1)[1],
                "compressed_policy": value.get("compressed_policy"),
                "safety_restoration_fraction": _ci_mean_or_fallback(
                    value, "safety_restoration_fraction"
                ),
                "safety_restoration_ci_low": (
                    value.get("safety_restoration_fraction_ci") or {}
                ).get("ci_low"),
                "safety_restoration_ci_high": (
                    value.get("safety_restoration_fraction_ci") or {}
                ).get("ci_high"),
                "refusal_restoration_fraction": _ci_mean_or_fallback(
                    value, "refusal_restoration_fraction"
                ),
                "refusal_restoration_ci_low": (
                    value.get("refusal_restoration_fraction_ci") or {}
                ).get("ci_low"),
                "refusal_restoration_ci_high": (
                    value.get("refusal_restoration_fraction_ci") or {}
                ).get("ci_high"),
                "leakage_avoidance_restoration_fraction": _ci_mean_or_fallback(
                    value, "leakage_avoidance_restoration_fraction"
                ),
                "leakage_avoidance_restoration_ci_low": (
                    value.get("leakage_avoidance_restoration_fraction_ci") or {}
                ).get("ci_low"),
                "leakage_avoidance_restoration_ci_high": (
                    value.get("leakage_avoidance_restoration_fraction_ci") or {}
                ).get("ci_high"),
            }
            for key, value in metrics.get("causal_restoration", {}).items()
        ]
        if restoration_rows:
            restoration_df = pd.DataFrame(restoration_rows)
            plot_df = restoration_df.dropna(subset=["safety_restoration_fraction"]).copy()
            if not plot_df.empty:
                plot_source_df = _safety_restoration_plot_subset(plot_df)
                if plot_source_df.empty:
                    plot_source_df = plot_df
                plot_df = plot_source_df.copy()
                plot_df = plot_df.sort_values("safety_restoration_fraction")
                plot_df = _with_restoration_display_bounds(plot_df)
                fig_height = max(4, 0.35 * len(plot_df))
                fig, ax = plt.subplots(figsize=(10, fig_height))
                labels = [_wrap_label(_clean_policy_label(row.policy), 34) for row in plot_df.itertuples()]
                colors = [_restoration_color(policy) for policy in plot_df["policy"]]
                ax.barh(labels, plot_df["safety_restoration_fraction"], color=colors, alpha=0.86)
                _draw_restoration_intervals(ax, plot_df)
                ax.axvline(0, color="black", linewidth=0.8)
                ax.axvline(1, color="0.6", linewidth=1, linestyle="--")
                _set_restoration_axis_limits(ax, plot_df)
                ax.set_xlabel("Restoration toward baseline (0 = compressed, 1 = baseline)")
                ax.set_title("Does patching system-cache tokens restore refusal safety?")
                ax.grid(axis="x", alpha=0.22)
                fig.tight_layout()
                _save_figure(
                    fig,
                    figures_dir,
                    "causal_restoration_fraction",
                    made,
                    data_rows=restoration_rows,
                )
                plt.close(fig)

                flow_df = _restoration_flow_rows(plot_df)
                if not flow_df.empty:
                    flow_df = flow_df.copy()
                    flow_df["safety_restoration_display_ci_low"] = flow_df[
                        "safety_restoration_ci_low"
                    ].map(_clip_unit_interval)
                    flow_df["safety_restoration_display_ci_high"] = flow_df[
                        "safety_restoration_ci_high"
                    ].map(_clip_unit_interval)
                    flow_df["plot_label"] = flow_df["policy"].map(
                        lambda policy: _wrap_label(_clean_policy_label(policy), 34)
                    )
                    fig_height = max(4.5, 0.42 * len(flow_df))
                    fig, ax = plt.subplots(figsize=(10, fig_height))
                    y_positions = list(range(len(flow_df)))
                    ax.axvline(0, color="0.2", linewidth=1)
                    ax.axvline(1, color="0.65", linewidth=1, linestyle="--")
                    ax.text(0, len(flow_df) + 0.15, "compressed", ha="center", va="bottom", fontsize=9)
                    ax.text(1, len(flow_df) + 0.15, "baseline", ha="center", va="bottom", fontsize=9)
                    for y, row in zip(y_positions, flow_df.itertuples(), strict=False):
                        color = _restoration_color(row.policy)
                        ci_width = max(
                            0.0,
                            row.safety_restoration_display_ci_high
                            - row.safety_restoration_display_ci_low,
                        )
                        line_width = 1.3 + 2.2 / (1.0 + 8.0 * ci_width)
                        ax.annotate(
                            "",
                            xy=(row.safety_restoration_fraction, y),
                            xytext=(0, y),
                            arrowprops={
                                "arrowstyle": "-|>",
                                "lw": line_width,
                                "color": color,
                                "alpha": 0.82,
                                "shrinkA": 0,
                                "shrinkB": 0,
                            },
                        )
                        ax.scatter(
                            [row.safety_restoration_fraction],
                            [y],
                            s=90,
                            color=color,
                            edgecolor="white",
                            linewidth=0.8,
                            zorder=3,
                        )
                        ax.errorbar(
                            [row.safety_restoration_fraction],
                            [y],
                            xerr=[
                                [
                                    max(
                                        0.0,
                                        row.safety_restoration_fraction
                                        - row.safety_restoration_display_ci_low,
                                    )
                                ],
                                [
                                    max(
                                        0.0,
                                        row.safety_restoration_display_ci_high
                                        - row.safety_restoration_fraction,
                                    )
                                ],
                            ],
                            fmt="none",
                            ecolor=color,
                            elinewidth=1.1,
                            capsize=3,
                            alpha=0.75,
                            zorder=2,
                        )
                    ax.set_yticks(y_positions, labels=flow_df["plot_label"])
                    _set_restoration_axis_limits(ax, flow_df)
                    ax.set_ylim(-0.8, len(flow_df) + 0.6)
                    ax.set_xlabel("Restoration toward baseline (0 = compressed, 1 = baseline)")
                    ax.set_title("Restoration flow for refusal-safety patches")
                    ax.grid(axis="x", alpha=0.22)
                    fig.tight_layout()
                    _save_figure(
                        fig,
                        figures_dir,
                        "causal_restoration_flow",
                        made,
                        data_rows=flow_df.to_dict(orient="records"),
                    )
                    plt.close(fig)

    cache_path = args.results_dir / "cache_stats.parquet"
    if cache_path.exists():
        cache_summaries = _stream_cache_summaries(cache_path)
        l2_grouped = cache_summaries["l2_rows"]
        if l2_grouped:
            grouped = pd.DataFrame(l2_grouped)
            if not grouped.empty:
                l2_y_limits = _one_axis_cluster_limits(
                    grouped["l2_retained_fraction"],
                    include_zero=False,
                )
                if l2_y_limits is None:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    _plot_cache_l2_panel(
                        ax,
                        grouped,
                        title="Cache L2 retained fraction over decoding",
                    )
                else:
                    fig, axes = plt.subplots(
                        1,
                        2,
                        figsize=(14, 5),
                        sharex=True,
                        gridspec_kw={"width_ratios": [0.95, 1.15]},
                    )
                    ax, zoom_ax = axes
                    _plot_cache_l2_panel(ax, grouped, title="All values")
                    _plot_cache_l2_panel(
                        zoom_ax,
                        grouped,
                        title="Zoomed central band",
                        ylim=l2_y_limits,
                        show_ylabel=False,
                    )
                    fig.suptitle("Cache L2 retained fraction over decoding", y=0.98)
                _legend_below(ax, title="Cache policy")
                fig.tight_layout()
                _save_figure(
                    fig,
                    figures_dir,
                    "cache_l2_retained_fraction",
                    made,
                    data_rows=l2_grouped,
                )
                plt.close(fig)
        role_rows = cache_summaries["role_rows"]
        if role_rows:
            role_df = pd.DataFrame(role_rows)
            pivot = role_df.pivot_table(
                index="role", columns="policy", values="retention_fraction", aggfunc="mean"
            )
            fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(pivot.columns)), 4.5))
            im = ax.imshow(pivot.fillna(0.0).values, aspect="auto", cmap="Oranges", vmin=0, vmax=1)
            ax.set_xticks(
                range(len(pivot.columns)),
                labels=[_wrap_label(_clean_policy_label(policy), 16) for policy in pivot.columns],
                rotation=0,
                ha="center",
            )
            ax.set_yticks(range(len(pivot.index)), labels=pivot.index)
            ax.set_title("Which prompt-token roles remain in cache?")
            fig.colorbar(im, ax=ax, label="retained / observed role tokens")
            fig.tight_layout()
            _save_figure(
                fig,
                figures_dir,
                "token_role_retention_heatmap",
                made,
                data_rows=role_rows,
            )
            plt.close(fig)
            atlas_rows = _safety_state_atlas_rows(selective_rows_for_atlas, role_rows)
            if atlas_rows:
                atlas_df = pd.DataFrame(atlas_rows)
                atlas_plot_df = atlas_df.sort_values(
                    "selective_safety_erasure_index", ascending=True
                ).reset_index(drop=True)
                fig_height = max(4.8, 0.44 * len(atlas_plot_df))
                atlas_x_limits = _combined_axis_cluster_limits(
                    atlas_plot_df,
                    [
                        "selective_safety_erasure_index",
                        "system_cache_loss",
                        "user_cache_loss",
                    ],
                )
                if atlas_x_limits is None:
                    fig, ax = plt.subplots(figsize=(10.5, fig_height))
                    _plot_safety_state_atlas_panel(
                        ax,
                        atlas_plot_df,
                        title="Behavior loss compared with cache-token loss",
                    )
                else:
                    fig, axes = plt.subplots(
                        1,
                        2,
                        figsize=(14, fig_height),
                        sharey=True,
                        gridspec_kw={"width_ratios": [0.95, 1.15]},
                    )
                    ax, zoom_ax = axes
                    _plot_safety_state_atlas_panel(ax, atlas_plot_df, title="All values")
                    _plot_safety_state_atlas_panel(
                        zoom_ax,
                        atlas_plot_df,
                        title="Zoomed central cluster",
                        xlim=atlas_x_limits,
                        show_yticklabels=False,
                    )
                    fig.suptitle("Behavior loss compared with cache-token loss", y=0.98)
                _legend_below(ax, title="Quantity", ncol=3)
                fig.tight_layout()
                _save_figure(
                    fig,
                    figures_dir,
                    "safety_state_atlas",
                    made,
                    data_rows=atlas_rows,
                )
                plt.close(fig)
        fingerprint_rows = _stream_cache_fingerprint(
            cache_path,
            args.results_dir / "prompts.jsonl",
            bin_count=48,
        )
        if fingerprint_rows:
            fingerprint_df = pd.DataFrame(fingerprint_rows)
            fingerprint_df["row_source_suffix"] = fingerprint_df["layer_source_label"].map(
                lambda label: "" if label == "explicit layer column" else " [row-order]"
            )
            fingerprint_df["row_label"] = (
                fingerprint_df["policy"].map(_clean_policy_label)
                + " / "
                + fingerprint_df["layer_label"]
                + fingerprint_df["row_source_suffix"]
            )
            fingerprint_df["column_label"] = (
                fingerprint_df["role"] + ":" + fingerprint_df["token_bin"].astype(str)
            )
            row_order = (
                fingerprint_df[["row_label", "policy", "layer_bin"]]
                .drop_duplicates()
                .sort_values(["policy", "layer_bin"])
            )
            column_order = (
                fingerprint_df[["column_label", "role", "token_bin"]]
                .drop_duplicates()
                .sort_values(
                    by=["role", "token_bin"],
                    key=lambda series: series.map(lambda value: ROLE_ORDER.get(str(value), 99))
                    if series.name == "role"
                    else series,
                )
            )
            pivot = fingerprint_df.pivot_table(
                index="row_label",
                columns="column_label",
                values="retention_fraction",
                aggfunc="mean",
            ).reindex(index=row_order["row_label"], columns=column_order["column_label"])
            fig_height = max(5, 0.22 * len(pivot.index))
            fig, ax = plt.subplots(figsize=(12, fig_height))
            im = ax.imshow(pivot.fillna(0.0).values, aspect="auto", cmap="Oranges", vmin=0, vmax=1)
            column_records = column_order.to_dict(orient="records")
            role_centers = _column_role_centers(column_records)
            ax.set_xticks(
                [center for _role, center in role_centers],
                labels=[role for role, _center in role_centers],
            )
            for boundary in _column_role_boundaries(column_records):
                ax.axvline(boundary, color="white", linewidth=0.5, alpha=0.6)
            ax.set_yticks(range(len(pivot.index)), labels=pivot.index, fontsize=8)
            ax.set_xlabel("Prompt-token role (columns are position bins inside each role)")
            ax.set_ylabel("Cache policy / layer band")
            ax.set_title("Which token roles stay in cache?")
            fig.colorbar(im, ax=ax, label="retained fraction")
            fig.tight_layout()
            _save_figure(
                fig,
                figures_dir,
                "cache_state_fingerprint",
                made,
                data_rows=fingerprint_rows,
            )
            plt.close(fig)

    write_json(
        figures_dir / "manifest.json",
        {
            "schema_version": 1,
            "source_artifacts": _source_artifacts(args.results_dir),
            "figures": made,
        },
    )
    print(f"Wrote {len(made)} figure(s) to {figures_dir}")


def _save_figure(
    fig: Any,
    figures_dir: Path,
    stem: str,
    made: list[dict[str, Any]],
    *,
    data_rows: list[dict[str, Any]] | None = None,
) -> None:
    png_path = figures_dir / f"{stem}.png"
    svg_path = figures_dir / f"{stem}.svg"
    pdf_path = figures_dir / f"{stem}.pdf"
    data_path = figures_dir / f"{stem}.csv"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    entry: dict[str, Any] = {
        "name": stem,
        "png": str(png_path),
        "png_sha256": file_sha256(png_path),
        "svg": str(svg_path),
        "svg_sha256": file_sha256(svg_path),
        "pdf": str(pdf_path),
        "pdf_sha256": file_sha256(pdf_path),
    }
    if data_rows is not None:
        _write_csv(data_path, data_rows)
        entry["data_csv"] = str(data_path)
        entry["data_csv_sha256"] = file_sha256(data_path)
        entry["data_row_count"] = len(data_rows)
    made.append(entry)


def _legend_below(
    ax: Any,
    *,
    handles: list[Any] | None = None,
    labels: list[str] | None = None,
    title: str | None = None,
    ncol: int = 3,
    y_anchor: float = -0.2,
) -> Any:
    if handles is None and labels is None:
        handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    if labels is None:
        labels = [handle.get_label() for handle in handles]
    unique_handles = []
    unique_labels = []
    seen_labels: set[str] = set()
    for handle, label in zip(handles, labels, strict=False):
        if not label or str(label).startswith("_") or label in seen_labels:
            continue
        unique_handles.append(handle)
        unique_labels.append(label)
        seen_labels.add(label)
    handles = unique_handles
    labels = unique_labels
    if not handles:
        return None
    return ax.legend(
        handles=handles,
        labels=labels,
        title=title,
        loc="upper center",
        bbox_to_anchor=(0.5, y_anchor),
        ncol=max(1, min(ncol, len(handles))),
        frameon=False,
        borderaxespad=0.0,
    )


def _plot_safety_capability_panel(
    ax: Any,
    plot_df: Any,
    *,
    title: str,
    annotate_count: int,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    show_ylabel: bool = True,
) -> None:
    ax.axline((0, 0), slope=1, color="0.6", linewidth=1, linestyle="--")
    for suite, suite_df in plot_df.groupby("suite"):
        ax.scatter(
            suite_df["x_plot"],
            suite_df["y_plot"],
            s=64,
            label=_clean_suite_label(suite),
            alpha=0.85,
        )
    labeled = plot_df[plot_df["index"].abs() > 0].copy()
    if not labeled.empty:
        labeled = labeled.sort_values("index", key=lambda series: series.abs(), ascending=False).head(
            annotate_count
        )
        for row in labeled.itertuples():
            if xlim is not None and not (xlim[0] <= row.x_plot <= xlim[1]):
                continue
            if ylim is not None and not (ylim[0] <= row.y_plot <= ylim[1]):
                continue
            ax.annotate(
                _short_policy_label(str(row.policy)),
                (row.x_plot, row.y_plot),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_title(title)
    ax.set_xlabel("Capability degradation")
    ax.set_ylabel("Safety degradation" if show_ylabel else "")
    ax.grid(alpha=0.2)


def _plot_phase_portrait_panel(
    ax: Any,
    phase_df: Any,
    *,
    title: str,
    xlim: tuple[float, float] | None = None,
    show_yticklabels: bool = True,
) -> None:
    y = list(range(len(phase_df)))
    ax.hlines(
        y,
        phase_df["capability_degradation"],
        phase_df["safety_degradation"],
        color="0.78",
        linewidth=1.2,
        zorder=1,
    )
    ax.scatter(
        phase_df["capability_degradation"],
        y,
        color=C_LIGHT,
        s=54,
        label="Capability loss",
        zorder=2,
    )
    ax.scatter(
        phase_df["safety_degradation"],
        y,
        color=C_ACCENT,
        s=54,
        label="Safety loss",
        zorder=3,
    )
    ax.axvline(0, color="0.15", linewidth=0.9)
    ax.set_yticks(y)
    if show_yticklabels:
        ax.set_yticklabels(
            [_wrap_label(_clean_policy_label(policy), 30) for policy in phase_df["policy"]]
        )
    else:
        ax.tick_params(axis="y", labelleft=False)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_xlabel("Change from baseline (positive = worse)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.22)


def _plot_policy_uncertainty_braid_panel(
    ax: Any,
    braid_df: Any,
    policy_order: list[str],
    y_lookup: dict[str, int],
    metric_offsets: dict[str, float],
    metric_colors: dict[str, str],
    *,
    title: str,
    xlim: tuple[float, float] | None = None,
    show_yticklabels: bool = True,
) -> None:
    ax.axvline(0, color="0.2", linewidth=0.9)
    for policy in policy_order:
        policy_df = braid_df[braid_df["policy"] == policy]
        connector = policy_df.sort_values("metric_rank")
        ax.plot(
            connector["mean"],
            [
                y_lookup[policy] + metric_offsets[metric]
                for metric in connector["metric"]
            ],
            color="0.72",
            linewidth=1.0,
            alpha=0.7,
            zorder=1,
        )
    for metric, metric_df in braid_df.groupby("metric"):
        y = [
            y_lookup[policy] + metric_offsets[metric]
            for policy in metric_df["policy"]
        ]
        xerr = [
            metric_df["mean"] - metric_df["ci_low"],
            metric_df["ci_high"] - metric_df["mean"],
        ]
        ax.errorbar(
            metric_df["mean"],
            y,
            xerr=xerr,
            fmt="o",
            color=metric_colors.get(metric, "0.2"),
            ecolor=metric_colors.get(metric, "0.2"),
            elinewidth=1.2,
            capsize=3,
            markersize=5.5,
            label=_clean_metric_label(metric),
            alpha=0.9,
            zorder=2,
        )
    ax.set_yticks(range(len(policy_order)))
    if show_yticklabels:
        ax.set_yticklabels(
            [_wrap_label(_clean_policy_label(policy), 30) for policy in policy_order]
        )
    else:
        ax.tick_params(axis="y", labelleft=False)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_xlabel("Estimate with 95% CI (positive = worse)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.22)


def _plot_cache_l2_panel(
    ax: Any,
    grouped: Any,
    *,
    title: str,
    ylim: tuple[float, float] | None = None,
    show_ylabel: bool = True,
) -> None:
    for policy, policy_df in grouped.groupby("policy"):
        ax.plot(
            policy_df["decode_step"],
            policy_df["l2_retained_fraction"],
            label=_clean_policy_label(policy),
            alpha=0.85,
        )
    if ylim is None:
        ax.set_ylim(0, max(1.05, float(grouped["l2_retained_fraction"].max()) * 1.02))
    else:
        ax.set_ylim(*ylim)
    ax.set_title(title)
    ax.set_xlabel("Decode step")
    ax.set_ylabel("L2 retained fraction" if show_ylabel else "")
    ax.grid(alpha=0.22)


def _plot_safety_state_atlas_panel(
    ax: Any,
    atlas_plot_df: Any,
    *,
    title: str,
    xlim: tuple[float, float] | None = None,
    show_yticklabels: bool = True,
) -> None:
    y = list(range(len(atlas_plot_df)))
    ax.axvline(0, color="0.15", linewidth=0.9)
    ax.scatter(
        atlas_plot_df["selective_safety_erasure_index"],
        y,
        color=C_ACCENT,
        s=58,
        label="SSeI",
        zorder=3,
    )
    ax.scatter(
        atlas_plot_df["system_cache_loss"],
        y,
        facecolors="none",
        edgecolors=C_DARK,
        s=74,
        linewidths=1.3,
        label="System-token cache loss",
        zorder=2,
    )
    ax.scatter(
        atlas_plot_df["user_cache_loss"],
        y,
        facecolors="none",
        edgecolors=C_LIGHT,
        s=74,
        linewidths=1.3,
        label="User-token cache loss",
        zorder=2,
    )
    ax.set_yticks(y)
    if show_yticklabels:
        ax.set_yticklabels(
            [_wrap_label(_clean_policy_label(policy), 30) for policy in atlas_plot_df["policy"]]
        )
    else:
        ax.tick_params(axis="y", labelleft=False)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_xlabel("Loss fraction or SSeI (larger = more loss)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.22)


def _plot_selective_ssei_bars_panel(
    ax: Any,
    bar_df: Any,
    *,
    title: str,
    xlim: tuple[float, float] | None = None,
    show_yticklabels: bool = True,
    label_suite: bool = False,
) -> None:
    y = list(range(len(bar_df)))
    colors = [C_ACCENT if value > 0 else C_LIGHT for value in bar_df["index"]]
    ax.barh(y, bar_df["index"], color=colors, alpha=0.88)
    ax.axvline(0, color="0.15", linewidth=0.9)
    ax.set_yticks(y)
    if show_yticklabels:
        if label_suite:
            labels = [
                _wrap_label(
                    f"{_clean_suite_label(row.suite)} | {_clean_policy_label(row.policy)}",
                    34,
                )
                for row in bar_df.itertuples()
            ]
        else:
            labels = [
                _wrap_label(_clean_policy_label(policy), 30)
                for policy in bar_df["policy"]
            ]
        ax.set_yticklabels(labels)
    else:
        ax.tick_params(axis="y", labelleft=False)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_xlabel("SSeI = safety loss - capability loss (positive = safety-specific loss)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.22)


def _cluster_zoom_limits(
    df: Any,
    *,
    x_column: str,
    y_column: str,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    x_limits = _one_axis_cluster_limits(df[x_column])
    y_limits = _one_axis_cluster_limits(df[y_column])
    if x_limits is None and y_limits is None:
        return None
    if x_limits is None:
        x_limits = _full_axis_limits(df[x_column])
    if y_limits is None:
        y_limits = _full_axis_limits(df[y_column])
    return x_limits, y_limits


def _combined_axis_cluster_limits(df: Any, columns: list[str]) -> tuple[float, float] | None:
    values = df[columns].melt(value_name="value")["value"]
    return _one_axis_cluster_limits(values)


def _one_axis_cluster_limits(
    values: Any,
    *,
    include_zero: bool = True,
) -> tuple[float, float] | None:
    series = values.dropna()
    if len(series) < 8:
        return None
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    if iqr <= 0:
        central = series[series != series.max()]
    else:
        central = series[(series >= q1 - 1.5 * iqr) & (series <= q3 + 1.5 * iqr)]
    if len(central) < max(4, int(0.5 * len(series))):
        return None
    full_span = float(series.max() - series.min())
    central_span = float(central.max() - central.min())
    if central_span <= 0 or full_span < central_span * 2.5:
        return None
    return _padded_limits(float(central.min()), float(central.max()), include_zero=include_zero)


def _full_axis_limits(values: Any) -> tuple[float, float]:
    series = values.dropna()
    if series.empty:
        return -0.01, 0.01
    return _padded_limits(float(series.min()), float(series.max()))


def _padded_limits(
    low: float,
    high: float,
    *,
    include_zero: bool = True,
) -> tuple[float, float]:
    if include_zero:
        low = min(low, 0.0)
        high = max(high, 0.0)
    span = high - low
    pad = max(0.01, span * 0.12)
    return low - pad, high + pad


def _source_artifacts(results_dir: Path) -> dict[str, dict[str, Any]]:
    artifacts = {}
    for name in ["manifest.json", "generations.jsonl", "metrics.json", "cache_stats.parquet"]:
        path = results_dir / name
        artifacts[name] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size if path.exists() else None,
        }
    return artifacts


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _selective_rows_for_figures(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    suite_rows = []
    for key, value in metrics.get("selective_safety_erasure", {}).items():
        index = _finite_float(value.get("selective_safety_erasure_index"))
        safety = _finite_float(value.get("safety_degradation"))
        capability = _finite_float(value.get("capability_degradation"))
        if index is None or safety is None or capability is None:
            continue
        suite, policy = key.split("::", 1)
        suite_rows.append(
            {
                "suite_policy": key,
                "suite": suite,
                "policy": policy,
                "contrast_scope": "suite",
                "index": index,
                "selective_safety_erasure_index": index,
                "safety_degradation": safety,
                "capability_degradation": capability,
            }
        )
    if suite_rows:
        return suite_rows

    policy_rows = []
    for policy, value in metrics.get("policy_level_contrasts", {}).items():
        safety_ci = value.get("safety_degradation_ci") or {}
        capability_ci = value.get("capability_degradation_ci") or {}
        ssei_ci = value.get("selective_safety_erasure_index_ci") or {}
        index = _finite_float(
            value.get("selective_safety_erasure_index")
            if value.get("selective_safety_erasure_index") is not None
            else ssei_ci.get("mean")
        )
        safety = _finite_float(safety_ci.get("mean"))
        capability = _finite_float(capability_ci.get("mean"))
        if index is None or safety is None or capability is None:
            continue
        policy_rows.append(
            {
                "suite_policy": f"global_policy_contrast::{policy}",
                "suite": "global_policy_contrast",
                "policy": policy,
                "contrast_scope": "policy_level",
                "index": index,
                "selective_safety_erasure_index": index,
                "safety_degradation": safety,
                "capability_degradation": capability,
                "safety_ci_low": safety_ci.get("ci_low"),
                "safety_ci_high": safety_ci.get("ci_high"),
                "capability_ci_low": capability_ci.get("ci_low"),
                "capability_ci_high": capability_ci.get("ci_high"),
                "selective_safety_erasure_index_ci_low": ssei_ci.get("ci_low"),
                "selective_safety_erasure_index_ci_high": ssei_ci.get("ci_high"),
                "safety_n": safety_ci.get("n") or ssei_ci.get("n_safety"),
                "capability_n": capability_ci.get("n") or ssei_ci.get("n_capability"),
            }
        )
    return policy_rows


def _paired_safety_forest_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for key, value in metrics.get("selective_safety_erasure", {}).items():
        ci = value.get("paired_safety_degradation_ci", {})
        if ci.get("mean") is None or ci.get("ci_low") is None or ci.get("ci_high") is None:
            continue
        rows.append(
            {
                "label": key.replace("::", " / "),
                "mean": ci["mean"],
                "ci_low": ci["ci_low"],
                "ci_high": ci["ci_high"],
                "cluster_n": ci.get("cluster_n"),
            }
        )
    return rows


def _policy_uncertainty_braid_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metric_specs = [
        ("capability", "capability_degradation_ci", 0),
        ("safety", "safety_degradation_ci", 1),
        ("ssei", "selective_safety_erasure_index_ci", 2),
    ]
    for policy, value in metrics.get("policy_level_contrasts", {}).items():
        policy_rows: list[dict[str, Any]] = []
        for metric, ci_key, rank in metric_specs:
            ci = value.get(ci_key) or {}
            mean = _finite_float(
                ci.get("mean")
                if ci.get("mean") is not None
                else value.get("selective_safety_erasure_index")
                if metric == "ssei"
                else None
            )
            ci_low = _finite_float(ci.get("ci_low"))
            ci_high = _finite_float(ci.get("ci_high"))
            if mean is None or ci_low is None or ci_high is None:
                policy_rows = []
                break
            policy_rows.append(
                {
                    "policy": policy,
                    "metric": metric,
                    "metric_rank": rank,
                    "mean": mean,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "ci_width": ci_high - ci_low,
                    "n_safety": ci.get("n_safety"),
                    "n_capability": ci.get("n_capability"),
                    "n": ci.get("n"),
                }
            )
        rows.extend(policy_rows)
    return rows


def _phase_portrait_rows(selective_df: Any) -> Any:
    import pandas as pd

    rows = []
    for row in selective_df.to_dict(orient="records"):
        safety = _finite_float(row.get("safety_degradation"))
        capability = _finite_float(row.get("capability_degradation"))
        if safety is None or capability is None:
            continue
        policy = str(row.get("policy"))
        family, budget_sort, budget_label = _policy_shape(policy)
        rows.append(
            {
                "suite": row.get("suite"),
                "policy": policy,
                "policy_family": family,
                "budget_sort": budget_sort,
                "budget_label": budget_label,
                "safety_degradation": safety,
                "capability_degradation": capability,
                "selective_safety_erasure_index": _finite_float(
                    row.get("selective_safety_erasure_index") or row.get("index")
                ),
            }
        )
    return pd.DataFrame(rows)


def _prompt_effect_constellation_rows(df: Any) -> list[dict[str, Any]]:
    import numpy as np

    rows = []
    baseline: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in df.to_dict(orient="records"):
        if row.get("policy") == "none":
            baseline[
                (
                    str(row.get("suite")),
                    str(row.get("prompt_id")),
                    int(row.get("seed", 0)),
                )
            ] = row
    metric_names = [
        "safety_score",
        "capability_score",
        "refusal_expected_accuracy",
        "leakage_avoidance_score",
        "generated_word_count",
    ]
    vectors = []
    for row in df.to_dict(orient="records"):
        policy = str(row.get("policy"))
        if policy == "none":
            continue
        key = (str(row.get("suite")), str(row.get("prompt_id")), int(row.get("seed", 0)))
        base = baseline.get(key)
        if base is None:
            continue
        vector = []
        raw: dict[str, float | None] = {}
        for metric in metric_names:
            base_value = _finite_float(base.get(metric))
            current_value = _finite_float(row.get(metric))
            if base_value is None or current_value is None:
                delta = None
                vector.append(0.0)
            elif metric == "generated_word_count":
                denominator = max(1.0, abs(base_value))
                delta = (current_value - base_value) / denominator
                vector.append(delta)
            else:
                delta = base_value - current_value
                vector.append(delta)
            raw[f"{metric}_delta"] = delta
        if not any(abs(value) > 1e-12 for value in vector):
            continue
        vectors.append(vector)
        rows.append(
            {
                "suite": key[0],
                "prompt_id": key[1],
                "seed": key[2],
                "policy": policy,
                **raw,
            }
        )
    if not rows:
        return []
    matrix = np.asarray(vectors, dtype=float)
    coords = _project_effect_vectors(matrix)
    magnitudes = np.linalg.norm(matrix, axis=1)
    max_magnitude = max(float(magnitudes.max()), 1e-12)
    for row, coord, magnitude in zip(rows, coords, magnitudes, strict=False):
        row["x"] = float(coord[0])
        row["y"] = float(coord[1])
        row["effect_magnitude"] = float(magnitude / max_magnitude)
    return rows


def _project_effect_vectors(matrix: Any) -> Any:
    import numpy as np

    if matrix.shape[0] == 1:
        return np.asarray([[matrix[0, 0], matrix[0, 1] if matrix.shape[1] > 1 else 0.0]])
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    if not np.any(np.abs(centered) > 1e-12):
        return matrix[:, :2] if matrix.shape[1] >= 2 else np.c_[matrix[:, 0], np.zeros(matrix.shape[0])]
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:2].T
    if basis.shape[1] < 2:
        return np.c_[centered @ basis[:, 0], np.zeros(centered.shape[0])]
    return centered @ basis


def _safety_state_atlas_rows(
    selective_rows: list[dict[str, Any]], role_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not selective_rows or not role_rows:
        return []
    role_lookup = {
        (str(row["policy"]), str(row["role"])): _finite_float(row.get("retention_fraction"))
        for row in role_rows
    }
    rows = []
    for row in selective_rows:
        policy = str(row.get("policy"))
        system_retention = role_lookup.get((policy, "system"))
        user_retention = role_lookup.get((policy, "user"))
        system_loss = None if system_retention is None else 1.0 - system_retention
        user_loss = None if user_retention is None else 1.0 - user_retention
        rows.append(
            {
                "suite": row.get("suite"),
                "policy": policy,
                "selective_safety_erasure_index": row.get("index")
                if row.get("index") is not None
                else row.get("selective_safety_erasure_index"),
                "safety_degradation": row.get("safety_degradation"),
                "capability_degradation": row.get("capability_degradation"),
                "retention_scope": "policy_global",
                "system_retention_fraction": system_retention,
                "user_retention_fraction": user_retention,
                "template_retention_fraction": role_lookup.get((policy, "template")),
                "generated_retention_fraction": role_lookup.get((policy, "generated")),
                "system_cache_loss": system_loss,
                "user_cache_loss": user_loss,
                "system_minus_user_cache_loss": (
                    None if system_loss is None or user_loss is None else system_loss - user_loss
                ),
            }
        )
    return rows


def _policy_shape(policy: str) -> tuple[str, float, str]:
    family = policy.split("__", 1)[0]
    budget_match = re.search(r"__budget(\d+)", policy)
    if budget_match:
        budget = float(budget_match.group(1))
        return family, budget, f"b={int(budget)}"
    if "int4" in policy:
        return family, 4.0, "4-bit"
    if "int8" in policy:
        return family, 8.0, "8-bit"
    if policy == "none":
        return family, 0.0, "base"
    return family, 1_000_000.0, ""


def _restoration_flow_rows(restoration_df: Any) -> Any:
    import pandas as pd

    rows = []
    for row in restoration_df.itertuples():
        fraction = _finite_float(row.safety_restoration_fraction)
        ci_low = _finite_float(getattr(row, "safety_restoration_ci_low", None))
        ci_high = _finite_float(getattr(row, "safety_restoration_ci_high", None))
        if fraction is None or ci_low is None or ci_high is None:
            continue
        rows.append(
            {
                "suite": row.suite,
                "policy": row.policy,
                "compressed_policy": row.compressed_policy,
                "safety_restoration_fraction": fraction,
                "safety_restoration_ci_low": ci_low,
                "safety_restoration_ci_high": ci_high,
                "safety_restoration_display_ci_low": _clip_unit_interval(ci_low),
                "safety_restoration_display_ci_high": _clip_unit_interval(ci_high),
                "safety_restoration_ci_width": ci_high - ci_low,
                "label": f"{row.suite} / {_short_policy_label(row.policy)}",
            }
        )
    return pd.DataFrame(rows).sort_values("safety_restoration_fraction") if rows else pd.DataFrame()


def _draw_restoration_intervals(ax: Any, plot_df: Any) -> None:
    if not {
        "safety_restoration_display_ci_low",
        "safety_restoration_display_ci_high",
    }.issubset(plot_df.columns):
        return
    valid = plot_df.dropna(
        subset=[
            "safety_restoration_fraction",
            "safety_restoration_display_ci_low",
            "safety_restoration_display_ci_high",
        ]
    )
    if valid.empty:
        return
    y_positions = list(range(len(valid)))
    ax.errorbar(
        valid["safety_restoration_fraction"],
        y_positions,
        xerr=[
            (
                valid["safety_restoration_fraction"]
                - valid["safety_restoration_display_ci_low"]
            ).clip(lower=0),
            (
                valid["safety_restoration_display_ci_high"]
                - valid["safety_restoration_fraction"]
            ).clip(lower=0),
        ],
        fmt="none",
        ecolor="0.15",
        elinewidth=1.0,
        capsize=3,
        alpha=0.75,
    )


def _with_restoration_display_bounds(df: Any) -> Any:
    bounded = df.copy()
    for source, target in [
        ("safety_restoration_ci_low", "safety_restoration_display_ci_low"),
        ("safety_restoration_ci_high", "safety_restoration_display_ci_high"),
    ]:
        bounded[target] = bounded[source].clip(lower=0.0, upper=1.0)
    return bounded


def _clip_unit_interval(value: float) -> float:
    return min(1.0, max(0.0, value))


def _short_policy_label(policy: str) -> str:
    return _clean_policy_label(policy)


def _clean_policy_label(policy: str) -> str:
    policy = str(policy)
    direct = {
        "none": "Baseline",
        "kv_int4_sim": "4-bit KV",
        "kv_int8_sim": "8-bit KV",
        "policy_pinned__budget128__sink8": "Policy-pinned cache",
        "random_matched__budget128__seed991": "Random matched",
        "sink_recent__budget128__sink8": "Sink+recent",
        "sliding_window__budget64": "Window 64",
        "sliding_window__budget128": "Window 128",
        "sliding_window__budget256": "Window 256",
    }
    if policy in direct:
        return direct[policy]
    if policy.startswith("kv_int4_sim__"):
        patch = "Patch"
        if "patchkey-value" in policy:
            patch = "Patch K+V"
        elif "patchkey" in policy:
            patch = "Patch keys"
        elif "patchvalue" in policy:
            patch = "Patch values"
        if "roleuser" in policy and "matchsystem" in policy:
            return f"{patch} on matched user tokens"
        if "rolesystem" in policy:
            return f"{patch} on system tokens"
    label = policy.replace("__", " / ")
    label = label.replace("budget", "budget ")
    label = label.replace("seed", "seed ")
    label = label.replace("sink", "sink ")
    label = label.replace("patchkey-value", "patch K+V")
    label = label.replace("patchkey", "patch keys")
    label = label.replace("patchvalue", "patch values")
    label = label.replace("rolesystem", "system")
    label = label.replace("roleuser", "user")
    return label


def _clean_suite_label(suite: str) -> str:
    suite = str(suite)
    return {
        "global_policy_contrast": "All public suites",
        "public_refusal_safety": "Refusal safety",
        "public_system_leakage": "Public system leakage",
        "system_leakage": "System leakage probe",
        "public_capability_arc": "Capability",
        "public_benign_overrefusal": "Benign over-refusal",
    }.get(suite, suite.replace("_", " ").title())


def _clean_metric_label(metric: str) -> str:
    return {
        "capability": "Capability loss",
        "safety": "Safety loss",
        "ssei": "SSeI",
    }.get(str(metric), str(metric).replace("_", " ").title())


def _wrap_label(text: Any, width: int) -> str:
    parts = str(text).split("\n")
    return "\n".join(
        "\n".join(textwrap.wrap(part, width=width, break_long_words=False) or [part])
        for part in parts
    )


def _clip_unit_interval(value: Any) -> float:
    number = _finite_float(value)
    if number is None:
        return 0.0
    return min(1.0, max(0.0, number))


def _safety_restoration_plot_subset(plot_df: Any) -> Any:
    public_refusal = plot_df[plot_df["suite"] == "public_refusal_safety"].copy()
    if not public_refusal.empty:
        return public_refusal
    refusal = plot_df[plot_df["suite"].astype(str).str.contains("refusal", na=False)].copy()
    if not refusal.empty:
        return refusal
    return plot_df.head(0)


def _set_restoration_axis_limits(ax: Any, plot_df: Any) -> None:
    values: list[float] = [0.0, 1.0]
    for column in [
        "safety_restoration_fraction",
        "safety_restoration_display_ci_low",
        "safety_restoration_display_ci_high",
    ]:
        if column not in plot_df:
            continue
        for value in plot_df[column].tolist():
            number = _finite_float(value)
            if number is not None:
                values.append(number)
    low = min(values)
    high = max(values)
    pad = max(0.08, 0.05 * (high - low))
    ax.set_xlim(low - pad, high + pad)


def _legacy_policy_label(policy: str) -> str:
    label = policy.replace("__", " / ")
    label = label.replace("patchkey-value", "patch K,V")
    label = label.replace("rolesystem", "system")
    label = label.replace("roleuser", "user")
    return label


def _restoration_color(policy: str) -> str:
    if "roleuser" in policy or "matchsystem" in policy:
        return C_LIGHT
    if "rolesystem" in policy or "system" in policy:
        return C_PRIMARY
    if "policy_pinned" in policy:
        return C_ACCENT
    return C_MUTED


def _stream_cache_fingerprint(
    cache_path: Path,
    prompts_path: Path,
    *,
    bin_count: int,
    layer_bin_count: int = 8,
) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise SystemExit("pyarrow is required to summarize cache_stats.parquet.") from exc
    if not prompts_path.exists():
        return []
    prompt_roles = _load_prompt_roles(prompts_path)
    parquet_file = pq.ParquetFile(cache_path)
    required = {
        "prompt_id",
        "policy",
        "decode_step",
        "original_seq_len",
        "retained_indices",
        "evicted_indices",
    }
    schema_names = set(parquet_file.schema.names)
    if not required.issubset(schema_names):
        return []
    optional = {"seed", "layer", "layer_count"} & schema_names
    counts: dict[tuple[str, int, str, str, str, str, int], list[float]] = defaultdict(
        lambda: [0.0, 0.0]
    )
    synthetic_layer_counters: dict[tuple[Any, ...], int] = defaultdict(int)
    columns = sorted(required | optional)
    for batch in parquet_file.iter_batches(columns=columns, batch_size=50_000):
        table = batch.to_pydict()
        for idx, raw_policy in enumerate(table.get("policy", [])):
            if int(_float_at(table, "decode_step", idx)) != 0:
                continue
            prompt_id = str(table["prompt_id"][idx])
            policy = str(raw_policy)
            seq_len = int(_float_at(table, "original_seq_len", idx))
            if seq_len <= 0:
                continue
            layer_count = max(1, int(_float_at(table, "layer_count", idx) or 1))
            layer, layer_source = _cache_stat_layer(
                table, idx, synthetic_layer_counters, layer_count=layer_count
            )
            layer_bin, layer_label = _layer_band(layer, layer_count, layer_bin_count)
            layer_source_label = _layer_source_label(layer_source)
            roles = prompt_roles.get(prompt_id, [])
            for token_idx in _parse_indices(table["retained_indices"][idx]):
                role = _role_at(roles, token_idx)
                token_bin = min(bin_count - 1, int((token_idx / seq_len) * bin_count))
                counts[
                    (policy, layer_bin, layer_label, layer_source, layer_source_label, role, token_bin)
                ][0] += 1.0
            for token_idx in _parse_indices(table["evicted_indices"][idx]):
                role = _role_at(roles, token_idx)
                token_bin = min(bin_count - 1, int((token_idx / seq_len) * bin_count))
                counts[
                    (policy, layer_bin, layer_label, layer_source, layer_source_label, role, token_bin)
                ][1] += 1.0
    rows = []
    for (
        policy,
        layer_bin,
        layer_label,
        layer_source,
        layer_source_label,
        role,
        token_bin,
    ), (retained, evicted) in sorted(
        counts.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            ROLE_ORDER.get(item[0][5], 99),
            item[0][6],
        ),
    ):
        total = retained + evicted
        if total <= 0:
            continue
        rows.append(
            {
                "policy": policy,
                "layer_bin": layer_bin,
                "layer_label": layer_label,
                "layer_source": layer_source,
                "layer_source_label": layer_source_label,
                "role": role,
                "token_bin": token_bin,
                "retained_count": retained,
                "evicted_count": evicted,
                "retention_fraction": retained / total,
            }
        )
    return rows


def _cache_stat_layer(
    table: dict[str, list[Any]],
    idx: int,
    synthetic_layer_counters: dict[tuple[Any, ...], int],
    *,
    layer_count: int,
) -> tuple[int, str]:
    if "layer" in table:
        return max(0, int(_float_at(table, "layer", idx))), "explicit_layer"
    key = (
        table.get("prompt_id", [None])[idx],
        table.get("seed", [None] * (idx + 1))[idx] if "seed" in table else None,
        table.get("policy", [None])[idx],
        table.get("decode_step", [None])[idx],
        table.get("original_seq_len", [None])[idx],
        table.get("retained_indices", [None])[idx],
        table.get("evicted_indices", [None])[idx],
    )
    layer = synthetic_layer_counters[key] % max(1, layer_count)
    synthetic_layer_counters[key] += 1
    source = "legacy_row_order" if layer_count > 1 else "unlayered_cache_rows"
    return layer, source


def _layer_source_label(layer_source: str) -> str:
    return {
        "explicit_layer": "explicit layer column",
        "legacy_row_order": "legacy row-order layer inference",
        "unlayered_cache_rows": "unlayered cache rows",
    }.get(layer_source, layer_source.replace("_", " "))


def _layer_band(layer: int, layer_count: int, layer_bin_count: int) -> tuple[int, str]:
    layer_count = max(1, layer_count)
    layer_bin_count = max(1, layer_bin_count)
    if layer_count <= layer_bin_count:
        layer = min(max(0, layer), layer_count - 1)
        return layer, f"L{layer:02d}"
    layer_bin = min(layer_bin_count - 1, int((max(0, layer) / layer_count) * layer_bin_count))
    start = math.floor(layer_bin * layer_count / layer_bin_count)
    end = max(start, math.floor((layer_bin + 1) * layer_count / layer_bin_count) - 1)
    return layer_bin, f"L{start:02d}-L{end:02d}"


def _column_role_centers(column_records: list[dict[str, Any]]) -> list[tuple[str, float]]:
    centers = []
    for role in sorted(
        {str(record["role"]) for record in column_records},
        key=lambda value: ROLE_ORDER.get(value, 99),
    ):
        indices = [
            index for index, record in enumerate(column_records) if str(record["role"]) == role
        ]
        if indices:
            centers.append((role, (indices[0] + indices[-1]) / 2))
    return centers


def _column_role_boundaries(column_records: list[dict[str, Any]]) -> list[float]:
    boundaries = []
    previous_role = None
    for index, record in enumerate(column_records):
        role = str(record["role"])
        if previous_role is not None and role != previous_role:
            boundaries.append(index - 0.5)
        previous_role = role
    return boundaries


def _load_prompt_roles(prompts_path: Path) -> dict[str, list[str]]:
    roles_by_prompt: dict[str, list[str]] = {}
    with prompts_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            rendered = record.get("rendered_prompt") or {}
            roles = rendered.get("token_roles") or []
            if roles:
                roles_by_prompt[str(record.get("prompt_id"))] = [str(role) for role in roles]
    return roles_by_prompt


def _parse_indices(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, list):
        return [int(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    return [int(part) for part in text.split(",") if part.strip()]


def _role_at(roles: list[str], idx: int) -> str:
    if 0 <= idx < len(roles):
        return roles[idx]
    return "unknown"


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _stream_cache_summaries(cache_path: Path) -> dict[str, list[dict[str, Any]]]:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise SystemExit("pyarrow is required to summarize cache_stats.parquet.") from exc
    parquet_file = pq.ParquetFile(cache_path)
    schema_names = set(parquet_file.schema.names)
    l2_columns = [
        column
        for column in ["policy", "decode_step", "cache_l2_before", "cache_l2_after"]
        if column in schema_names
    ]
    role_columns = [
        column
        for column in schema_names
        if (
            column.startswith("retained_")
            and column.endswith("_tokens")
            and f"evicted_{column[len('retained_'):]}" in schema_names
        )
    ]
    columns = sorted(set(l2_columns + role_columns + ["policy"] + [
        f"evicted_{column[len('retained_'):]}" for column in role_columns
    ]))
    if "policy" not in columns:
        return {"l2_rows": [], "role_rows": []}

    l2_sums: dict[tuple[str, int], list[float]] = {}
    role_sums: dict[tuple[str, str], list[float]] = {}
    for batch in parquet_file.iter_batches(columns=columns, batch_size=100_000):
        table = batch.to_pydict()
        policies = table.get("policy", [])
        for idx, raw_policy in enumerate(policies):
            policy = str(raw_policy)
            if {"decode_step", "cache_l2_before", "cache_l2_after"}.issubset(table):
                before = _float_at(table, "cache_l2_before", idx)
                after = _float_at(table, "cache_l2_after", idx)
                if before:
                    key = (policy, int(_float_at(table, "decode_step", idx)))
                    l2_sums.setdefault(key, [0.0, 0.0])
                    l2_sums[key][0] += after / before
                    l2_sums[key][1] += 1.0
            for retained_col in role_columns:
                role = retained_col[len("retained_") : -len("_tokens")]
                evicted_col = f"evicted_{role}_tokens"
                retained = _float_at(table, retained_col, idx)
                evicted = _float_at(table, evicted_col, idx)
                if retained or evicted:
                    key = (policy, role)
                    role_sums.setdefault(key, [0.0, 0.0])
                    role_sums[key][0] += retained
                    role_sums[key][1] += evicted

    l2_rows = [
        {
            "policy": policy,
            "decode_step": decode_step,
            "l2_retained_fraction": total / count if count else None,
        }
        for (policy, decode_step), (total, count) in sorted(l2_sums.items())
    ]
    role_rows = []
    for (policy, role), (retained, evicted) in sorted(role_sums.items()):
        total = retained + evicted
        if total <= 0:
            continue
        role_rows.append(
            {
                "policy": policy,
                "role": role,
                "retention_fraction": retained / total,
                "retained_count": retained,
                "evicted_count": evicted,
            }
        )
    return {"l2_rows": l2_rows, "role_rows": role_rows}


def _float_at(table: dict[str, list[Any]], column: str, idx: int) -> float:
    values = table.get(column)
    if values is None:
        return 0.0
    value = values[idx]
    if value is None:
        return 0.0
    return float(value)


if __name__ == "__main__":
    main()
