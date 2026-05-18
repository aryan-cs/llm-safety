"""Generate three prototype visualizations for the paper.

1. SSEI Heatmap (models × policies)
2. Safety vs Capability Scatter
3. Causal Restoration Waterfall

Run: uv run python scripts/prototype_visuals.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

C_PRIMARY = "#FFB347"
C_DARK = "#E8943A"
C_LIGHT = "#FFD699"
C_MUTED = "#F5C28A"
C_ACCENT = "#FF8C42"

RESULTS = Path("results")
OUT = Path("docs/generated/prototype_visuals")
OUT.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = [
    "qwen2_5_7b_base",
    "qwen2_5_7b_instruct",
    "qwen2_5_14b_instruct",
    "qwen2_5_14b_msm_rules",
    "qwen2_5_14b_msm_value_aug",
    "qwen3_5_9b",
    "llama3_1_8b_instruct",
    "gemma2_9b_it",
    "mistral_7b_instruct_v0_3",
    "olmo3_7b_instruct",
    "phi4",
    "gpt_oss_20b",
]

MODEL_LABELS = {
    "qwen2_5_7b_base": "Qwen2.5-7B base",
    "qwen2_5_7b_instruct": "Qwen2.5-7B-Instruct",
    "qwen2_5_14b_instruct": "Qwen2.5-14B-Instruct",
    "qwen2_5_14b_msm_rules": "Qwen14B + MSM-Rules",
    "qwen2_5_14b_msm_value_aug": "Qwen14B + MSM-ValueAug",
    "qwen3_5_9b": "Qwen3-8B",
    "llama3_1_8b_instruct": "Llama-3.1-8B-Instruct",
    "gemma2_9b_it": "Gemma-2-9B-IT",
    "mistral_7b_instruct_v0_3": "Mistral-7B-Instruct-v0.3",
    "olmo3_7b_instruct": "OLMo-3-7B-Instruct",
    "phi4": "Phi-4",
    "gpt_oss_20b": "GPT-OSS-20B",
}

POLICY_ORDER = [
    "sliding_window__budget128",
    "sink_recent__budget128__sink8",
    "policy_pinned__budget128__sink8",
    "user_pinned__budget128__sink8",
    "random_matched__budget128__seed991",
]
POLICY_LABELS = {
    "sliding_window__budget128": "Sliding\nWindow",
    "sink_recent__budget128__sink8": "Sink+\nRecent",
    "policy_pinned__budget128__sink8": "Policy-\nPinned",
    "user_pinned__budget128__sink8": "User-\nPinned",
    "random_matched__budget128__seed991": "Random\nMatched",
}


def load_all_data():
    """Load SSEI and degradation data for all models."""
    all_data = {}
    for d in sorted(RESULTS.iterdir()):
        if not d.name.startswith("selectivity_h200_powered_"):
            continue
        model = d.name.removeprefix("selectivity_h200_powered_")
        mp = d / "metrics.json"
        if not mp.exists():
            continue
        metrics = json.loads(mp.read_text())
        contrasts = metrics.get("policy_level_contrasts", {})
        all_data[model] = contrasts
    return all_data


def make_heatmap(all_data: dict) -> None:
    """Figure 1: SSEI heatmap across models × policies."""
    models = [m for m in MODEL_ORDER if m in all_data]
    policies = POLICY_ORDER

    matrix = np.full((len(models), len(policies)), np.nan)
    ci_low_matrix = np.full((len(models), len(policies)), np.nan)
    # Track cells that are structurally N/A (role-based policies on base models)
    na_cells = set()
    role_policies = {"policy_pinned__budget128__sink8", "user_pinned__budget128__sink8"}

    for i, model in enumerate(models):
        for j, policy in enumerate(policies):
            if "base" in model and policy in role_policies:
                na_cells.add((i, j))
                continue
            payload = all_data[model].get(policy, {})
            ssei = payload.get("selective_safety_erasure_index")
            ci = payload.get("selective_safety_erasure_index_ci", {})
            if ssei is not None:
                matrix[i, j] = ssei
                ci_low_matrix[i, j] = ci.get("ci_low", np.nan)

    # Diverging orange colormap: white at 0, deep orange for positive, cool gray for negative
    cmap_colors = [
        (0.0, "#4A6274"),    # negative: steel blue-gray
        (0.15, "#8FAABB"),   # light blue-gray
        (0.4, "#E8E8E8"),    # near-zero: light gray
        (0.5, "#FFFFFF"),    # zero: white
        (0.6, "#FFE8C8"),    # slight positive: cream
        (0.75, "#FFB347"),   # moderate: primary orange
        (1.0, "#CC5500"),    # strong positive: burnt orange
    ]
    cmap = mcolors.LinearSegmentedColormap.from_list("ssei_diverging", cmap_colors)

    vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)))
    vmin = -vmax

    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    # Annotate cells
    for i in range(len(models)):
        for j in range(len(policies)):
            if (i, j) in na_cells:
                # Hatched background for N/A cells
                from matplotlib.patches import FancyBboxPatch
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                     facecolor="#F0F0F0", edgecolor="none", zorder=1)
                ax.add_patch(rect)
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=7,
                        color="#AAAAAA", fontstyle="italic")
                continue
            val = matrix[i, j]
            ci_lo = ci_low_matrix[i, j]
            if np.isnan(val):
                ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="#999999")
                continue
            # Bold + star if CI excludes zero (positive)
            sig = ci_lo > 0 if not np.isnan(ci_lo) else False
            color = "black" if abs(val) < 0.06 else "white" if val > 0.06 else "white"
            weight = "bold" if sig else "normal"
            _v = val if round(val, 3) != 0.0 else 0.0
            label = f"{_v:+.3f}"
            if sig:
                label += " *"
            ax.text(j, i, label, ha="center", va="center", fontsize=7.5,
                    color=color, fontweight=weight)

    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels([POLICY_LABELS[p] for p in policies], fontsize=9, ha="center")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([MODEL_LABELS[m] for m in models], fontsize=9)

    ax.set_xlabel("Cache Eviction Policy", fontsize=11, labelpad=10)
    ax.set_title("Selective Safety Erasure Index by Model and Policy", fontsize=13, pad=15)

    # Add family separators
    family_breaks = []
    families = []
    for i, m in enumerate(models):
        fam = m.split("_")[0]
        if not families or fam != families[-1]:
            if families:
                family_breaks.append(i - 0.5)
            families.append(fam)
    for yb in family_breaks:
        ax.axhline(yb, color="#666666", linewidth=0.8, linestyle="-")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("SSEI", fontsize=10)

    # Legend note
    ax.text(0.5, -0.18, "* = 95% CI lower bound excludes zero (statistically significant positive selectivity)",
            transform=ax.transAxes, ha="center", fontsize=8, color="#666666")

    fig.tight_layout()
    fig.savefig(OUT / "01_ssei_heatmap.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "01_ssei_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT / '01_ssei_heatmap.png'}")


def make_scatter(all_data: dict) -> None:
    """Figure 2: Safety delta vs Capability delta scatter for sliding_window."""
    from adjustText import adjust_text

    policy = "sliding_window__budget128"

    fig, ax = plt.subplots(figsize=(9, 7.5))

    model_colors = {
        "qwen2_5_7b_base": C_LIGHT,
        "qwen2_5_7b_instruct": C_PRIMARY,
        "qwen2_5_14b_instruct": C_DARK,
        "qwen2_5_14b_msm_rules": "#CC7A2E",
        "qwen2_5_14b_msm_value_aug": "#B36B20",
        "qwen3_5_9b": C_ACCENT,
        "llama3_1_8b_instruct": "#E87A00",
        "gemma2_9b_it": "#FFCC80",
        "mistral_7b_instruct_v0_3": C_MUTED,
        "olmo3_7b_instruct": "#D4944A",
        "phi4": "#FF6B00",
        "gpt_oss_20b": "#8B5E3C",
    }

    xs, ys, labels, colors = [], [], [], []
    for model in MODEL_ORDER:
        if model not in all_data:
            continue
        payload = all_data[model].get(policy, {})
        safety_ci = payload.get("safety_degradation_ci", {})
        cap_ci = payload.get("capability_degradation_ci", {})
        sd = safety_ci.get("mean")
        cd = cap_ci.get("mean")
        if sd is None or cd is None:
            continue
        xs.append(cd)
        ys.append(sd)
        labels.append(MODEL_LABELS[model])
        colors.append(model_colors.get(model, C_PRIMARY))

    # Set axis limits first so fills cover the full area
    lim_max = max(max(abs(v) for v in xs + ys) * 1.5, 0.02)
    ax.set_xlim(-lim_max, lim_max)
    ax.set_ylim(-lim_max, lim_max)

    # Fill triangles edge-to-edge using polygon patches
    big = lim_max * 2  # oversized to guarantee full coverage after clipping
    upper_tri = plt.Polygon(
        [(-big, -big), (big, big), (-big, big)],
        closed=True, facecolor=C_DARK, alpha=0.08, zorder=0, clip_on=True,
    )
    lower_tri = plt.Polygon(
        [(-big, -big), (big, big), (big, -big)],
        closed=True, facecolor="#4A6274", alpha=0.05, zorder=0, clip_on=True,
    )
    ax.add_patch(upper_tri)
    ax.add_patch(lower_tri)

    # Diagonal line (SSEI = 0)
    ax.plot([-big, big], [-big, big], color="#CCCCCC", linewidth=1, linestyle="--", zorder=1)

    # Region labels
    ax.text(0.03, 0.97, "Safety degrades more\nthan capability (SSEI > 0)",
            transform=ax.transAxes, fontsize=8, color=C_DARK, va="top", fontstyle="italic")
    ax.text(0.97, 0.03, "Capability degrades more\nthan safety (SSEI < 0)",
            transform=ax.transAxes, fontsize=8, color="#4A6274", ha="right", fontstyle="italic")

    # Manual label placement: specify which side each label goes on
    # Left-side labels for points on the left side of the plot or where right would collide
    left_labels = {
        "Qwen3-8B", "OLMo-3-7B-Instruct", "Qwen2.5-7B-Instruct",
        "Qwen14B + MSM-Rules", "GPT-OSS-20B",
    }

    bbox_props = dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.2)
    arrow_props = dict(arrowstyle="-", color="#AAAAAA", linewidth=0.6, zorder=2)

    # Compute offset in data coords (scaled to axis range)
    x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    h_offset = x_range * 0.04
    v_offset = y_range * 0.015

    for x, y, label in zip(xs, ys, labels):
        if label in left_labels:
            ha = "right"
            tx = x - h_offset
        else:
            ha = "left"
            tx = x + h_offset
        ty = y + v_offset

        ax.annotate(
            label, xy=(x, y), xytext=(tx, ty),
            fontsize=6.5, color="#333333", ha=ha, va="center",
            zorder=4,
            bbox=bbox_props,
            arrowprops=arrow_props,
        )

    # Draw scatter points AFTER labels+arrows so they sit on top (zorder=5)
    for x, y, color in zip(xs, ys, colors):
        ax.scatter(x, y, s=100, c=color, edgecolors="white", linewidth=1, zorder=5)

    ax.set_xlabel("Capability Degradation (higher = more capability lost)", fontsize=10)
    ax.set_ylabel("Safety Degradation (higher = more safety lost)", fontsize=10)
    ax.set_title("Safety vs. Capability Degradation\n(Sliding Window, budget=128)", fontsize=12)
    ax.axhline(0, color="#DDDDDD", linewidth=0.5, zorder=0)
    ax.axvline(0, color="#DDDDDD", linewidth=0.5, zorder=0)

    fig.tight_layout()
    fig.savefig(OUT / "02_safety_vs_capability_scatter.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "02_safety_vs_capability_scatter.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT / '02_safety_vs_capability_scatter.png'}")


def make_waterfall() -> None:
    """Figure 3: Causal restoration waterfall for each model."""
    # Data from causal patching results (refusal restoration fractions)
    # For each model: compressed baseline (0), +system patch, +user patch, policy-pinned (1.0)
    models_data = {}

    for model_dir in ["h200_causal_patch_qwen7b", "h200_causal_patch_qwen3_5_9b",
                       "h200_causal_patch_llama3_1_8b_instruct"]:
        mp = RESULTS / model_dir / "metrics.json"
        if not mp.exists():
            continue
        data = json.loads(mp.read_text())
        cr = data.get("causal_restoration", {})

        sys_key = [k for k in cr if "refusal_safety" in k and "rolesystem" in k and "patchkey-value" in k and "kv_int4" in k]
        user_key = [k for k in cr if "refusal_safety" in k and "roleuser" in k and "patchkey-value" in k and "kv_int4" in k]
        pinned_key = [k for k in cr if "refusal_safety" in k and "policy_pinned" in k]

        sys_rf = cr[sys_key[0]]["safety_restoration_fraction"] if sys_key and "safety_restoration_fraction" in cr[sys_key[0]] else (
            cr[sys_key[0]].get("refusal_restoration_fraction") if sys_key else None
        )
        user_rf = cr[user_key[0]]["safety_restoration_fraction"] if user_key and "safety_restoration_fraction" in cr[user_key[0]] else (
            cr[user_key[0]].get("refusal_restoration_fraction") if user_key else None
        )
        pinned_rf = cr[pinned_key[0]]["safety_restoration_fraction"] if pinned_key and "safety_restoration_fraction" in cr[pinned_key[0]] else (
            cr[pinned_key[0]].get("refusal_restoration_fraction") if pinned_key else None
        )

        # Get CI for sys and user
        sys_ci = {}
        user_ci = {}
        if sys_key:
            sys_ci = cr[sys_key[0]].get("safety_restoration_fraction_ci",
                      cr[sys_key[0]].get("refusal_restoration_fraction_ci", {}))
        if user_key:
            user_ci = cr[user_key[0]].get("safety_restoration_fraction_ci",
                       cr[user_key[0]].get("refusal_restoration_fraction_ci", {}))

        label_map = {
            "h200_causal_patch_qwen7b": "Qwen2.5-7B",
            "h200_causal_patch_qwen3_5_9b": "Qwen3-8B",
            "h200_causal_patch_llama3_1_8b_instruct": "Llama-3.1-8B",
        }
        models_data[label_map.get(model_dir, model_dir)] = {
            "sys": sys_rf,
            "user": user_rf,
            "pinned": pinned_rf,
            "sys_ci_low": sys_ci.get("ci_low"),
            "sys_ci_high": sys_ci.get("ci_high"),
            "user_ci_low": user_ci.get("ci_low"),
            "user_ci_high": user_ci.get("ci_high"),
        }

    if not models_data:
        print("No causal patching data found, skipping waterfall.")
        return

    fig, axes = plt.subplots(1, len(models_data), figsize=(4 * len(models_data), 5), sharey=True)
    if len(models_data) == 1:
        axes = [axes]

    bar_colors = [
        "#E0E0E0",   # compressed (gray)
        C_DARK,      # +system patch
        C_PRIMARY,   # +user patch
        C_ACCENT,    # policy-pinned
    ]

    for idx, (model_name, vals) in enumerate(models_data.items()):
        ax = axes[idx]

        stages = ["Compressed\n(baseline)", "  + System-role\n  K+V patch", "  + User-role\n  K+V patch", "Policy-pinned\n(full retention)"]
        heights = [0.0, vals["sys"], vals["user"], vals["pinned"] if vals["pinned"] else 1.0]

        # Clamp to [0, 1] for display
        heights_display = [max(0, min(1, h)) if h is not None else 0 for h in heights]

        bars = ax.bar(range(len(stages)), heights_display, color=bar_colors, edgecolor="white",
                      linewidth=1.5, width=0.7, zorder=2)

        # Add value labels on bars
        for i, (bar, h) in enumerate(zip(bars, heights)):
            if h is not None and h != 0:
                label_y = min(max(h, 0), 1)
                ax.text(bar.get_x() + bar.get_width() / 2, label_y + 0.03,
                        f"{h:.2f}" if abs(h) <= 1 else f"{h:.1f}",
                        ha="center", va="bottom", fontsize=9, fontweight="bold",
                        color="#333333")
            elif i == 0:
                ax.text(bar.get_x() + bar.get_width() / 2, 0.03,
                        "0.00", ha="center", va="bottom", fontsize=9,
                        color="#999999")

        # Dashed line at 1.0 (full restoration)
        ax.axhline(1.0, color="#CCCCCC", linewidth=0.8, linestyle="--", zorder=1)
        ax.text(3.4, 1.02, "full\nbaseline", fontsize=7, color="#999999", ha="center")

        ax.set_xticks(range(len(stages)))
        ax.set_xticklabels(stages, fontsize=7.5)
        ax.set_title(model_name, fontsize=11, fontweight="bold", color=C_DARK)
        ax.set_ylim(-0.15, 1.25)

        if idx == 0:
            ax.set_ylabel("Restoration toward baseline\n(0 = compressed, 1 = full)", fontsize=9)

        # Add grid
        ax.yaxis.grid(True, alpha=0.3, linestyle="-")
        ax.set_axisbelow(True)

    fig.suptitle("Causal Restoration: How Much Safety Does Each Intervention Recover?",
                 fontsize=13, y=1.02)

    fig.tight_layout()
    fig.savefig(OUT / "03_causal_waterfall.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "03_causal_waterfall.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT / '03_causal_waterfall.png'}")


def main():
    all_data = load_all_data()
    make_heatmap(all_data)
    make_scatter(all_data)
    make_waterfall()
    print(f"\nAll prototypes saved to {OUT}/")


if __name__ == "__main__":
    main()
