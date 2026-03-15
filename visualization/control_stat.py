"""
Controlled Statistical Comparison of ATD Values by Word Order

This script performs statistical tests to determine whether languages with
the same word order as a focal language have significantly different ATD
(Attention Transport Distance) values compared to languages with different
word orders, after excluding genetically/areally related languages.

Statistical Tests Used:
1. Mann-Whitney U test (primary): Non-parametric test comparing two groups
2. Permutation test (robustness): Distribution-free significance test
3. Effect size (Cohen's d and rank-biserial correlation)

Author: Generated for LLMVis project
"""

import os
import sys
import pickle
import numpy as np
from scipy import stats

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import seaborn as sns

from utils import (
    get_all_lang,
    get_focal_language_groups,
    get_comparison_groups_for_focal,
    get_word_order_for_lang,
)


# =====================================================================
# Configuration
# =====================================================================

MODEL, TOP_K, THR = "m2m", 2000, 0.2
RESULTS_DIR = "results_m2m"
SUFFIX = "_fixed_ot"

# For LLaMA model:
# MODEL, TOP_K, THR = "llama3", 100, 0.0
# RESULTS_DIR = "results_llama3"
# SUFFIX = "_fixed_clip_layer"

OUTPUT_DIR = f"control_stat_results/{MODEL}_{TOP_K}_{THR}"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================================================
# Statistical Test Functions
# =====================================================================


def mann_whitney_test(group1, group2):
    """
    Perform Mann-Whitney U test (Wilcoxon rank-sum test).

    This is a non-parametric test that compares whether two samples
    come from the same distribution.

    Returns:
        statistic: U statistic
        p_value: Two-sided p-value
        rank_biserial: Effect size (rank-biserial correlation, -1 to 1)
    """
    if len(group1) == 0 or len(group2) == 0:
        return np.nan, np.nan, np.nan

    statistic, p_value = stats.mannwhitneyu(group1, group2, alternative="two-sided")

    # Calculate rank-biserial correlation as effect size
    # r = 1 - (2U)/(n1*n2), where U is the smaller of U1 and U2
    n1, n2 = len(group1), len(group2)
    # The statistic from mannwhitneyu is U for the first sample
    rank_biserial = 1 - (2 * statistic) / (n1 * n2)

    return statistic, p_value, rank_biserial


def permutation_test(group1, group2, n_permutations=10000, seed=42):
    """
    Perform permutation test for difference in means.

    This test makes no distributional assumptions and tests whether
    the observed difference in means could have occurred by chance.

    Returns:
        observed_diff: Observed difference in means (group1 - group2)
        p_value: Two-sided p-value
    """
    if len(group1) == 0 or len(group2) == 0:
        return np.nan, np.nan

    np.random.seed(seed)

    # Observed difference
    observed_diff = np.mean(group1) - np.mean(group2)

    # Combine data
    combined = np.concatenate([group1, group2])
    n1 = len(group1)
    n_total = len(combined)

    # Generate permutation distribution
    perm_diffs = np.zeros(n_permutations)
    for i in range(n_permutations):
        np.random.shuffle(combined)
        perm_diffs[i] = np.mean(combined[:n1]) - np.mean(combined[n1:])

    # Two-sided p-value
    p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))

    return observed_diff, p_value


def cohens_d(group1, group2):
    """
    Calculate Cohen's d effect size.

    Interpretation:
        |d| < 0.2: negligible
        0.2 <= |d| < 0.5: small
        0.5 <= |d| < 0.8: medium
        |d| >= 0.8: large
    """
    if len(group1) == 0 or len(group2) == 0:
        return np.nan

    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return np.nan

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def bootstrap_ci(group1, group2, n_bootstrap=10000, ci=0.95, seed=42):
    """
    Calculate bootstrap confidence interval for difference in means.

    Returns:
        mean_diff: Point estimate of difference
        ci_low: Lower bound of CI
        ci_high: Upper bound of CI
    """
    if len(group1) == 0 or len(group2) == 0:
        return np.nan, np.nan, np.nan

    np.random.seed(seed)

    boot_diffs = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        boot1 = np.random.choice(group1, size=len(group1), replace=True)
        boot2 = np.random.choice(group2, size=len(group2), replace=True)
        boot_diffs[i] = np.mean(boot1) - np.mean(boot2)

    alpha = 1 - ci
    ci_low = np.percentile(boot_diffs, 100 * alpha / 2)
    ci_high = np.percentile(boot_diffs, 100 * (1 - alpha / 2))
    mean_diff = np.mean(group1) - np.mean(group2)

    return mean_diff, ci_low, ci_high


# =====================================================================
# Data Loading
# =====================================================================


def load_distance_data():
    """Load the pre-computed distance matrices."""
    print("=" * 60)
    print("Loading distance data")
    print("=" * 60)

    with open(f"{RESULTS_DIR}/selected_{MODEL}_{TOP_K}_{THR}.pkl", "rb") as f:
        sel = pickle.load(f)
    with open(f"{RESULTS_DIR}/all_{MODEL}_distance_dict{SUFFIX}.pkl", "rb") as f:
        all_dist = pickle.load(f)

    all_lang_dict, all_lang_list = get_all_lang()
    idx_sel = sel["selected_indices"]
    lang_sel = sel["selected_languages"]
    lang_idx = [all_lang_list.index(l) for l in lang_sel]

    print(f"Number of selected sentences: {len(idx_sel)}")
    print(f"Number of languages: {len(lang_sel)}")
    print(f"Languages: {lang_sel}")

    # Filter valid indices
    valid_indices = [i for i in idx_sel if i in all_dist]
    if len(valid_indices) < len(idx_sel):
        print(f"Warning: {len(idx_sel) - len(valid_indices)} missing indices")

    # Build distance tensor
    sub = [all_dist[i][lang_idx][:, lang_idx] for i in valid_indices]
    tensor = np.stack(sub, axis=0)
    # tensor shape: (num_sentences, num_langs, num_langs, num_layers)

    print(f"Distance tensor shape: {tensor.shape}")

    return tensor, lang_sel, all_lang_dict


def get_atd_values_for_focal(tensor, lang_sel, focal_lang, target_langs):
    """
    Extract mean ATD values between focal language and target languages.
    Returns one value per target language (averaged across sentences and layers).

    Args:
        tensor: Distance tensor (sentences, langs, langs, layers)
        lang_sel: List of language codes in the tensor
        focal_lang: The focal language code
        target_langs: List of target language codes

    Returns:
        Array of mean ATD values (one per target language)
    """
    if focal_lang not in lang_sel:
        print(f"Warning: Focal language {focal_lang} not in selected languages")
        return np.array([])

    focal_idx = lang_sel.index(focal_lang)
    target_indices = [lang_sel.index(l) for l in target_langs if l in lang_sel]

    if len(target_indices) == 0:
        return np.array([])

    atd_values = []
    for target_idx in target_indices:
        mean_atd = tensor[:, focal_idx, target_idx, :].mean()
        atd_values.append(mean_atd)
    return np.array(atd_values)


def get_mean_atd_per_language(tensor, lang_sel, focal_lang, target_langs):
    """
    Get mean ATD value for each target language (averaged across sentences and layers).

    Returns:
        dict: {lang_code: mean_atd_value}
    """
    if focal_lang not in lang_sel:
        return {}

    focal_idx = lang_sel.index(focal_lang)
    result = {}

    for lang in target_langs:
        if lang in lang_sel:
            target_idx = lang_sel.index(lang)
            # Mean across sentences and layers
            mean_atd = tensor[:, focal_idx, target_idx, :].mean()
            result[lang] = mean_atd

    return result


# =====================================================================
# Main Analysis
# =====================================================================


def run_controlled_comparison(tensor, lang_sel, all_lang_dict):
    """
    Run controlled comparison for all focal languages.
    """
    results = {}
    focal_groups = get_focal_language_groups()

    for focal_lang, info in focal_groups.items():
        print("\n" + "=" * 70)
        print(
            f"Focal Language: {info['name']} ({focal_lang}) - Word Order: {info['word_order']}"
        )
        print("=" * 70)

        # Get comparison groups
        groups = get_comparison_groups_for_focal(focal_lang, available_langs=lang_sel)

        print(f"\nExcluded (related): {groups['excluded']}")
        print(f"  Reason: {groups['excluded_reason']}")
        print(
            f"\nSame word order ({groups['focal_word_order']}): {len(groups['same_word_order'])} languages"
        )
        print(f"  {groups['same_word_order']}")
        print(f"\nDifferent word order: {len(groups['diff_word_order'])} languages")
        print(f"  {groups['diff_word_order']}")

        # Check if focal language is in the data
        if focal_lang not in lang_sel:
            print(f"\nWarning: {focal_lang} not in selected languages, skipping...")
            continue

        # Get ATD values for each group
        atd_same_wo = get_atd_values_for_focal(
            tensor, lang_sel, focal_lang, groups["same_word_order"]
        )
        atd_diff_wo = get_atd_values_for_focal(
            tensor, lang_sel, focal_lang, groups["diff_word_order"]
        )

        print(f"\n--- ATD Statistics (per-language mean values) ---")
        print(
            f"Same word order: n={len(atd_same_wo)} languages, mean={np.mean(atd_same_wo):.4f}, "
            f"std={np.std(atd_same_wo):.4f}, median={np.median(atd_same_wo):.4f}"
        )
        print(
            f"Diff word order: n={len(atd_diff_wo)} languages, mean={np.mean(atd_diff_wo):.4f}, "
            f"std={np.std(atd_diff_wo):.4f}, median={np.median(atd_diff_wo):.4f}"
        )

        # Statistical tests
        print(f"\n--- Statistical Tests ---")

        # Mann-Whitney U test (non-parametric, good for small samples)
        u_stat, mw_p, rank_biserial = mann_whitney_test(atd_same_wo, atd_diff_wo)
        print(f"Mann-Whitney U test:")
        print(f"  U statistic: {u_stat:.2f}")
        print(f"  p-value: {mw_p:.4f}")
        print(f"  Rank-biserial r: {rank_biserial:.4f}")

        # Independent t-test (for comparison)
        t_stat, t_p = stats.ttest_ind(atd_same_wo, atd_diff_wo)
        print(f"\nIndependent t-test:")
        print(f"  t statistic: {t_stat:.4f}")
        print(f"  p-value: {t_p:.4f}")

        # Effect size
        d = cohens_d(atd_same_wo, atd_diff_wo)
        print(f"\nCohen's d: {d:.4f}")
        if abs(d) < 0.2:
            effect_interp = "negligible"
        elif abs(d) < 0.5:
            effect_interp = "small"
        elif abs(d) < 0.8:
            effect_interp = "medium"
        else:
            effect_interp = "large"
        print(f"  Interpretation: {effect_interp}")

        # Per-language mean ATD
        print(f"\n--- Per-Language Mean ATD ---")
        same_wo_means = get_mean_atd_per_language(
            tensor, lang_sel, focal_lang, groups["same_word_order"]
        )
        diff_wo_means = get_mean_atd_per_language(
            tensor, lang_sel, focal_lang, groups["diff_word_order"]
        )

        print(f"\nSame word order languages (sorted by ATD):")
        for lang, atd in sorted(same_wo_means.items(), key=lambda x: x[1]):
            lang_name = all_lang_dict.get(lang, lang)
            print(f"  {lang} ({lang_name}): {atd:.4f}")

        print(f"\nDifferent word order languages (sorted by ATD):")
        for lang, atd in sorted(diff_wo_means.items(), key=lambda x: x[1]):
            lang_name = all_lang_dict.get(lang, lang)
            wo = get_word_order_for_lang(lang)
            print(f"  {lang} ({lang_name}, {wo}): {atd:.4f}")

        # Store results
        results[focal_lang] = {
            "focal_name": info["name"],
            "focal_word_order": info["word_order"],
            "excluded": groups["excluded"],
            "same_word_order_langs": groups["same_word_order"],
            "diff_word_order_langs": groups["diff_word_order"],
            "n_same_wo": len(atd_same_wo),
            "n_diff_wo": len(atd_diff_wo),
            "atd_same_wo_mean": np.mean(atd_same_wo),
            "atd_same_wo_std": np.std(atd_same_wo),
            "atd_diff_wo_mean": np.mean(atd_diff_wo),
            "atd_diff_wo_std": np.std(atd_diff_wo),
            "mann_whitney_p": mw_p,
            "t_test_p": t_p,
            "cohens_d": d,
            "same_wo_per_lang": same_wo_means,
            "diff_wo_per_lang": diff_wo_means,
        }

    return results


def print_summary_table(results):
    """Print a summary table of all results."""
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)

    print(
        f"\n{'Focal Lang':<12} {'Word Order':<10} {'Same WO Mean':<14} "
        f"{'Diff WO Mean':<14} {'MW p-value':<12} {'Cohen d':<10} {'Effect':<10}"
    )
    print("-" * 80)

    for focal, res in results.items():
        effect = (
            "negligible"
            if abs(res["cohens_d"]) < 0.2
            else (
                "small"
                if abs(res["cohens_d"]) < 0.5
                else "medium" if abs(res["cohens_d"]) < 0.8 else "large"
            )
        )

        print(
            f"{focal:<12} {res['focal_word_order']:<10} "
            f"{res['atd_same_wo_mean']:<14.4f} {res['atd_diff_wo_mean']:<14.4f} "
            f"{res['mann_whitney_p']:<12.2e} {res['cohens_d']:<10.4f} {effect:<10}"
        )


def generate_latex_table(results, all_lang_dict):
    """Generate LaTeX table for paper."""
    latex = []
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"\centering")
    latex.append(
        r"\caption{Controlled comparison of ATD values by word order. "
        r"For each focal language, we exclude related languages and compare "
        r"ATD values to languages with same vs. different word orders. "
        r"Significance tested via Mann-Whitney U test.}"
    )
    latex.append(r"\label{tab:word_order_comparison}")
    latex.append(r"\begin{tabular}{lcccccc}")
    latex.append(r"\toprule")
    latex.append(
        r"\textbf{Focal} & \textbf{WO} & \textbf{Same WO} & \textbf{Diff WO} & "
        r"\textbf{p-value} & \textbf{Cohen's d} & \textbf{Effect} \\"
    )
    latex.append(r"\midrule")

    for focal, res in results.items():
        effect = (
            "negligible"
            if abs(res["cohens_d"]) < 0.2
            else (
                "small"
                if abs(res["cohens_d"]) < 0.5
                else "medium" if abs(res["cohens_d"]) < 0.8 else "large"
            )
        )

        p_str = (
            f"{res['mann_whitney_p']:.2e}"
            if res["mann_whitney_p"] < 0.001
            else f"{res['mann_whitney_p']:.3f}"
        )

        latex.append(
            f"{res['focal_name']} & {res['focal_word_order']} & "
            f"{res['atd_same_wo_mean']:.3f} & {res['atd_diff_wo_mean']:.3f} & "
            f"{p_str} & {res['cohens_d']:.3f} & {effect} \\\\"
        )

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    return "\n".join(latex)


def generate_grouping_latex_table(all_lang_dict, lang_sel):
    """Generate LaTeX table showing language groupings for each focal language."""
    focal_groups = get_focal_language_groups()

    latex = []
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"\centering")
    latex.append(r"\small")
    latex.append(
        r"\caption{Language groupings for controlled word order comparison. "
        r"For each focal language, we exclude genetically/areally related languages, "
        r"then compare ATD values between languages with the same word order (WO) "
        r"versus different word orders.}"
    )
    latex.append(r"\label{tab:focal_language_groups}")
    latex.append(r"\begin{tabular}{llp{9cm}}")
    latex.append(r"\toprule")
    latex.append(r"\textbf{Focal Language} & \textbf{Category} & \textbf{Languages} \\")
    latex.append(r"\midrule")

    for focal_lang, info in focal_groups.items():
        groups = get_comparison_groups_for_focal(focal_lang, available_langs=lang_sel)

        # Format language lists with full names
        def format_langs(lang_list, max_show=8):
            formatted = []
            for l in lang_list[:max_show]:
                name = all_lang_dict.get(l, l)
                formatted.append(f"{name} ({l})")
            if len(lang_list) > max_show:
                formatted.append(f"... (+{len(lang_list) - max_show} more)")
            return ", ".join(formatted)

        excluded_str = format_langs(groups["excluded"], max_show=10)
        same_wo_str = format_langs(groups["same_word_order"], max_show=6)
        diff_wo_str = format_langs(groups["diff_word_order"], max_show=6)

        # First row with multirow
        n_rows = 3
        latex.append(
            f"\\multirow{{{n_rows}}}{{*}}{{\\textbf{{{info['name']} ({info['word_order']})}}}} "
        )
        latex.append(f"  & Excluded & {excluded_str} \\\\")
        latex.append(f"  & Same WO ({info['word_order']}) & {same_wo_str} \\\\")
        latex.append(f"  & Different WO & {diff_wo_str} \\\\")
        latex.append(r"\midrule")

    # Remove last midrule
    latex[-1] = r"\bottomrule"

    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    return "\n".join(latex)


# =====================================================================
# Visualization Functions
# =====================================================================


def _compute_uniform_jitter(atd_values, max_jitter=0.3, y_tolerance=0.05):
    """
    Compute x jitter so that points with similar y values are uniformly distributed.

    Args:
        atd_values: Array of ATD values (y positions)
        max_jitter: Maximum jitter range (±max_jitter/2)
        y_tolerance: Points within this y distance are considered "same row"

    Returns:
        Array of jitter values for each point
    """
    n = len(atd_values)
    if n == 0:
        return np.array([])

    # Sort by y value to group nearby points
    sorted_indices = np.argsort(atd_values)
    jitter = np.zeros(n)

    i = 0
    while i < n:
        # Find all points in the same "row" (within y_tolerance)
        row_start = i
        row_indices = [sorted_indices[i]]
        current_y = atd_values[sorted_indices[i]]

        while i + 1 < n and abs(atd_values[sorted_indices[i + 1]] - current_y) < y_tolerance:
            i += 1
            row_indices.append(sorted_indices[i])

        # Distribute points in this row uniformly
        row_count = len(row_indices)
        if row_count == 1:
            # Single point: center it
            jitter[row_indices[0]] = 0
        else:
            # Multiple points: distribute uniformly from -max_jitter/2 to +max_jitter/2
            positions = np.linspace(-max_jitter / 2, max_jitter / 2, row_count)
            for j, idx in enumerate(row_indices):
                jitter[idx] = positions[j]

        i += 1

    return jitter


def _prepare_plot_data(tensor, lang_sel, focal_lang, groups):
    """
    Prepare data for plotting: ATD values, language codes, and jitter positions.
    Returns consistent data for both scatter and label plots.
    """
    same_wo_langs = [l for l in groups["same_word_order"] if l in lang_sel]
    diff_wo_langs = [l for l in groups["diff_word_order"] if l in lang_sel]

    atd_same_wo = get_atd_values_for_focal(
        tensor, lang_sel, focal_lang, same_wo_langs
    )
    atd_diff_wo = get_atd_values_for_focal(
        tensor, lang_sel, focal_lang, diff_wo_langs
    )

    # Compute uniform jitter based on y value clustering
    # y_tolerance=0.03 to avoid grouping points that are slightly apart
    jitter_same = _compute_uniform_jitter(atd_same_wo, max_jitter=0.8, y_tolerance=0.03)
    jitter_diff = _compute_uniform_jitter(atd_diff_wo, max_jitter=0.8, y_tolerance=0.03)

    return {
        "same_wo_langs": same_wo_langs,
        "diff_wo_langs": diff_wo_langs,
        "atd_same_wo": atd_same_wo,
        "atd_diff_wo": atd_diff_wo,
        "jitter_same": jitter_same,
        "jitter_diff": jitter_diff,
    }


def plot_atd_comparison(tensor, lang_sel, all_lang_dict, save_dir, figsize=(14, 5)):
    """
    Create strip plot + box plot showing per-language ATD values for all focal languages.
    Each point = one language's mean ATD to the focal language.
    """
    import pandas as pd

    focal_groups = get_focal_language_groups()
    n_focal = len(focal_groups)

    fig, axes = plt.subplots(1, n_focal, figsize=figsize)
    if n_focal == 1:
        axes = [axes]

    colors = {"Same WO": "#E41A1C", "Diff WO": "#377EB8"}

    # Collect all ATD values to determine consistent y range
    all_atd = []
    plot_data_list = []

    for focal_lang, info in focal_groups.items():
        if focal_lang not in lang_sel:
            plot_data_list.append(None)
            continue
        groups = get_comparison_groups_for_focal(focal_lang, available_langs=lang_sel)
        plot_data = _prepare_plot_data(tensor, lang_sel, focal_lang, groups)
        plot_data_list.append(plot_data)
        all_atd.extend(plot_data["atd_same_wo"].tolist())
        all_atd.extend(plot_data["atd_diff_wo"].tolist())

    # Fixed y limits
    y_lim = (1.0, 3.0)

    for idx, (focal_lang, info) in enumerate(focal_groups.items()):
        ax = axes[idx]
        plot_data = plot_data_list[idx]

        if plot_data is None:
            ax.text(
                0.5, 0.5, f"{focal_lang} not in data",
                ha="center", va="center", transform=ax.transAxes,
            )
            continue

        atd_same_wo = plot_data["atd_same_wo"]
        atd_diff_wo = plot_data["atd_diff_wo"]
        jitter_same = plot_data["jitter_same"]
        jitter_diff = plot_data["jitter_diff"]

        # Build dataframe for box plot
        data = []
        for val in atd_same_wo:
            data.append({"Group": "Same WO", "ATD": val})
        for val in atd_diff_wo:
            data.append({"Group": "Diff WO", "ATD": val})
        df = pd.DataFrame(data)

        # Box plot only
        sns.boxplot(
            data=df, x="Group", y="ATD", palette=colors, ax=ax,
            width=0.9, showfliers=False,
        )

        # Scatter points (scale jitter to match box width 0.8)
        ax.scatter(0 + jitter_same, atd_same_wo, color=colors["Same WO"],
                   s=60, alpha=0.7, zorder=3)
        ax.scatter(1 + jitter_diff, atd_diff_wo, color=colors["Diff WO"],
                   s=60, alpha=0.7, zorder=3)

        # Set consistent y limits
        ax.set_ylim(y_lim)

        # Stats
        mean_same = np.mean(atd_same_wo)
        mean_diff = np.mean(atd_diff_wo)
        focal_name = info["name"]
        focal_wo = info["word_order"]

        ax.set_title(
            f"{focal_name} ({focal_wo})\n"
            f"n={len(atd_same_wo)} vs {len(atd_diff_wo)}, "
            f"$\\mu$={mean_same:.2f} vs {mean_diff:.2f}",
            fontsize=11,
        )
        ax.set_xlabel("")
        if idx == 0:
            ax.set_ylabel("ATD", fontsize=11)
        else:
            ax.set_ylabel("")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "atd_comparison_boxstrip.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot saved: {save_path}")


def plot_atd_comparison_with_labels(tensor, lang_sel, all_lang_dict, save_dir, figsize=(14, 5)):
    """
    Create box plot with language text labels instead of dots.
    Each label = one language's code, positioned at its mean ATD value.
    """
    import pandas as pd

    focal_groups = get_focal_language_groups()
    n_focal = len(focal_groups)

    fig, axes = plt.subplots(1, n_focal, figsize=figsize)
    if n_focal == 1:
        axes = [axes]

    # Light colors for box, dark colors for text
    box_colors = {"Same WO": "#FFCCCC", "Diff WO": "#CCE5FF"}  # Light red, light blue
    text_colors = {"Same WO": "#B22222", "Diff WO": "#1E3A8A"}  # Dark red, dark blue

    # Collect all ATD values to determine consistent y range
    all_atd = []
    plot_data_list = []

    for focal_lang, info in focal_groups.items():
        if focal_lang not in lang_sel:
            plot_data_list.append(None)
            continue
        groups = get_comparison_groups_for_focal(focal_lang, available_langs=lang_sel)
        plot_data = _prepare_plot_data(tensor, lang_sel, focal_lang, groups)
        plot_data_list.append(plot_data)
        all_atd.extend(plot_data["atd_same_wo"].tolist())
        all_atd.extend(plot_data["atd_diff_wo"].tolist())

    # Fixed y limits
    y_lim = (1.0, 3.0)

    for idx, (focal_lang, info) in enumerate(focal_groups.items()):
        ax = axes[idx]
        plot_data = plot_data_list[idx]

        if plot_data is None:
            ax.text(
                0.5, 0.5, f"{focal_lang} not in data",
                ha="center", va="center", transform=ax.transAxes,
            )
            continue

        same_wo_langs = plot_data["same_wo_langs"]
        diff_wo_langs = plot_data["diff_wo_langs"]
        atd_same_wo = plot_data["atd_same_wo"]
        atd_diff_wo = plot_data["atd_diff_wo"]
        jitter_same = plot_data["jitter_same"]
        jitter_diff = plot_data["jitter_diff"]

        # Build dataframe for box plot
        data = []
        for val in atd_same_wo:
            data.append({"Group": "Same WO", "ATD": val})
        for val in atd_diff_wo:
            data.append({"Group": "Diff WO", "ATD": val})
        df = pd.DataFrame(data)

        # Box plot with light colors (wider box to match text spread)
        sns.boxplot(
            data=df, x="Group", y="ATD", palette=box_colors, ax=ax,
            width=0.9, showfliers=False,
        )

        # Add language labels - Same WO group
        for i, (lang, atd_val) in enumerate(zip(same_wo_langs, atd_same_wo)):
            ax.text(
                0 + jitter_same[i], atd_val, lang,
                fontsize=5, ha="center", va="center",
                color=text_colors["Same WO"], fontweight="bold",
            )

        # Add language labels - Diff WO group
        for i, (lang, atd_val) in enumerate(zip(diff_wo_langs, atd_diff_wo)):
            ax.text(
                1 + jitter_diff[i], atd_val, lang,
                fontsize=5, ha="center", va="center",
                color=text_colors["Diff WO"], fontweight="bold",
            )

        # Set consistent y limits
        ax.set_ylim(y_lim)

        # Stats
        mean_same = np.mean(atd_same_wo)
        mean_diff = np.mean(atd_diff_wo)
        focal_name = info["name"]
        focal_wo = info["word_order"]

        ax.set_title(
            f"{focal_name} ({focal_wo})\n"
            f"n={len(atd_same_wo)} vs {len(atd_diff_wo)}, "
            f"$\\mu$={mean_same:.2f} vs {mean_diff:.2f}",
            fontsize=11,
        )
        ax.set_xlabel("")
        if idx == 0:
            ax.set_ylabel("ATD", fontsize=11)
        else:
            ax.set_ylabel("")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "atd_comparison_boxlabels.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot with labels saved: {save_path}")


def _prepare_plot_data_3groups(tensor, lang_sel, focal_lang, groups):
    """
    Prepare data for 3-group plotting: Same WO, Diff WO, and Excluded languages.
    """
    same_wo_langs = [l for l in groups["same_word_order"] if l in lang_sel]
    diff_wo_langs = [l for l in groups["diff_word_order"] if l in lang_sel]
    excluded_langs = [l for l in groups["excluded"] if l in lang_sel]

    atd_same_wo = get_atd_values_for_focal(
        tensor, lang_sel, focal_lang, same_wo_langs
    )
    atd_diff_wo = get_atd_values_for_focal(
        tensor, lang_sel, focal_lang, diff_wo_langs
    )
    atd_excluded = get_atd_values_for_focal(
        tensor, lang_sel, focal_lang, excluded_langs
    )

    # Compute uniform jitter based on y value clustering
    jitter_same = _compute_uniform_jitter(atd_same_wo, max_jitter=0.8, y_tolerance=0.03)
    jitter_diff = _compute_uniform_jitter(atd_diff_wo, max_jitter=0.8, y_tolerance=0.03)
    jitter_excluded = _compute_uniform_jitter(atd_excluded, max_jitter=0.8, y_tolerance=0.03)

    return {
        "same_wo_langs": same_wo_langs,
        "diff_wo_langs": diff_wo_langs,
        "excluded_langs": excluded_langs,
        "atd_same_wo": atd_same_wo,
        "atd_diff_wo": atd_diff_wo,
        "atd_excluded": atd_excluded,
        "jitter_same": jitter_same,
        "jitter_diff": jitter_diff,
        "jitter_excluded": jitter_excluded,
    }


def plot_atd_comparison_3groups(tensor, lang_sel, all_lang_dict, save_dir, figsize=(18, 5)):
    """
    Create box plot with 3 groups: Same WO, Diff WO, and Excluded (deleted) languages.
    Shows language text labels instead of dots.
    """
    import pandas as pd

    focal_groups = get_focal_language_groups()
    n_focal = len(focal_groups)

    fig, axes = plt.subplots(1, n_focal, figsize=figsize)
    if n_focal == 1:
        axes = [axes]

    # Light colors for box, dark colors for text
    box_colors = {"Same WO": "#FFCCCC", "Diff WO": "#CCE5FF", "Excluded": "#D4EDDA"}  # Light red, blue, green
    text_colors = {"Same WO": "#B22222", "Diff WO": "#1E3A8A", "Excluded": "#155724"}  # Dark red, blue, green

    # Fixed y limits
    y_lim = (1.0, 3.0)

    for idx, (focal_lang, info) in enumerate(focal_groups.items()):
        ax = axes[idx]

        if focal_lang not in lang_sel:
            ax.text(
                0.5, 0.5, f"{focal_lang} not in data",
                ha="center", va="center", transform=ax.transAxes,
            )
            continue

        groups = get_comparison_groups_for_focal(focal_lang, available_langs=lang_sel)
        plot_data = _prepare_plot_data_3groups(tensor, lang_sel, focal_lang, groups)

        same_wo_langs = plot_data["same_wo_langs"]
        diff_wo_langs = plot_data["diff_wo_langs"]
        excluded_langs = plot_data["excluded_langs"]
        atd_same_wo = plot_data["atd_same_wo"]
        atd_diff_wo = plot_data["atd_diff_wo"]
        atd_excluded = plot_data["atd_excluded"]
        jitter_same = plot_data["jitter_same"]
        jitter_diff = plot_data["jitter_diff"]
        jitter_excluded = plot_data["jitter_excluded"]

        # Build dataframe for box plot
        data = []
        for val in atd_same_wo:
            data.append({"Group": "Same WO", "ATD": val})
        for val in atd_diff_wo:
            data.append({"Group": "Diff WO", "ATD": val})
        for val in atd_excluded:
            data.append({"Group": "Excluded", "ATD": val})
        df = pd.DataFrame(data)

        # Box plot with light colors
        sns.boxplot(
            data=df, x="Group", y="ATD", palette=box_colors, ax=ax,
            order=["Same WO", "Diff WO", "Excluded"],
            width=0.9, showfliers=False,
        )

        # Add language labels - Same WO group (x=0)
        for i, (lang, atd_val) in enumerate(zip(same_wo_langs, atd_same_wo)):
            ax.text(
                0 + jitter_same[i], atd_val, lang,
                fontsize=5, ha="center", va="center",
                color=text_colors["Same WO"], fontweight="bold",
            )

        # Add language labels - Diff WO group (x=1)
        for i, (lang, atd_val) in enumerate(zip(diff_wo_langs, atd_diff_wo)):
            ax.text(
                1 + jitter_diff[i], atd_val, lang,
                fontsize=5, ha="center", va="center",
                color=text_colors["Diff WO"], fontweight="bold",
            )

        # Add language labels - Excluded group (x=2)
        for i, (lang, atd_val) in enumerate(zip(excluded_langs, atd_excluded)):
            ax.text(
                2 + jitter_excluded[i], atd_val, lang,
                fontsize=5, ha="center", va="center",
                color=text_colors["Excluded"], fontweight="bold",
            )

        # Set consistent y limits
        ax.set_ylim(y_lim)

        # Stats
        mean_same = np.mean(atd_same_wo) if len(atd_same_wo) > 0 else 0
        mean_diff = np.mean(atd_diff_wo) if len(atd_diff_wo) > 0 else 0
        mean_excluded = np.mean(atd_excluded) if len(atd_excluded) > 0 else 0
        focal_name = info["name"]
        focal_wo = info["word_order"]

        ax.set_title(
            f"{focal_name} ({focal_wo})\n"
            f"n={len(atd_same_wo)}/{len(atd_diff_wo)}/{len(atd_excluded)}, "
            f"$\\mu$={mean_same:.2f}/{mean_diff:.2f}/{mean_excluded:.2f}",
            fontsize=10,
        )
        ax.set_xlabel("")
        if idx == 0:
            ax.set_ylabel("ATD", fontsize=11)
        else:
            ax.set_ylabel("")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "atd_comparison_3groups.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"3-group comparison plot saved: {save_path}")


# =====================================================================
# Main
# =====================================================================


if __name__ == "__main__":
    # Load data
    tensor, lang_sel, all_lang_dict = load_distance_data()

    # Run analysis
    results = run_controlled_comparison(tensor, lang_sel, all_lang_dict)

    # Print summary
    print_summary_table(results)

    # Generate LaTeX tables
    print("\n" + "=" * 80)
    print("LaTeX Table: Results Summary")
    print("=" * 80)
    print(generate_latex_table(results, all_lang_dict))

    print("\n" + "=" * 80)
    print("LaTeX Table: Language Groupings")
    print("=" * 80)
    print(generate_grouping_latex_table(all_lang_dict, lang_sel))

    # Save results
    output_file = f"{OUTPUT_DIR}/control_comparison_results.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {output_file}")

    # Generate visualization
    print("\n" + "=" * 80)
    print("Generating Visualization")
    print("=" * 80)

    plot_atd_comparison(tensor, lang_sel, all_lang_dict, OUTPUT_DIR)
    plot_atd_comparison_with_labels(tensor, lang_sel, all_lang_dict, OUTPUT_DIR)
    plot_atd_comparison_3groups(tensor, lang_sel, all_lang_dict, OUTPUT_DIR)

    print(f"\nAll visualizations saved to {OUTPUT_DIR}/")
