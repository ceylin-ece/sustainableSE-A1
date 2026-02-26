#!/usr/bin/env python3
"""
Comprehensive analysis and visualization of LLM quantization energy experiment results.

Statistical methodology:
  - Shapiro-Wilk normality tests
  - Z-score outlier removal (|x - mean| > 3*std)
  - Welch's t-test (normal data) or Mann-Whitney U (non-normal data)
  - Effect size: Cohen's d / percent change / median difference
  - Energy per correct solution as a quality-aware energy metric
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy import stats
from itertools import combinations

# ── Configuration ──────────────────────────────────────────────────────────────
RESULTS_DIR = "results"
ENERGY_DIR = os.path.join(RESULTS_DIR, "energy")
EVAL_DIR = os.path.join(RESULTS_DIR, "eval")
SUMMARY_FILE = os.path.join(RESULTS_DIR, "total_energy_summary.csv")
OUTPUT_DIR = "plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_LABELS = {
    "Llama-3.2-3B-Instruct-Q4_K_M": "Llama 3.2 3B\n(Q4_K_M)",
    "Phi-3-mini-4k-instruct-q4": "Phi-3 Mini\n(Q4)",
    "qwen2.5-3b-instruct-q4_k_m": "Qwen 2.5 3B\n(Q4_K_M)",
}
MODEL_ORDER = [
    "Llama-3.2-3B-Instruct-Q4_K_M",
    "Phi-3-mini-4k-instruct-q4",
    "qwen2.5-3b-instruct-q4_k_m",
]
PALETTE = {
    "Llama-3.2-3B-Instruct-Q4_K_M": "#4C72B0",
    "Phi-3-mini-4k-instruct-q4": "#DD8452",
    "qwen2.5-3b-instruct-q4_k_m": "#55A868",
}

sns.set_theme(style="whitegrid", font_scale=1.15)
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_summary():
    df = pd.read_csv(SUMMARY_FILE)
    df = df.drop_duplicates(subset=["Model_Name", "Trial_Number"], keep="first")
    df["Label"] = df["Model_Name"].map(MODEL_LABELS)
    return df


def load_eval():
    frames = []
    for fp in sorted(glob.glob(os.path.join(EVAL_DIR, "eval_results_*.csv"))):
        frames.append(pd.read_csv(fp))
    df = pd.concat(frames, ignore_index=True)
    df["Model_Clean"] = df["Model_Name"].str.replace(".gguf", "", regex=False)
    return df


def load_energy_trace(model_name, trial=1):
    pattern = os.path.join(ENERGY_DIR, f"energy_results_{model_name}_trial_{trial}.csv")
    files = glob.glob(pattern)
    if not files:
        return None
    df = pd.read_csv(files[0])
    df["Time_Sec"] = df["Delta"].cumsum() / 1000.0
    return df


def compute_eval_metrics(eval_df):
    grouped = eval_df.groupby(["Model_Clean", "Trial"]).agg(
        n_tasks=("Passed", "count"),
        n_passed=("Passed", "sum"),
        total_completion_tokens=("Completion_Tokens", "sum"),
        total_tokens=("Total_Tokens", "sum"),
        total_duration=("Duration_Sec", "sum"),
    ).reset_index()
    grouped["pass_rate"] = grouped["n_passed"] / grouped["n_tasks"]
    grouped["tokens_per_sec"] = grouped["total_completion_tokens"] / grouped["total_duration"]
    return grouped


def merge_data(summary_df, eval_metrics):
    merged = summary_df.merge(
        eval_metrics,
        left_on=["Model_Name", "Trial_Number"],
        right_on=["Model_Clean", "Trial"],
        how="inner",
    )
    # Approximate metric – energy is for the whole trial, not per-token
    merged["energy_per_token_approx"] = merged["Total_Joules"] / merged["total_completion_tokens"]
    merged["avg_power"] = merged["Total_Joules"] / merged["Total_Execution_Time_Sec"]
    # Quality-aware energy metric: energy cost to produce one correct solution
    merged["energy_per_correct"] = merged["Total_Joules"] / merged["n_passed"]
    # Energy Delay Product: penalizes slow runs (Metrics Report slide 14)
    # EDP = E × t  (Joule-seconds)
    merged["edp"] = merged["Total_Joules"] * merged["Total_Execution_Time_Sec"]
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# OUTLIER REMOVAL  (z-score > 3)
# ══════════════════════════════════════════════════════════════════════════════

def remove_outliers_zscore(df, column, group_col="Model_Name", z_thresh=3):
    """Remove rows where |value - group_mean| > z_thresh * group_std."""
    before = len(df)
    mask = pd.Series(True, index=df.index)
    removed_info = []
    for model, group in df.groupby(group_col):
        vals = group[column]
        z_scores = np.abs((vals - vals.mean()) / vals.std()) if vals.std() > 0 else pd.Series(0, index=vals.index)
        outliers = z_scores > z_thresh
        if outliers.any():
            removed_info.append((model, outliers.sum(), group.loc[outliers, column].tolist()))
        mask.loc[group.index] = ~outliers
    df_clean = df[mask].copy()
    after = len(df_clean)
    return df_clean, before - after, removed_info


# ══════════════════════════════════════════════════════════════════════════════
# NORMALITY TESTING  (Shapiro-Wilk)
# ══════════════════════════════════════════════════════════════════════════════

def test_normality(df, column, group_col="Model_Name"):
    """Run Shapiro-Wilk test per group. Returns dict: model -> (W, p, is_normal)."""
    results = {}
    for model in MODEL_ORDER:
        vals = df[df[group_col] == model][column].astype(float).values
        if len(vals) < 3:
            results[model] = (np.nan, np.nan, False)
            continue
        w_stat, p_val = stats.shapiro(vals)
        results[model] = (w_stat, p_val, p_val >= 0.05)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICAL TESTS
# ══════════════════════════════════════════════════════════════════════════════

def cohens_d(a, b):
    """Compute Cohen's d effect size."""
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(((na - 1) * np.std(a, ddof=1)**2 + (nb - 1) * np.std(b, ddof=1)**2) / (na + nb - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled_std


def percent_change(a, b):
    """Percent change from a to b: (mean_b - mean_a) / mean_a * 100."""
    ma = np.mean(a)
    if ma == 0:
        return np.nan
    return (np.mean(b) - ma) / ma * 100


def interpret_cohens_d(d):
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def run_pairwise_tests(df, column, group_col, normality_results, label):
    """Run pairwise comparisons: Welch's t-test if both normal, else Mann-Whitney U."""
    print(f"\n  Pairwise comparisons – {label}:")
    print(f"  {'─' * 65}")
    for m1, m2 in combinations(MODEL_ORDER, 2):
        a = df[df[group_col] == m1][column].astype(float).values
        b = df[df[group_col] == m2][column].astype(float).values
        l1 = MODEL_LABELS[m1].replace("\n", " ")
        l2 = MODEL_LABELS[m2].replace("\n", " ")

        both_normal = normality_results[m1][2] and normality_results[m2][2]

        try:
            if both_normal:
                t_stat, p_val = stats.ttest_ind(a, b, equal_var=False, alternative="two-sided")
                test_name = "Welch's t"
                stat_label = f"t={t_stat:.3f}"
            else:
                u_stat, p_val = stats.mannwhitneyu(a, b, alternative="two-sided")
                test_name = "Mann-Whitney U"
                stat_label = f"U={u_stat:.0f}"
        except Exception as e:
            print(f"    {l1} vs {l2}: Could not compute ({e})")
            continue

        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        d = cohens_d(a, b)
        pct = percent_change(a, b)
        d_interp = interpret_cohens_d(d)
        med_diff = np.median(b) - np.median(a)

        print(f"    {l1} vs {l2}")
        print(f"      Test: {test_name}, {stat_label}, p={p_val:.2e} {sig}")
        print(f"      Cohen's d = {d:.3f} ({d_interp}), Δmedian = {med_diff:+,.1f}, Δ% = {pct:+.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_energy_boxplot(summary):
    """Box + strip plot of total energy consumption per model."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=summary, x="Model_Name", y="Total_Joules", hue="Model_Name",
                order=MODEL_ORDER, hue_order=MODEL_ORDER, palette=PALETTE,
                width=0.5, ax=ax, legend=False)
    sns.stripplot(data=summary, x="Model_Name", y="Total_Joules", order=MODEL_ORDER,
                  color="black", alpha=0.35, size=4, jitter=True, ax=ax)
    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
    ax.set_ylabel("Total Energy (Joules)")
    ax.set_xlabel("")
    ax.set_title("Total Energy Consumption per Model (20 Trials)")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.savefig(os.path.join(OUTPUT_DIR, "energy_boxplot.png"))
    plt.close(fig)
    print("  ✓ energy_boxplot.png")


def plot_time_boxplot(summary):
    """Box + strip plot of total execution time per model."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=summary, x="Model_Name", y="Total_Execution_Time_Sec", hue="Model_Name",
                order=MODEL_ORDER, hue_order=MODEL_ORDER, palette=PALETTE,
                width=0.5, ax=ax, legend=False)
    sns.stripplot(data=summary, x="Model_Name", y="Total_Execution_Time_Sec", order=MODEL_ORDER,
                  color="black", alpha=0.35, size=4, jitter=True, ax=ax)
    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
    ax.set_ylabel("Total Execution Time (seconds)")
    ax.set_xlabel("")
    ax.set_title("Execution Time per Model (20 Trials)")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.savefig(os.path.join(OUTPUT_DIR, "time_boxplot.png"))
    plt.close(fig)
    print("  ✓ time_boxplot.png")


def plot_pass_rate(eval_metrics):
    """Bar chart of average pass rate per model with std error bars."""
    fig, ax = plt.subplots(figsize=(8, 5))
    means, stds, labels = [], [], []
    for m in MODEL_ORDER:
        subset = eval_metrics[eval_metrics["Model_Clean"] == m]
        means.append(subset["pass_rate"].mean() * 100)
        stds.append(subset["pass_rate"].std() * 100)
        labels.append(MODEL_LABELS[m])
    bars = ax.bar(labels, means, yerr=stds, capsize=5,
                  color=[PALETTE[m] for m in MODEL_ORDER], edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Pass Rate (%)")
    ax.set_title("MBPP Benchmark Pass Rate per Model (Mean ± SD)")
    ax.set_ylim(0, 100)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{mean:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    fig.savefig(os.path.join(OUTPUT_DIR, "pass_rate_bar.png"))
    plt.close(fig)
    print("  ✓ pass_rate_bar.png")


def plot_energy_violin(summary):
    """Violin + box plot of energy distribution per model."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.violinplot(data=summary, x="Model_Name", y="Total_Joules", hue="Model_Name",
                   order=MODEL_ORDER, hue_order=MODEL_ORDER, palette=PALETTE,
                   inner="box", ax=ax, legend=False)
    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
    ax.set_ylabel("Total Energy (Joules)")
    ax.set_xlabel("")
    ax.set_title("Energy Distribution per Model (Violin + Box)")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.savefig(os.path.join(OUTPUT_DIR, "energy_violin.png"))
    plt.close(fig)
    print("  ✓ energy_violin.png")


def plot_energy_per_token(merged):
    """Box plot of (approximate) energy per completion token."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=merged, x="Model_Name", y="energy_per_token_approx", hue="Model_Name",
                order=MODEL_ORDER, hue_order=MODEL_ORDER, palette=PALETTE,
                width=0.5, ax=ax, legend=False)
    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
    ax.set_ylabel("Energy per Completion Token (J/token)")
    ax.set_xlabel("")
    ax.set_title("Approx. Energy per Token (Total Joules ÷ Total Completion Tokens)")
    fig.savefig(os.path.join(OUTPUT_DIR, "energy_per_token_approx.png"))
    plt.close(fig)
    print("  ✓ energy_per_token_approx.png")


def plot_energy_per_correct(merged):
    """Box plot of energy per correct solution – the quality-aware energy metric."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=merged, x="Model_Name", y="energy_per_correct", hue="Model_Name",
                order=MODEL_ORDER, hue_order=MODEL_ORDER, palette=PALETTE,
                width=0.5, ax=ax, legend=False)
    sns.stripplot(data=merged, x="Model_Name", y="energy_per_correct", order=MODEL_ORDER,
                  color="black", alpha=0.35, size=4, jitter=True, ax=ax)
    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
    ax.set_ylabel("Energy per Correct Solution (J)")
    ax.set_xlabel("")
    ax.set_title("Energy Cost per Correct Solution (Total Joules ÷ Passed Tasks)")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.savefig(os.path.join(OUTPUT_DIR, "energy_per_correct.png"))
    plt.close(fig)
    print("  ✓ energy_per_correct.png")


def plot_throughput(merged):
    """Box plot of inference throughput (tokens/sec)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=merged, x="Model_Name", y="tokens_per_sec", hue="Model_Name",
                order=MODEL_ORDER, hue_order=MODEL_ORDER, palette=PALETTE,
                width=0.5, ax=ax, legend=False)
    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
    ax.set_ylabel("Completion Tokens / Second")
    ax.set_xlabel("")
    ax.set_title("Inference Throughput per Model")
    fig.savefig(os.path.join(OUTPUT_DIR, "throughput_boxplot.png"))
    plt.close(fig)
    print("  ✓ throughput_boxplot.png")


def plot_power_traces(trial=10):
    """System power over time for one trial of each model."""
    fig, ax = plt.subplots(figsize=(12, 5))
    for model in MODEL_ORDER:
        trace = load_energy_trace(model, trial=trial)
        if trace is not None:
            ax.plot(trace["Time_Sec"], trace["SYSTEM_POWER (Watts)"],
                    label=MODEL_LABELS[model].replace("\n", " "),
                    color=PALETTE[model], alpha=0.8, linewidth=0.6)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("System Power (Watts)")
    ax.set_title(f"System Power Over Time (Trial {trial})")
    ax.legend()
    fig.savefig(os.path.join(OUTPUT_DIR, "power_traces.png"))
    plt.close(fig)
    print("  ✓ power_traces.png")


def plot_cpu_temp_traces(trial=10):
    """Average CPU temperature over time for one trial of each model."""
    fig, ax = plt.subplots(figsize=(12, 5))
    temp_cols = [f"CPU_TEMP_{i}" for i in range(10)]
    for model in MODEL_ORDER:
        trace = load_energy_trace(model, trial=trial)
        if trace is not None:
            available = [c for c in temp_cols if c in trace.columns]
            trace["avg_cpu_temp"] = trace[available].mean(axis=1)
            ax.plot(trace["Time_Sec"], trace["avg_cpu_temp"],
                    label=MODEL_LABELS[model].replace("\n", " "),
                    color=PALETTE[model], alpha=0.8, linewidth=0.6)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Average CPU Temperature (°C)")
    ax.set_title(f"CPU Temperature Over Time (Trial {trial})")
    ax.legend()
    fig.savefig(os.path.join(OUTPUT_DIR, "cpu_temp_traces.png"))
    plt.close(fig)
    print("  ✓ cpu_temp_traces.png")


def plot_energy_vs_passrate(merged):
    """Scatter plot of total energy vs pass rate per trial."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for model in MODEL_ORDER:
        subset = merged[merged["Model_Name"] == model]
        ax.scatter(subset["pass_rate"] * 100, subset["Total_Joules"],
                   label=MODEL_LABELS[model].replace("\n", " "),
                   color=PALETTE[model], s=50, alpha=0.7, edgecolors="black", linewidth=0.4)
    ax.set_xlabel("Pass Rate (%)")
    ax.set_ylabel("Total Energy (Joules)")
    ax.set_title("Energy Consumption vs. Code Quality (Pass Rate)")
    ax.legend()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.savefig(os.path.join(OUTPUT_DIR, "energy_vs_passrate.png"))
    plt.close(fig)
    print("  ✓ energy_vs_passrate.png")


def plot_combined_summary(merged):
    """4-panel summary: energy, time, pass rate, energy per correct solution."""
    agg = merged.groupby("Model_Name").agg(
        mean_energy=("Total_Joules", "mean"),
        mean_time=("Total_Execution_Time_Sec", "mean"),
        mean_pass=("pass_rate", "mean"),
        mean_epc=("energy_per_correct", "mean"),
    ).reindex(MODEL_ORDER)

    metrics = ["mean_energy", "mean_time", "mean_pass", "mean_epc"]
    metric_labels = ["Mean Energy (J)", "Mean Time (s)", "Mean Pass Rate",
                     "Mean Energy per\nCorrect Solution (J)"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    for ax, metric, mlabel in zip(axes, metrics, metric_labels):
        vals = agg[metric].values
        colors = [PALETTE[m] for m in MODEL_ORDER]
        labels = [MODEL_LABELS[m] for m in MODEL_ORDER]
        bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_title(mlabel, fontsize=11)
        ax.tick_params(axis="x", labelsize=8)
        if metric == "mean_pass":
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                        f"{v:.1%}", ha="center", fontsize=9, fontweight="bold")
        else:
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, v * 1.02,
                        f"{v:,.0f}", ha="center", fontsize=9, fontweight="bold")
    fig.suptitle("Model Comparison Summary", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "combined_summary.png"))
    plt.close(fig)
    print("  ✓ combined_summary.png")


def plot_avg_power(merged):
    """Box + strip plot of average power (W) per model."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=merged, x="Model_Name", y="avg_power", hue="Model_Name",
                order=MODEL_ORDER, hue_order=MODEL_ORDER, palette=PALETTE,
                width=0.5, ax=ax, legend=False)
    sns.stripplot(data=merged, x="Model_Name", y="avg_power", order=MODEL_ORDER,
                  color="black", alpha=0.35, size=4, jitter=True, ax=ax)
    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
    ax.set_ylabel("Average Power (Watts)")
    ax.set_xlabel("")
    ax.set_title("Average Power Consumption per Model (Total Energy ÷ Total Time)")
    fig.savefig(os.path.join(OUTPUT_DIR, "avg_power_boxplot.png"))
    plt.close(fig)
    print("  ✓ avg_power_boxplot.png")


def plot_edp(merged):
    """Box + strip plot of Energy Delay Product per model."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=merged, x="Model_Name", y="edp", hue="Model_Name",
                order=MODEL_ORDER, hue_order=MODEL_ORDER, palette=PALETTE,
                width=0.5, ax=ax, legend=False)
    sns.stripplot(data=merged, x="Model_Name", y="edp", order=MODEL_ORDER,
                  color="black", alpha=0.35, size=4, jitter=True, ax=ax)
    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
    ax.set_ylabel("Energy Delay Product (J·s)")
    ax.set_xlabel("")
    ax.set_title("Energy Delay Product per Model (E × t — penalizes slow runs)")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.savefig(os.path.join(OUTPUT_DIR, "edp_boxplot.png"))
    plt.close(fig)
    print("  ✓ edp_boxplot.png")


def plot_normality_histograms(summary, merged):
    """Histograms + KDE of energy and energy-per-correct for normality visual check."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for col_idx, model in enumerate(MODEL_ORDER):
        label = MODEL_LABELS[model].replace("\n", " ")
        # Row 0: Total Energy
        vals_e = summary[summary["Model_Name"] == model]["Total_Joules"].values
        ax = axes[0, col_idx]
        ax.hist(vals_e, bins=8, density=True, alpha=0.6, color=PALETTE[model], edgecolor="black")
        if len(vals_e) > 2 and np.std(vals_e) > 0:
            xmin, xmax = vals_e.min(), vals_e.max()
            x = np.linspace(xmin - 0.1 * (xmax - xmin), xmax + 0.1 * (xmax - xmin), 100)
            ax.plot(x, stats.norm.pdf(x, np.mean(vals_e), np.std(vals_e)),
                    color="black", linewidth=1.5, linestyle="--", label="Normal fit")
        w, p = stats.shapiro(vals_e) if len(vals_e) >= 3 else (np.nan, np.nan)
        normal_sym = "Normal" if p >= 0.05 else "Not normal"
        ax.set_title(f"{label}\nShapiro p={p:.3f} ({normal_sym})", fontsize=10)
        ax.set_xlabel("Total Energy (J)")
        if col_idx == 0:
            ax.set_ylabel("Probability Density")

        # Row 1: Energy per correct solution
        vals_c = merged[merged["Model_Name"] == model]["energy_per_correct"].astype(float).values
        ax = axes[1, col_idx]
        ax.hist(vals_c, bins=8, density=True, alpha=0.6, color=PALETTE[model], edgecolor="black")
        if len(vals_c) > 2 and np.std(vals_c) > 0:
            xmin, xmax = vals_c.min(), vals_c.max()
            x = np.linspace(xmin - 0.1 * (xmax - xmin), xmax + 0.1 * (xmax - xmin), 100)
            ax.plot(x, stats.norm.pdf(x, np.mean(vals_c), np.std(vals_c)),
                    color="black", linewidth=1.5, linestyle="--", label="Normal fit")
        w, p = stats.shapiro(vals_c) if len(vals_c) >= 3 else (np.nan, np.nan)
        normal_sym = "Normal" if p >= 0.05 else "Not normal"
        ax.set_title(f"Shapiro p={p:.3f} ({normal_sym})", fontsize=10)
        ax.set_xlabel("Energy per Correct Solution (J)")
        if col_idx == 0:
            ax.set_ylabel("Probability Density")

    fig.suptitle("Normality Check: Histograms with Normal Fit", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "normality_histograms.png"))
    plt.close(fig)
    print("  ✓ normality_histograms.png")


# ══════════════════════════════════════════════════════════════════════════════
# PRINTED REPORTS
# ══════════════════════════════════════════════════════════════════════════════

def print_descriptive_stats(summary, merged):
    print("\n" + "=" * 70)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 70)
    for m in MODEL_ORDER:
        label = MODEL_LABELS[m].replace("\n", " ")
        s = summary[summary["Model_Name"] == m]
        me = merged[merged["Model_Name"] == m]
        print(f"\n{'─' * 55}")
        print(f"  {label}")
        print(f"{'─' * 55}")
        print(f"  Trials:                    {len(s):>5}")
        print(f"  Total Energy (J):     {s['Total_Joules'].mean():>10,.1f} ± {s['Total_Joules'].std():>8,.1f}")
        print(f"  Execution Time (s):   {s['Total_Execution_Time_Sec'].mean():>10,.1f} ± {s['Total_Execution_Time_Sec'].std():>8,.1f}")
        print(f"  Average Power (W):    {me['avg_power'].mean():>10,.1f} ± {me['avg_power'].std():>8,.1f}")
        if len(me) > 0:
            print(f"  Pass Rate (%):        {me['pass_rate'].mean() * 100:>10.1f} ± {me['pass_rate'].std() * 100:>8.1f}")
            print(f"  Correct Solutions:    {me['n_passed'].mean():>10.1f} ± {me['n_passed'].std():>8.1f}  (out of {me['n_tasks'].iloc[0]})")
            print(f"  Energy/Correct (J):   {me['energy_per_correct'].mean():>10,.1f} ± {me['energy_per_correct'].std():>8,.1f}")
            print(f"  Energy/Token* (J):    {me['energy_per_token_approx'].mean():>10.2f} ± {me['energy_per_token_approx'].std():>8.2f}  (*approx.)")
            print(f"  Throughput (tok/s):   {me['tokens_per_sec'].mean():>10.1f} ± {me['tokens_per_sec'].std():>8.1f}")
            print(f"  EDP (J·s):            {me['edp'].mean():>10,.0f} ± {me['edp'].std():>8,.0f}")


def print_outlier_report(summary, merged):
    print("\n" + "=" * 70)
    print("OUTLIER REMOVAL (z-score > 3, applied to ALL metrics)")
    print("=" * 70)

    # --- Summary-level metrics ---
    summary_metrics = ["Total_Joules", "Total_Execution_Time_Sec"]
    summary_clean = summary.copy()
    total_summary_removed = 0
    for col in summary_metrics:
        summary_clean, n_removed, info = remove_outliers_zscore(summary_clean, col)
        total_summary_removed += n_removed
        if n_removed > 0:
            col_label = "Total Energy (J)" if col == "Total_Joules" else "Execution Time (s)"
            print(f"  {col_label}: removed {n_removed} outlier(s):")
            for model, n, vals in info:
                print(f"    {MODEL_LABELS[model].replace(chr(10), ' ')}: {n} point(s) = {vals}")

    # --- Merged-level metrics ---
    merged_metrics = ["energy_per_correct", "avg_power", "edp"]
    merged_clean = merged.copy()
    total_merged_removed = 0
    for col in merged_metrics:
        merged_clean, n_removed, info = remove_outliers_zscore(merged_clean, col)
        total_merged_removed += n_removed
        if n_removed > 0:
            col_labels = {"energy_per_correct": "Energy/Correct (J)",
                          "avg_power": "Average Power (W)",
                          "edp": "EDP (J·s)"}
            print(f"  {col_labels[col]}: removed {n_removed} outlier(s):")
            for model, n, vals in info:
                print(f"    {MODEL_LABELS[model].replace(chr(10), ' ')}: {n} point(s) = {vals}")

    if total_summary_removed == 0 and total_merged_removed == 0:
        print("  No outliers detected (|z| > 3) in any metric.")
    else:
        print(f"\n  Summary after outlier removal:")
        print(f"    Summary rows: {len(summary)} -> {len(summary_clean)}  (removed {len(summary) - len(summary_clean)})")
        print(f"    Merged  rows: {len(merged)} -> {len(merged_clean)}  (removed {len(merged) - len(merged_clean)})")

    return summary_clean, merged_clean


def print_normality_report(summary, merged):
    print("\n" + "=" * 70)
    print("NORMALITY TESTS (Shapiro-Wilk, alpha = 0.05)")
    print("=" * 70)

    metrics_to_test = [
        ("Total Energy (J)", summary, "Total_Joules", "Model_Name"),
        ("Execution Time (s)", summary, "Total_Execution_Time_Sec", "Model_Name"),
        ("Energy per Correct (J)", merged, "energy_per_correct", "Model_Name"),
        ("EDP (J·s)", merged, "edp", "Model_Name"),
        ("Average Power (W)", merged, "avg_power", "Model_Name"),
    ]

    normality_cache = {}
    for metric_name, df, col, gcol in metrics_to_test:
        results = test_normality(df, col, gcol)
        normality_cache[(col, gcol)] = results
        print(f"\n  {metric_name}:")
        for model in MODEL_ORDER:
            w, p, is_normal = results[model]
            label = MODEL_LABELS[model].replace("\n", " ")
            status = "NORMAL" if is_normal else "NOT NORMAL"
            print(f"    {label:30s}  W={w:.4f}, p={p:.4f}  -> {status}")

    return normality_cache


def print_statistical_tests(summary, merged, normality_cache):
    print("\n" + "=" * 70)
    print("STATISTICAL TESTS (Welch's t if normal, Mann-Whitney U otherwise)")
    print("  + Effect size: Cohen's d, median difference, percent change")
    print("=" * 70)

    # --- Total Energy ---
    norm_energy = normality_cache.get(("Total_Joules", "Model_Name"),
                                       test_normality(summary, "Total_Joules"))
    run_pairwise_tests(summary, "Total_Joules", "Model_Name", norm_energy, "Total Energy (J)")

    # --- Execution Time ---
    norm_time = normality_cache.get(("Total_Execution_Time_Sec", "Model_Name"),
                                     test_normality(summary, "Total_Execution_Time_Sec"))
    run_pairwise_tests(summary, "Total_Execution_Time_Sec", "Model_Name", norm_time, "Execution Time (s)")

    # --- Energy per Correct Solution ---
    norm_epc = normality_cache.get(("energy_per_correct", "Model_Name"),
                                    test_normality(merged, "energy_per_correct"))
    run_pairwise_tests(merged, "energy_per_correct", "Model_Name", norm_epc, "Energy per Correct Solution (J)")

    # --- Energy Delay Product ---
    norm_edp = normality_cache.get(("edp", "Model_Name"),
                                    test_normality(merged, "edp"))
    run_pairwise_tests(merged, "edp", "Model_Name", norm_edp, "Energy Delay Product (J·s)")

    # --- Average Power ---
    norm_pow = normality_cache.get(("avg_power", "Model_Name"),
                                    test_normality(merged, "avg_power"))
    run_pairwise_tests(merged, "avg_power", "Model_Name", norm_pow, "Average Power (W)")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  LLM Quantization Energy Analysis")
    print("  Statistical analysis of energy, time, and accuracy")
    print("=" * 70)

    # ── Load ──
    print("\n[1/6] Loading data...")
    summary = load_summary()
    eval_df = load_eval()
    eval_metrics = compute_eval_metrics(eval_df)
    merged = merge_data(summary, eval_metrics)
    print(f"  Summary: {len(summary)} rows  |  Eval: {len(eval_df)} rows  |  Merged: {len(merged)} rows")
    print(f"  Models: {list(MODEL_ORDER)}")
    print(f"  Trials per model: {summary.groupby('Model_Name')['Trial_Number'].nunique().to_dict()}")

    # ── Outlier removal ──
    print("\n[2/6] Checking for outliers...")
    summary_clean, merged_clean = print_outlier_report(summary, merged)

    # ── Normality ──
    print("\n[3/6] Normality testing...")
    normality_cache = print_normality_report(summary_clean, merged_clean)

    # ── Descriptive stats ──
    print("\n[4/6] Computing descriptive statistics...")
    print_descriptive_stats(summary_clean, merged_clean)

    # ── Statistical tests ──
    print("\n[5/6] Running statistical tests...")
    print_statistical_tests(summary_clean, merged_clean, normality_cache)

    # ── Plots ──
    print(f"\n[6/6] Generating plots -> {OUTPUT_DIR}/")
    plot_energy_boxplot(summary_clean)
    plot_time_boxplot(summary_clean)
    plot_pass_rate(eval_metrics)
    plot_energy_violin(summary_clean)
    plot_energy_per_token(merged_clean)
    plot_energy_per_correct(merged_clean)
    plot_throughput(merged_clean)
    plot_power_traces(trial=10)
    plot_cpu_temp_traces(trial=10)
    plot_energy_vs_passrate(merged_clean)
    plot_combined_summary(merged_clean)
    plot_avg_power(merged_clean)
    plot_edp(merged_clean)
    plot_normality_histograms(summary_clean, merged_clean)

    print(f"\n{'=' * 70}")
    print(f"  Done! 14 figures saved to: {OUTPUT_DIR}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
