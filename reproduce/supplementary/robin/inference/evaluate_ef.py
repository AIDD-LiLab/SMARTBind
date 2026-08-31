"""Enrichment factors for the leakage-filtered ROBIN SMM benchmark.

Turns per-binder rank percentiles into the numbers behind the published panel:
per-target EF at four screening depths, the mean across the 21 targets, and a paired
two-sided Wilcoxon signed-rank test of SMARTBind against each baseline.

    EF@t%  = (fraction of a target's binders with rank% <= t) / (t / 100)
    AUROC  = 1 - mean(rank%) / 100
    rank%  = (# decoys scoring strictly higher) / n_decoys * 100      lower is better

SMARTBind's per-binder file is the one `run_smartbind.py` writes; if it has not been
run yet the reference copy shipped in `reference/` is used instead, so the evaluation
can be inspected without a GPU. The three baselines are shipped as per-binder files
under `baselines/` - they were produced by their own authors' code on this exact
decoy pool and cannot be re-run from this repository (see baselines/README.md).

    python evaluate_ef.py
"""
import inspect
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# EF values are heavily tied (many targets score 0 at the shallowest depths), so the
# exact permutation null is not valid here; the published tests use the normal
# approximation without the continuity correction. scipy renamed this keyword from
# `mode` to `method` in 1.13.
_METHOD_KW = "method" if "method" in inspect.signature(wilcoxon).parameters else "mode"
_WILCOXON_KW = {"zero_method": "wilcox", "correction": False, _METHOD_KW: "approx"}

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

REFERENCE = "SMARTBind"
DEPTHS = [0.5, 1, 3, 5]
ALTERNATIVE = "two-sided"          # H1: SMARTBind differs from the baseline

SMARTBIND_OUT = os.path.join(HERE, "smartbind_smm_filtered_per_binder.csv")
SMARTBIND_REF = os.path.join(HERE, "reference", "smartbind_smm_filtered_per_binder.csv")
BASELINES = {
    "GerNA-Bind": "gernabind_smm_filtered_per_binder.csv",
    "DrugBAN": "drugban_fold10_smm_filtered_per_binder.csv",
    "GraphDTA": "graphdta_fold7_smm_filtered_per_binder.csv",
}


def load_per_binder():
    """({model -> DataFrame(target, SMILES, score, rank_percentile)}, is_fresh_screen)."""
    fresh = os.path.exists(SMARTBIND_OUT)
    path = SMARTBIND_OUT if fresh else SMARTBIND_REF
    source = "this run" if fresh else "shipped reference (run_smartbind.py has not been run)"
    print(f"SMARTBind per-binder scores: {source}\n  {path}")
    frames = {REFERENCE: pd.read_csv(path)}
    for model, filename in BASELINES.items():
        frames[model] = pd.read_csv(os.path.join(HERE, "baselines", filename))
    return frames, fresh


def per_target_table(frames):
    rows = []
    for model, df in frames.items():
        for target, block in df.groupby("target"):
            rank = block.rank_percentile.to_numpy(dtype=float)
            row = {
                "library": "SMM",
                "model": model,
                "target": target,
                "n_binders": len(block),
                "AUROC": 1.0 - rank.mean() / 100.0,
            }
            for depth in DEPTHS:
                row[f"EF@{depth:g}%"] = (rank <= depth).mean() / (depth / 100.0)
            rows.append(row)
    return pd.DataFrame(rows)


def significance(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def wilcoxon_table(per_target):
    metrics = [f"EF@{d:g}%" for d in DEPTHS] + ["AUROC"]
    rows = []
    for metric in metrics:
        wide = per_target.pivot(index="target", columns="model", values=metric)
        reference = wide[REFERENCE].to_numpy(dtype=float)
        for baseline in BASELINES:
            other = wide[baseline].to_numpy(dtype=float)
            statistic, p = wilcoxon(reference, other, alternative=ALTERNATIVE,
                                    **_WILCOXON_KW)
            rows.append({
                "library": "SMM",
                "metric": metric,
                "baseline": baseline,
                "n_targets": len(wide),
                "smartbind_mean": reference.mean(),
                "baseline_mean": other.mean(),
                "mean_paired_difference": (reference - other).mean(),
                "wilcoxon_statistic": statistic,
                "p_value": p,
                "wilcoxon_alternative": ALTERNATIVE,
                "p_adjustment": "none",
                "significance": significance(p),
            })
    return pd.DataFrame(rows)


def draw_panel(per_target, stats, path):
    """A plain rendering of the published SMM enrichment panel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = [REFERENCE] + list(BASELINES)
    colors = {"SMARTBind": "#5B5EA6", "GerNA-Bind": "#B9BCDA",
              "DrugBAN": "#C4614C", "GraphDTA": "#EFC670"}
    metrics = [f"EF@{d:g}%" for d in DEPTHS]
    means = {m: [per_target[(per_target.model == m) & (per_target.library == "SMM")][k].mean()
                 for k in metrics] for m in models}
    sds = {m: [per_target[(per_target.model == m) & (per_target.library == "SMM")][k].std(ddof=1)
               for k in metrics] for m in models}

    x = np.arange(len(metrics))
    width = 0.8 / len(models)
    offsets = (np.arange(len(models)) - (len(models) - 1) / 2) * width
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for model, offset in zip(models, offsets):
        ax.bar(x + offset, means[model], width=width, color=colors[model],
               edgecolor="black" if model == REFERENCE else "#3A3A3A",
               linewidth=1.1 if model == REFERENCE else 0.4, label=model, zorder=3)
        for xpos, mean, sd in zip(x + offset, means[model], sds[model]):
            ax.text(xpos, mean + 0.05, f"{mean:.2f}±{sd:.2f}", ha="center",
                    va="bottom", rotation=90, fontsize=8, zorder=5)
    ax.axhline(1, color="black", linestyle="--", linewidth=1.1, zorder=1)
    ax.text(len(metrics) - 0.45, 0.97, "EF = 1", ha="right", va="top", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12, fontweight="bold")
    ax.set_ylabel("Average enrichment factor", fontsize=12, fontweight="bold")
    ax.set_title("ROBIN SMM library, 21 leakage-filtered targets", fontsize=12)
    ax.legend(frameon=True, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OUT] {path}")


def compare_with_reference(per_target, stats, fresh):
    """Confirm the recomputation matches the published tables.

    Re-evaluating the shipped per-binder file must agree bit-for-bit. A fresh screen is
    allowed a little numerical drift: torch reductions are not bit-reproducible across
    devices and thread counts, which moves a handful of near-tied binders past a decoy
    and shifts their rank percentile in the third decimal. EF is a threshold count and
    is unaffected; AUROC, being a mean of rank percentiles, moves in the seventh decimal.
    """
    ref_pt = os.path.join(HERE, "reference", "robin_leakage_filtered_per_target.csv")
    ref_wx = os.path.join(HERE, "reference", "robin_leakage_filtered_wilcoxon.csv")
    if not (os.path.exists(ref_pt) and os.path.exists(ref_wx)):
        return
    print("\n--- Agreement with the published tables ---")
    published = pd.read_csv(ref_pt)
    published = published[published.library == "SMM"]
    merged = per_target.merge(published, on=["library", "model", "target"],
                              suffixes=("_new", "_published"))
    worst = 0.0
    for column in ["AUROC"] + [f"EF@{d:g}%" for d in DEPTHS]:
        delta = (merged[f"{column}_new"] - merged[f"{column}_published"]).abs().max()
        worst = max(worst, delta)
        print(f"  {column:10s} max |difference| = {delta:.2e}  over {len(merged)} target-model rows")
    published_wx = pd.read_csv(ref_wx)
    published_wx = published_wx[published_wx.library == "SMM"]
    merged_wx = stats.merge(published_wx, on=["library", "metric", "baseline"],
                            suffixes=("_new", "_published"))
    dp = (merged_wx.p_value_new - merged_wx.p_value_published).abs().max()
    ds = (merged_wx.wilcoxon_statistic_new - merged_wx.wilcoxon_statistic_published).abs().max()
    print(f"  {'Wilcoxon W':10s} max |difference| = {ds:.2e}  over {len(merged_wx)} comparisons")
    print(f"  {'Wilcoxon p':10s} max |difference| = {dp:.2e}")
    tol = 1e-3 if fresh else 1e-9
    ok = worst < tol and ds == 0 and dp < tol
    mode = "fresh screen" if fresh else "shipped reference scores"
    print(f"\n  {'PASS' if ok else 'MISMATCH'}: the recomputed numbers "
          f"{'reproduce' if ok else 'do NOT reproduce'} the published tables "
          f"({mode}; tolerance {tol:g}; worst metric {worst:.2e}, W {ds:.2e}, p {dp:.2e})")
    return ok


def main():
    os.makedirs(RESULTS, exist_ok=True)
    frames, fresh = load_per_binder()
    n_targets = frames[REFERENCE].target.nunique()
    n_pairs = len(frames[REFERENCE])
    print(f"{n_targets} targets | {n_pairs:,} binder pairs | "
          f"{frames[REFERENCE].SMILES.nunique():,} distinct binder molecules")

    per_target = per_target_table(frames)
    stats = wilcoxon_table(per_target)
    per_target.to_csv(os.path.join(RESULTS, "per_target_ef.csv"), index=False)
    stats.to_csv(os.path.join(RESULTS, "wilcoxon.csv"), index=False)
    print(f"\n[OUT] {os.path.join(RESULTS, 'per_target_ef.csv')}")
    print(f"[OUT] {os.path.join(RESULTS, 'wilcoxon.csv')}")

    print("\n--- Mean EF across the 21 targets ---")
    summary = (per_target.groupby("model")[[f"EF@{d:g}%" for d in DEPTHS]]
               .agg(["mean", "std"]).round(4))
    order = [REFERENCE] + list(BASELINES)
    print(summary.loc[order].to_string())

    draw_panel(per_target, stats, os.path.join(RESULTS, "robin_smm_ef_leakage_filtered.png"))
    ok = compare_with_reference(per_target, stats, fresh)
    # non-zero on MISMATCH: run.sh runs under `set -euo pipefail`, so without this a
    # mismatch would still print the wrapper's success footer
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
