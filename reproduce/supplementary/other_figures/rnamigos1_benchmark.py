#!/usr/bin/env python3
"""Fig. 2a and Supplementary Fig. 17a - SMARTBind against RNAmigos1 on the RNAmigos1
10-fold random split.

Replaces the notebook of the same name, which read four pickles from a `rnamigos1_swap/`
directory that is not part of the release.  The same per-binder values ship as Source Data,
so this reads `raw_data/` and runs from a clean clone with no weights and no inference.

  Fig. 2a                SMARTBind vs RNAmigos1, two screening libraries
  Supplementary Fig. 17a the same plus SMARTBind-noAug

RNAmigos1's own bars are the values its paper reports; they are read from the summary
tables rather than recomputed, because no per-binder record for RNAmigos1 exists.

Usage
    python rnamigos1_benchmark.py
    python rnamigos1_benchmark.py --out-dir figures
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATASETS = ["PDB Ligand", "DecoyFinder"]
COLOR = {"SMARTBind": "#64599C", "SMARTBind-noAug": "#A8C8E8", "RNAmigos1": "#F4E9B3"}


def find_raw_data():
    for folder in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
        if (folder / "raw_data").is_dir():
            return folder / "raw_data"
    raise SystemExit(
        "\nCould not find `raw_data/`.\n\n"
        "Download the Source Data archive from the link given in the repository\n"
        "README and unpack it at the repository root, next to `smartbind/`.\n")


def series(per_binder, model):
    """mean and sample s.d. of the per-binder rank percentiles, one entry per library."""
    means, sds, ns = [], [], []
    for lib in DATASETS:
        v = per_binder.loc[(per_binder.model == model) &
                           (per_binder.screening_library == lib), "rank_percentile"].values
        means.append(float(v.mean()))
        sds.append(float(v.std(ddof=1)))
        ns.append(len(v))
    return means, sds, ns


def from_summary(summary, model, mean_col, sd_col, lib_col):
    """RNAmigos1's published bars, which have no per-binder record to recompute from."""
    means, sds = [], []
    for lib in DATASETS:
        row = summary[(summary.model == model) & (summary[lib_col] == lib)].iloc[0]
        means.append(float(row[mean_col]))
        sds.append(float(row[sd_col]))
    return means, sds


def draw(path, bars, figsize, ylabel_size, tick_size, xtick_size, legend_cols):
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(DATASETS))
    width = 0.25 if len(bars) == 3 else 0.35
    offsets = ([-width, 0.0, width] if len(bars) == 3 else [-width / 2, width / 2])
    alpha = 1.0 if len(bars) == 3 else 0.7
    for (label, means, sds), off in zip(bars, offsets):
        ax.bar(x + off, means, width, label=label, color=COLOR[label], alpha=alpha,
               edgecolor="black", linewidth=1, yerr=sds, capsize=5, ecolor="black")
    ax.set_ylim(0.0, 1.12)
    ax.set_yticks(np.arange(0.0, 1.09, 0.2))
    ax.set_ylabel("Rank Percentile", fontsize=ylabel_size, fontweight="bold")
    ax.tick_params(axis="y", labelsize=tick_size, width=1, length=6)
    ax.tick_params(axis="x", labelsize=xtick_size, width=1, length=6)
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS, fontsize=xtick_size, fontweight="bold")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1)
        ax.spines[side].set_color("black")
    ax.grid(False)
    ax.legend(loc="upper center", fontsize=12, frameon=False,
              ncol=legend_cols, bbox_to_anchor=(0.5, 1.05))
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(f"{path}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "figures"))
    ap.add_argument("--tolerance", type=float, default=5e-4)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    RAW = find_raw_data()
    fig2a = pd.read_csv(RAW / "Fig2" / "Fig2a_rank_percentile_per_binder.csv")
    fig2a_sum = pd.read_csv(RAW / "Fig2" / "Fig2a_summary.csv")
    s17a = pd.read_csv(RAW / "SupplementaryFig17" /
                       "SupplementaryFig17a_rnamigos1_rank_percentile_per_sample.csv")
    s17a_sum = pd.read_csv(RAW / "SupplementaryFig17" / "SupplementaryFig17a_summary.csv")

    sb_mean, sb_sd, sb_n = series(fig2a, "SMARTBind")
    rm_mean, rm_sd = from_summary(fig2a_sum, "RNAmigos1",
                                  "plotted_bar_height", "plotted_error_bar", "screening_library")
    na_mean, na_sd, na_n = series(s17a, "SMARTBind-noAug")

    draw(out / "Fig2a_rnamigos1_benchmark",
         [("SMARTBind", sb_mean, sb_sd), ("RNAmigos1", rm_mean, rm_sd)],
         (7, 9), 24, 20, 22, 2)
    draw(out / "SupplementaryFig17a_rnamigos1_benchmark",
         [("SMARTBind", sb_mean, sb_sd), ("SMARTBind-noAug", na_mean, na_sd),
          ("RNAmigos1", rm_mean, rm_sd)],
         (7, 6), 18, 14, 18, 3)
    print(f"[OUT] {out}/Fig2a_rnamigos1_benchmark.png|.pdf")
    print(f"[OUT] {out}/SupplementaryFig17a_rnamigos1_benchmark.png|.pdf")

    # ---- self-check: every drawn bar against the shipped summary tables ----
    failures, n_checks = [], 0

    def compare(label, got, want, tol=None):
        nonlocal n_checks
        n_checks += 1
        if abs(float(got) - float(want)) > (a.tolerance if tol is None else tol):
            failures.append(f"{label}: recomputed {float(got):.6f} vs shipped {float(want):.6f}")

    print("\nFig. 2a - bar height and error bar, recomputed from the per-binder table")
    for i, lib in enumerate(DATASETS):
        row = fig2a_sum[(fig2a_sum.model == "SMARTBind") &
                        (fig2a_sum.screening_library == lib)].iloc[0]
        compare(f"2a SMARTBind/{lib} n", sb_n[i], row.n_samples, 0)
        compare(f"2a SMARTBind/{lib} mean", sb_mean[i], row.plotted_bar_height)
        compare(f"2a SMARTBind/{lib} sd", sb_sd[i], row.plotted_error_bar)
        print(f"  {lib:<14} n={sb_n[i]:<5} {sb_mean[i]:.4f} +/- {sb_sd[i]:.4f}   "
              f"shipped {row.plotted_bar_height:.4f} +/- {row.plotted_error_bar:.4f}")

    print("\nSupplementary Fig. 17a - the same plus SMARTBind-noAug")
    for model, means, sds, ns in (("SMARTBind", sb_mean, sb_sd, sb_n),
                                  ("SMARTBind-noAug", na_mean, na_sd, na_n)):
        for i, lib in enumerate(DATASETS):
            row = s17a_sum[(s17a_sum.model == model) &
                           (s17a_sum.screening_library == lib)].iloc[0]
            compare(f"17a {model}/{lib} n", ns[i], row.n, 0)
            compare(f"17a {model}/{lib} mean", means[i], row.plotted_bar_height_mean)
            compare(f"17a {model}/{lib} sd", sds[i], row.errorbar_value_sd_sample_ddof1)
            print(f"  {model:<16} {lib:<14} n={ns[i]:<5} {means[i]:.4f} +/- {sds[i]:.4f}   "
                  f"shipped {row.plotted_bar_height_mean:.4f} +/- "
                  f"{row.errorbar_value_sd_sample_ddof1:.4f}")

    print()
    if failures:
        for f in failures:
            print("  " + f)
        print(f"MISMATCH - {len(failures)} of {n_checks} values differ by more than {a.tolerance}")
        return 1
    print(f"PASS - every drawn bar reproduces the shipped summary tables ({n_checks} checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
