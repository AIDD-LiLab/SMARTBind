#!/usr/bin/env bash
# One-click reproduction of the SMARTBind result on the leakage-filtered ROBIN
# SMM benchmark (Supplementary Fig. 10, SMM panel).
#
#   bash run.sh              screen with the 10-model ensemble, then evaluate
#   bash run.sh --eval-only  skip the screen, evaluate the shipped reference scores
#
# Requirements
#   * the SMARTBind conda environment (environment.yml at the repository root)
#   * SMARTBind_weight/ - the 10-model ensemble from the Zenodo record in the README,
#     or point SMARTBIND_WEIGHTS at it
#
# Runtime for the screen: ~2 min on one GPU, ~15 min on CPU (21 targets x 25,049
# molecules x 10 models). The evaluation takes a couple of seconds.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"

if [[ "${1:-}" != "--eval-only" ]]; then
    echo "=== 1/2  SMARTBind virtual screen ==============================="
    "$PYTHON" run_smartbind.py
    echo
else
    echo "=== 1/2  skipped (--eval-only): using the shipped reference scores"
    echo
fi

echo "=== 2/2  Enrichment factors and significance ==================="
"$PYTHON" evaluate_ef.py

echo
echo "Outputs are in results/:"
echo "  per_target_ef.csv                    per-target EF and AUROC, all four methods"
echo "  wilcoxon.csv                         paired two-sided Wilcoxon vs SMARTBind"
echo "  robin_smm_ef_leakage_filtered.png    the enrichment panel"
