#!/usr/bin/env bash
set -euo pipefail

# ---- config you might tweak ----
FLOW="ifcb_flow.py"   # or "label_flow.py" if you named it that
ROOTS="roots.json"
MIN_TRAIN=50
TRAIN1="SYKE_plankton_IFCB_Utö_2021"
TRAIN2="smhi_ifcb_tångesund"
OVERRIDES="manual_overrides.csv"   # e.g., OVERRIDES="manual_overrides.csv"

# ---- setup ----
python -m pip install --user --quiet pandas requests

# ---- 1) Scan (non-recursive): counts + manifests ----
# rm -rf out_scan out_std vocab_out vocab_min${MIN_TRAIN} out_filtered_min${MIN_TRAIN} || true
# python "$FLOW" scan --roots "$ROOTS" --out out_scan --emit-manifests

# ---- 2) Build mapping with WoRMS (and optional overrides) ----
python ifcb_flow.py map --classes out_scan/all_unique_classes.csv \
  --out class_mapping.csv --cache worms_cache.json \
  --overrides manual_overrides.csv

# (Optional) If you manually reviewed and saved a fixed file, point to it here:
MAPPING="class_mapping.csv"
# MAPPING="class_mapping_manually_reviewed.csv"

# ---- 3) Apply mapping to summaries + manifests ----
python "$FLOW" apply-map --scan-dir out_scan --mapping "$MAPPING" --out out_std

# ---- 4) Build training vocabulary from the two training sets ----
python "$FLOW" vocab --std-dir out_std \
  --training-datasets "$TRAIN1" "$TRAIN2" \
  --min-per-train "$MIN_TRAIN" \
  --exclude-classes "Unclassifiable" \
  --out "vocab_min${MIN_TRAIN}"

# Sanity check
echo "Classes in vocab:"
wc -l "vocab_min${MIN_TRAIN}/training_vocab.txt" || true

# ---- 5) Filter ALL datasets to the training vocab (closed-set eval) ----
python "$FLOW" filter --std-dir out_std \
  --vocab "vocab_min${MIN_TRAIN}/training_vocab.txt" \
  --out "out_filtered_min${MIN_TRAIN}"

echo "✅ Done. Key outputs:"
echo " - out_scan/: raw class counts & combined_manifest.csv"
echo " - class_mapping.csv (+ *_REVIEW.csv if unresolved)"
echo " - out_std/: standardized summaries & manifests"
echo " - vocab_min${MIN_TRAIN}/training_vocab.txt, label_index.json"
echo " - out_filtered_min${MIN_TRAIN}/: filtered summaries, pivot, removal report"
