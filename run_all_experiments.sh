#!/usr/bin/env bash
set -euo pipefail

### ------------------------------------------------------------
### IFCB Domain Adaptation — batch runner
### Runs: source_only, CORAL, BN-adapt for each source→target pair
### Then aggregates all results into a single CSV.
### ------------------------------------------------------------

# ---- Python interpreter (use your working ROCm env Python)
PYTHON="/cfs/klemming/projects/supr/snic2020-6-126/environments/Karin/amime_uv_env/bin/python"

# ---- Common training knobs
EPOCHS="${EPOCHS:-12}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-1e-3}"
CORAL_LAMBDA="${CORAL_LAMBDA:-0.5}"
BN_PASSES="${BN_PASSES:-1}"
EXCLUDE_CLASSES=("Unclassifiable")
# Set STRONG_AUG=1 to turn on TrivialAugmentWide
STRONG_AUG="${STRONG_AUG:-0}"

# ---- Paths (update if your symlink root changes)
SYKE="/cfs/klemming/projects/supr/snic2020-6-126/projects/amime/DDLS_course_project/03_dataset_creation/exports/dataset_symlinked/SYKE_plankton_IFCB_Utö_2021"
TANGESUND="/cfs/klemming/projects/supr/snic2020-6-126/projects/amime/DDLS_course_project/03_dataset_creation/exports/dataset_symlinked/smhi_ifcb_tångesund"
BALTIC_PROPER="/cfs/klemming/projects/supr/snic2020-6-126/projects/amime/DDLS_course_project/03_dataset_creation/exports/dataset_symlinked/smhi_ifcb_svea_baltic_proper"
SKAGERRAK="/cfs/klemming/projects/supr/snic2020-6-126/projects/amime/DDLS_course_project/03_dataset_creation/exports/dataset_symlinked/smhi_ifcb_svea_skagerrak_kattegat"

# ---- Script locations
TRAIN_EVAL="05_model_training_and_inference/train_eval.py"
RUNS_ROOT="05_model_training_and_inference/runs"

# ---- Helper to join array into CLI words
join_excludes() {
  local arr=("$@")
  # shellcheck disable=SC2145
  echo "${arr[@]}"
}

# ---- Run one configuration
run_job() {
  local src="$1"
  local tgt="$2"
  local tag="$3"      # e.g., syke_to_baltic, tangesund_to_skagerrak
  local method="$4"   # source_only | coral | bn_adapt

  local out_dir="${RUNS_ROOT}/${tag}/${method}"
  mkdir -p "${out_dir}"

  echo ""
  echo "=============================="
  echo "Run: ${tag} | method=${method}"
  echo "src=${src}"
  echo "tgt=${tgt}"
  echo "out=${out_dir}"
  echo "=============================="

  local args_common=(
    "${TRAIN_EVAL}" train
    --source-root "${src}"
    --target-root "${tgt}"
    --run-dir "${out_dir}"
    --method "${method}"
    --epochs "${EPOCHS}"
    --batch-size "${BATCH_SIZE}"
    --lr "${LR}"
    --exclude-classes $(join_excludes "${EXCLUDE_CLASSES[@]}")
  )

  # optional strong aug
  if [[ "${STRONG_AUG}" == "1" ]]; then
    args_common+=("--strong-aug")
  fi

  # method-specific flags
  if [[ "${method}" == "coral" ]]; then
    args_common+=(--coral-lambda "${CORAL_LAMBDA}")
  elif [[ "${method}" == "bn_adapt" ]]; then
    args_common+=(--bn-passes "${BN_PASSES}")
  fi

  # launch
  "${PYTHON}" "${args_common[@]}"
}

### ------------------------------------------------------------
### Experiment matrix
### ------------------------------------------------------------
# We run:
# 1) Tångesund → Skagerrak/Kattegat
# 2) SYKE → Baltic Proper
# 3) SYKE → Skagerrak/Kattegat

# 1) Tångesund → Skagerrak/Kattegat
# run_job "${TANGESUND}" "${SKAGERRAK}" "tangesund_to_skagerrak" "source_only"
# run_job "${TANGESUND}" "${SKAGERRAK}" "tangesund_to_skagerrak" "coral"
run_job "${TANGESUND}" "${SKAGERRAK}" "tangesund_to_skagerrak" "bn_adapt"

# 2) SYKE → Baltic Proper
run_job "${SYKE}" "${BALTIC_PROPER}" "syke_to_baltic" "source_only"
run_job "${SYKE}" "${BALTIC_PROPER}" "syke_to_baltic" "coral"
run_job "${SYKE}" "${BALTIC_PROPER}" "syke_to_baltic" "bn_adapt"

# 3) SYKE → Skagerrak/Kattegat
#run_job "${SYKE}" "${SKAGERRAK}" "syke_to_skagerrak" "source_only"
#run_job "${SYKE}" "${SKAGERRAK}" "syke_to_skagerrak" "coral"
#run_job "${SYKE}" "${SKAGERRAK}" "syke_to_skagerrak" "bn_adapt"

# 4) Tångesund → Baltic Proper
run_job "${TANGESUND}" "${BALTIC_PROPER}" "tangesund_to_baltic" "source_only"
run_job "${TANGESUND}" "${BALTIC_PROPER}" "tangesund_to_baltic" "coral"
run_job "${TANGESUND}" "${BALTIC_PROPER}" "tangesund_to_baltic" "bn_adapt"

### ------------------------------------------------------------
### Aggregate results to CSV
### ------------------------------------------------------------
echo ""
echo "Aggregating results to ${RUNS_ROOT}/summary.csv ..."
"${PYTHON}" - << 'PY'
import json, glob, os, csv
rows=[]
paths = glob.glob("05_model_training_and_inference/runs/*/*/results.json")
if not paths:
    print("No results.json found under runs/*/*/")
else:
    for path in paths:
        with open(path) as f:
            r=json.load(f)
        parts = path.split(os.sep)
        # runs/<src>_to_<tgt>/<method>/results.json
        tag = parts[-3]
        method = parts[-2]
        rows.append({
            "run": tag,
            "method": method,
            "src_macroF1": r["source_val"]["macro_f1"],
            "src_acc": r["source_val"]["accuracy"],
            "tgt_macroF1": r.get("target_eval",{}).get("macro_f1", None),
            "tgt_acc": r.get("target_eval",{}).get("accuracy", None),
            "file": path
        })
    rows.sort(key=lambda x:(x["run"], x["method"]))
    os.makedirs("05_model_training_and_inference/runs", exist_ok=True)
    out_csv = "05_model_training_and_inference/runs/summary.csv"
    with open(out_csv,"w",newline="") as f:
        w=csv.DictWriter(f, fieldnames=["run","method","src_macroF1","src_acc","tgt_macroF1","tgt_acc","file"])
        w.writeheader(); w.writerows(rows)
    print("Wrote", out_csv)
PY

echo "✅ All done."
