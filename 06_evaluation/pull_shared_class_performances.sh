PYTHON="/cfs/klemming/projects/supr/snic2020-6-126/environments/Karin/amime_uv_env/bin/python"

echo ""
echo "Aggregating overlap-class results with run-specific overlap files..."
"$PYTHON" - << 'PY'
import json, glob, os, csv, statistics, re

# ---- Absolute base path to project ----
PROJECT_ROOT = "/cfs/klemming/projects/supr/snic2020-6-126/projects/amime/DDLS_course_project"

RUNS_ROOT = os.path.join(PROJECT_ROOT, "05_model_training_and_inference/runs")
RESULTS_PATTERN = os.path.join(RUNS_ROOT, "*", "*", "results.json")

OVERLAP_FILES_BY_RUN = {
    # Baltic ↔ Baltic
    "syke_to_baltic": os.path.join(
        PROJECT_ROOT,
        "02_data_set_visualization/dataset_overlap_reports/same-location-baltic.txt"
    ),

    # West coast ↔ West coast
    "tangesund_to_skagerrak": os.path.join(
        PROJECT_ROOT,
        "02_data_set_visualization/dataset_overlap_reports/same-location-west-coast.txt"
    ),

    # West coast ↔ Baltic (different locations)
    "tangesund_to_baltic": os.path.join(
        PROJECT_ROOT,
        "02_data_set_visualization/dataset_overlap_reports/different-location-west-coast-to-baltic.txt"
    ),
    "syke_to_skagerrak": os.path.join(
        PROJECT_ROOT,
        "02_data_set_visualization/dataset_overlap_reports/different-location-west-coast-to-baltic.txt"
    ),
}

SPECIAL_KEYS = {"accuracy", "macro avg", "weighted avg"}

def norm_name(name: str) -> str:
    return re.sub(r'[^0-9A-Za-z]+', '', name).lower()

overlap_cache = {}  # path -> (raw_names, normalized_set)

def get_overlap_for_run(run_tag: str):
    path = OVERLAP_FILES_BY_RUN.get(run_tag)
    if path is None:
        return None, None

    if path in overlap_cache:
        return overlap_cache[path]

    with open(path) as f:
        raw = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

    norm_set = {norm_name(c) for c in raw}
    overlap_cache[path] = (raw, norm_set)
    print(f"[INFO] Loaded {len(norm_set)} overlap classes for {run_tag} from {path}")
    return raw, norm_set

def extract_overlap_metrics(src_report, tgt_report, overlap_norm):
    overlap_classes_found = []
    per_class_rows = []

    for cls, src_stats in src_report.items():
        if cls in SPECIAL_KEYS:
            continue

        cls_norm = norm_name(cls)
        if cls_norm not in overlap_norm:
            continue

        if cls not in tgt_report:
            continue

        tgt_stats = tgt_report[cls]

        try:
            src_support = float(src_stats.get("support", 0.0))
            tgt_support = float(tgt_stats.get("support", 0.0))
        except (TypeError, ValueError):
            continue

        if tgt_support <= 0:
            continue

        overlap_classes_found.append(cls)
        per_class_rows.append({
            "class": cls,
            "src_precision": src_stats.get("precision", None),
            "src_recall": src_stats.get("recall", None),
            "src_f1": src_stats.get("f1-score", None),
            "src_support": src_support,
            "tgt_precision": tgt_stats.get("precision", None),
            "tgt_recall": tgt_stats.get("recall", None),
            "tgt_f1": tgt_stats.get("f1-score", None),
            "tgt_support": tgt_support,
        })

    if not overlap_classes_found:
        return None, None, [], []

    src_macro_f1 = statistics.mean(
        r["src_f1"] for r in per_class_rows if r["src_f1"] is not None
    )
    tgt_macro_f1 = statistics.mean(
        r["tgt_f1"] for r in per_class_rows if r["tgt_f1"] is not None
    )

    return src_macro_f1, tgt_macro_f1, overlap_classes_found, per_class_rows


rows = []
paths = glob.glob(RESULTS_PATTERN)

print(f"[INFO] Looking for results.json under: {RESULTS_PATTERN}")
print(f"[INFO] Found {len(paths)} files.")

if not paths:
    print(f"No results.json found under {RESULTS_PATTERN}")
else:
    for path in sorted(paths):
        with open(path) as f:
            r = json.load(f)

        parts = path.split(os.sep)
        run_tag = parts[-3]
        method = parts[-2]

        raw_overlap, overlap_norm = get_overlap_for_run(run_tag)
        if overlap_norm is None:
            print(f"[WARN] No overlap file configured for run_tag={run_tag}, skipping {path}")
            continue

        src_report = r.get("source_val", {}).get("report", {})
        tgt_report = r.get("target_eval", {}).get("report", {})

        if not src_report or not tgt_report:
            print(f"[WARN] Missing source_val or target_eval report in {path}, skipping.")
            continue

        src_macro_overlap, tgt_macro_overlap, overlap_classes_found, per_class_rows = extract_overlap_metrics(
            src_report, tgt_report, overlap_norm
        )

        if not overlap_classes_found:
            print(f"[WARN] No overlap classes with target support > 0 for {path}")
            continue

        out_dir = os.path.dirname(path)
        per_run_csv = os.path.join(out_dir, "overlap_classes_metrics.csv")
        with open(per_run_csv, "w", newline="") as fcsv:
            fieldnames = [
                "class",
                "src_precision", "src_recall", "src_f1", "src_support",
                "tgt_precision", "tgt_recall", "tgt_f1", "tgt_support",
            ]
            w = csv.DictWriter(fcsv, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(per_class_rows)

        rows.append({
            "run": run_tag,
            "method": method,
            "overlap_file": OVERLAP_FILES_BY_RUN.get(run_tag),
            "n_overlap_classes": len(overlap_classes_found),
            "overlap_classes": ";".join(overlap_classes_found),
            "src_macroF1_overlap": src_macro_overlap,
            "tgt_macroF1_overlap": tgt_macro_overlap,
            "file": path,
        })

    rows.sort(key=lambda x: (x["run"], x["method"]))
    os.makedirs(RUNS_ROOT, exist_ok=True)
    out_csv = os.path.join(RUNS_ROOT, "summary_overlap_classes.csv")
    with open(out_csv, "w", newline="") as f:
        fieldnames = [
            "run",
            "method",
            "overlap_file",
            "n_overlap_classes",
            "overlap_classes",
            "src_macroF1_overlap",
            "tgt_macroF1_overlap",
            "file",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print("Wrote", out_csv)
PY

echo "✅ Overlap-class aggregation done."
