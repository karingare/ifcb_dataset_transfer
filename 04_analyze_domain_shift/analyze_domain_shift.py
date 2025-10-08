#!/usr/bin/env python3



# example use:
""" python analyze_domain_shift.py \
  --manifest "/cfs/klemming/projects/supr/snic2020-6-126/projects/amime/DDLS_course_project/01_data_prep/out_filtered_min50/combined_manifest_filtered.csv" \
  --out "/cfs/klemming/projects/supr/snic2020-6-126/projects/amime/DDLS_course_project/04_analyze_domain_shift/reports/domain_shift" \
  --exclude-class "Unclassifiable" \
  --max-per-dataset 10000 \
  --bins 256 \
  --workers 8"""

import argparse, os, math, json
from pathlib import Path
import multiprocessing as mp
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

DEF_ENCODING = "utf-8"

def js_divergence(p, q, eps=1e-12):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    m = 0.5 * (p + q)
    def kl(a, b):
        a = np.clip(a, eps, None)
        b = np.clip(b, eps, None)
        return np.sum(a * np.log(a / b))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

def percentile(a, p):
    if a.size == 0: return np.nan
    return float(np.percentile(a, p))

def analyze_one(args):
    row, bins = args
    ds = row["dataset"]
    fpath = row["filepath"]
    cls = row.get("standardized_class", row.get("class_name", ""))

    try:
        with Image.open(fpath) as im:
            # convert to grayscale [0..255]
            g = im.convert("L")
            arr = np.array(g, dtype=np.uint8)
            h, w = arr.shape
            flat = arr.reshape(-1).astype(np.float32)
            # metrics
            mean = float(flat.mean()) / 255.0           # brightness [0..1]
            std = float(flat.std()) / 255.0             # contrast [0..1]
            p1 = percentile(flat, 1.0) / 255.0
            p99 = percentile(flat, 99.0) / 255.0
            pct_black = float((flat <= 1).mean())       # near-zero clip
            pct_white = float((flat >= 254).mean())     # near-255 clip
            aspect = (w / h) if h else np.nan
            mpix = (w * h) / 1e6

            # histograms
            hist_b, _ = np.histogram(flat, bins=bins, range=(0, 255))
            # for "contrast histogram", use per-image std as a sample -> filled upstream

            return {
                "ok": True,
                "dataset": ds,
                "class": cls,
                "filepath": fpath,
                "width": w,
                "height": h,
                "aspect_ratio": aspect,
                "megapixels": mpix,
                "brightness_mean": mean,
                "contrast_std": std,
                "p1": p1,
                "p99": p99,
                "pct_black": pct_black,
                "pct_white": pct_white,
                "hist_brightness": hist_b.astype(np.int64),
            }
    except Exception as e:
        return {"ok": False, "dataset": ds, "filepath": fpath, "error": str(e)}

def plot_hist(data_by_ds, title, xlabel, out_png, bins=None):
    plt.figure(figsize=(9,4.5))
    for ds, vals in data_by_ds.items():
        plt.hist(vals, bins=50 if bins is None else bins, alpha=0.5, label=ds, density=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    if len(data_by_ds) <= 10:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    plt.close()

def plot_scatter_size(samples_by_ds, out_png):
    plt.figure(figsize=(6.5,6))
    for ds, wh in samples_by_ds.items():
        if not wh: 
            continue
        w = [x[0] for x in wh]
        h = [x[1] for x in wh]
        plt.scatter(w, h, s=4, alpha=0.4, label=ds)
    plt.xlabel("Width (px)")
    plt.ylabel("Height (px)")
    plt.title("Image size scatter")
    if len(samples_by_ds) <= 10:
        plt.legend(markerscale=3, frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    plt.close()

def heatmap(matrix_df, title, out_png):
    plt.figure(figsize=(6.5,5.5))
    im = plt.imshow(matrix_df.values, interpolation="nearest")
    plt.xticks(range(len(matrix_df.columns)), matrix_df.columns, rotation=45, ha="right")
    plt.yticks(range(len(matrix_df.index)), matrix_df.index)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Analyze size/brightness/contrast per dataset to spot domain shifts.")
    ap.add_argument("--manifest", required=True, help="combined_manifest_filtered.csv")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--exclude-class", action="append", default=[], help="Exclude any rows with this standardized_class (repeatable)")
    ap.add_argument("--max-per-dataset", type=int, default=0, help="Cap images per dataset (0 = all)")
    ap.add_argument("--bins", type=int, default=256, help="Histogram bins for brightness")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count()//2))
    args = ap.parse_args()

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.manifest, encoding=DEF_ENCODING)
    need_cols = {"dataset", "filepath"}
    if not need_cols.issubset(df.columns):
        raise SystemExit(f"Manifest must have columns at least: {need_cols}")
    if args.exclude_class:
        std_col = "standardized_class" if "standardized_class" in df.columns else None
        if std_col:
            df = df[~df[std_col].isin(set(args.exclude_class))].copy()

    # sample per dataset if requested
    if args.max_per_dataset > 0:
        df = df.groupby("dataset", group_keys=False).apply(lambda g: g.sample(min(len(g), args.max_per_dataset), random_state=42))

    # process
    rows = df.to_dict("records")
    work = [(r, args.bins) for r in rows]

    results = []
    with mp.Pool(processes=args.workers) as pool:
        for out in pool.imap_unordered(analyze_one, work, chunksize=64):
            results.append(out)

    # split ok/errors
    ok = [r for r in results if r.get("ok")]
    errs = [r for r in results if not r.get("ok")]
    if errs:
        pd.DataFrame(errs).to_csv(outdir / "errors_reading_images.csv", index=False)

    if not ok:
        print("No images processed successfully.")
        return

    # per-image metrics CSV
    # Flatten hist for CSV only if you want (we’ll keep separate JSON for hists)
    per_image = pd.DataFrame([{k:v for k,v in r.items() if k not in ("hist_brightness",)} for r in ok])
    per_image.to_csv(outdir / "per_image_metrics.csv", index=False)

    # aggregate histograms per dataset
    datasets = sorted(per_image["dataset"].unique())
    H_brightness = {}
    contrast_samples = defaultdict(list)
    width_height = defaultdict(list)
    aspect_samples = defaultdict(list)
    for r in ok:
        ds = r["dataset"]
        H_brightness.setdefault(ds, np.zeros(args.bins, dtype=np.int64))
        H_brightness[ds] += r["hist_brightness"]
        contrast_samples[ds].append(r["contrast_std"])
        width_height[ds].append((r["width"], r["height"]))
        aspect_samples[ds].append(r["aspect_ratio"])

    # per-dataset summary stats
    summaries = []
    for ds, g in per_image.groupby("dataset"):
        summaries.append({
            "dataset": ds,
            "n_images": len(g),
            "width_mean": g["width"].mean(), "width_median": g["width"].median(),
            "height_mean": g["height"].mean(), "height_median": g["height"].median(),
            "megapixels_mean": g["megapixels"].mean(), "megapixels_median": g["megapixels"].median(),
            "brightness_mean_mean": g["brightness_mean"].mean(),
            "brightness_mean_median": g["brightness_mean"].median(),
            "contrast_std_mean": g["contrast_std"].mean(),
            "contrast_std_median": g["contrast_std"].median(),
            "p1_mean": g["p1"].mean(), "p99_mean": g["p99"].mean(),
            "pct_black_mean": g["pct_black"].mean(), "pct_white_mean": g["pct_white"].mean(),
        })
    pd.DataFrame(summaries).to_csv(outdir / "per_dataset_summary.csv", index=False)

    # save hist JSON (optional)
    hist_json = {ds: H_brightness[ds].tolist() for ds in datasets}
    (outdir / "brightness_histograms.json").write_text(json.dumps(hist_json, indent=2))

    # plots: brightness histograms (overlay) & per-dataset
    # overlay brightness
    brightness_samples = {}
    for ds in datasets:
        # Turn hist into pseudo-samples for overlay? Instead use per-image means instead:
        brightness_samples[ds] = per_image[per_image["dataset"]==ds]["brightness_mean"].values
    plot_hist(brightness_samples, "Brightness (per-image mean) by dataset", "Brightness (0..1)",
              outdir / "brightness_per_image_mean_overlay.png")

    # contrast overlay
    plot_hist(contrast_samples, "Contrast (per-image std) by dataset", "Contrast (0..1)",
              outdir / "contrast_per_image_std_overlay.png")

    # aspect ratio overlay
    plot_hist(aspect_samples, "Aspect ratio by dataset", "Aspect ratio (W/H)",
              outdir / "aspect_ratio_overlay.png")

    # size scatter
    plot_scatter_size(width_height, outdir / "size_scatter.png")

    # JS divergence matrices (brightness hist, contrast KDE via per-image std histogram)
    # brightness JS
    B = args.bins
    bright_mat = np.zeros((len(datasets), len(datasets)), dtype=float)
    for i, d1 in enumerate(datasets):
        for j, d2 in enumerate(datasets):
            bright_mat[i, j] = js_divergence(H_brightness[d1], H_brightness[d2])
    bright_df = pd.DataFrame(bright_mat, index=datasets, columns=datasets)
    bright_df.to_csv(outdir / "brightness_js_divergence.csv")
    heatmap(bright_df, "JS divergence — brightness hist", outdir / "brightness_js_divergence.png")

    # contrast JS: bin per-image std into histogram (0..1)
    contrast_H = {}
    for ds in datasets:
        cs = np.asarray(contrast_samples[ds], dtype=float)
        contrast_H[ds], _ = np.histogram(cs, bins=50, range=(0.0, 1.0))
    contr_mat = np.zeros((len(datasets), len(datasets)), dtype=float)
    for i, d1 in enumerate(datasets):
        for j, d2 in enumerate(datasets):
            contr_mat[i, j] = js_divergence(contrast_H[d1], contrast_H[d2])
    contr_df = pd.DataFrame(contr_mat, index=datasets, columns=datasets)
    contr_df.to_csv(outdir / "contrast_js_divergence.csv")
    heatmap(contr_df, "JS divergence — contrast", outdir / "contrast_js_divergence.png")

    print(f"✅ Wrote reports to: {outdir}")

if __name__ == "__main__":
    main()
