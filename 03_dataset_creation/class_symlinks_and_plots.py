#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create symlinked datasets and plot class distributions per dataset.

Examples
--------
# Just plots (top 30) from your filtered pivot:
python class_symlinks_and_plots.py \
  --pivot "/cfs/klemming/projects/supr/snic2020-6-126/projects/amime/DDLS_course_project/01_data_prep/out_filtered_min50/pivot_classes_x_datasets.csv" \
  --plots-out "/cfs/klemming/projects/supr/snic2020-6-126/projects/amime/DDLS_course_project/03_dataset_creation/reports/class_distributions" \
  --top-n 30

# Also build symlinked datasets first (from filtered manifest):
python class_symlinks_and_plots.py \
  --pivot "/cfs/klemming/projects/supr/snic2020-6-126/projects/amime/DDLS_course_project/01_data_prep/out_filtered_min50/pivot_classes_x_datasets.csv" \
  --manifest "/cfs/klemming/projects/supr/snic2020-6-126/projects/amime/DDLS_course_project/01_data_prep/out_filtered_min50/combined_manifest_filtered.csv" \
  --symlink-root "/cfs/klemming/projects/supr/snic2020-6-126/projects/amime/DDLS_course_project/03_dataset_creation/exports/dataset_symlinked" \
  --plots-out "/cfs/klemming/projects/supr/snic2020-6-126/projects/amime/DDLS_course_project/03_dataset_creation/reports/class_distributions" \
  --do-symlinks \
  --top-n 30
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
from typing import List

import pandas as pd
import matplotlib.pyplot as plt


def slug(s: str) -> str:
    """Make a filename-friendly slug (keep unicode, just avoid path-breaking chars)."""
    return re.sub(r"[^0-9A-Za-z._\-åäöÅÄÖ]+", "_", s, flags=re.UNICODE)


def ensure_cols(df: pd.DataFrame, required: List[str], where: str):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in {where}: {missing}\nAvailable: {list(df.columns)}")


def load_pivot(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    # Some pivots write the class in the first unnamed column
    if "standardized_class" not in df.columns:
        first = df.columns[0]
        df = df.rename(columns={first: "standardized_class"})
    ensure_cols(df, ["standardized_class"], "pivot")
    return df


def load_manifest(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    ensure_cols(df, ["dataset", "standardized_class", "filepath"], "manifest")
    # Drop rows where file vanished
    exists_mask = df["filepath"].apply(lambda p: Path(p).exists())
    if not exists_mask.all():
        missing = (~exists_mask).sum()
        print(f"⚠️  Skipping {missing} rows with missing files (not found on disk).")
        df = df[exists_mask].copy()
    return df


def safe_symlink(target: Path, link: Path):
    """Create symlink; if link exists with different target, add a numeric suffix."""
    link.parent.mkdir(parents=True, exist_ok=True)
    if not link.exists():
        link.symlink_to(target)
        return

    # If already correct, skip
    try:
        if link.is_symlink() and link.resolve() == target.resolve():
            return
    except Exception:
        pass

    stem, suffix = link.stem, link.suffix
    for i in range(1, 10000):
        alt = link.with_name(f"{stem}__{i}{suffix}")
        if not alt.exists():
            alt.symlink_to(target)
            return
    raise RuntimeError(f"Could not create unique link for {link}")


def build_symlinks(manifest_csv: str, out_root: str):
    df = load_manifest(manifest_csv)
    out_root = Path(out_root)
    n = 0
    for _, r in df.iterrows():
        ds = str(r["dataset"])
        cls = str(r["standardized_class"])
        src = Path(r["filepath"])

        # Keep original filename but ensure path-safe
        fn = slug(src.name)
        link = out_root / slug(ds) / slug(cls) / fn
        try:
            safe_symlink(src, link)
            n += 1
        except Exception as e:
            print(f"⚠️  Could not link: {src} -> {link}: {e}")
    print(f"✅ Symlinks created under {out_root} (total links: {n})")


def plot_distributions(pivot_csv: str, plots_out: str, top_n: int = 30):
    df = load_pivot(pivot_csv)
    plots_dir = Path(plots_out)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # dataset columns are all except 'standardized_class'
    ds_cols = [c for c in df.columns if c != "standardized_class"]
    if not ds_cols:
        raise SystemExit("No dataset columns found in pivot.")

    for ds in ds_cols:
        if ds not in df.columns:
            continue
        # Series: class -> count for this dataset
        s = df.set_index("standardized_class")[ds].astype(int)
        s = s[s > 0].sort_values(ascending=False)

        if s.empty:
            print(f"ℹ️  No classes for dataset '{ds}' (all zeros). Skipping plot.")
            continue

        top = s.head(top_n)
        plt.figure(figsize=(12, max(4, 0.4 * len(top))))  # one plot per chart
        top.plot(kind="barh")  # no custom colors/styles
        plt.gca().invert_yaxis()
        plt.xlabel("Image count")
        plt.title(f"Top {min(top_n, len(s))} classes — {ds}")
        plt.tight_layout()

        out_png = plots_dir / f"class_dist_{slug(ds)}.png"
        plt.savefig(out_png, dpi=200)
        plt.close()
        print(f"📈 Wrote {out_png}")

    # Also write a CSV with per-dataset totals & unique-class counts
    summary_rows = []
    for ds in ds_cols:
        s = df[ds].astype(int)
        total = int(s.sum())
        n_classes = int((s > 0).sum())
        summary_rows.append({"dataset": ds, "total_images": total, "n_classes": n_classes})
    pd.DataFrame(summary_rows).to_csv(plots_dir / "per_dataset_summary.csv", index=False, encoding="utf-8")
    print(f"📝 Wrote {plots_dir / 'per_dataset_summary.csv'}")


def main():
    ap = argparse.ArgumentParser(description="Make symlinked datasets and plot class distributions per dataset.")
    ap.add_argument("--pivot", required=True, help="Path to pivot_classes_x_datasets.csv (filtered).")
    ap.add_argument("--plots-out", required=True, help="Output directory for charts and summaries.")
    ap.add_argument("--top-n", type=int, default=30, help="Top-N classes to show per dataset (default 30).")
    ap.add_argument("--do-symlinks", action="store_true", help="Create symlinked dataset tree before plotting.")
    ap.add_argument("--manifest", help="Path to combined_manifest_filtered.csv (required if --do-symlinks).")
    ap.add_argument("--symlink-root", help="Root directory to create symlinked datasets (required if --do-symlinks).")
    args = ap.parse_args()

    if args.do_symlinks:
        if not args.manifest or not args.symlink_root:
            raise SystemExit("--do-symlinks requires --manifest and --symlink-root")
        build_symlinks(args.manifest, args.symlink_root)

    plot_distributions(args.pivot, args.plots_out, top_n=args.top_n)


if __name__ == "__main__":
    main()
