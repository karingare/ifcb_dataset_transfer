#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd

PIVOT_CSV = "/cfs/klemming/projects/supr/snic2020-6-126/projects/amime/DDLS_course_project/01_data_prep/out_filtered_min50/pivot_classes_x_datasets.csv"
OUT_DIR = Path("dataset_overlap_reports")

COL_SYKE       = "SYKE_plankton_IFCB_Utö_2021"
COL_BALTIC     = "smhi_ifcb_svea_baltic_proper"
COL_TANGESUND  = "smhi_ifcb_tångesund"
COL_SKAGERRAK  = "smhi_ifcb_svea_skagerrak_kattegat"

PAIRS = {
    # renamed here:
    "same-location-baltic.txt":                   (COL_SYKE, COL_BALTIC),
    "same-location-west-coast.txt":               (COL_TANGESUND, COL_SKAGERRAK),

    # kept these names as-is (different datasets/locations):
    "different-location-baltic-to-west-coast.txt": (COL_SYKE, COL_SKAGERRAK),
    "different-location-west-coast-to-baltic.txt": (COL_TANGESUND, COL_BALTIC),
}

def load_pivot(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    if "standardized_class" not in df.columns:
        df = df.rename(columns={df.columns[0]: "standardized_class"})
    return df

def classes_present_in(df: pd.DataFrame, *cols: str) -> list[str]:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in pivot: {missing}\nAvailable: {list(df.columns)}")
    mask = (df[list(cols)] > 0).all(axis=1)
    return df.loc[mask, "standardized_class"].sort_values().tolist()

def write_list(lines: list[str], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for x in lines:
            f.write(x + "\n")

def main():
    df = load_pivot(PIVOT_CSV)

    summary = []
    for filename, cols in PAIRS.items():
        names = classes_present_in(df, *cols)
        out_path = OUT_DIR / filename
        write_list(names, out_path)
        summary.append((filename, len(names)))
        print(f"Wrote {len(names):3d} classes -> {out_path}")

    # boolean flags CSV (column names mirror the filenames)
    flags = df[["standardized_class"]].copy()
    for filename, cols in PAIRS.items():
        key = filename.replace(".txt", "")
        flags[key] = (df[list(cols)] > 0).all(axis=1)
    flags.to_csv(OUT_DIR / "overlap_flags.csv", index=False, encoding="utf-8")
    print(f"Wrote boolean flags -> {OUT_DIR / 'overlap_flags.csv'}")

if __name__ == "__main__":
    main()
