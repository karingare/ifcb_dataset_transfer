#!/usr/bin/env python3
"""
IFCB plankton label flow — end‑to‑end, reproducible.

Subcommands:
  scan        -> read dataset roots (non‑recursive), build class summaries + per‑image manifests
  map         -> build class_mapping.csv via WoRMS + normalization + manual overrides
  apply-map   -> apply mapping to summaries/manifests to produce standardized outputs
  vocab       -> build training vocabulary from standardized summaries of training datasets
  filter      -> filter ALL datasets to training vocab (closed‑set); emit pivots & reports

Typical usage (with your datasets):
  1) Create roots.json (dataset name -> root folder with class subdirs):
     {
       "smhi_ifcb_svea_baltic_proper": "/cfs/klemming/projects/supr/snic2020-6-126/projects/amime/manually_classified_ifcb_sets/SMHI_IFCB_Plankton_Image_Reference_Library_v4/smhi_ifcb_baltic_annotated_images",
       "smhi_ifcb_svea_skagerrak_kattegat": "/cfs/klemming/projects/supr/snic2020-6-126/projects/amime/manually_classified_ifcb_sets/SMHI_IFCB_Plankton_Image_Reference_Library_v4/smhi_ifcb_skagerrak-kattegat_annotated_images",
       "smhi_ifcb_tångesund": "/cfs/klemming/projects/supr/snic2020-6-126/projects/amime/manually_classified_ifcb_sets/SMHI_IFCB_Plankton_Image_Reference_Library_v4/smhi_ifcb_tangesund_annotated_images",
       "SYKE_plankton_IFCB_Utö_2021": "/cfs/klemming/projects/supr/snic2020-6-126/projects/amime/manually_classified_ifcb_sets/SYKE_2021"
     }

  2) Scan (non‑recursive):
     python ifcb_flow.py scan --roots roots.json --out out_scan --emit-manifests

  3) Build mapping with WoRMS (review unresolved later):
     python ifcb_flow.py map --classes out_scan/all_unique_classes.csv \
       --out class_mapping.csv --cache worms_cache.json

  4) Apply mapping to summaries + manifests:
     python ifcb_flow.py apply-map --scan-dir out_scan --mapping class_mapping.csv --out out_std

  5) Build training vocabulary from standardized training sets:
     python ifcb_flow.py vocab --std-dir out_std \
       --training-datasets "SYKE_plankton_IFCB_Utö_2021" "smhi_ifcb_tångesund" \
       --min-per-train 1 --out vocab_out

  6) Filter all datasets to training vocab (closed‑set evaluation):
     python ifcb_flow.py filter --std-dir out_std --vocab vocab_out/training_vocab.txt --out out_filtered

Requires: pandas, requests
"""

from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import re
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from urllib.parse import quote

# ---------------------------- Utils -----------------------------------------

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
DEF_ENCODING = "utf-8"


def read_json(path: str | Path) -> dict:
    with open(path, "r", encoding=DEF_ENCODING) as f:
        return json.load(f)


def write_df(df: pd.DataFrame, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# ---------------------------- scan ------------------------------------------

def scan_one_dataset(name: str, root: str, emit_manifest: bool) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Root not found: {root}")

    class_dirs = [p for p in root_path.iterdir() if p.is_dir()]
    rows = []
    manifest_rows = []

    for cdir in class_dirs:
        cls = cdir.name
        # NON-RECURSIVE: only files directly inside the class dir
        files = [p for p in cdir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
        n = len(files)
        rows.append({"dataset": name, "class_name": cls, "n_images": n})
        if emit_manifest and n > 0:
            for p in files:
                manifest_rows.append({
                    "dataset": name,
                    "class_name": cls,
                    "filepath": str(p)
                })

    summary = pd.DataFrame(rows).sort_values(["dataset", "class_name"]).reset_index(drop=True)
    manifest = pd.DataFrame(manifest_rows) if emit_manifest else None
    return summary, manifest


def cmd_scan(args):
    roots = read_json(args.roots)
    all_summaries = []
    all_manifests = []

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    for ds, path in roots.items():
        print(f"Scanning {ds} -> {path}")
        summ, mani = scan_one_dataset(ds, path, emit_manifest=args.emit_manifests)
        write_df(summ, outdir / f"{ds}_summary.csv")
        all_summaries.append(summ)
        if mani is not None:
            write_df(mani, outdir / f"{ds}_manifest.csv")
            all_manifests.append(mani)

    combined = pd.concat(all_summaries, ignore_index=True)
    write_df(combined, outdir / "combined_summary.csv")

    unique_classes = (
        combined[["class_name"]].drop_duplicates().sort_values("class_name")
    )
    write_df(unique_classes, outdir / "all_unique_classes.csv")

    if all_manifests:
        combined_mani = pd.concat(all_manifests, ignore_index=True)
        write_df(combined_mani, outdir / "combined_manifest.csv")

    print(f"✅ wrote scan outputs to {outdir}")


# ---------------------------- mapping (WoRMS) -------------------------------

WORMS_BASE = "https://www.marinespecies.org/rest"
HEADERS = {"User-Agent": "ifcb-mapper/1.0 (contact: you@example.com)"}
SLEEP_SEC = 0.3
TIMEOUT = 30

# Manual overrides (extend as needed)
MANUAL_OVERRIDES: Dict[str, Dict[str, Optional[str]]] = {
    # buckets / morphology -> keep without ID
    "Unclassifiable": {"standardized_class": "Unclassifiable", "aphia_id": None, "notes": "Non-taxonomic label"},
    "Pennales_sp_thick": {"standardized_class": "Pennales (morphological group)", "aphia_id": None, "notes": "morphology bucket"},
    "Pennales_sp_thin": {"standardized_class": "Pennales (morphological group)", "aphia_id": None, "notes": "morphology bucket"},
    "Centrales": {"standardized_class": "Centrales (morphological group)", "aphia_id": None, "notes": "morphology bucket"},
    "Centrales_sp": {"standardized_class": "Centrales (morphological group)", "aphia_id": None, "notes": "morphology bucket"},

    # common harmonizations
    "Ciliata": {"standardized_class": "Ciliophora", "notes": "synonym in usage"},
    "Cryptophyceae-Teleaulax": {"standardized_class": "Teleaulax", "notes": "mapped to genus"},
    "Chaetoceros_sp": {"standardized_class": "Chaetoceros", "notes": "genus-level"},
    "Chaetoceros_sp_single": {"standardized_class": "Chaetoceros", "notes": "genus-level"},
    "Scrippsiella_group": {"standardized_class": "Scrippsiella", "notes": "group → genus"},
    "Gymnodinium_like": {"standardized_class": "Gymnodinium", "notes": "'-like' stripped"},
    "Dinobryon_spp": {"standardized_class": "Dinobryon", "notes": "genus-level"},
    "Eutreptiella_spp": {"standardized_class": "Eutreptiella", "notes": "genus-level"},
    "Pyramimonas_sp": {"standardized_class": "Pyramimonas", "notes": "genus-level"},
    "Actinocyclus_spp": {"standardized_class": "Actinocyclus", "notes": "genus-level"},
    "Aphanizomenon_spp_bundle": {"standardized_class": "Aphanizomenon", "notes": "bundle removed"},
    "Aphanizomenon_spp_filament": {"standardized_class": "Aphanizomenon", "notes": "filament removed"},
    "Peridiniella_catenata_chain": {"standardized_class": "Peridiniella catenata", "notes": "chain removed"},
    "Peridiniella_catenata_single": {"standardized_class": "Peridiniella catenata", "notes": "single removed"},
    "Diatoma_tenuis-like_chain": {"standardized_class": "Diatoma", "notes": "'-like' stripped → genus"},
    "Diatoma_tenuis-like_single_cell": {"standardized_class": "Diatoma", "notes": "'-like' stripped → genus"},
}

UNSCORED_SUFFIXES = [
    "_sp_single", "_sp", "_spp", "_group", "_pair", "_bundle", "_filament",
    "_chain", "_single_cell", "_smaller_than_30", "_larger_than_30"
]
REPLACEMENTS = [
    (re.compile(r"_"), " "),
    (re.compile(r"(?i)-like"), ""),
    (re.compile(r"(?i)flosaquae"), "flos-aquae"),
]
NON_TAXON_BUCKETS = {"Unclassifiable", "Pennales (morphological group)", "Centrales (morphological group)"}


def normalize_label(raw: str) -> tuple[str, list[str]]:
    if raw in MANUAL_OVERRIDES and MANUAL_OVERRIDES[raw].get("standardized_class"):
        return MANUAL_OVERRIDES[raw]["standardized_class"], ["manual_override"]

    notes = []
    s = raw
    for suf in UNSCORED_SUFFIXES:
        if s.endswith(suf):
            s = s[: -len(suf)]
            notes.append(f"removed '{suf}'")
    for pat, repl in REPLACEMENTS:
        s2 = pat.sub(repl, s)
        if s2 != s:
            s = s2
    s = re.sub(r"\s+", " ", s).strip()

    # remove trailing qualifiers like -coiled, -bundle, -filament, -chain, -single_cell, -pair
    QUALIFIER_RE = re.compile(
        r"[-_](?:coiled?|bundle|filament|chain|single(?:_cell)?|pair|smaller_than_\\d+|larger_than_\\d+)$",
        flags=re.IGNORECASE,
    )
    while True:
        s2 = QUALIFIER_RE.sub("", s)
        if s2 == s:
            break
        s = s2.strip()
    # genus-level if contains sp/spp
    if re.search(r"\b(spp|sp)\b", s, flags=re.IGNORECASE):
        genus = s.split()[0]
        notes.append("genus-level (sp/spp)")
        return genus, notes

    return s, notes


def worms_get(url: str) -> Optional[dict]:
    r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    if r.status_code == 204:
        return None
    r.raise_for_status()
    return r.json()


def worms_by_name(name: str, like=False, fuzzy=False) -> list[dict]:
    url = (
        f"{WORMS_BASE}/AphiaRecordsByName/{quote(name)}?like={'true' if like else 'false'}"
        f"&fuzzy={'true' if fuzzy else 'false'}&marine_only=false&extant_only=true"
    )
    data = worms_get(url)
    return data or []


def worms_by_id(aphia_id: int) -> Optional[dict]:
    return worms_get(f"{WORMS_BASE}/AphiaRecordByAphiaID/{aphia_id}")


def choose_best(cands: list[dict]) -> Optional[dict]:
    if not cands:
        return None
    accepted = [c for c in cands if (c.get("status") or "").lower() == "accepted"]
    pool = accepted or cands
    pool = sorted(pool, key=lambda c: (c.get("valid_AphiaID") is None, c.get("AphiaID")))
    return pool[0]


def resolve_record(source_class: str) -> tuple[str, Optional[int], str]:
    # -------- 1) Manual override wins (optionally fill missing AphiaID) --------
    if source_class in MANUAL_OVERRIDES:
        ov = MANUAL_OVERRIDES[source_class]
        std = ov.get("standardized_class", source_class)
        aid = ov.get("aphia_id")
        notes = [ov.get("notes", "manual_override")]
        if aid is None and std:
            # Try to look up AphiaID for the chosen standardized name,
            # but DO NOT change the chosen name.
            recs = worms_by_name(std, like=False, fuzzy=False)
            best = choose_best(recs) if recs else None
            if best and (best.get("status") or "").lower() != "accepted" and best.get("valid_AphiaID"):
                best = worms_by_id(best["valid_AphiaID"]) or best
            if best:
                aid = best.get("AphiaID")
                notes.append("aphia_from_std")
        return std, aid, "; ".join(notes)

    # -------- 2) Normalize raw label (remove suffixes, fix spellings, etc.) ----
    norm, notes = normalize_label(source_class)
    if norm in NON_TAXON_BUCKETS:
        return norm, None, "; ".join(["non-taxonomic bucket"] + notes)

    # -------- 3) WoRMS lookup: exact → like → fuzzy ---------------------------
    for like, fuzzy in [(False, False), (True, False), (True, True)]:
        recs = worms_by_name(norm, like=like, fuzzy=fuzzy)
        if recs:
            best = choose_best(recs)
            if best and (best.get("status") or "").lower() != "accepted" and best.get("valid_AphiaID"):
                best = worms_by_id(best["valid_AphiaID"]) or best
            sci = best.get("scientificname") or norm
            aid = best.get("AphiaID")
            info = [
                f"status={best.get('status')}",
                f"rank={best.get('rank')}",
                "match=like/fuzzy" if (like or fuzzy) else "match=exact"
            ]
            return sci, aid, "; ".join(info + notes)
        time.sleep(SLEEP_SEC)

    # -------- 4) Fallback if nothing found ------------------------------------
    return norm, None, "; ".join(["NO_MATCH_WORMS"] + notes)



def cmd_map(args):
    # Read classes and sanity-check
    classes_df = pd.read_csv(args.classes, encoding=DEF_ENCODING)
    if "class_name" not in classes_df.columns:
        raise SystemExit("--classes CSV must have a 'class_name' column")
    classes_df = classes_df.copy()
    classes_df["class_name"] = classes_df["class_name"].astype(str).str.strip()

    # ---- Load manual overrides (CSV) and merge into MANUAL_OVERRIDES ----
    ov = {}
    if args.overrides:
        o = pd.read_csv(args.overrides, encoding=DEF_ENCODING)
        need = {"source_class", "standardized_class", "WoRMS_ID", "notes"}
        if not need.issubset(o.columns):
            raise SystemExit(f"--overrides must have columns {need}")

        for _, r in o.iterrows():
            src = str(r["source_class"]).strip()
            std = "" if pd.isna(r["standardized_class"]) else str(r["standardized_class"]).strip()

            # robust AphiaID parsing: allow blank/NaN/"578476"/"578476.0"
            aid_raw = r.get("WoRMS_ID")
            aid = None
            if aid_raw is not None and not (isinstance(aid_raw, float) and pd.isna(aid_raw)):
                s = str(aid_raw).strip()
                if s:
                    try:
                        aid = int(float(s))
                    except ValueError:
                        aid = None

            notes = str(r.get("notes", "manual_override")).strip()
            ov[src] = {"standardized_class": (std or src), "aphia_id": aid, "notes": notes}

    # file overrides win over the in-code defaults
    MANUAL_OVERRIDES.update(ov)

    # ---- Map every unique class name ----
    unique = sorted(set(classes_df["class_name"]))

    # warm cache
    cache: Dict[str, Tuple[str, Optional[int], str]] = {}
    if args.cache and Path(args.cache).exists():
        cache = json.loads(Path(args.cache).read_text(encoding=DEF_ENCODING))

    rows = []
    for src in unique:
        # 1) OVERRIDE FIRST (wins over cache & WoRMS)
        if src in MANUAL_OVERRIDES:
            ov = MANUAL_OVERRIDES[src]
            std = ov.get("standardized_class", src)
            aid = ov.get("aphia_id")
            notes = ov.get("notes", "manual_override")
        # 2) CACHE NEXT
        elif src in cache:
            std, aid, notes = cache[src]
        # 3) OTHERWISE USE WORMS
        else:
            std, aid, notes = resolve_record(src)
            if args.cache:
                cache[src] = (std, aid, notes)

        rows.append({
            "source_class": src,
            "standardized_class": std,
            "WoRMS_ID": aid,
            "notes": notes
        })
        time.sleep(SLEEP_SEC)

    out_df = pd.DataFrame(rows).sort_values("source_class")
    write_df(out_df, args.out)

    if args.cache:
        Path(args.cache).write_text(
            json.dumps(cache, indent=2, ensure_ascii=False),
            encoding=DEF_ENCODING
        )

    # unresolved → review file
    rev = out_df[out_df["WoRMS_ID"].isna()].copy()
    if not rev.empty:
        write_df(rev, Path(args.out).with_name(Path(args.out).stem + "_REVIEW.csv"))

    print(f"✅ wrote mapping: {args.out} (overrides={len(ov)}, unresolved={len(rev)})")



# ---------------------------- apply-map -------------------------------------

def apply_mapping_to_summary(df: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    m = mapping[["source_class", "standardized_class", "WoRMS_ID", "notes"]].copy()
    m.rename(columns={"source_class": "class_name"}, inplace=True)
    out = df.merge(m, on="class_name", how="left")
    # if no mapping, fall back to original
    out["standardized_class"].fillna(out["class_name"], inplace=True)
    return out


def apply_mapping_to_manifest(df: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    m = mapping[["source_class", "standardized_class", "WoRMS_ID", "notes"]].copy()
    m.rename(columns={"source_class": "class_name"}, inplace=True)
    out = df.merge(m, on="class_name", how="left")
    out["standardized_class"].fillna(out["class_name"], inplace=True)
    return out


def cmd_apply_map(args):
    scan_dir = Path(args.scan_dir)
    mapping = pd.read_csv(args.mapping, encoding=DEF_ENCODING)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # apply to per-dataset summaries
    for csv_path in scan_dir.glob("*_summary.csv"):
        df = pd.read_csv(csv_path, encoding=DEF_ENCODING)
        std = apply_mapping_to_summary(df, mapping)
        write_df(std, outdir / (csv_path.stem + "_std.csv"))

    # combined summary
    comb = pd.read_csv(scan_dir / "combined_summary.csv", encoding=DEF_ENCODING)
    comb_std = apply_mapping_to_summary(comb, mapping)
    write_df(comb_std, outdir / "combined_summary_std.csv")

    # manifests if present
    for csv_path in scan_dir.glob("*_manifest.csv"):
        df = pd.read_csv(csv_path, encoding=DEF_ENCODING)
        std = apply_mapping_to_manifest(df, mapping)
        write_df(std, outdir / (csv_path.stem + "_std.csv"))

    if (scan_dir / "combined_manifest.csv").exists():
        mani = pd.read_csv(scan_dir / "combined_manifest.csv", encoding=DEF_ENCODING)
        mani_std = apply_mapping_to_manifest(mani, mapping)
        write_df(mani_std, outdir / "combined_manifest_std.csv")

    print(f"✅ wrote standardized outputs to {outdir}")


# ---------------------------- vocab -----------------------------------------

def cmd_vocab(args):
    std_dir = Path(args.std_dir)
    comb_std = pd.read_csv(std_dir / "combined_summary_std.csv", encoding=DEF_ENCODING)

    train_names = set(args.training_datasets)
    train_df = comb_std[comb_std["dataset"].isin(train_names)].copy()

    # NEW: exclusions
    exclude = set(args.exclude_classes or [])
    if getattr(args, "exclude_non_taxonomy", False):
        exclude |= NON_TAXON_BUCKETS
    if exclude:
        train_df = train_df[~train_df["standardized_class"].isin(exclude)].copy()

    keep = (
        train_df[train_df["n_images"] >= args.min_per_train]
        .groupby("standardized_class")["dataset"].nunique()
        .index.tolist()
    )
    keep = sorted(set(keep))

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # vocab + label index
    (outdir / "training_vocab.txt").write_text("\n".join(keep) + "\n", encoding=DEF_ENCODING)
    label_index = {cls: i for i, cls in enumerate(keep)}
    (outdir / "label_index.json").write_text(json.dumps(label_index, indent=2, ensure_ascii=False), encoding=DEF_ENCODING)

    print(f"✅ wrote training vocab ({len(keep)} classes) to {outdir}")


# ---------------------------- filter ----------------------------------------

def cmd_filter(args):
    std_dir = Path(args.std_dir)
    vocab = [l.strip() for l in Path(args.vocab).read_text(encoding=DEF_ENCODING).splitlines() if l.strip()]
    vocab_set = set(vocab)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # Filter standardized summaries for each dataset
    kept_all = []
    removed_all = []

    for csv_path in std_dir.glob("*_summary_std.csv"):
        df = pd.read_csv(csv_path, encoding=DEF_ENCODING)
        ds = df["dataset"].iloc[0] if not df.empty else csv_path.stem
        in_vocab = df["standardized_class"].isin(vocab_set)
        kept = df[in_vocab].copy()
        removed = df[~in_vocab].copy()
        kept_all.append(kept)
        if not removed.empty:
            removed["remove_reason"] = "not_in_training_vocab"
            removed_all.append(removed)
        write_df(kept, outdir / f"{ds}_summary_filtered.csv")

    if kept_all:
        comb_kept = pd.concat(kept_all, ignore_index=True)
        write_df(comb_kept, outdir / "combined_summary_filtered.csv")
        # pivot
        pivot = comb_kept.pivot_table(index="standardized_class", columns="dataset", values="n_images", aggfunc="sum", fill_value=0)
        write_df(pivot.reset_index(), outdir / "pivot_classes_x_datasets.csv")

    if removed_all:
        removed = pd.concat(removed_all, ignore_index=True)
        write_df(removed, outdir / "classes_removed_report.csv")

    # Filter manifests if present
    if (std_dir / "combined_manifest_std.csv").exists():
        mani = pd.read_csv(std_dir / "combined_manifest_std.csv", encoding=DEF_ENCODING)
        kept_mani = mani[mani["standardized_class"].isin(vocab_set)].copy()
        write_df(kept_mani, outdir / "combined_manifest_filtered.csv")

    print(f"✅ wrote filtered outputs to {outdir}")


# ---------------------------- CLI ------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description="IFCB plankton label flow")
    sub = p.add_subparsers(dest="cmd", required=True)

    # scan
    s = sub.add_parser("scan", help="Scan dataset roots (non-recursive)")
    s.add_argument("--roots", required=True, help="JSON file: {dataset: root_path}")
    s.add_argument("--out", required=True, help="Output directory for scan CSVs")
    s.add_argument("--emit-manifests", action="store_true", help="Also write per-image manifests")
    s.set_defaults(func=cmd_scan)

    # map
    m = sub.add_parser("map", help="Build class_mapping.csv via WoRMS")
    m.add_argument("--classes", required=True, help="CSV with column 'class_name' (e.g., out_scan/all_unique_classes.csv)")
    m.add_argument("--out", required=True, help="Output mapping CSV path")
    m.add_argument("--cache", default=None, help="Optional JSON cache for WoRMS lookups")
    m.add_argument("--overrides", default=None,
               help="CSV with columns source_class,standardized_class,WoRMS_ID,notes")

    m.set_defaults(func=cmd_map)

    # apply-map
    a = sub.add_parser("apply-map", help="Apply mapping to scan summaries/manifests")
    a.add_argument("--scan-dir", required=True, help="Directory produced by 'scan'")
    a.add_argument("--mapping", required=True, help="class_mapping.csv")
    a.add_argument("--out", required=True, help="Output standardized directory")
    a.set_defaults(func=cmd_apply_map)

    # vocab
    v = sub.add_parser("vocab", help="Build training vocab from standardized summaries of training datasets")
    v.add_argument("--std-dir", required=True, help="Directory produced by 'apply-map'")
    v.add_argument("--training-datasets", nargs="+", required=True, help="Names of training datasets (match 'dataset' column)")
    v.add_argument("--min-per-train", type=int, default=1, help="Min images in a training dataset to keep a class (default 1)")
    v.add_argument("--out", required=True, help="Output directory for vocab + label index")
    v.add_argument("--exclude-classes", nargs="*", default=[],
                help="Standardized class names to exclude from the training vocab")
    v.add_argument("--exclude-non-taxonomy", action="store_true",
                help="Also exclude NON_TAXON_BUCKETS (e.g., Unclassifiable, Pennales/Centrales groups)")

    v.set_defaults(func=cmd_vocab)

    # filter
    f = sub.add_parser("filter", help="Filter ALL datasets to the training vocab (closed-set)")
    f.add_argument("--std-dir", required=True, help="Directory produced by 'apply-map'")
    f.add_argument("--vocab", required=True, help="training_vocab.txt from 'vocab'")
    f.add_argument("--out", required=True, help="Output directory for filtered results")
    f.set_defaults(func=cmd_filter)
    

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
