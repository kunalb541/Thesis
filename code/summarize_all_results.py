#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
summarize_all_results.py — Aggregate evaluation summaries into a single table.
- Scans a directory tree for evaluation_summary.json files
- Outputs combined CSV and JSON
Usage:
  python summarize_all_results.py --root results --out results/summary_all
"""
from __future__ import annotations

import argparse, os, json
from pathlib import Path
from typing import List, Dict

def find_summaries(root: str) -> List[str]:
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f == "evaluation_summary.json":
                paths.append(os.path.join(dirpath, f))
    return sorted(paths)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root directory to scan for evaluation_summary.json")
    ap.add_argument("--out", required=True, help="Output directory for combined CSV/JSON")
    args = ap.parse_args()

    files = find_summaries(args.root)
    if not files:
        print("No evaluation_summary.json files found.")
        return 0

    rows: List[Dict] = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                obj = json.load(f)
            m = obj.get("metrics", {})
            ed = obj.get("early_detection", {})
            rows.append({
                "path": fp,
                "model": obj.get("model", ""),
                "data": obj.get("data", ""),
                "accuracy": m.get("accuracy", None),
                "roc_auc": m.get("roc_auc", None),
                "pr_auc": m.get("pr_auc", None),
                "average_precision": m.get("average_precision", None),
                "tn": m.get("tn", None),
                "fp": m.get("fp", None),
                "fn": m.get("fn", None),
                "tp": m.get("tp", None),
                "early_detection": ed,
            })
        except Exception as e:
            print(f"Failed to parse {fp}: {e}")

    os.makedirs(args.out, exist_ok=True)

    # JSON dump
    json_out = os.path.join(args.out, "all_results.json")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"Wrote {json_out}")

    # CSV dump (flat columns; early_detection as compact string)
    import csv
    csv_out = os.path.join(args.out, "all_results.csv")
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path","model","data","accuracy","roc_auc","pr_auc","average_precision","tn","fp","fn","tp","early_detection"])
        for r in rows:
            ed = r.get("early_detection", {})
            ed_str = "; ".join([f"{k}:{v:.4f}" for k,v in ed.items()]) if isinstance(ed, dict) else str(ed)
            w.writerow([r.get("path",""), r.get("model",""), r.get("data",""),
                        r.get("accuracy",""), r.get("roc_auc",""), r.get("pr_auc",""),
                        r.get("average_precision",""), r.get("tn",""), r.get("fp",""),
                        r.get("fn",""), r.get("tp",""), ed_str])
    print(f"Wrote {csv_out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
