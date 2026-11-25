"""Bước 1a: tạo manifest gdc-client cho slide chẩn đoán (FFPE, DX) của một cohort TCGA.

Ví dụ:
    python scripts/01_make_manifest.py --cohort TCGA-LUAD
    python scripts/01_make_manifest.py --cohort TCGA-LUAD --limit 5   # tải thử vài slide
"""
import argparse
import json
import os

import requests

GDC_FILES = "https://api.gdc.cancer.gov/files"


def query_slides(cohort):
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": [cohort]}},
            {"op": "in", "content": {"field": "data_type", "value": ["Slide Image"]}},
            {"op": "in", "content": {"field": "experimental_strategy", "value": ["Diagnostic Slide"]}},
            {"op": "in", "content": {"field": "data_format", "value": ["SVS"]}},
        ],
    }
    params = {
        "filters": json.dumps(filters),
        "fields": "file_id,file_name,md5sum,file_size,state",
        "format": "JSON",
        "size": "100000",
    }
    r = requests.get(GDC_FILES, params=params, timeout=120)
    r.raise_for_status()
    return r.json()["data"]["hits"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, help="vd. TCGA-LUAD")
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--limit", type=int, default=None, help="chỉ lấy N slide đầu (để test)")
    args = ap.parse_args()

    hits = sorted(query_slides(args.cohort), key=lambda h: h["file_name"])
    if not hits:
        raise SystemExit(f"Không tìm thấy slide nào cho {args.cohort}")
    if args.limit:
        hits = hits[: args.limit]

    out_dir = os.path.join(args.data_root, args.cohort)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "manifest.txt")
    with open(out, "w") as f:
        f.write("id\tfilename\tmd5\tsize\tstate\n")
        for h in hits:
            f.write(f"{h['file_id']}\t{h['file_name']}\t{h['md5sum']}\t{h['file_size']}\t{h['state']}\n")

    total_gb = sum(h["file_size"] for h in hits) / 1e9
    n_cases = len({h["file_name"][:12] for h in hits})
    print(f"{args.cohort}: {len(hits)} slide, {n_cases} bệnh nhân, {total_gb:.1f} GB -> {out}")


if __name__ == "__main__":
    main()
