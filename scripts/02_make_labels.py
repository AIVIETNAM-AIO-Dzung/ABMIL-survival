"""Bước 2: lấy nhãn sống còn (OS) từ TCGA-CDR và ghép với slide đã tải.

TCGA-CDR (Liu et al., Cell 2018): file TCGA-CDR-SupplementalTableS1.xlsx, tải ở
https://gdc.cancer.gov/about-data/publications/pancanatlas

Ví dụ:
    python scripts/02_make_labels.py --cohort TCGA-LUAD --cdr TCGA-CDR-SupplementalTableS1.xlsx
Output: data/<cohort>/labels/os.csv với cột case_id, slide_ids, time, event, site
"""
import argparse
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--cdr", required=True, help="đường dẫn TCGA-CDR-SupplementalTableS1.xlsx")
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--endpoint", default="OS", choices=["OS", "DSS", "DFI", "PFI"])
    args = ap.parse_args()

    root = os.path.join(args.data_root, args.cohort)
    cancer = args.cohort.replace("TCGA-", "")
    ev_col, t_col = args.endpoint, f"{args.endpoint}.time"

    cdr = pd.read_excel(args.cdr, sheet_name="TCGA-CDR")
    cdr = cdr[cdr["type"] == cancer]
    if "Redaction" in cdr.columns:  # bệnh nhân bị TCGA thu hồi dữ liệu
        cdr = cdr[cdr["Redaction"].isna()]
    cdr = cdr[["bcr_patient_barcode", ev_col, t_col]]
    cdr = cdr.rename(columns={"bcr_patient_barcode": "case_id", ev_col: "event", t_col: "time"})
    cdr["event"] = pd.to_numeric(cdr["event"], errors="coerce")
    cdr["time"] = pd.to_numeric(cdr["time"], errors="coerce")
    n0 = len(cdr)
    cdr = cdr.dropna(subset=["event", "time"])
    cdr = cdr[cdr["time"] > 0]
    print(f"CDR {cancer}: {n0} bệnh nhân, còn {len(cdr)} sau khi bỏ thiếu nhãn / time<=0")

    slide_dir = os.path.join(root, "slides")
    slides = sorted(f[:-4] for f in os.listdir(slide_dir) if f.endswith(".svs"))
    sdf = pd.DataFrame({"slide_id": slides})
    sdf["case_id"] = sdf["slide_id"].str[:12]
    grouped = sdf.groupby("case_id")["slide_id"].apply(lambda s: ";".join(s)).rename("slide_ids")

    df = cdr.merge(grouped, left_on="case_id", right_index=True, how="inner")
    df["event"] = df["event"].astype(int)
    df["site"] = df["case_id"].str.split("-").str[1]  # tissue source site
    df = df[["case_id", "slide_ids", "time", "event", "site"]].sort_values("case_id")

    out_dir = os.path.join(root, "labels")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{args.endpoint.lower()}.csv")
    df.to_csv(out, index=False)
    print(f"{len(df)} bệnh nhân có slide + nhãn ({df['event'].sum()} biến cố, "
          f"{df['site'].nunique()} site) -> {out}")
    missing = set(sdf["case_id"]) - set(df["case_id"])
    if missing:
        print(f"Bỏ {len(missing)} bệnh nhân có slide nhưng không có nhãn")


if __name__ == "__main__":
    main()
