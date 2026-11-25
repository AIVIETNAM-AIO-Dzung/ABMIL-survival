"""Bước 3a: chia slide thành hai process list theo magnification gốc, để CLAM cắt
cùng trường nhìn 256 px @ 20x:
  - slide 20x -> patch_size 256 ở level 0
  - slide 40x -> patch_size 512 ở level 0 (lúc trích feature resize về 224)

Ghi vào <save_dir>/: process_list_20x.csv, process_list_40x.csv, slide_list.csv (mọi slide hợp lệ),
                    skipped_mag.csv (slide không đọc được magnification).
"""
import argparse
import os

import openslide
import pandas as pd


def base_magnification(path):
    props = openslide.OpenSlide(path).properties
    mag = props.get("aperio.AppMag") or props.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
    if mag:
        return round(float(mag))
    mpp = props.get(openslide.PROPERTY_NAME_MPP_X)
    if mpp:
        mpp = float(mpp)
        if 0.2 <= mpp < 0.35:
            return 40
        if 0.4 <= mpp < 0.6:
            return 20
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="thư mục .svs")
    ap.add_argument("--save_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    rows = []
    for f in sorted(os.listdir(args.source)):
        if not f.endswith(".svs"):
            continue
        try:
            mag = base_magnification(os.path.join(args.source, f))
        except Exception as e:  # file hỏng
            print(f"Lỗi đọc {f}: {e}")
            mag = None
        rows.append({"slide_id": f, "mag": mag})
    df = pd.DataFrame(rows)

    ok = df["mag"].isin([20, 40])
    for m in (20, 40):
        df.loc[df["mag"] == m, ["slide_id"]].to_csv(
            os.path.join(args.save_dir, f"process_list_{m}x.csv"), index=False)
    df.loc[ok, ["slide_id"]].to_csv(os.path.join(args.save_dir, "slide_list.csv"), index=False)
    df.loc[~ok].to_csv(os.path.join(args.save_dir, "skipped_mag.csv"), index=False)
    print(df["mag"].value_counts(dropna=False).to_string())
    if (~ok).any():
        print(f"Bỏ qua {(~ok).sum()} slide (magnification không phải 20x/40x) -> skipped_mag.csv")


if __name__ == "__main__":
    main()
