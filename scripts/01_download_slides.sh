#!/usr/bin/env bash
# Bước 1b: tải slide bằng gdc-client rồi gom .svs ra thư mục phẳng (CLAM cần).
# Cần cài gdc-client: https://gdc.cancer.gov/access-data/gdc-data-transfer-tool
# Dùng: bash scripts/01_download_slides.sh TCGA-LUAD [n_processes]
set -euo pipefail

COHORT=${1:?"Cần tên cohort, vd. TCGA-LUAD"}
NPROC=${2:-8}
ROOT=data/${COHORT}
RAW=${ROOT}/raw_download
SLIDES=${ROOT}/slides

mkdir -p "${RAW}" "${SLIDES}"
gdc-client download -m "${ROOT}/manifest.txt" -d "${RAW}" -n "${NPROC}"

# raw_download/<uuid>/<file>.svs -> slides/<file>.svs (symlink, không tốn thêm dung lượng)
find "$(realpath "${RAW}")" -name "*.svs" | while read -r f; do
    ln -sf "$f" "${SLIDES}/$(basename "$f")"
done
echo "Có $(ls "${SLIDES}" | wc -l) slide trong ${SLIDES}"
