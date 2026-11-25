#!/usr/bin/env bash
# Bước 3: tách mô khỏi nền + cắt patch 256x256 @ 20x bằng CLAM create_patches_fp.py
# Dùng: bash scripts/03_patch.sh TCGA-LUAD
set -euo pipefail

COHORT=${1:?"Cần tên cohort, vd. TCGA-LUAD"}
PROJ=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-${PROJ}/.venv/bin/python}
SRC=${PROJ}/data/${COHORT}/slides
OUT=${PROJ}/data/${COHORT}/patches

"${PY}" "${PROJ}/scripts/03_split_by_mag.py" --source "${SRC}" --save_dir "${OUT}"

cd "${PROJ}/third_party/CLAM"   # CLAM đọc presets/ theo đường dẫn tương đối
for MAG in 20 40; do
    LIST=process_list_${MAG}x.csv
    [ "$(wc -l < "${OUT}/${LIST}")" -le 1 ] && continue   # không có slide nào ở mag này
    PS=$(( MAG == 40 ? 512 : 256 ))
    echo "== ${MAG}x: patch_size=${PS} =="
    "${PY}" create_patches_fp.py --source "${SRC}" --save_dir "${OUT}" \
        --patch_size "${PS}" --step_size "${PS}" --patch_level 0 \
        --preset tcga.csv --seg --patch --stitch --process_list "${LIST}"
    cp "${OUT}/process_list_autogen.csv" "${OUT}/process_list_autogen_${MAG}x.csv"
done
echo "Xong. Kiểm tra bằng mắt: ${OUT}/stitches/ và ${OUT}/masks/"
