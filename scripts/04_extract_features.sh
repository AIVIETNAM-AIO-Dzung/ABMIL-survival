#!/usr/bin/env bash
# Bước 4: trích feature UNI (1024-d) cho mỗi patch bằng CLAM extract_features_fp.py
#
# Chuẩn bị (một lần):
#   1. Xin quyền truy cập https://huggingface.co/MahmoodLab/UNI
#   2. huggingface-cli login
#   3. Script tự tải pytorch_model.bin về data/checkpoints/uni nếu UNI_CKPT_PATH chưa đặt.
#
# Dùng: bash scripts/04_extract_features.sh TCGA-LUAD [batch_size]
set -euo pipefail

COHORT=${1:?"Cần tên cohort, vd. TCGA-LUAD"}
BS=${2:-256}
PROJ=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-${PROJ}/.venv/bin/python}
ROOT=${PROJ}/data/${COHORT}

if [ -z "${UNI_CKPT_PATH:-}" ]; then
    export UNI_CKPT_PATH=$("${PY}" -c "
from huggingface_hub import hf_hub_download
print(hf_hub_download('MahmoodLab/UNI', 'pytorch_model.bin', local_dir='${PROJ}/data/checkpoints/uni'))")
fi
echo "UNI checkpoint: ${UNI_CKPT_PATH}"

cd "${PROJ}/third_party/CLAM"
"${PY}" extract_features_fp.py \
    --data_h5_dir "${ROOT}/patches" --data_slide_dir "${ROOT}/slides" \
    --csv_path "${ROOT}/patches/slide_list.csv" --slide_ext .svs \
    --feat_dir "${ROOT}/features" \
    --model_name uni_v1 --target_patch_size 224 --batch_size "${BS}"
echo "Feature: ${ROOT}/features/pt_files/<slide_id>.pt, shape (N, 1024)"
