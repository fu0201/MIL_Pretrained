#!/usr/bin/env bash
set -euo pipefail

# ===== Fixed settings =====
PFM_NAMES=(${PFM_NAMES:-conch_v1_5})
SLIDE_NAMES=(${SLIDE_NAMES:-transmil})
DTYPE="${DTYPE:-fp32}"

# ===== Paths =====
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
BENCHMARK_PY="${PROJECT_DIR}/benchmark_pretrain.py"
JSON_DIR="${PROJECT_DIR}/downstream_task_jsons"
JOB_DIR="${JOB_DIR:-${PROJECT_DIR}/results_scratch}"

# ===== Data roots =====
DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/data}"

configure_slide_model() {
  local slide_name="$1"
  case "${slide_name}" in
    abmil)
      SLIDE_NAME="abmil"
      SLIDE_DIR_NAME="abmil"
      DIM_HIDDEN=384
      ;;
    amdmil)
      SLIDE_NAME="amdmil"
      SLIDE_DIR_NAME="amdmil"
      DIM_HIDDEN=512
      ;;
    clam_mb)
      SLIDE_NAME="clam_mb"
      SLIDE_DIR_NAME="clam_mb"
      DIM_HIDDEN=512
      ;;
    clam_sb)
      SLIDE_NAME="clam_sb"
      SLIDE_DIR_NAME="clam_sb"
      DIM_HIDDEN=512
      ;;
    wikg)
      SLIDE_NAME="wikg"
      SLIDE_DIR_NAME="wikg"
      DIM_HIDDEN=512
      ;;
    aemmil)
      SLIDE_NAME="aemmil"
      SLIDE_DIR_NAME="aemmil"
      DIM_HIDDEN=512
      ;;
    2dmamba)
      SLIDE_NAME="2dmamba"
      SLIDE_DIR_NAME="2dmamba"
      DIM_HIDDEN=128
      ;;
    dagmil)
      SLIDE_NAME="dagmil"
      SLIDE_DIR_NAME="dagmil"
      DIM_HIDDEN=512
      ;;
    gdfmil)
      SLIDE_NAME="gdfmil"
      SLIDE_DIR_NAME="gdfmil"
      DIM_HIDDEN=512
      ;;
    transmil)
      SLIDE_NAME="transmil"
      SLIDE_DIR_NAME="transmil"
      DIM_HIDDEN=512
      ;;
    *)
      echo "Unsupported slide model: ${slide_name}" >&2
      exit 1
      ;;
  esac
}

run_one() {
  local data_dir="$1"
  local json_path="$2"
  local seed="$3"
  local dataset_name="${4:-}"

  local ds
  if [[ -n "${dataset_name}" ]]; then
    ds="${dataset_name}"
  else
    ds="$(basename "${json_path}" .json)"
  fi
  if [[ ! -f "${json_path}" ]]; then
    echo "[skip] missing task split file: ${json_path}"
    return 0
  fi
  local metrics_file="${JOB_DIR}/${ds}/${PFM_NAME}/${SLIDE_DIR_NAME}/${seed}/benchmark/all_test_metrics.json"
  if [[ -f "${metrics_file}" ]]; then
    echo "[skip] already done: ${metrics_file}"
    return 0
  fi

  local ds_name_arg=""
  if [[ -n "${dataset_name}" ]]; then
    ds_name_arg="--dataset_name ${dataset_name}"
  fi

  local max_patches_arg=""
  if [[ "${SLIDE_NAME}" == "dagmil" ]]; then
    max_patches_arg="--max_patches_per_sample 40000"
  fi

  python "${BENCHMARK_PY}" \
    --data_dir "${data_dir}" \
    --json_path "${json_path}" \
    --pfm_name "${PFM_NAME}" \
    --slide_name "${SLIDE_NAME}" \
    --job_dir "${JOB_DIR}" \
    --seed "${seed}" \
    --gpu_id "${GPU_ID:-0}" \
    --dtype "${DTYPE}" \
    --best_metrics "bal_accuracy" \
    --epochs 50 \
    --batch_size 1 \
    --num_workers 12 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --grad_accum_steps 1 \
    --early_stop_patience 5 \
    --dim_hidden "${DIM_HIDDEN}" \
    --slide_dir_name "${SLIDE_DIR_NAME}" \
    ${max_patches_arg} \
    ${ds_name_arg}
}

run_five_seeds() {
  local data_dir="$1"
  local json_path="$2"
  local dataset_name="${3:-}"

  run_one "${data_dir}" "${json_path}" 2077 "${dataset_name}"
  run_one "${data_dir}" "${json_path}" 2078 "${dataset_name}"
  run_one "${data_dir}" "${json_path}" 2079 "${dataset_name}"
  run_one "${data_dir}" "${json_path}" 2080 "${dataset_name}"
  run_one "${data_dir}" "${json_path}" 2081 "${dataset_name}"
}

# ===== Run each PFM in turn =====
for PFM_NAME in "${PFM_NAMES[@]}"; do
  for slide_model in "${SLIDE_NAMES[@]}"; do
    configure_slide_model "${slide_model}"
    echo "===== PFM: ${PFM_NAME} | slide: ${SLIDE_NAME} ====="
    run_five_seeds "${DATA_ROOT}/bcnb" "${JSON_DIR}/bcnb_er.json"
  done
done
