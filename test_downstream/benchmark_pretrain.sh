#!/usr/bin/env bash
set -euo pipefail

# ===== Fixed settings =====
PFM_NAMES=(${PFM_NAMES:-conch_v1_5})
SLIDE_NAMES=(${SLIDE_NAMES:-dagmil})
DTYPE="${DTYPE:-fp32}"

# ===== Paths =====
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
BENCHMARK_PY="${PROJECT_DIR}/benchmark_pretrain.py"
JSON_DIR="${PROJECT_DIR}/downstream_task_jsons"
JOB_DIR="${JOB_DIR:-${PROJECT_DIR}/results_distill}"

# ===== Data roots =====
DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/data}"
REPO_ROOT="${REPO_ROOT:-$(cd "${PROJECT_DIR}/.." && pwd)}"
PRETRAINED_WEIGHTS_DIR="${PRETRAINED_WEIGHTS_DIR:-${REPO_ROOT}/pretrained_weight}"

resolve_pretrained_weights() {
  local slide_name="$1"
  local env_var="PRETRAINED_WEIGHTS_${slide_name^^}"
  env_var="${env_var//[^A-Z0-9_]/_}"

  if [[ -n "${!env_var:-}" ]]; then
    printf '%s\n' "${!env_var}"
    return 0
  fi

  if [[ -n "${PRETRAINED_WEIGHTS_DIR}" ]]; then
    local candidates=(
      "${PRETRAINED_WEIGHTS_DIR}/pretrained_${slide_name}.pt"
      "${PRETRAINED_WEIGHTS_DIR}/pretrained_${slide_name}"
      "${PRETRAINED_WEIGHTS_DIR}/${slide_name}.pt"
      "${PRETRAINED_WEIGHTS_DIR}/${slide_name}"
    )
    for candidate in "${candidates[@]}"; do
      if [[ -f "${candidate}" ]]; then
        printf '%s\n' "${candidate}"
        return 0
      fi
    done
    printf '%s\n' "${candidates[0]}"
    return 0
  fi

  printf '%s\n' ""
}

configure_slide_model() {
  local slide_name="$1"
  local slide_dir_prefix=""
  case "${slide_name}" in
    abmil)
      SLIDE_NAME="abmil"
      slide_dir_prefix="abmil_distillinit"
      DIM_HIDDEN=384
      ;;
    transmil)
      SLIDE_NAME="transmil"
      slide_dir_prefix="transmil_distillinit"
      DIM_HIDDEN=512
      ;;
    aemmil)
      SLIDE_NAME="aemmil"
      slide_dir_prefix="aemmil_distillinit"
      DIM_HIDDEN=512
      ;;
    2dmamba)
      SLIDE_NAME="2dmamba"
      slide_dir_prefix="2dmamba_distillinit"
      DIM_HIDDEN=128
      ;;
    amdmil)
      SLIDE_NAME="amdmil"
      slide_dir_prefix="amdmil_distillinit"
      DIM_HIDDEN=512
      ;;
    clam_mb)
      SLIDE_NAME="clam_mb"
      slide_dir_prefix="clam_mb_distillinit"
      DIM_HIDDEN=512
      ;;
    wikg)
      SLIDE_NAME="wikg"
      slide_dir_prefix="wikg_distillinit"
      DIM_HIDDEN=512
      ;;
    clam_sb)
      SLIDE_NAME="clam_sb"
      slide_dir_prefix="clam_sb_distillinit"
      DIM_HIDDEN=512
      ;;
    dagmil)
      SLIDE_NAME="dagmil"
      slide_dir_prefix="dagmil_distillinit"
      DIM_HIDDEN=256
      ;;
    gdfmil)
      SLIDE_NAME="gdfmil"
      slide_dir_prefix="gdfmil_distillinit"
      DIM_HIDDEN=256
      ;;
    *)
      echo "Unsupported slide model: ${slide_name}" >&2
      exit 1
      ;;
  esac
  PRETRAINED_WEIGHTS="$(resolve_pretrained_weights "${SLIDE_NAME}")"
  SLIDE_DIR_NAME="${slide_dir_prefix}"
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
  if [[ -z "${PRETRAINED_WEIGHTS}" ]]; then
    echo "[skip] no pretrained weights configured for ${SLIDE_NAME}; set PRETRAINED_WEIGHTS_DIR or PRETRAINED_WEIGHTS_${SLIDE_NAME^^}"
    return 0
  fi
  if [[ ! -f "${PRETRAINED_WEIGHTS}" ]]; then
    echo "[skip] missing pretrained weights for ${SLIDE_NAME}"
    return 0
  fi

  local ds_name_arg=""
  if [[ -n "${dataset_name}" ]]; then
    ds_name_arg="--dataset_name ${dataset_name}"
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
    --pretrained_weights "${PRETRAINED_WEIGHTS}" \
    --slide_dir_name "${SLIDE_DIR_NAME}" \
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
    echo "===== PFM: ${PFM_NAME} | slide: ${SLIDE_NAME} | init: ${SLIDE_DIR_NAME} ====="
    run_five_seeds "${DATA_ROOT}/bcnb" "${JSON_DIR}/bcnb_er.json"
  done
done
