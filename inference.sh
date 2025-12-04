#!/usr/bin/env bash
set -euo pipefail

# ------------- User-configurable paths -------------
PYTHON_BIN="python"  # or absolute path to python
INFER_SCRIPT="/home/joty/code/flow_matching/inference_lut.py"
DATA_ROOT="/home/joty/code/flow_matching/data"
CKPT_PATH="/home/joty/code/flow_matching/exp/checkpoints/test7/epoch=7-val_loss=0.167420.ckpt"
LUT_DIR="/home/joty/code/flow_matching/database"
SAVE_DIR="/home/joty/code/flow_matching/test_results/lut16"

# ------------- Inference hyperparameters -----------
NUM_GPUS=1                 # set 0 for CPU
PRECISION="bf16"           # bf16 | bf16-mixed | 32 | 16-mixed
NUM_WORKERS=12
INPUT_LEN=8
PRED_LEN=8
NUM_FLOW_STEPS=100
EPS=0.0
T_SCHEDULE="stratified_uniform"
LIMIT_TEST_BATCHES=1.0     # 1.0 means all, or e.g. 0.1 for 10%

# ------------- Batching (loader compatibility) ----
TRAIN_BS=32                # not used in test, kept for compatibility
VAL_BS=32                  # kept for compatibility with loader signature
TEST_BS=32                 # will try to override test loader batch_size if supported

# ------------- Logging (W&B) -----------------------
ENABLE_WANDB=1             # 1 to enable, 0 to disable
WANDB_PROJECT="flow_matching"
WANDB_GROUP="flow_matching"
WANDB_ID=""                # leave empty for a new run
WANDB_OFFLINE=0            # 1 for offline mode


YEARS_TRAIN=(2019 2020)
YEARS_VAL=(2021)
YEARS_TEST=(2022)
# ------------- Reproducibility ---------------------
SEED=42
BS=16
STRIDE=1
FORECAST=true
USE_POSSIBLE_STARTS=true
LUT_TOPK=1
LUT_TAU=10.0
LUT_BETA=0
LUT_AGG="weighted_mean"
# ------------- Build CLI ---------------------------
CLI_ARGS=(
  --ckpt_path "${CKPT_PATH}"
  --data_path "${DATA_ROOT}"
  --years_train "${YEARS_TRAIN[@]}" \
  --years_val "${YEARS_VAL[@]}" \
  --years_test "${YEARS_TEST[@]}" \
  --batch_size ${BS} \
  --save_results_dir "${SAVE_DIR}"
  --input_len "${INPUT_LEN}"
  --pred_len "${PRED_LEN}"
  --stride ${STRIDE} \
  --num_gpus "${NUM_GPUS}"
  --precision "${PRECISION}"
  --num_workers "${NUM_WORKERS}"
  --num_flow_steps "${NUM_FLOW_STEPS}"
  --eps "${EPS}"
  --t_schedule "${T_SCHEDULE}"
  --lut_path "${LUT_DIR}" \
  --use_lut \
  --lut_topk ${LUT_TOPK} \
  --lut_tau ${LUT_TAU} \
  --lut_beta ${LUT_BETA} \
  --lut_agg "${LUT_AGG}"
)

if [[ "${ENABLE_WANDB}" -eq 1 ]]; then
  CLI_ARGS+=( --enable_wandb --wandb_project_name "${WANDB_PROJECT}" )
  [[ -n "${WANDB_GROUP}" ]] && CLI_ARGS+=( --wandb_group "${WANDB_GROUP}" )
  [[ -n "${WANDB_ID}" ]] && CLI_ARGS+=( --wandb_id "${WANDB_ID}" )
  [[ "${WANDB_OFFLINE}" -eq 1 ]] && CLI_ARGS+=( --wandb_offline )
fi

# ------------- Run -------------------------------
export CUDA_VISIBLE_DEVICES=0
echo "[Run] ${PYTHON_BIN} ${INFER_SCRIPT}"
"${PYTHON_BIN}" "${INFER_SCRIPT}" "${CLI_ARGS[@]}"