#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="python"
TRAIN_PY="/home/joty/code/train.py"

DATA_PATH=""
VAE_CKPT=""
COT_VAE_CKPT=""

NUM_GPUS=1
NUM_WORKERS=12  
BS=16   
MAX_EPOCHS=1000
PRECISION="bf16"
FLOW_STEPS=100
CHECK_VAL_EVERY=1
EMA_DECAY=0.9999
EPS=0.0
T_SCHEDULE="stratified_uniform"
WEIGHT_DECAY=1e-2
LR=5e-4
INPUT_LEN=8
PRED_LEN=1
OT_METHOD="sinkhorn"
OT_REG=0.2
INPUT_LEN=8
PRED_LEN=8
STRIDE=1
FORECAST=true
USE_POSSIBLE_STARTS=true

YEARS_TRAIN=(2017 2018 2019 2020) 
YEARS_VAL=(2021)
YEARS_TEST=(2022)
# W&B
WANDB_PROJECT="flow_matching"
WANDB_GROUP="WIPFlow"
WANDB_ID=""           

# ====================================

export WANDB_MODE=online
export WANDB_DIR="${HOME}/wandb_logs"
export CUDA_VISIBLE_DEVICES=1
${PYTHON_BIN} "${TRAIN_PY}" \
  --precision "${PRECISION}" \
  --data_path "${DATA_PATH}" \
  --num_flow_steps ${FLOW_STEPS} \
  --num_gpus ${NUM_GPUS} \
  --num_workers ${NUM_WORKERS} \
  --batch_size ${BS} \
  --check_val_every_n_epoch ${CHECK_VAL_EVERY} \
  --years_train "${YEARS_TRAIN[@]}" \
  --years_val "${YEARS_VAL[@]}" \
  --years_test "${YEARS_TEST[@]}" \
  --input_len ${INPUT_LEN} \
  --pred_len ${PRED_LEN} \
  --stride ${STRIDE} \
  --max_epochs ${MAX_EPOCHS} \
  --ema_decay ${EMA_DECAY} \
  --eps ${EPS} \
  --weight_decay ${WEIGHT_DECAY} \
  --lr ${LR} \
  --wandb_project_name "${WANDB_PROJECT}" \
  --wandb_group "${WANDB_GROUP}" \
  $( [ -n "${WANDB_ID}" ] && echo --wandb_id "${WANDB_ID}" ) \
  --vae_ckpt_path "${VAE_CKPT}" \
  --cot_vae_path "${COT_VAE_CKPT}" \