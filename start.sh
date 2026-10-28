#!/usr/bin/env bash
set -euo pipefail

# DeepSeek-V4.1-Flash on SGLang (8x H200 141GB verified; larger GPUs auto-detected)
#
# Recipe from the SGLang cookbook (H200 cell, verified):
#   https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1
#
# Notes:
#   - MoE model (fp8 dense + fp4 experts), ~510GB weights. TP = EP = GPU count.
#     8 GPUs is the verified cookbook cell. 4 GPUs (~564GB) is tight: it needs
#     MEM_FRACTION 0.92, --max-running-requests 16, --chunked-prefill-size 1024
#     and a capped --context-length (CONTEXT_LENGTH, default 65536) to start;
#     those are the 4-GPU defaults (EXTRA_ARGS still overrides them). 8 GPUs
#     run the model's full window. Clients like Hermes need >= 64K.
#     2 GPUs (~282GB) cannot hold the weights and is rejected up front.
#   - Not in an SGLang release yet: needs the preview image
#     lmsysorg/sglang:dev-dsv41.
#   - Cookbook pins --attention-backend dsv4, --moe-runner-backend
#     flashinfer_mxfp4 and --cuda-graph-max-bs-decode 64 on H200 (the
#     derived decode graph batch list does not fit in 140 GiB).
#   - Per-GPU memory is read from nvidia-smi (override: GPU_MEM_GB, in GiB).
#     Fewer than 160 GiB per GPU (H200) keeps the pins above. At 160+ GiB
#     (B200, B300, ...) the pins are dropped so SGLang resolves backends and the
#     decode graph list itself (the cookbook says not to override them; the graph
#     cap only exists for 140 GiB) and MEM_FRACTION defaults to 0.9. Total memory
#     decides the rest: < 500 GiB is rejected, < 600 GiB gets the squeeze caps.
#     Not yet run on B200/B300 here: check the startup log for resolved backends.
#   - --reasoning-parser auto (-> deepseek-v41) splits thinking into
#     reasoning_content. Thinking is OFF by default; send reasoning_effort
#     per request, or THINKING=1 turns it on server-wide.
#   - --tool-call-parser auto (-> deepseekv41); just send `tools`.
#   - DSPARK=1 turns on the model's bundled DSpark speculative draft
#     (block size 5). The verified H200 cells run without it, so it is off
#     by default.

# Load optional .env overrides. Shell env wins; .env fills gaps; defaults last.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/.env" ]]; then
  while IFS='=' read -r key value || [[ -n "${key}" ]]; do
    key="${key%$'\r'}"; value="${value%$'\r'}"
    key="${key#"${key%%[![:space:]]*}"}"; key="${key%"${key##*[![:space:]]}"}"
    [[ -z "${key}" || "${key}" == \#* ]] && continue
    if [[ -z "${!key:-}" ]]; then
      export "${key}=${value}"
    fi
  done < "${SCRIPT_DIR}/.env"
fi

MODEL_ID="${MODEL_ID:-deepseek-ai/DeepSeek-V4.1-Flash}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-DeepSeek-V4.1-Flash}"
IMAGE="${IMAGE:-lmsysorg/sglang:dev-dsv41}"
GPU_ID="${GPU_ID:-0,1,2,3,4,5,6,7}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
THINKING="${THINKING:-0}"
# Tight (< 600 GiB total) only: the KV cache is what is left after weights, so the window is capped.
CONTEXT_LENGTH="${CONTEXT_LENGTH:-65536}"
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"
# Free-form extra SGLang flags, appended LAST (argparse last-wins):
#   EXTRA_ARGS="--max-running-requests 256" ./start.sh
read -ra EXTRA_ARGS_ARR <<< "${EXTRA_ARGS:-}"

TP_SIZE="$(tr ',' '\n' <<< "${GPU_ID}" | grep -c '[0-9]' || true)"
# Per-GPU GiB, smallest of the selected GPUs; falls back to H200 (the verified recipe).
GPU_MEM_GB="${GPU_MEM_GB:-$(nvidia-smi -i "${GPU_ID}" --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | sort -n | awk 'NR==1{print int($1/1024)}' || true)}"
GPU_MEM_GB="${GPU_MEM_GB:-141}"
TOTAL_GB=$(( GPU_MEM_GB * TP_SIZE ))

case "${TP_SIZE}" in 2|4|8) ;; *) echo "GPU_ID='${GPU_ID}' gives ${TP_SIZE} GPUs; use 2, 4 or 8"; exit 1 ;; esac
if (( TOTAL_GB < 500 )); then
  echo "${TP_SIZE} GPUs x ${GPU_MEM_GB}GiB = ${TOTAL_GB}GiB; DeepSeek-V4.1-Flash weights are ~510GB. Use more GPUs"; exit 1
fi

if (( GPU_MEM_GB >= 160 )); then
  PIN_ARGS=(); DEFAULT_MEM=0.9
else
  PIN_ARGS=(--attention-backend dsv4 --moe-runner-backend flashinfer_mxfp4 --cuda-graph-max-bs-decode 64)
  DEFAULT_MEM=0.8
fi

if (( TOTAL_GB < 600 )); then
  # Required to fit; placed before EXTRA_ARGS so it can override.
  DEFAULT_MEM=0.92
  EXTRA_ARGS_ARR=(--max-running-requests 16 --chunked-prefill-size 1024 --context-length "${CONTEXT_LENGTH}" ${EXTRA_ARGS_ARR[@]+"${EXTRA_ARGS_ARR[@]}"})
  echo "WARNING: ${TP_SIZE}x ${GPU_MEM_GB}GiB is tight; capped at 16 running requests and ${CONTEXT_LENGTH} context"
fi
MEM_FRACTION="${MEM_FRACTION:-${DEFAULT_MEM}}"

# DP attention: DP_SIZE=4 on 8 GPUs = attention DP4 x TP2, experts still EP8.
# Pays off at high concurrency; raise --max-running-requests via EXTRA_ARGS too.
DP_SIZE="${DP_SIZE:-1}"
if ! [[ "${DP_SIZE}" =~ ^[0-9]+$ ]] || (( DP_SIZE < 1 || TP_SIZE % DP_SIZE != 0 )); then
  echo "DP_SIZE='${DP_SIZE}' must be a positive divisor of the GPU count (${TP_SIZE})"; exit 1
fi
DP_ARGS=()
if (( DP_SIZE > 1 )); then
  DP_ARGS=(--dp-size "${DP_SIZE}" --enable-dp-attention)
  echo "WARNING: DP attention crashed on the first request on dev-dsv41 (8x B300); see README"
fi

DSPARK="${DSPARK:-0}"
case "${DSPARK}" in
  0) SPEC_ARGS=() ;;
  1) SPEC_ARGS=(--speculative-algorithm DSPARK --speculative-dspark-block-size 5) ;;
  *) echo "DSPARK must be 0 or 1, got '${DSPARK}'"; exit 1 ;;
esac

CONTAINER_NAME="deepseek-v4.1-flash-sglang"
PID_FILE=".sglang.pid"
LOG_FILE=".sglang.log"
READY_URL="http://127.0.0.1:${PORT}/v1/models"

command -v docker >/dev/null 2>&1 || { echo "docker is not on PATH"; exit 1; }
command -v curl >/dev/null 2>&1 || { echo "curl is not on PATH"; exit 1; }

mkdir -p "${HF_CACHE}"

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Container ${CONTAINER_NAME} is already running"
    echo "Log: ${LOG_FILE}"
    exit 0
  fi
  docker rm "${CONTAINER_NAME}" >/dev/null
fi

echo "Starting SGLang container for ${MODEL_ID}"
echo "GPUs: ${GPU_ID} (${GPU_MEM_GB}GiB each, TP${TP_SIZE} + EP${TP_SIZE}, DP-attn ${DP_SIZE}, mem-fraction ${MEM_FRACTION})"
echo "Default thinking: $( (( THINKING )) && echo on || echo off)"
echo "Spec decode: $( (( DSPARK )) && echo DSpark || echo off)"
echo "Image: ${IMAGE}"
echo "Served model name: ${SERVED_MODEL_NAME}"
echo "Listening on ${HOST}:${PORT}"
echo "Writing progress to ${LOG_FILE}"

echo "[$(date -Is)] launching SGLang container" > "${LOG_FILE}"

docker run -d \
  --name "${CONTAINER_NAME}" \
  --network host \
  --ipc host \
  --cap-add SYS_NICE \
  --gpus "\"device=${GPU_ID}\"" \
  --shm-size 32g \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e SGLANG_DEFAULT_THINKING="${THINKING}" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  "${IMAGE}" \
  python3 -m sglang.launch_server \
  --model-path "${MODEL_ID}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  --tp "${TP_SIZE}" \
  --ep-size "${TP_SIZE}" \
  --mem-fraction-static "${MEM_FRACTION}" \
  "${PIN_ARGS[@]}" \
  --reasoning-parser auto \
  --tool-call-parser auto \
  "${SPEC_ARGS[@]}" \
  "${DP_ARGS[@]}" \
  --enable-metrics \
  --host "${HOST}" \
  --port "${PORT}" \
  "${EXTRA_ARGS_ARR[@]}" \
  >/dev/null

docker inspect -f '{{.Id}}' "${CONTAINER_NAME}" > "${PID_FILE}"
echo "Spawned container ${CONTAINER_NAME} ($(cat "${PID_FILE}"))"

log_follow_pid=""
cleanup() {
  if [[ -n "${log_follow_pid}" ]] && kill -0 "${log_follow_pid}" 2>/dev/null; then
    kill "${log_follow_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

docker logs -f "${CONTAINER_NAME}" 2>&1 | tee -a "${LOG_FILE}" &
log_follow_pid=$!

# First boot downloads ~510GB of weights; that alone can take a while.
echo "Waiting for HTTP readiness at ${READY_URL}"
heartbeat=0
until curl -fsS "${READY_URL}" >/dev/null 2>&1; do
  if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "SGLang container exited before becoming ready"
    tail -n 200 "${LOG_FILE}" || true
    exit 1
  fi
  if (( heartbeat % 6 == 0 )); then
    echo "  still starting..."
  fi
  heartbeat=$((heartbeat + 1))
  sleep 5
done

echo "SGLang is ready"
echo "OpenAI base URL: http://${HOST}:${PORT}/v1"
echo "Served model name: ${SERVED_MODEL_NAME}"
echo "Stop with ./stop.sh; logs: docker logs -f ${CONTAINER_NAME}"
