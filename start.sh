#!/usr/bin/env bash
set -euo pipefail

# DeepSeek-V4.1-Flash on SGLang (8x H200 141GB)
#
# Recipe from the SGLang cookbook (H200 cell, verified):
#   https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1
#
# Notes:
#   - MoE model (fp8 dense + fp4 experts), ~510GB weights. TP = EP = GPU count.
#     8 GPUs is the verified cookbook cell. 4 GPUs (~564GB) is unverified and
#     tight: MEM_FRACTION defaults to 0.9 there, leaving little KV cache.
#     2 GPUs (~282GB) cannot hold the weights and is rejected up front.
#   - Not in an SGLang release yet: needs the preview image
#     lmsysorg/sglang:dev-dsv41.
#   - Cookbook pins --attention-backend dsv4, --moe-runner-backend
#     flashinfer_mxfp4 and --cuda-graph-max-bs-decode 64 on H200 (the
#     derived decode graph batch list does not fit in 140 GiB).
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
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"
# Free-form extra SGLang flags, appended LAST (argparse last-wins):
#   EXTRA_ARGS="--max-running-requests 256" ./start.sh
read -ra EXTRA_ARGS_ARR <<< "${EXTRA_ARGS:-}"

TP_SIZE="$(tr ',' '\n' <<< "${GPU_ID}" | grep -c '[0-9]' || true)"
case "${TP_SIZE}" in
  8) MEM_FRACTION="${MEM_FRACTION:-0.8}" ;;
  4) MEM_FRACTION="${MEM_FRACTION:-0.9}"
     echo "WARNING: 4x H200 is not a verified recipe; ~510GB weights leave little KV cache" ;;
  2) echo "GPU_ID='${GPU_ID}' gives 2 GPUs (~282GB); DeepSeek-V4.1-Flash weights are ~510GB. Use 4 or 8"; exit 1 ;;
  *) echo "GPU_ID='${GPU_ID}' gives ${TP_SIZE} GPUs; use 4 or 8 (2 is too small for the weights)"; exit 1 ;;
esac

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
echo "GPUs: ${GPU_ID} (TP${TP_SIZE} + EP${TP_SIZE}, mem-fraction ${MEM_FRACTION})"
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
  --attention-backend dsv4 \
  --moe-runner-backend flashinfer_mxfp4 \
  --cuda-graph-max-bs-decode 64 \
  --reasoning-parser auto \
  --tool-call-parser auto \
  "${SPEC_ARGS[@]}" \
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
