#!/usr/bin/env bash
# GPU topology study on `bench` (2x RTX PRO 6000 Blackwell 96GB, sm_120):
# single-GPU vs two independent single-GPU instances (dual) vs tensor-parallel
# TP=2, for Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 served through vLLM.
#
# Run ONE subcommand at a time (this is a human-in-the-loop runbook, not a
# single unattended pipeline):
#
#   gpu2_topology_study.sh s1-single
#   gpu2_topology_study.sh s2-dual
#   gpu2_topology_study.sh s3-tp2
#   gpu2_topology_study.sh s4-restore
#   gpu2_topology_study.sh report
#
# Executed on pg. All bench-side actions go through `ssh -o BatchMode=yes
# bench`. Probe scripts (scripts/gpu_endpoint_bench.py,
# scripts/moa_prefix_cache_probe.py) run from pg, inside the
# hermes-moa-proxy-test docker image (--network host), against bench's
# LAN IP (10.0.120.249) -- mirrors the existing
# tests/integration/docker/run-moa-proxy-tests.sh mount convention
# (scripts/ ro at /app/scripts, an output dir at /out).
#
# Containers this script is allowed to touch on bench:
#   vllm-topo-single, vllm-topo-dual0, vllm-topo-tp2   (created/removed here)
#   vllm-agents-a1-bf16-gpu0-200k                       (stopped in s2, started in s4 -- someone
#                                                        else's idle deployment, must come back)
#   vllm-q80-gpu1                                       (s4: permanent arbiter service)
# NEVER touch any other container on bench (e.g. hf-prefetch-80b).

set -euo pipefail

BENCH_HOST="bench"
BENCH_IP="10.0.120.249"
HF_CACHE="/home/turq/.cache/huggingface"
MODEL="Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
MODEL_HUB_DIR="${HF_CACHE}/hub/models--Qwen--Qwen3-Next-80B-A3B-Instruct-FP8"
SERVED_NAME="q80"
IMAGE_BENCH="local/vllm-openai:cu130-gemma4"
IMAGE_PROBE="hermes-moa-proxy-test"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTDIR="${HOME}/opt/moa-runs/gpu2"

# vLLM flags shared by every stage's server (only --gpus, --name, -p, and
# --tensor-parallel-size/--disable-custom-all-reduce vary).
COMMON_ARGS=(
  --model "${MODEL}"
  --served-model-name "${SERVED_NAME}"
  --host 0.0.0.0
  --max-model-len 262144
  --kv-cache-dtype fp8_e4m3
  --gpu-memory-utilization 0.92
  --enable-prefix-caching
  --max-num-seqs 8
)

log() { printf '\n=== %s ===\n' "$*"; }

ssh_bench() {
  ssh -o BatchMode=yes "${BENCH_HOST}" "$@"
}

setup() {
  mkdir -p "${OUTDIR}"
}

# Guard: refuse to launch anything if the HF cache copy of the model is
# still downloading (any *.incomplete blob) or implausibly small.
model_ready() {
  local incomplete size_gb
  incomplete=$(ssh_bench "find '${MODEL_HUB_DIR}/blobs' -name '*.incomplete' 2>/dev/null | wc -l") || incomplete=""
  if [[ -z "${incomplete}" ]]; then
    echo "FAIL: could not stat ${MODEL_HUB_DIR} on bench" >&2
    return 1
  fi
  if [[ "${incomplete}" != "0" ]]; then
    echo "FAIL: model still downloading on bench (${incomplete} .incomplete blob(s) under ${MODEL_HUB_DIR}/blobs)" >&2
    return 1
  fi
  size_gb=$(ssh_bench "du -s --block-size=1G '${MODEL_HUB_DIR}' 2>/dev/null | cut -f1") || size_gb=""
  if [[ -z "${size_gb}" || "${size_gb}" -lt 40 ]]; then
    echo "FAIL: model cache at ${MODEL_HUB_DIR} is only ${size_gb:-0}GB (expected a completed FP8 80B-A3B download) -- aborting stage" >&2
    return 1
  fi
  echo "model ready: ${MODEL_HUB_DIR} = ${size_gb}GB, 0 incomplete blobs"
}

# wait_health <port> <cap_seconds>
wait_health() {
  local port="$1" cap="$2" waited=0 interval=10
  log "waiting for http://${BENCH_IP}:${port}/health (cap ${cap}s)"
  until curl -sf --max-time 5 "http://${BENCH_IP}:${port}/health" -o /dev/null; do
    if (( waited >= cap )); then
      echo "FAIL: http://${BENCH_IP}:${port}/health did not come up within ${cap}s" >&2
      return 1
    fi
    sleep "${interval}"
    waited=$(( waited + interval ))
    echo "  ...still waiting (${waited}/${cap}s)"
  done
  echo "healthy after ~${waited}s"
}

# probe_container <extra docker-run args...> -- runs the given bash -c
# command string inside IMAGE_PROBE with scripts/ mounted ro and OUTDIR
# mounted at /out, --network host so it can reach bench's LAN IP directly.
run_probe() {
  local cmd="$1"
  docker run --rm --network host \
    -v "${REPO_DIR}/scripts:/app/scripts:ro" \
    -v "${OUTDIR}:/out" \
    "${IMAGE_PROBE}" \
    bash -c "${cmd}"
}

endpoint_bench_cmd() {
  local base="$1" label="$2" out_base="$3"
  echo "python scripts/gpu_endpoint_bench.py --base ${base} --model ${SERVED_NAME} --label ${label} --out /out/${out_base}.json 2>&1 | tee /out/${out_base}.txt"
}

prefix_probe_cmd() {
  local base="$1" prefix_tokens="$2" warm_calls="$3" out_file="$4" metrics="${5:-}" timeout="${6:-}"
  local extra=""
  [[ -n "${metrics}" ]] && extra="${extra} --metrics ${metrics}"
  [[ -n "${timeout}" ]] && extra="${extra} --timeout ${timeout}"
  echo "python scripts/moa_prefix_cache_probe.py --base ${base} --model ${SERVED_NAME} --prefix-tokens ${prefix_tokens} --warm-calls ${warm_calls}${extra} --out /out/${out_file}"
}

s1_single() {
  setup
  log "s1-single: guard model download state"
  model_ready

  log "s1-single: launch vllm-topo-single on bench GPU 1, port 18151"
  ssh_bench bash -s <<REMOTE
set -euo pipefail
docker run -d --name vllm-topo-single --gpus '"device=1"' \\
  -v ${HF_CACHE}:/root/.cache/huggingface \\
  -p 0.0.0.0:18151:8000 --ipc=host \\
  ${IMAGE_BENCH} \\
  --model ${MODEL} --served-model-name ${SERVED_NAME} --host 0.0.0.0 \\
  --max-model-len 262144 --kv-cache-dtype fp8_e4m3 \\
  --gpu-memory-utilization 0.92 --enable-prefix-caching --max-num-seqs 8
REMOTE

  wait_health 18151 900

  log "s1-single: (a) endpoint bench -> s1-endpoint.{json,txt}"
  run_probe "$(endpoint_bench_cmd "http://${BENCH_IP}:18151/v1" s1-single s1-endpoint)"

  log "s1-single: (b) prefix-cache probe (60k) -> s1-cache.json"
  run_probe "$(prefix_probe_cmd "http://${BENCH_IP}:18151/v1" 60000 3 s1-cache.json "http://${BENCH_IP}:18151/metrics")"

  log "s1-single: (c) long-context smoke (180k, warm-calls=1) -> s1-longctx.json"
  run_probe "$(prefix_probe_cmd "http://${BENCH_IP}:18151/v1" 180000 1 s1-longctx.json "" 900)"

  log "s1-single done. Container vllm-topo-single left running for s2."
}

s2_dual() {
  setup
  log "s2-dual: guard model download state"
  model_ready

  log "s2-dual: stop vllm-agents-a1-bf16-gpu0-200k to free GPU 0"
  ssh_bench "docker stop vllm-agents-a1-bf16-gpu0-200k"

  log "s2-dual: launch vllm-topo-dual0 on bench GPU 0, port 18152"
  ssh_bench bash -s <<REMOTE
set -euo pipefail
docker run -d --name vllm-topo-dual0 --gpus '"device=0"' \\
  -v ${HF_CACHE}:/root/.cache/huggingface \\
  -p 0.0.0.0:18152:8000 --ipc=host \\
  ${IMAGE_BENCH} \\
  --model ${MODEL} --served-model-name ${SERVED_NAME} --host 0.0.0.0 \\
  --max-model-len 262144 --kv-cache-dtype fp8_e4m3 \\
  --gpu-memory-utilization 0.92 --enable-prefix-caching --max-num-seqs 8
REMOTE

  wait_health 18152 900

  log "s2-dual: aggregate endpoint bench, BOTH endpoints simultaneously"
  run_probe "$(endpoint_bench_cmd "http://${BENCH_IP}:18151/v1" s2-dual-A s2-endpointA)" &
  local pidA=$!
  run_probe "$(endpoint_bench_cmd "http://${BENCH_IP}:18152/v1" s2-dual-B s2-endpointB)" &
  local pidB=$!
  wait "${pidA}"
  wait "${pidB}"

  log "s2-dual: prefix-cache probe on the new instance (18152) -> s2-cache.json"
  run_probe "$(prefix_probe_cmd "http://${BENCH_IP}:18152/v1" 60000 3 s2-cache.json "http://${BENCH_IP}:18152/metrics")"

  log "s2-dual done. vllm-topo-single (18151) and vllm-topo-dual0 (18152) both left running for s3 teardown."
}

s3_tp2() {
  setup
  log "s3-tp2: guard model download state"
  model_ready

  log "s3-tp2: remove the two single-GPU instances to free both GPUs"
  ssh_bench "docker rm -f vllm-topo-single vllm-topo-dual0"

  log "s3-tp2: launch vllm-topo-tp2 (TP=2, both GPUs), port 18153"
  ssh_bench bash -s <<REMOTE
set -euo pipefail
docker run -d --name vllm-topo-tp2 --gpus all \\
  -v ${HF_CACHE}:/root/.cache/huggingface \\
  -p 0.0.0.0:18153:8000 --ipc=host \\
  ${IMAGE_BENCH} \\
  --model ${MODEL} --served-model-name ${SERVED_NAME} --host 0.0.0.0 \\
  --max-model-len 262144 --kv-cache-dtype fp8_e4m3 \\
  --gpu-memory-utilization 0.92 --enable-prefix-caching --max-num-seqs 8 \\
  --tensor-parallel-size 2 --disable-custom-all-reduce
REMOTE

  wait_health 18153 1200

  log "s3-tp2: endpoint bench -> s3-endpoint.{json,txt}"
  run_probe "$(endpoint_bench_cmd "http://${BENCH_IP}:18153/v1" s3-tp2 s3-endpoint)"

  log "s3-tp2: prefix-cache probe (60k) -> s3-cache.json"
  run_probe "$(prefix_probe_cmd "http://${BENCH_IP}:18153/v1" 60000 3 s3-cache.json "http://${BENCH_IP}:18153/metrics")"

  log "s3-tp2: long-context smoke (180k) -> s3-longctx.json"
  run_probe "$(prefix_probe_cmd "http://${BENCH_IP}:18153/v1" 180000 1 s3-longctx.json "" 900)"

  log "s3-tp2: long-context attempt (240k, TP2 has more KV room) -> s3-longctx240.json"
  run_probe "$(prefix_probe_cmd "http://${BENCH_IP}:18153/v1" 240000 1 s3-longctx240.json "" 1200)"

  log "s3-tp2 done. vllm-topo-tp2 left running; torn down in s4-restore."
}

s4_restore() {
  setup
  log "s4-restore: remove all vllm-topo-* study containers (idempotent)"
  ssh_bench "docker rm -f vllm-topo-tp2 vllm-topo-dual0 vllm-topo-single 2>/dev/null || true"

  log "s4-restore: bring back vllm-agents-a1-bf16-gpu0-200k (someone else's deployment)"
  ssh_bench "docker start vllm-agents-a1-bf16-gpu0-200k"

  log "s4-restore: guard model download state before relaunching the arbiter"
  model_ready

  log "s4-restore: relaunch the s1 single-GPU config as the permanent arbiter, vllm-q80-gpu1 (GPU 1, port 18151)"
  ssh_bench bash -s <<REMOTE
set -euo pipefail
docker run -d --name vllm-q80-gpu1 --restart unless-stopped --gpus '"device=1"' \\
  -v ${HF_CACHE}:/root/.cache/huggingface \\
  -p 0.0.0.0:18151:8000 --ipc=host \\
  ${IMAGE_BENCH} \\
  --model ${MODEL} --served-model-name ${SERVED_NAME} --host 0.0.0.0 \\
  --max-model-len 262144 --kv-cache-dtype fp8_e4m3 \\
  --gpu-memory-utilization 0.92 --enable-prefix-caching --max-num-seqs 8
REMOTE

  wait_health 18151 900

  log "s4-restore: final state"
  ssh_bench "docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Ports}}\t{{.Status}}'"
  ssh_bench "nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv"
}

report() {
  python3 - "${OUTDIR}" <<'PYEOF'
import json, sys, glob, os

outdir = sys.argv[1]

def load(name):
    path = os.path.join(outdir, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def endpoint_row(label, name):
    d = load(name)
    if not d:
        return [label, "-", "-", "-"]
    ttft = d["single_streams"][0]["ttft_s"] if d.get("single_streams") else None
    return [
        label,
        d.get("single_decode_tok_s_best", "-"),
        ttft if ttft is not None else "-",
        [c.get("aggregate_tok_s") for c in d.get("concurrent", [])],
    ]

def cache_row(label, name):
    d = load(name)
    if not d:
        return [label, "-", "-"]
    calls = d.get("calls", [])
    cold = calls[0]["ttft_s"] if calls else None
    warm = d.get("warm_over_cold_ttft")
    return [label, cold, warm]

def longctx_row(label, name):
    d = load(name)
    if not d:
        return [label, "-", "-"]
    calls = d.get("calls", [])
    cold = calls[0]["ttft_s"] if calls else None
    ok = bool(calls)
    return [label, cold, ok]

print("=== decode / TTFT / aggregate tok/s ===")
rows = [
    endpoint_row("single (gpu1)", "s1-endpoint.json"),
    endpoint_row("dual-A (gpu1, aggregate run)", "s2-endpointA.json"),
    endpoint_row("dual-B (gpu0, aggregate run)", "s2-endpointB.json"),
    endpoint_row("tp2 (both gpus)", "s3-endpoint.json"),
]
print(f"{'config':32} {'best decode tok/s':18} {'ttft_s (single#1)':18} {'aggregate_tok_s by concurrency'}")
for r in rows:
    print(f"{r[0]:32} {str(r[1]):18} {str(r[2]):18} {r[3]}")

print("\n=== prefix-cache warm/cold TTFT (60k prefix) ===")
for r in [cache_row("single", "s1-cache.json"), cache_row("dual (new inst.)", "s2-cache.json"), cache_row("tp2", "s3-cache.json")]:
    print(f"{r[0]:32} cold_ttft_s={r[1]!s:10} warm/cold_ratio={r[2]}")

print("\n=== long-context smoke ===")
for label, name in [
    ("single 180k", "s1-longctx.json"),
    ("tp2 180k", "s3-longctx.json"),
    ("tp2 240k", "s3-longctx240.json"),
]:
    r = longctx_row(label, name)
    print(f"{r[0]:20} cold_ttft_s={r[1]!s:10} completed={r[2]}")

print(f"\n(raw files in {outdir}: {sorted(os.path.basename(p) for p in glob.glob(os.path.join(outdir, '*')))})")
PYEOF
}

usage() {
  cat <<EOF
Usage: $(basename "$0") <subcommand>
  s1-single    launch single-GPU (bench GPU1, port 18151), run probes
  s2-dual      stop the idle deployment, launch a second single-GPU instance (GPU0, port 18152), run aggregate probes
  s3-tp2       remove both single-GPU instances, launch TP=2 (both GPUs, port 18153), run probes
  s4-restore   tear down study containers, restore the idle deployment, relaunch the arbiter as vllm-q80-gpu1
  report       parse ${OUTDIR}/*.json and print a comparison table
EOF
}

main() {
  case "${1:-}" in
    s1-single) s1_single ;;
    s2-dual) s2_dual ;;
    s3-tp2) s3_tp2 ;;
    s4-restore) s4_restore ;;
    report) report ;;
    *) usage; exit 1 ;;
  esac
}

main "$@"
