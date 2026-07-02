#!/usr/bin/env bash
# Build the MoA proxy test image and run the unit suite, then (when
# OPENROUTER_API_KEY is set in the environment) the live integration suite.
#
# Usage, from the repo root:
#   OPENROUTER_API_KEY=sk-or-... tests/integration/docker/run-moa-proxy-tests.sh
set -euo pipefail

cd "$(dirname "$0")/../../.."

IMAGE=hermes-moa-proxy-test

docker build -f tests/integration/docker/Dockerfile.moa-proxy -t "$IMAGE" .

# tests/ is excluded from the image (.dockerignore) — mount it read-only.
TESTS_MOUNT=(-v "$PWD/tests:/app/tests:ro")

echo "=== MoA proxy unit tests (no network) ==="
docker run --rm "${TESTS_MOUNT[@]}" "$IMAGE" \
    pytest tests/hermes_cli/test_moa_proxy_server.py -v

if [[ -n "${OPENROUTER_API_KEY:-}" ]]; then
    echo "=== MoA proxy LIVE integration tests (OpenRouter) ==="
    docker run --rm "${TESTS_MOUNT[@]}" -e OPENROUTER_API_KEY "$IMAGE" \
        pytest -m integration tests/integration/test_moa_proxy_live.py -v
else
    echo "OPENROUTER_API_KEY not set — skipping live integration tests."
fi
