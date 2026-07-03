# Minimal image for the OpenAI-compatible MoA endpoint (`hermes moa serve`).
#
# Serves your configured Mixture-of-Agents presets as models on
# /v1/chat/completions so ANY OpenAI-SDK client (aider, mini-swe-agent,
# Harbor/terminus-2, OpenCode, plain openai-python) can use Hermes MoA as
# its upstream model. Deliberately slim: no gateway, no dashboard, no
# messaging stacks — just the CLI plus aiohttp (the one dependency `moa
# serve` needs beyond the core pins).
#
# Build (from the repo root):
#   docker build -f docker/moa-proxy/Dockerfile -t hermes-moa-proxy .
#
# Run (config mounted read-only; see docs/moa-openai-endpoint.md for the
# config shape — presets, optional router block, save_traces):
#   docker run -d --name moa-proxy \
#     -v $PWD/moa-home:/moa-home:ro \
#     -e OPENROUTER_API_KEY \
#     -p 127.0.0.1:8646:8646 \
#     hermes-moa-proxy
#
# Or: docker compose -f docker-compose.moa.yml up -d

FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# aiohttp pin matches the repo's messaging extra (see pyproject.toml).
RUN python -m pip install --upgrade pip && pip install aiohttp==3.14.1

# The build context excludes *.md (.dockerignore) but pyproject declares
# readme = "README.md" — stub it so the install works.
COPY . .
RUN touch README.md && pip install .

# HERMES_HOME carries the MoA config (config.yaml with moa.presets /
# moa.router). Mount your prepared directory here. Traces (if enabled via
# moa.save_traces) are written under it too — mount read-write in that case
# and set moa.trace_dir if you want them elsewhere.
ENV HERMES_HOME=/moa-home

EXPOSE 8646

# --host 0.0.0.0 is safe INSIDE the container; publish to loopback
# (-p 127.0.0.1:8646:8646) or set HERMES_MOA_API_KEY / --api-key when
# exposing further. The server warns on non-loopback binds without a key.
ENTRYPOINT ["hermes", "moa", "serve", "--host", "0.0.0.0", "--port", "8646"]
