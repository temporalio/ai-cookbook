#!/usr/bin/env bash
# Build the sandbox image and, if REGISTRY is set, push it where the OpenShell provider can pull it.
#   sandbox-image/build.sh                      # local Docker (Apple Silicon with Docker Desktop, or a Linux host)
#   REGISTRY=ghcr.io/<org> sandbox-image/build.sh
set -euo pipefail
cd "$(dirname "$0")/.."
TAG=${SANDBOX_IMAGE:-durable-simulation-agent-sandbox:0.1}
docker build -f sandbox-image/Dockerfile -t "$TAG" .
if [ -n "${REGISTRY:-}" ]; then
  docker tag "$TAG" "$REGISTRY/$TAG"
  docker push "$REGISTRY/$TAG"
  echo "pushed $REGISTRY/$TAG; start the worker with SANDBOX_IMAGE=$REGISTRY/$TAG"
fi
