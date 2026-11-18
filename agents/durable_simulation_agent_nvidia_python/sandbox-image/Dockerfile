# The sandbox image, used for BOTH the exec sandbox and the tools sandbox; the OpenShell policy is the difference.
# Requirements from OpenShell: a non-root USER, a writable /sandbox workdir, iproute2, not distroless.
# Build from the recipe directory (the context needs sceneops/ and site/):
#   docker build -f sandbox-image/Dockerfile -t durable-simulation-agent-sandbox:0.1 .
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends iproute2 ca-certificates && rm -rf /var/lib/apt/lists/* \
 && useradd -m -u 1000 -s /bin/bash agent && mkdir -p /sandbox /opt/site && chown agent:agent /sandbox
RUN pip install --no-cache-dir "usd-core>=26.11,<27"
COPY sceneops /usr/local/lib/python3.12/site-packages/sceneops
COPY site /opt/site
RUN chmod -R a-w /opt/site /usr/local/lib/python3.12/site-packages/sceneops
USER agent
WORKDIR /sandbox
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
# OpenShell replaces ENTRYPOINT/CMD; the sandbox is created with `-- sleep infinity` and every operation is an exec.
CMD ["sleep", "infinity"]
