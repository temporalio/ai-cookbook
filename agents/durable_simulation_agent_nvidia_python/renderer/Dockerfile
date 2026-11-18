# The renderer service: Isaac Sim 6.1 plus this recipe's renderer package, on a host with an NVIDIA RTX GPU. It runs
# outside OpenShell and holds no credentials. Build from the recipe directory (the context needs renderer/, validate/, site/):
#   docker build -f renderer/Dockerfile -t durable-simulation-agent-renderer:0.1 .
# Pulling the base image needs an NGC login: docker login nvcr.io (user $oauthtoken, password: your NGC API key).
FROM nvcr.io/nvidia/isaac-sim:6.1.0
ENV ACCEPT_EULA=Y PRIVACY_CONSENT=Y \
    ISAACSIM_ASSET_ROOT=/opt/assets RENDERER_HOME=/opt/renderer RENDERER_SITE_LAYER=/opt/renderer/site/000_site.usda \
    PYTHONPATH=/opt/renderer
COPY renderer /opt/renderer/renderer
COPY validate /opt/renderer/validate
COPY site /opt/renderer/site
WORKDIR /opt/renderer
ENTRYPOINT ["/isaac-sim/python.sh", "-m", "renderer.run_service", "--backend", "kit", "--host", "127.0.0.1", "--port", "8799", "--out", "/srv/renderer/out"]
