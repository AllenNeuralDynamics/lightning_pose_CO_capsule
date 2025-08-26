# hash:sha256:efe04cd2567bb9241fc6b72a3fb36d96271710494fb18df0bd31e189420457d0
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/cuda-miniconda-jupyterlab:latest

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install -U --no-cache-dir \
    lightning-pose==1.6.0 \
    moviepy==1.0.3 \
    ipywidgets==8.1.2