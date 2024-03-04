# hash:sha256:57ba09552714e05bdcdafd78335375c1c2a9aafe8375afcae951bc59956d2beb
FROM registry.codeocean.allenneuraldynamics.org/codeocean/miniconda3:4.9.2-cuda11.7.0-cudnn8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y


COPY postInstall /
RUN /postInstall
