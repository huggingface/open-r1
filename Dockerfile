FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
  git \
  && rm -rf /var/lib/apt/lists/*
