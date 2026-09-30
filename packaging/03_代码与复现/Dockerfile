# Rebuild path for the HyperSeg-UAV round-two environment.
#
# The authoritative, already-verified environment is the root filesystem snapshot
# shipped in ../04_镜像 (restore with `docker import`); this Dockerfile is the
# secondary path that rebuilds an equivalent environment from public packages.
#
# Pinned to the versions actually recorded on server3 (see docker/environment.txt
# and docker/requirements.lock.txt):
#   Ubuntu 22.04.3, Python 3.12.3, torch 2.3.0+cu121, CUDA 12.1, cuDNN 8.9.2,
#   transformers 4.57.6.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        ca-certificates \
        curl \
        git \
        libgl1 \
        libglib2.0-0 \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

# CPU-only bookkeeping during the build avoids OpenMP thread thrashing on hosts
# whose cgroup quota is far below the visible core count. Override at run time.
ENV OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4

WORKDIR /opt/hyperseg
COPY hyperseg/ /opt/hyperseg/
COPY docker/ /opt/hyperseg/docker/

RUN python3.12 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        -r docker/requirements.docker.txt

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/opt/hyperseg

# Inference needs the MiT-B3 backbone. The package ships it at
# hyperseg/models/nvidia--mit-b3 (already COPYed into /opt/hyperseg above), and
# hyperseg_uav/model.py prefers that local directory over the hub, so a rebuild works
# offline. The hub name is only used when the shipped copy is missing.
RUN python -c "\
from pathlib import Path; \
from transformers import SegformerModel; \
local = Path('/opt/hyperseg/models/nvidia--mit-b3'); \
src = str(local) if local.is_dir() else 'nvidia/mit-b3'; \
SegformerModel.from_pretrained(src); \
print('backbone ready from', src)"

# The competition dataset is never part of an image. Mount it instead:
#   -v /host/low_altitude_2026:/opt/hyperseg/dataset/low_altitude_2026:ro
VOLUME ["/opt/hyperseg/dataset"]

CMD ["/bin/bash"]
