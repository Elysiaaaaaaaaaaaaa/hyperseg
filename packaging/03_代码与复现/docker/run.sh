#!/usr/bin/env bash
# Run the rebuilt image with a GPU and the competition data mounted read-only.
#
# Usage: docker/run.sh /absolute/path/to/low_altitude_2026
set -euo pipefail

cd "$(dirname "$0")/.."
TAG="${TAG:-hyperseg-uav:round2}"
DATA="${1:?usage: docker/run.sh /path/to/low_altitude_2026}"

docker run --rm -it --gpus all --shm-size=8g \
    -v "$DATA:/opt/hyperseg/dataset/low_altitude_2026:ro" \
    -v "$PWD/reproduced:/opt/hyperseg/reproduced" \
    "$TAG" /bin/bash
