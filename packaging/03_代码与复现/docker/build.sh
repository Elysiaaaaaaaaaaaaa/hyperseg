#!/usr/bin/env bash
# Build the rebuild-path image from this package.
#
# Run from 03_代码与复现/ (this file's parent's parent).
set -euo pipefail

cd "$(dirname "$0")/.."
TAG="${1:-hyperseg-uav:round2}"

docker build --tag "$TAG" --file Dockerfile .
echo "built $TAG"
echo "smoke test:"
docker run --rm "$TAG" python -c "
from hyperseg_uav import HyperSegUAV, translate_legacy_keys
import torch
state = torch.load('models/hyperseg_b3_best.pt', map_location='cpu', weights_only=False)
_, translated = translate_legacy_keys(state['model'])
print('translated legacy encoder keys:', translated)
print('expected: 628')
"
