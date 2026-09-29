import importlib
import json
import os
import sys
from pathlib import Path

print('CONTAINER_STARTED', flush=True)
print(json.dumps({'cwd': os.getcwd(), 'python': sys.executable, 'version': sys.version}), flush=True)
for name in ['torch', 'numpy', 'PIL', 'mmcv', 'mmcv.ops', 'mmengine', 'mmseg', 'transformers', 'hyperseg_uav.swin_l_model']:
    module = importlib.import_module(name)
    print('IMPORT_OK', name, getattr(module, '__version__', ''), flush=True)
for name in [
    '/root/hyperseg/experiment/mask2former_uav/infer_h3_hyperseg.py',
    '/root/hyperseg/runs/mask2former_uav/h3_swin_l_hyperseg_160k_seed3407_fp32/best.pt',
]:
    path = Path(name)
    assert path.is_file(), name
    print('FILE_OK', name, path.stat().st_size, flush=True)
print('STARTUP_CHECK_PASSED (no inference, no checkpoint loading)', flush=True)
