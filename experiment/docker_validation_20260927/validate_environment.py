"""Validate the restored H3 environment using its real prediction entry point."""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path('/root/hyperseg')
WORK = Path('/validation')
CHECKPOINT = PROJECT / 'runs/mask2former_uav/h3_swin_l_hyperseg_160k_seed3407_fp32/best.pt'


def log(stage, **details):
    print(json.dumps({'time': datetime.now(timezone.utc).isoformat(),
                      'stage': stage, **details}), flush=True)


def environment():
    import numpy as np
    import torch
    import mmcv
    import mmengine
    import mmseg
    import transformers
    import PIL
    from mmcv.ops import get_compiling_cuda_version, nms
    from hyperseg_uav.swin_l_model import SwinLHyperSeg

    boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.], [20., 20., 30., 30.]])
    scores = torch.tensor([0.9, 0.8, 0.7])
    _, keep = nms(boxes, scores, 0.5)
    assert keep.tolist() == [0, 2], keep
    state = torch.load(CHECKPOINT, map_location='cpu', weights_only=False, mmap=True)
    assert state['model_config']['classes'] == 9
    assert state['step'] == 137200
    assert all(isinstance(value, torch.Tensor) for value in state['model'].values())
    result = {
        'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
        'mmcv': mmcv.__version__, 'mmengine': mmengine.__version__,
        'mmseg': mmseg.__version__, 'transformers': transformers.__version__,
        'pillow': PIL.__version__, 'mmcv_compiled_cuda': get_compiling_cuda_version(),
        'cuda_available': torch.cuda.is_available(), 'mmcv_cpu_nms': 'passed',
        'model_module_import': 'passed', 'checkpoint_step': state['step'],
        'checkpoint_best_val_miou': state['best_mIoU'],
        'checkpoint_state_entries': len(state['model']),
        'checkpoint_model_config': state['model_config'],
    }
    (WORK / 'environment.json').write_text(json.dumps(result, indent=2) + '\n')
    log('environment_passed', **result)


def create_input():
    import numpy as np
    from PIL import Image
    image_dir = WORK / 'synthetic_input'
    image_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(3407)
    array = rng.integers(0, 256, (1024, 1024, 3), dtype=np.uint8)
    Image.fromarray(array, mode='RGB').save(image_dir / 'synthetic_1024.png')
    log('synthetic_input_created', size=[1024, 1024])


def prediction_check():
    import numpy as np
    from PIL import Image
    prediction = WORK / 'predictions/synthetic_1024.png'
    with Image.open(prediction) as image:
        image.load()
        labels = np.asarray(image)
        assert image.mode == 'L', image.mode
        assert image.size == (1024, 1024), image.size
        assert labels.min() >= 0 and labels.max() <= 8
        values, counts = np.unique(labels, return_counts=True)
        result = {'format': image.format, 'mode': image.mode, 'size': list(image.size),
                  'labels': {str(int(v)): int(c) for v, c in zip(values, counts)}}
    (WORK / 'prediction_check.json').write_text(json.dumps(result, indent=2) + '\n')
    log('prediction_check_passed', **result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['environment', 'input', 'prediction'])
    args = parser.parse_args()
    if args.phase:
        return {'environment': environment, 'input': create_input,
                'prediction': prediction_check}[args.phase]()

    WORK.mkdir(exist_ok=True)
    os.chdir(PROJECT)
    started = time.monotonic()
    result = {'status': 'running', 'device': 'cpu', 'dataset': 'synthetic',
              'source_archive': 'server2-system-no-dataset-20260926.rootfs.tar.gz'}
    commands = [
        [sys.executable, '-u', __file__, '--phase', 'environment'],
        [sys.executable, '-u', __file__, '--phase', 'input'],
        [sys.executable, '-u', str(PROJECT / 'experiment/mask2former_uav/infer_h3_hyperseg.py'),
         '--checkpoint', str(CHECKPOINT), '--input', str(WORK / 'synthetic_input'),
         '--output', str(WORK / 'predictions'), '--size', '512', '--overlap', '0.5'],
        [sys.executable, '-u', str(PROJECT / 'tools/check_submission.py'),
         str(WORK / 'predictions'), str(WORK / 'synthetic_input')],
        [sys.executable, '-u', __file__, '--phase', 'prediction'],
        [sys.executable, str(PROJECT / 'experiment/mask2former_uav/train_h3_hyperseg.py'), '--help'],
    ]
    try:
        for index, command in enumerate(commands, 1):
            log('command_started', index=index, command=command)
            child = subprocess.Popen(command)
            while True:
                try:
                    code = child.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    log('command_running', index=index,
                        elapsed_seconds=round(time.monotonic() - started))
            if code:
                raise RuntimeError(f'Command {index} failed with exit code {code}')
            log('command_passed', index=index)
        result.update(status='passed', commands_passed=len(commands),
                      checkpoint_strict_load='passed via actual inference entry point',
                      inference='passed: 1024x1024 input, 512 window, 0.5 overlap',
                      training_entry_import='passed (--help only)',
                      gpu_validation='not performed: no NVIDIA GPU on this Windows host',
                      accuracy_validation='not performed: synthetic input has no ground truth')
    except BaseException as error:
        result.update(status='failed', error=str(error))
        raise
    finally:
        result.update(elapsed_seconds=round(time.monotonic() - started),
                      finished_utc=datetime.now(timezone.utc).isoformat())
        (WORK / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        log('validation_finished', **result)


if __name__ == '__main__':
    main()
