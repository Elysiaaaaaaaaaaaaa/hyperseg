# ENVIRONMENT

Environment of the machine that produced the submitted predictions, plus what it
takes to rebuild it.

## Recorded versions (server3)

| item | value |
| --- | --- |
| OS | Ubuntu 22.04.3 LTS, linux/amd64 |
| Python | 3.12.3 |
| venv | `/root/hyperseg/.venv-mask2former` |
| GPU | NVIDIA GeForce RTX 4090, 24 GiB |
| NVIDIA driver | 580.76.05 |
| CUDA (torch build) | 12.1 |
| cuDNN | 8.9.2 (build 8902) |
| torch | 2.3.0+cu121 |
| torchvision | 0.18.0+cu121 |
| transformers | 4.57.6 |
| tokenizers | 0.22.2 |
| huggingface_hub | 0.36.2 |
| safetensors | 0.8.0 |
| numpy | 2.5.3 |
| pillow | 10.3.0 |
| opencv-python | 5.0.0.93 |

`docker/requirements.lock.txt` is the raw `pip freeze` (179 entries);
`docker/requirements.docker.txt` is the installable subset, with torch pulled from
the cu121 wheel index.

## Three ways to get a working environment

1. **Root filesystem snapshot (preferred, verified).** `04_镜像` ships a
   `docker import`-able snapshot of the machine above. Everything is already
   installed and the project lives at `/root/hyperseg`.
2. **`Dockerfile` (rebuild).** Bakes an equivalent environment from public
   packages. Not yet executed end to end in a clean Docker daemon — see the
   caveat below.
3. **Manual venv.** Ubuntu 22.04 + Python 3.12, then
   `pip install -r docker/requirements.docker.txt` with
   `--extra-index-url https://download.pytorch.org/whl/cu121`.

## Two things that will bite you

**1. `OMP_NUM_THREADS` on CPU-only hosts.** torch defaults to one thread per
*visible* core. server3 exposes 128 cores but its no-GPU cgroup quota is 0.5 core
with 2 GiB RAM, and under that mismatch the threads fight each other: the same
`--check-only` preflight takes **15 s with `OMP_NUM_THREADS=2`** and roughly
**4 minutes** with the default. The export scripts set it explicitly.

**2. `transformers` major version.** `models/hyperseg_b3_best.pt` was produced by a
transformers 5 release, whose SegFormer backbone names its tensors
`encoder.backbone.stages[i].blocks[j]...`. transformers 4 (which this environment
uses) expects `encoder.backbone.encoder.block[i][j]...`, so a plain
`load_state_dict` fails with **628 missing keys**. `hyperseg_uav.model.translate_legacy_keys`
renames them in memory before loading; the checkpoint file itself is untouched and
keeps its SHA-256.

Related trap: `tools/test_hyperseg.py` loads with `strict=False`, so before the
translation was added it silently dropped the whole backbone and still printed a
number. It now translates first, but the `strict=False` remains — check the
`missing=`/`unexpected=` line it prints, and treat a non-zero count as a failure.

## Common errors

| symptom | cause | fix |
| --- | --- | --- |
| `Missing key(s) ... encoder.backbone.encoder.block.0.0...` | loading the checkpoint without translation | route through `translate_legacy_keys` (as `tools/infer_hyperseg.py` does) |
| `RuntimeError: No GPU allocated` | launched on a CPU-only container | enable GPU mode, or pass `--check-only` for a CPU preflight |
| a run that seems to hang for minutes | thread thrashing under a low CPU quota | `export OMP_NUM_THREADS=4` |
| `Expected 1300 round-two images, found N` | wrong `--input` directory | point at `test_2/images` |
