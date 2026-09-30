# OPS — commands, parameters, cost

Two working directories are used, and mixing them up is the easiest way to get a
`ModuleNotFoundError`:

* **package code root** `03_代码与复现/` — for the numbered step scripts
  (`scripts/0X_*.sh`). Each script resolves its own location via `$(dirname "$0")`
  and `cd`s into `hyperseg/` itself, so it does not matter which directory you
  invoke it from.
* **project root** `03_代码与复现/hyperseg/` — for the direct `python ...` commands
  below. From the package root that is `cd 03_代码与复现/hyperseg`.

`$DATA` is the mounted competition data root (`low_altitude_2026`).

> **PYTHONPATH.** `python experiment/b3_test2_20260929/run_inference.py` puts only the
> script's own directory on `sys.path`, not the project root, so `import hyperseg_uav`
> fails unless the root is importable. The shipped copy of `run_inference.py`
> bootstraps `sys.path` itself, as do `val_sanity.py` and `tools/infer_hyperseg.py`.
> The copy **inside the image** is one revision older and does not, so inside the
> imported container export `PYTHONPATH=/root/hyperseg` first (this is exactly how the
> verification run was executed). The step scripts export it for you.

## 0. Preflight (no GPU needed, ~15 s with the thread cap)

```bash
cd 03_代码与复现/hyperseg
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python experiment/b3_test2_20260929/run_inference.py --check-only
```

Prints a JSON block. Must contain:

```
"checkpoint_sha256": "fb124302bafaec18aaa3f695dc021efead7bd00f710348d040888bb270148968"
"checkpoint_epoch": 141
"translated_legacy_encoder_keys": 628
"image_count": 1300
```

`translated_legacy_encoder_keys` is the important one: 0 means the backbone was
not translated, and the load would have failed or been silently incomplete.

## 1. Rebuild the working data layout

Steps 1–5 run `scripts/0X_*.sh`, i.e. paths relative to the **package code root**
`03_代码与复现/`. All of them locate themselves via `$(dirname "$0")`, so the invoking
directory does not matter; 02–05 additionally `cd` into `hyperseg/` and export
`PYTHONPATH` for you.

```bash
scripts/01_prepare_data.sh "$DATA"
```

## 2. Reproduce the recorded validation mIoU

```bash
scripts/03_eval.sh "$DATA"
```

Expect `replayed_val_miou ≈ 0.7420232` against `recorded_val_miou = 0.7420227`
(absolute difference 5.5e-7). Cost: 699 images, 768×768, batch 2, ~29 s on an
RTX 4090.

## 3. Export the round-two predictions

```bash
scripts/04_infer.sh "$DATA"
```

| parameter | value |
| --- | --- |
| `--size` | 768 (matches training resolution) |
| `--overlap` | 0.5 |
| `--tta` | not passed (off) |
| precision | fp32 |
| input | 1300 images, 1024×1024 |
| observed cost | 224.83 s on an RTX 4090 → 5.78 images/s, 4 windows per image |
| peak VRAM | ~3 GB (batch 1 per window) |

Do not add TTA to reproduce the submission: the submitted archive was produced
without it, and turning it on changes the masks.

## 4. Check the export

```bash
scripts/05_check_submission.sh "$DATA"
```

Expect `OK: 1300 predictions`. The stricter block-level review (PNG chunk parsing,
palette detection, member-list diff against the official test-set archive) produced
`PASS: zip matches the manual section 13 requirements` for the submitted archive;
its full output is in `05_实验证据/logs/manual_compliance_20260929.json`.

## 5. Training (not needed to reproduce the submission)

```bash
scripts/01_prepare_data.sh "$DATA"
scripts/02_train.sh
```

60 epochs with the code defaults is roughly 12 h on an RTX 4090 at 768×768,
batch 2, fp32. The submitted checkpoint is epoch 141, i.e. the original run used a
larger `--epochs`; see `REPRODUCE.md` for what is and is not recoverable.

## Cost summary for the submitted run

| step | wall clock | device |
| --- | --- | --- |
| preflight | 15 s (CPU, capped threads) | CPU |
| validation replay | 28.76 s | RTX 4090 |
| export 1300 images | 224.83 s | RTX 4090 |
| submission check | 2 s | CPU |
