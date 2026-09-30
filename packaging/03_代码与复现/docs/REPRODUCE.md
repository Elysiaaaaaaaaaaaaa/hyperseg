# REPRODUCE

What can be reproduced exactly, what cannot, and how to tell the two apart.

## Summary

| claim | status | how to verify |
| --- | --- | --- |
| The submitted masks come from `models/hyperseg_b3_best.pt` | **exact** | re-export and compare per-image equality with `01_预测结果/predictions_test2_round2.zip` |
| The checkpoint loads correctly despite the transformers version gap | **exact** | `translate_legacy_keys` returns 628; `load_state_dict` is strict and raises on any mismatch |
| The checkpoint's recorded validation mIoU is real | **exact** | `scripts/03_eval.sh` → 0.7420232 vs recorded 0.7420227 (5.5e-7) |
| The export passes the manual's format rules | **exact** | `scripts/05_check_submission.sh` |
| Retraining reproduces the same checkpoint | **no** | not claimed; see below |

## Exact path (inference)

```bash
# 1. environment: either import ../04_镜像 or build the Dockerfile
#    (both are entered at the project root 03_代码与复现/hyperseg, i.e. /root/hyperseg)
# 2. data: mount the official low_altitude_2026 read-only
# 3. preflight — must print translated_legacy_encoder_keys: 628
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python experiment/b3_test2_20260929/run_inference.py --check-only

# 4. replay the recorded validation mIoU
../scripts/03_eval.sh /path/to/low_altitude_2026
#    expect replayed 0.7420232149377345 vs recorded 0.7420226665631505

# 5. export and compare with the archive in ../01_预测结果
../scripts/04_infer.sh /path/to/low_altitude_2026
```

Steps 1–5 are run from `03_代码与复现/hyperseg/`; `scripts/` sits one level up, hence
`../scripts/`. The scripts themselves are location-independent.

Inside the imported image, export `PYTHONPATH=/root/hyperseg` before step 3: the copy
of `run_inference.py` baked into the image predates the `sys.path` bootstrap that the
package copy carries, so without it `import hyperseg_uav` raises `ModuleNotFoundError`.
The verification run recorded in `05_实验证据` was executed exactly that way.

Step 4 is the load-bearing one. If the key translation were misaligned — one tensor
attached to the wrong module — the mIoU would collapse instead of landing within
5.5e-7 of the recorded value. `experiment/b3_test2_20260929/val_sanity.py` prints
both numbers and their difference, and `05_实验证据/logs/val_sanity.json` holds the
recorded run.

Step 5 should reproduce the archive bit for bit in the pixel data. The archive
itself is not byte-comparable (zip timestamps and compression differ), so compare
the PNG payloads, e.g.:

```bash
unzip -qq -d /tmp/ref  01_预测结果/predictions_test2_round2.zip
unzip -qq -d /tmp/new  runs/b3_test2_reproduce/predictions.zip
diff -r /tmp/ref /tmp/new && echo "pixels identical"
```

## Expected numbers

| quantity | value |
| --- | --- |
| images | 1300 |
| input size | 1024×1024 |
| output | single-channel 8-bit grey PNG, same size, values in 1..8 |
| archive bytes | 12 655 462 |
| archive SHA-256 | `ff0a2fb0e0580bafb1b7f41f82d29861ad42bf181689f7efff43df3e84b0ef8a` |
| checkpoint SHA-256 | `fb124302bafaec18aaa3f695dc021efead7bd00f710348d040888bb270148968` |
| recorded validation mIoU | 0.7420226665631505 (74.2023 %) |
| replayed validation mIoU | 0.7420232149377345 |
| export wall clock | 224.83 s on an RTX 4090 |

Class share of the submitted predictions: background 33.94 %, vegetation 20.94 %,
building 18.52 %, agricultural 10.30 %, water 7.34 %, road 6.89 %, barren 1.45 %,
vehicle 0.60 %. Class 0 (Ignore) is never predicted, which is correct — it is a
label-only class.

## What is *not* reproducible

**Retraining.** The checkpoint stores only `model`, `model_config`, `epoch` and
`val_miou`. It does not record the argument vector of the run that produced it.
`--epochs` certainly differed from the 60-epoch default, because `epoch` is 141, but
the exact schedule cannot be recovered from the artifact, and the original training
log was not kept in the repository. A retrain with the documented defaults
(`configs/train_config.json`) will land near, not exactly on, this checkpoint.

This is stated rather than papered over: the submitted predictions are fully
reproducible from the shipped weights, and the weights' provenance (epoch, validation
score, architecture, hash) is fully recorded. What is missing is the training
recipe's provenance.

**TTA.** `tools/infer_hyperseg.py` supports `--tta` (a single horizontal flip, not a
multi-model ensemble). It was **not** used. Adding it produces different masks.

## Residual risks

* `tools/test_hyperseg.py` still calls `load_state_dict(..., strict=False)`. It now
  translates keys first, so a correct checkpoint loads fully, but a wrong checkpoint
  would still only print a `missing=`/`unexpected=` count instead of raising. Read
  that line.
* **The `docker import` path has now been executed** on 2026-09-30 against a real
  Docker daemon: the 13 079 718 593-byte archive imported as `hyperseg-uav:submit`
  (image ID `9b65ec6f9121`, 34.3 GB unpacked on a linux/amd64 host), and the in-container
  preflight printed `translated_legacy_encoder_keys: 628` with the checkpoint loading
  strictly. The transcript is `05_实验证据/logs/container_smoke_20260930.log`; see
  section 6 of `04_镜像/README_IMAGE.md`. What remains untested is only the
  `Dockerfile` rebuild path (step 1b of the quick start), which no claim depends on.
* Only one training seed was ever used for this model, so no seed variance is
  available; `0.7420` is a single-run number on the fixed validation split.
