# DATA ACCESS — what is and is not in this package

## Why no images ship with the submission

The competition rules forbid leaking or redistributing the dataset, in the package
and inside the image alike. So:

* **no competition image or mask is included anywhere**, and the root filesystem
  snapshot in `04_镜像` was exported with the dataset directory explicitly excluded
  (verified by scanning the archive: zero members under `/root/hyperseg/dataset`);
* what *does* ship is the **split manifest** — three plain lists of file names —
  because that is what makes the split reproducible without carrying any pixels.

## Official layout after you download the data

```
low_altitude_2026/
├── train/images/          6996 labelled images (1024x1024 RGB PNG)
├── train/masks/           6996 single-channel masks, values 0..8, 0 = Ignore
├── images/                500 unlabelled competition images (round one test set)
└── test_2/images/         1300 unlabelled competition images (round two test set)
```

`Label.txt` lists the class names; `stratified_split_90_10.json` is the organisers'
own stratified split and is **not** the split this project uses.

## The split this project uses

`06_数据说明/splits/{train,val,test}.txt`, one bare file stem per line (for example
`test1_123`, no extension). Image and mask names must match exactly.

| file | ids | role |
| --- | --- | --- |
| `train.txt` | 5598 | parameter learning |
| `val.txt` | 699 | checkpoint selection (this is what `0.7420` is measured on) |
| `test.txt` | 699 | offline evaluation only, never used for model selection |

The three sets are pairwise disjoint and their union is the 6996 labelled images.

Verify after staging:

```bash
cd 06_数据说明
wc -l splits/*.txt                       # 5598 / 699 / 699
sha256sum splits/*.txt
```

Recorded SHA-256 (identical on the development machine and inside the image):

| file | sha256 |
| --- | --- |
| `train.txt` | `4d19ea1ece87be870f17fb04d760475f68a5344cfcb674386e1f37c5fc484622` |
| `val.txt` | `39d349bffd31a2dd7bdccd03dac3225a5a30aa11c5e2c97fe7f0367a7ef940f8` |
| `test.txt` | `09068e956dd175da0f417ba590844876ea542ec7f210261f5647272b34635e4f` |

## Two different things both called "test"

This trips people up, so to be explicit:

| name | what it is | can you compute mIoU on it? |
| --- | --- | --- |
| `runs/splits/test.txt` | 699 **labelled** images held out from the training data | yes — offline evaluation |
| `low_altitude_2026/test_2/images` | 1300 **unlabelled** round-two images | **no** — no ground truth exists |

Every mIoU in this package (74.2023 % recorded, 74.2023 % replayed) is measured on
the labelled validation split. The round-two export produced numbers only because
the export script statically checks image count, sizes and the output format — it
computes no accuracy, and claiming one would be dishonest.

## Class palette

| id | class | counted in mIoU |
| --- | --- | --- |
| 0 | Ignore | no |
| 1 | Background | yes |
| 2 | Building | yes |
| 3 | Road | yes |
| 4 | Water | yes |
| 5 | Barren | yes |
| 6 | Vegetation | yes |
| 7 | Agricultural | yes |
| 8 | Vehicle | yes |

Masks and predictions are both written as single-channel 8-bit grey PNG; no colour
palette is used, and the manual forbids it explicitly.
