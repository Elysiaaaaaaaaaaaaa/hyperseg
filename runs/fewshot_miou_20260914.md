# Few-shot mIoU（seed=3407）

数值为 mIoU，范围 0–1；SUIM 测试集为 110 张 TEST 图像。

| 数据集 | 实验 | steps | 1-shot | 2-shot | 5-shot | 10-shot |
|---|---|---:|---:|---:|---:|---:|
| LoveDA | semantic-head | 200 | — | — | — | **0.535868** |
| SUIM | adapter | 200 | 0.202700 | 0.217207 | 0.203407 | 0.184149 |
| SUIM | adapter | 2000 | 0.271339 | 0.333996 | 0.402581 | **0.435610** |
| SUIM | semantic-head | 200 | 0.020888 | 0.021952 | 0.022911 | 0.023910 |
| SUIM | semantic-head | 2000 | 0.165610 | **0.174845** | 0.160630 | 0.162551 |

原始日志和完整 JSON 结果见 `logs/fewshot_20260914/`、`runs/loveda_latest_10shot_semantic_head_200_seed3407/` 和 `runs/suim_fewshot_v1_seed3407/`。
