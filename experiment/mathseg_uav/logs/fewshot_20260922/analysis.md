<!-- FEWSHOT_20260922_START -->
## 2026-09-22 few-shot 服务器运行追踪

更新 UTC：2026-09-22T08:56:58+00:00；实测队列状态：completed (all planned result files present)；当前计划 seed=[3407]，完成 128/128 次，历史全部种子共完成 206 次。
队列最后记录组：{'variant': 'm3', 'dataset': 'loveda'}；原队列状态仅作历史记录，完成情况以结果文件为准。本节由 sync_fewshot_logs.py 自动维护。

LoveDA：原始 1..7→0..6，原始 0/255→255 Ignore，7 通道；SUIM：8 类全部有效。
同一数据集复用固定支持集；最终步评估，不使用评估集选 checkpoint。
下表按当前计划的种子汇总；单种子结果不代表跨种子稳定性。

| 数据集 | 模型 | 设置 | K | seeds | mIoU % | 标准差 pp | 相对 M0 pp |
|---|---|---|---:|---:|---:|---:|---:|
| loveda | m0 | adapter_200 | 1 | 1 | 53.6099 | — | +0.0000 |
| loveda | m0 | adapter_200 | 2 | 1 | 52.0685 | — | +0.0000 |
| loveda | m0 | adapter_200 | 5 | 1 | 53.2180 | — | +0.0000 |
| loveda | m0 | adapter_200 | 10 | 1 | 54.2450 | — | +0.0000 |
| loveda | m0 | adapter_2000 | 1 | 1 | 50.9653 | — | +0.0000 |
| loveda | m0 | adapter_2000 | 2 | 1 | 52.2592 | — | +0.0000 |
| loveda | m0 | adapter_2000 | 5 | 1 | 53.5632 | — | +0.0000 |
| loveda | m0 | adapter_2000 | 10 | 1 | 55.9167 | — | +0.0000 |
| loveda | m0 | semantic_head_200 | 1 | 1 | 53.5595 | — | +0.0000 |
| loveda | m0 | semantic_head_200 | 2 | 1 | 53.5214 | — | +0.0000 |
| loveda | m0 | semantic_head_200 | 5 | 1 | 53.5147 | — | +0.0000 |
| loveda | m0 | semantic_head_200 | 10 | 1 | 53.5722 | — | +0.0000 |
| loveda | m0 | semantic_head_2000 | 1 | 1 | 54.2025 | — | +0.0000 |
| loveda | m0 | semantic_head_2000 | 2 | 1 | 53.5228 | — | +0.0000 |
| loveda | m0 | semantic_head_2000 | 5 | 1 | 53.9736 | — | +0.0000 |
| loveda | m0 | semantic_head_2000 | 10 | 1 | 54.1877 | — | +0.0000 |
| loveda | m1 | adapter_200 | 1 | 1 | 53.6349 | — | +0.0250 |
| loveda | m1 | adapter_200 | 2 | 1 | 52.0800 | — | +0.0116 |
| loveda | m1 | adapter_200 | 5 | 1 | 53.2266 | — | +0.0086 |
| loveda | m1 | adapter_200 | 10 | 1 | 54.2562 | — | +0.0112 |
| loveda | m1 | adapter_2000 | 1 | 1 | 50.9753 | — | +0.0100 |
| loveda | m1 | adapter_2000 | 2 | 1 | 52.2672 | — | +0.0080 |
| loveda | m1 | adapter_2000 | 5 | 1 | 53.5530 | — | -0.0102 |
| loveda | m1 | adapter_2000 | 10 | 1 | 55.9286 | — | +0.0119 |
| loveda | m1 | semantic_head_200 | 1 | 1 | 53.5694 | — | +0.0098 |
| loveda | m1 | semantic_head_200 | 2 | 1 | 53.5309 | — | +0.0095 |
| loveda | m1 | semantic_head_200 | 5 | 1 | 53.5239 | — | +0.0092 |
| loveda | m1 | semantic_head_200 | 10 | 1 | 53.5795 | — | +0.0073 |
| loveda | m1 | semantic_head_2000 | 1 | 1 | 54.2064 | — | +0.0039 |
| loveda | m1 | semantic_head_2000 | 2 | 1 | 53.5270 | — | +0.0042 |
| loveda | m1 | semantic_head_2000 | 5 | 1 | 53.9776 | — | +0.0040 |
| loveda | m1 | semantic_head_2000 | 10 | 1 | 54.1909 | — | +0.0032 |
| loveda | m2 | adapter_200 | 1 | 1 | 53.3276 | — | -0.2823 |
| loveda | m2 | adapter_200 | 2 | 1 | 52.2721 | — | +0.2037 |
| loveda | m2 | adapter_200 | 5 | 1 | 53.4083 | — | +0.1903 |
| loveda | m2 | adapter_200 | 10 | 1 | 54.4553 | — | +0.2103 |
| loveda | m2 | adapter_2000 | 1 | 1 | 50.2012 | — | -0.7641 |
| loveda | m2 | adapter_2000 | 2 | 1 | 52.5243 | — | +0.2651 |
| loveda | m2 | adapter_2000 | 5 | 1 | 53.7070 | — | +0.1439 |
| loveda | m2 | adapter_2000 | 10 | 1 | 56.0366 | — | +0.1199 |
| loveda | m2 | semantic_head_200 | 1 | 1 | 53.1727 | — | -0.3868 |
| loveda | m2 | semantic_head_200 | 2 | 1 | 53.1604 | — | -0.3609 |
| loveda | m2 | semantic_head_200 | 5 | 1 | 53.1677 | — | -0.3470 |
| loveda | m2 | semantic_head_200 | 10 | 1 | 53.2086 | — | -0.3636 |
| loveda | m2 | semantic_head_2000 | 1 | 1 | 53.9564 | — | -0.2461 |
| loveda | m2 | semantic_head_2000 | 2 | 1 | 53.4557 | — | -0.0671 |
| loveda | m2 | semantic_head_2000 | 5 | 1 | 53.7408 | — | -0.2328 |
| loveda | m2 | semantic_head_2000 | 10 | 1 | 53.9914 | — | -0.1963 |
| loveda | m3 | adapter_200 | 1 | 1 | 53.3049 | — | -0.3050 |
| loveda | m3 | adapter_200 | 2 | 1 | 52.2575 | — | +0.1890 |
| loveda | m3 | adapter_200 | 5 | 1 | 53.4374 | — | +0.2194 |
| loveda | m3 | adapter_200 | 10 | 1 | 54.4718 | — | +0.2268 |
| loveda | m3 | adapter_2000 | 1 | 1 | 50.2827 | — | -0.6826 |
| loveda | m3 | adapter_2000 | 2 | 1 | 52.4255 | — | +0.1663 |
| loveda | m3 | adapter_2000 | 5 | 1 | 53.6510 | — | +0.0879 |
| loveda | m3 | adapter_2000 | 10 | 1 | 56.0184 | — | +0.1017 |
| loveda | m3 | semantic_head_200 | 1 | 1 | 53.1988 | — | -0.3607 |
| loveda | m3 | semantic_head_200 | 2 | 1 | 53.1902 | — | -0.3312 |
| loveda | m3 | semantic_head_200 | 5 | 1 | 53.1978 | — | -0.3170 |
| loveda | m3 | semantic_head_200 | 10 | 1 | 53.2378 | — | -0.3344 |
| loveda | m3 | semantic_head_2000 | 1 | 1 | 53.9609 | — | -0.2416 |
| loveda | m3 | semantic_head_2000 | 2 | 1 | 53.4850 | — | -0.0378 |
| loveda | m3 | semantic_head_2000 | 5 | 1 | 53.7689 | — | -0.2047 |
| loveda | m3 | semantic_head_2000 | 10 | 1 | 54.0178 | — | -0.1699 |
| suim | m0 | adapter_200 | 1 | 1 | 8.7447 | — | +0.0000 |
| suim | m0 | adapter_200 | 2 | 1 | 13.8163 | — | +0.0000 |
| suim | m0 | adapter_200 | 5 | 1 | 22.6809 | — | +0.0000 |
| suim | m0 | adapter_200 | 10 | 1 | 19.0294 | — | +0.0000 |
| suim | m0 | adapter_2000 | 1 | 1 | 24.0393 | — | +0.0000 |
| suim | m0 | adapter_2000 | 2 | 1 | 35.2833 | — | +0.0000 |
| suim | m0 | adapter_2000 | 5 | 1 | 42.7040 | — | +0.0000 |
| suim | m0 | adapter_2000 | 10 | 1 | 49.6607 | — | +0.0000 |
| suim | m0 | semantic_head_200 | 1 | 1 | 1.0853 | — | +0.0000 |
| suim | m0 | semantic_head_200 | 2 | 1 | 1.0716 | — | +0.0000 |
| suim | m0 | semantic_head_200 | 5 | 1 | 1.1936 | — | +0.0000 |
| suim | m0 | semantic_head_200 | 10 | 1 | 1.2450 | — | +0.0000 |
| suim | m0 | semantic_head_2000 | 1 | 1 | 7.7005 | — | +0.0000 |
| suim | m0 | semantic_head_2000 | 2 | 1 | 16.6973 | — | +0.0000 |
| suim | m0 | semantic_head_2000 | 5 | 1 | 19.7174 | — | +0.0000 |
| suim | m0 | semantic_head_2000 | 10 | 1 | 18.7913 | — | +0.0000 |
| suim | m1 | adapter_200 | 1 | 1 | 19.5531 | — | +10.8084 |
| suim | m1 | adapter_200 | 2 | 1 | 21.9327 | — | +8.1164 |
| suim | m1 | adapter_200 | 5 | 1 | 32.0343 | — | +9.3535 |
| suim | m1 | adapter_200 | 10 | 1 | 27.0278 | — | +7.9985 |
| suim | m1 | adapter_2000 | 1 | 1 | 27.2384 | — | +3.1991 |
| suim | m1 | adapter_2000 | 2 | 1 | 35.6439 | — | +0.3607 |
| suim | m1 | adapter_2000 | 5 | 1 | 42.5488 | — | -0.1552 |
| suim | m1 | adapter_2000 | 10 | 1 | 49.8313 | — | +0.1706 |
| suim | m1 | semantic_head_200 | 1 | 1 | 6.1065 | — | +5.0212 |
| suim | m1 | semantic_head_200 | 2 | 1 | 6.1379 | — | +5.0663 |
| suim | m1 | semantic_head_200 | 5 | 1 | 6.1402 | — | +4.9466 |
| suim | m1 | semantic_head_200 | 10 | 1 | 6.1645 | — | +4.9195 |
| suim | m1 | semantic_head_2000 | 1 | 1 | 15.4783 | — | +7.7778 |
| suim | m1 | semantic_head_2000 | 2 | 1 | 20.1131 | — | +3.4158 |
| suim | m1 | semantic_head_2000 | 5 | 1 | 22.6879 | — | +2.9705 |
| suim | m1 | semantic_head_2000 | 10 | 1 | 20.7270 | — | +1.9357 |
| suim | m2 | adapter_200 | 1 | 1 | 11.7889 | — | +3.0442 |
| suim | m2 | adapter_200 | 2 | 1 | 16.3884 | — | +2.5721 |
| suim | m2 | adapter_200 | 5 | 1 | 17.4047 | — | -5.2761 |
| suim | m2 | adapter_200 | 10 | 1 | 25.0570 | — | +6.0276 |
| suim | m2 | adapter_2000 | 1 | 1 | 25.0558 | — | +1.0166 |
| suim | m2 | adapter_2000 | 2 | 1 | 34.3009 | — | -0.9824 |
| suim | m2 | adapter_2000 | 5 | 1 | 41.0923 | — | -1.6117 |
| suim | m2 | adapter_2000 | 10 | 1 | 49.9326 | — | +0.2719 |
| suim | m2 | semantic_head_200 | 1 | 1 | 5.3301 | — | +4.2448 |
| suim | m2 | semantic_head_200 | 2 | 1 | 5.1769 | — | +4.1053 |
| suim | m2 | semantic_head_200 | 5 | 1 | 5.8172 | — | +4.6237 |
| suim | m2 | semantic_head_200 | 10 | 1 | 5.4758 | — | +4.2308 |
| suim | m2 | semantic_head_2000 | 1 | 1 | 10.6773 | — | +2.9768 |
| suim | m2 | semantic_head_2000 | 2 | 1 | 12.8206 | — | -3.8767 |
| suim | m2 | semantic_head_2000 | 5 | 1 | 18.7612 | — | -0.9562 |
| suim | m2 | semantic_head_2000 | 10 | 1 | 19.3797 | — | +0.5884 |
| suim | m3 | adapter_200 | 1 | 1 | 11.8626 | — | +3.1180 |
| suim | m3 | adapter_200 | 2 | 1 | 16.1748 | — | +2.3586 |
| suim | m3 | adapter_200 | 5 | 1 | 17.4112 | — | -5.2696 |
| suim | m3 | adapter_200 | 10 | 1 | 24.8155 | — | +5.7861 |
| suim | m3 | adapter_2000 | 1 | 1 | 25.0430 | — | +1.0037 |
| suim | m3 | adapter_2000 | 2 | 1 | 33.9782 | — | -1.3051 |
| suim | m3 | adapter_2000 | 5 | 1 | 41.1429 | — | -1.5611 |
| suim | m3 | adapter_2000 | 10 | 1 | 49.7929 | — | +0.1322 |
| suim | m3 | semantic_head_200 | 1 | 1 | 5.2369 | — | +4.1516 |
| suim | m3 | semantic_head_200 | 2 | 1 | 5.0655 | — | +3.9939 |
| suim | m3 | semantic_head_200 | 5 | 1 | 5.6764 | — | +4.4828 |
| suim | m3 | semantic_head_200 | 10 | 1 | 5.2924 | — | +4.0474 |
| suim | m3 | semantic_head_2000 | 1 | 1 | 10.4255 | — | +2.7250 |
| suim | m3 | semantic_head_2000 | 2 | 1 | 12.4002 | — | -4.2971 |
| suim | m3 | semantic_head_2000 | 5 | 1 | 18.6849 | — | -1.0325 |
| suim | m3 | semantic_head_2000 | 10 | 1 | 19.0645 | — | +0.2732 |

日志和逐次指标：`logs/fewshot_20260922/`。压缩权重留在服务器，源 UAV checkpoint 必须保留。
<!-- FEWSHOT_20260922_END -->
