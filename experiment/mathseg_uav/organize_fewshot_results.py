"""Audit and summarize the single-seed MathSeg SUIM/LoveDA results already downloaded."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = ('adapter_200', 'adapter_2000', 'semantic_head_200', 'semantic_head_2000')
VARIANTS = ('m0', 'm1', 'm2', 'm3')
SHOTS = (1, 2, 5, 10)
START = '<!-- FEWSHOT_SINGLE_SEED_REVIEW_START -->'
END = '<!-- FEWSHOT_SINGLE_SEED_REVIEW_END -->'


def write_csv(path, rows, fieldnames=None):
    with path.open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(root, seed=3407):
    results, classes, missing, by_key, hashes = [], [], [], {}, {}
    evaluation_pixels = {}
    for dataset in ('suim', 'loveda'):
        for variant in VARIANTS:
            for setting in SETTINGS:
                mode = 'adapter' if setting.startswith('adapter') else 'semantic-head'
                steps = int(setting.rsplit('_', 1)[1])
                for k in SHOTS:
                    folder = root / dataset / variant / setting / f'{k}shot_seed{seed}'
                    if dataset == 'loveda':
                        nested = folder / f'{mode}_{k}shot_seed{seed}'
                        assert not ((folder / 'summary.json').exists() and (nested / 'summary.json').exists()), folder
                        if not (folder / 'summary.json').exists():
                            folder = nested
                    summary = folder / 'summary.json'
                    if not summary.exists():
                        history = folder / 'history.jsonl'
                        last_step = 0
                        if history.exists():
                            for line in history.read_text().splitlines():
                                try:
                                    last_step = json.loads(line)['step']
                                except json.JSONDecodeError:
                                    pass
                        missing.append(dict(dataset=dataset, variant=variant, setting=setting, shots=k,
                            seed=seed, last_step=last_step, state='incomplete' if folder.exists() else 'not_started',
                            directory=str(folder.relative_to(root))))
                        continue
                    data = json.loads(summary.read_text())
                    info = json.loads((folder / 'run_config.json').read_text())
                    assert (data['seed'], data['shots_per_class'], data['mode']) == (seed, k, mode), summary
                    assert info['transfer']['variant'].lower() == variant, summary
                    assert info['config']['steps'] == steps, summary
                    hist = [json.loads(line) for line in (folder / 'history.jsonl').read_text().splitlines() if line]
                    assert [row['step'] for row in hist] == list(range(1, steps + 1)), summary
                    metrics = data['metrics']
                    expected_classes = 7 if dataset == 'loveda' else 8
                    assert len(metrics['confusion']) == expected_classes, summary
                    assert len(metrics['per_class_iou']) == expected_classes, summary
                    assert all(len(row) == expected_classes for row in metrics['confusion']), summary
                    matrix = metrics['confusion']
                    pixels = sum(map(sum, matrix))
                    assert pixels > 0
                    evaluation_pixels.setdefault(dataset, set()).add(pixels)
                    ious = []
                    for c in range(expected_classes):
                        union = sum(matrix[c]) + sum(row[c] for row in matrix) - matrix[c][c]
                        if union:
                            ious.append(matrix[c][c] / union)
                    assert abs(statistics.mean(ious)-metrics['mIoU']) < 1e-9, summary
                    assert abs(sum(matrix[c][c] for c in range(expected_classes))/pixels-metrics['pixel_accuracy']) < 1e-9, summary
                    if dataset == 'loveda':
                        assert info['ignore_index'] == 255 and info['label_policy'] == 'standard', summary
                    hashes.setdefault(dataset, set()).add(info.get('protocol_sha256') or info['config']['protocol_sha256'])
                    row = dict(dataset=dataset, variant=variant, setting=setting, shots=k, seed=seed,
                        steps=steps, support_images=data['support_images'], evaluation_images=data['evaluation_images'],
                        miou_percent=metrics['mIoU']*100, pixel_accuracy_percent=metrics['pixel_accuracy']*100,
                        final_loss=hist[-1]['loss'], mean_last20_loss=statistics.mean(x['loss'] for x in hist[-20:]),
                        source_summary=str(summary.relative_to(root)))
                    results.append(row)
                    by_key[dataset, variant, setting, k] = row
                    for name, value in metrics['per_class_iou'].items():
                        classes.append(dict(dataset=dataset, variant=variant, setting=setting, shots=k, seed=seed,
                            class_name=name, iou_percent=None if value is None else value*100))
    assert all(len(values) == 1 for values in hashes.values()), 'Protocol differs between compared runs'
    assert all(len(values) == 1 for values in evaluation_pixels.values()), 'Evaluation pixels differ between compared runs'
    comparisons = []
    for dataset in ('suim', 'loveda'):
        for setting in SETTINGS:
            for k in SHOTS:
                row = dict(dataset=dataset, setting=setting, shots=k, seed=seed)
                for variant in VARIANTS:
                    result = by_key.get((dataset, variant, setting, k))
                    row[variant+'_miou_percent'] = result['miou_percent'] if result else None
                for first, second in (('m1', 'm0'), ('m2', 'm0'), ('m3', 'm2'), ('m3', 'm0')):
                    a, b = row[first+'_miou_percent'], row[second+'_miou_percent']
                    row[first+'_minus_'+second+'_pp'] = None if a is None or b is None else a-b
                comparisons.append(row)
    output = root / 'organized_seed3407'
    output.mkdir(exist_ok=True)
    write_csv(output / 'results.csv', results)
    write_csv(output / 'comparison.csv', comparisons)
    write_csv(output / 'per_class_iou.csv', classes)
    write_csv(output / 'remaining.csv', missing,
              ['dataset', 'variant', 'setting', 'shots', 'seed', 'last_step', 'state', 'directory'])
    health = json.loads((root / 'server_health.json').read_text()) if (root / 'server_health.json').exists() else {}
    audit = dict(updated_utc=datetime.now(timezone.utc).isoformat(), seed=seed, completed=len(results), expected=128,
                 missing=missing, protocol_hashes={k: next(iter(v)) for k, v in hashes.items()}, server_health=health,
                 checks='run identity, variant, budget, consecutive history steps, confusion shape, class count, protocol, ignore policy, recomputed mIoU/accuracy and identical evaluation pixel counts')
    (output / 'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    complete = len(results) == 128 and not missing
    title = '最终总结' if complete else '阶段报告'
    lines = [f'# MathSeg few-shot 单种子{title}', '', f'seed={seed}；完成 {len(results)}/128；生成 UTC：{audit["updated_utc"]}。', '',
        '只汇总 seed=3407，历史 3408/3409 不进入对比。每次最终步评估，未用评估集选择 checkpoint。',
        'LoveDA 原始标签 1..7 映射到 0..6，原始 0/255 均映射到 Ignore=255，不参与训练损失和评估；固定 1599 张 Val 留出集。SUIM 为 8 类全部有效，官方 TEST 110 张。',
        'SUIM 使用随机语义头，LoveDA 使用 UAV 语义映射头，两个数据集的绝对成绩不作横向比较。', '',
        '## 完成情况', '', '| 数据集 | M0 | M1 | M2 | M3 |', '|---|---:|---:|---:|---:|']
    for dataset in ('suim', 'loveda'):
        counts = [sum(r['dataset']==dataset and r['variant']==v for r in results) for v in VARIANTS]
        lines.append('| '+dataset+' | '+' | '.join(f'{n}/16' for n in counts)+' |')
    if health.get('queue_alive') is False:
        lines += ['', '原队列进程已退出，queue_status.json 的 running 是残留状态。',
                  '128 项均有最终指标，训练步数及混淆矩阵复算检查通过，以结果文件确认全部完成。' if complete
                  else '仍有缺失结果，尚不能认为全部完成。']
    lines += ['', '## 受控对比', '', '下表差值单位为百分点。均值仅是列出的配置组合的描述性均值，不是多种子均值或显著性检验。', '',
              '| 数据集 | 对比 | 完成配对数 | 平均差 pp | 提升次数 |', '|---|---|---:|---:|---:|']
    for dataset in ('suim','loveda'):
        for pair in ('m1_minus_m0_pp', 'm2_minus_m0_pp', 'm3_minus_m2_pp'):
            values = [r[pair] for r in comparisons if r['dataset']==dataset and r[pair] is not None]
            if values:
                lines.append(f'| {dataset} | {pair} | {len(values)} | {statistics.mean(values):+.4f} | {sum(v>0 for v in values)}/{len(values)} |')
    lines += ['', '## 结果解读', '',
        '1. 数学先验参与尺度融合（M3 对 M2）尚未显示稳定收益。应按相同数据集、K、更新范围和预算逐行比较，不能用不同配置的最高分替代消融。',
        '2. SUIM 的 M1 对 M0 数值优势不能直接归因于数学先验：load_mathseg 在构建原模型之后才随机初始化目标头，'
        '不同变体构造时消耗的随机数不同，同 seed 不保证目标头初始化相同；两者还继承各自的 UAV 训练权重。'
        'M2/M3 结构与参数形状一致，更适合检查先验输入的影响。当前结果如实记录，不改变已完成实验。', '']
    for dataset in ('suim', 'loveda'):
        complete = [by_key[dataset, v, 'adapter_2000', 10]['miou_percent'] for v in VARIANTS]
        gains = [by_key[dataset, v, 'adapter_2000', 1]['miou_percent']-
                 by_key[dataset, v, 'adapter_200', 1]['miou_percent'] for v in VARIANTS]
        lines.append(f'- {dataset}：10-shot/adapter/2000 步的四模型 mIoU 为 {min(complete):.4f}～{max(complete):.4f}%；'
                     f'1-shot 下将 adapter 从 200 增至 2000 步，四模型平均变化 {statistics.mean(gains):+.4f} 个百分点。')
    lines += ['', 'LoveDA 1-shot 增加训练预算后下降，与小支持集过度适配相符，但没有独立验证轨迹，不能单凭最终结果确认原因。',
              'LoveDA 的 M1/M0 差异极小；不把单种子的微小差异解释成可靠提升。SUIM 的仅语义头适配明显弱于 adapter，说明当前固定特征加随机头的适配能力有限。']
    lines += ['', '## 完整 mIoU 对照（%）', '', '| 数据集 | 设置 | K | M0 | M1 | M2 | M3 | M3−M2 pp |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in comparisons:
        values = [row[v+'_miou_percent'] for v in VARIANTS]+[row['m3_minus_m2_pp']]
        lines.append(f'| {row["dataset"]} | {row["setting"]} | {row["shots"]} | '+' | '.join('未完成' if v is None else f'{v:.4f}' for v in values)+' |')
    lines += ['', '## 剩余项目', '']
    if complete:
        lines.append('无。128/128 项单种子实验全部完成，未继续运行 seed=3408/3409。')
    for item in missing:
        lines.append(f'- {item["dataset"]}/{item["variant"]}/{item["setting"]}/{item["shots"]}-shot：{item["state"]}，已记录 {item["last_step"]} 步。')
    lines += ['', '结果 CSV、逐类 IoU、剩余清单和审计 JSON 位于 `experiment/mathseg_uav/logs/fewshot_20260922/organized_seed3407/`。原始日志按 dataset/variant/setting/shot_seed 保留在 `logs/fewshot_20260922/`。',
              '单种子只能说明这次固定支持集下的结果；缺失配置不填零、不计入均值。训练损失使用各支持集类别权重，不直接跨 K 比较。']
    report = '\n'.join(lines)+'\n'
    (output/'report.md').write_text(report, encoding='utf-8')
    document = ROOT/'experiment/mathseg_uav/experiment.md'
    text = document.read_text(encoding='utf-8')
    section = START+'\n'+report.replace(f'# MathSeg few-shot 单种子{title}',f'## 11. MathSeg few-shot 单种子{title}',1).replace('\n## ', '\n### ')+'\n'+END
    if START in text:
        before, rest = text.split(START, 1)
        _, after = rest.split(END, 1)
        text = before+section+after
    else:
        text += '\n\n'+section+'\n'
    document.write_text(text, encoding='utf-8')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT/'experiment/mathseg_uav/logs/fewshot_20260922')
    args = parser.parse_args()
    summarize(args.root)
