# -*- coding: utf-8 -*-
"""逐图 delta 分析：碎裂/建筑侵占/车辆误检在 test_2 上到底集中在哪些图。

对 1300 张逐图计算 baseline / aug / loss / full 的类别份额与边界密度，
回答三个问题：
1. full 相对 baseline 的背景份额下降 (d_bg<0) 与建筑份额上升 (d_bld>0) 是否相关？
   （验证"背景被压掉后质量流向建筑"的机制假说）
2. 碎裂增量 (d_bd) 是否集中在被"救回"的图（h3 bg>0.6 子集）？
3. 车辆误检：哪些图车辆像素数在 full 下异常增加？组件是否细小？

另输出 top  offender 的对比拼图（原图 + baseline + full 叠加）。
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PRED = HERE / "results" / "predictions"
IMAGES = ROOT / "dataset" / "low_altitude_2026" / "test_2" / "images"
H3_ZIP = ROOT / "experiment" / "mask2former_uav" / "test2_20260926" / "swin_l_hyperseg_复赛.zip"

SETS = ["baseline", "aug", "loss", "full"]
DIRS = {s: PRED / f"pred_test2_{s}_20k" for s in SETS}

PALETTE = np.array([
    (0, 0, 0), (80, 80, 80), (220, 50, 47), (38, 139, 210), (42, 161, 152),
    (181, 101, 29), (67, 160, 71), (238, 190, 39), (155, 89, 182),
], dtype=np.uint8)


def boundary_density(mask):
    valid = mask != 0
    dh = (mask[:, 1:] != mask[:, :-1]) & valid[:, 1:] & valid[:, :-1]
    dv = (mask[1:, :] != mask[:-1, :]) & valid[1:, :] & valid[:-1, :]
    tot = valid[:, 1:].sum() + valid[1:, :].sum()
    return float((dh.sum() + dv.sum()) / tot) if tot else 0.0


def shares(mask):
    n = mask.size
    return np.bincount(mask.ravel(), minlength=9)[:9] / n


def overlay(img, mask, alpha=0.45):
    rgb = np.asarray(img.convert("RGB")).astype(np.float32)
    tint = PALETTE[mask].astype(np.float32)
    m = mask > 0
    rgb[m] = rgb[m] * (1 - alpha) + tint[m] * alpha
    return Image.fromarray(rgb.clip(0, 255).astype(np.uint8))


def main():
    names = sorted(p.name for p in DIRS["baseline"].glob("*.png"))
    print(f"{len(names)} images")

    # h3 背景占比（分桶依据）
    import zipfile, io
    h3_bg = {}
    with zipfile.ZipFile(H3_ZIP) as z:
        lut = {Path(n).name: n for n in z.namelist() if n.lower().endswith(".png")}
        for nm in names:
            a = np.array(Image.open(io.BytesIO(z.read(lut[nm]))))
            h3_bg[nm] = float((a == 1).mean())

    per = {s: {} for s in SETS}  # per[set][name] = (shares, bd)
    for s in SETS:
        print(f"[{s}] ...", flush=True)
        for nm in names:
            m = np.array(Image.open(DIRS[s] / nm))
            per[s][nm] = (shares(m), boundary_density(m))

    rows = []
    for nm in names:
        r = {"name": nm, "h3_bg": h3_bg[nm]}
        for s in SETS:
            sh, bd = per[s][nm]
            r[f"{s}_bg"] = float(sh[1])
            r[f"{s}_bld"] = float(sh[2])
            r[f"{s}_road"] = float(sh[3])
            r[f"{s}_veh"] = float(sh[8])
            r[f"{s}_bd"] = bd
        rows.append(r)

    def corr(a, b):
        a, b = np.array(a), np.array(b)
        if a.std() == 0 or b.std() == 0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    for s in ("loss", "full", "aug"):
        d_bg = [r[f"{s}_bg"] - r["baseline_bg"] for r in rows]
        d_bld = [r[f"{s}_bld"] - r["baseline_bld"] for r in rows]
        d_veh = [r[f"{s}_veh"] - r["baseline_veh"] for r in rows]
        d_bd = [r[f"{s}_bd"] - r["baseline_bd"] for r in rows]
        print(f"\n== {s} vs baseline ==")
        print(f"  corr(d_bg, d_bld) = {corr(d_bg, d_bld):+.3f}   corr(d_bg, d_veh) = {corr(d_bg, d_veh):+.3f}")
        print(f"  corr(d_bg, d_bd)  = {corr(d_bg, d_bd):+.3f}")
        # 质量流向：背景份额总下降量里，各类吸收多少
        tot_drop = -sum(d_bg)
        flow = {}
        for c, key in [(2, "bld"), (3, "road"), (8, "veh")]:
            flow[key] = sum(max(0.0, r[f"{s}_{key}"] - r[f"baseline_{key}"]) for r in rows)
        for c in (4, 5, 6, 7):
            key = f"c{c}"
            flow[key] = sum(max(0.0, float(per[s][nm][0][c] - per["baseline"][nm][0][c])) for nm in names)
        tot_flow = sum(flow.values())
        print(f"  背景净流出 {tot_drop/len(names)*100:+.2f}pt/图；各类净流入占比: " +
              ", ".join(f"{k} {v/tot_flow*100:.1f}%" for k, v in sorted(flow.items(), key=lambda x: -x[1])))

        # 被救回子集 (h3 bg>0.6) vs 其余
        rescued = [r for r in rows if r["h3_bg"] > 0.6]
        rest = [r for r in rows if r["h3_bg"] <= 0.6]
        for grp, gname in ((rescued, "rescued(h3bg>0.6)"), (rest, "rest")):
            mbd = np.mean([r[f"{s}_bd"] - r["baseline_bd"] for r in grp])
            mbld = np.mean([r[f"{s}_bld"] - r["baseline_bld"] for r in grp])
            mveh = np.mean([r[f"{s}_veh"] - r["baseline_veh"] for r in grp])
            print(f"  {gname:<18} n={len(grp):4d}  d_bd={mbd:+.5f}  d_bld={mbld*100:+.2f}pt  d_veh={mveh*100:+.3f}pt")

    # top offenders
    def top(key_fn, k=10):
        return sorted(rows, key=key_fn, reverse=True)[:k]

    tops = {
        "d_bld_full": [r["name"] for r in top(lambda r: r["full_bld"] - r["baseline_bld"])],
        "d_veh_full": [r["name"] for r in top(lambda r: r["full_veh"] - r["baseline_veh"])],
        "d_bd_full": [r["name"] for r in top(lambda r: r["full_bd"] - r["baseline_bd"])],
    }
    print("\ntop d_bld (full-base):", tops["d_bld_full"])
    print("top d_veh (full-base):", tops["d_veh_full"])
    print("top d_bd  (full-base):", tops["d_bd_full"])

    out = {"rows": rows, "tops": tops}
    (HERE / "results" / "per_image_delta_20k.json").write_text(
        json.dumps(out, ensure_ascii=False), encoding="utf-8", newline="\n")

    # 拼图：三类 top 各取前 2，行=图，列=原图/baseline/full
    picks = []
    for k in ("d_bld_full", "d_veh_full", "d_bd_full"):
        for nm in tops[k][:2]:
            if nm not in picks:
                picks.append(nm)
    picks = picks[:6]
    cell = 341
    sheet = Image.new("RGB", (cell * 3, cell * (len(picks) + 1)), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    for j, t in enumerate(("image", "baseline", "full")):
        draw.text((cell * j + 6, 4), t, fill=(240, 240, 240))
    for i, nm in enumerate(picks):
        img = Image.open(IMAGES / nm).resize((cell, cell))
        m_b = np.array(Image.open(DIRS["baseline"] / nm).resize((cell, cell), Image.NEAREST))
        m_f = np.array(Image.open(DIRS["full"] / nm).resize((cell, cell), Image.NEAREST))
        b = overlay(img, m_b)
        f = overlay(img, m_f)
        y = cell * (i + 1)
        sheet.paste(img, (0, y))
        sheet.paste(b, (cell, y))
        sheet.paste(f, (cell * 2, y))
        draw.text((6, y + 4), nm, fill=(255, 255, 0))
    sheet_path = HERE / "results" / "delta_montage_20k.png"
    sheet.save(sheet_path)
    print(f"montage -> {sheet_path}")


if __name__ == "__main__":
    sys.exit(main())
