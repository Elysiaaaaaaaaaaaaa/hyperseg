"""Browse LoveDA Val overlays and select ordered, per-class few-shot supports.

CPU only: python tools/select_loveda_gui.py --data-root LoveDA
"""
from __future__ import annotations

import argparse
import json
import queue
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("Ignore", "Background 背景", "Building 建筑", "Road 道路", "Water 水体",
         "Barren 裸地", "Forest 林地", "Agriculture 农田")
COLORS = np.array([(0, 0, 0), (220, 220, 220), (255, 65, 65), (255, 210, 0),
                   (30, 140, 255), (200, 130, 65), (30, 210, 100), (210, 80, 240)], dtype=np.uint8)
SHOTS = (0, 1, 2, 5, 10)


def child(parent: Path, name: str) -> Path:
    matches = [p for p in parent.iterdir() if p.is_dir() and p.name.lower() == name.lower()]
    if len(matches) != 1:
        raise ValueError(f"{parent} 下应有一个 {name} 目录")
    return matches[0]


def read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        mask = np.array(im)
    if mask.ndim != 2:
        raise ValueError(f"需要单通道标签 mask，实际为 {mask.shape}: {path}")
    mask[mask == 255] = 0
    if not np.isin(mask, np.arange(8)).all():
        raise ValueError(f"标签必须为 0..7 或 255（Ignore）: {path}")
    return mask.astype(np.uint8)


def scan_dataset(root: Path, progress=lambda message: None) -> tuple[Path, list[dict]]:
    val = root if root.name.lower() == "val" else child(root, "Val")
    records = []
    for domain in ("Urban", "Rural"):
        folder = child(val, domain)
        images, masks = child(folder, "images_png"), child(folder, "masks_png")
        files = sorted(p for p in images.iterdir() if p.suffix.lower() == ".png")
        for image in files:
            mask_path = masks / image.name
            mask = read_mask(mask_path)
            with Image.open(image) as im:
                if im.size != (mask.shape[1], mask.shape[0]):
                    raise ValueError(f"图像与 mask 尺寸不一致: {image}")
            counts = np.bincount(mask.ravel(), minlength=8).tolist()
            records.append(dict(key=f"{domain.lower()}/{image.name}", domain=domain,
                                image=image, mask=mask_path, counts=counts,
                                total=int(mask.size), valid=int(mask.size - counts[0])))
            if len(records) % 25 == 0:
                progress(f"已统计 {len(records)} 张图片……")
    if not records:
        raise ValueError(f"没有找到图像: {val}")
    return val.resolve(), records


def ratio(record: dict, cid: int, valid: bool = False) -> float:
    return record["counts"][cid] / max(1, record["valid" if valid else "total"]) * 100


def overlay(rgb: Image.Image, mask: np.ndarray, cid: int, alpha: float, mode: str) -> Image.Image:
    if mode == "原图":
        return rgb.copy()
    active = mask == cid if mode == "仅目标叠加" else mask != 0
    pixels = np.array(rgb, dtype=np.float32)
    pixels[active] = pixels[active] * (1 - alpha) + COLORS[mask[active]] * alpha
    return Image.fromarray(pixels.astype(np.uint8))


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def validate_selection(selected: dict, classes: list[int], by_key: dict) -> dict:
    if not isinstance(selected, dict) or set(selected) != {str(c) for c in classes}:
        raise ValueError("保存的目标类别与当前 --classes 不一致，请使用相同类别或新的 --output")
    for cid in classes:
        keys = selected[str(cid)]
        if not isinstance(keys, list) or any(not isinstance(k, str) for k in keys):
            raise ValueError("选择清单格式错误")
        if len(keys) > 10 or len(set(keys)) != len(keys):
            raise ValueError(f"类别 {cid} 的样本重复或超过 10 张")
        for key in keys:
            if key not in by_key or by_key[key]["counts"][cid] == 0:
                raise ValueError(f"已选图片不存在或不包含类别 {cid}: {key}")
    return selected


def export_manifests(output: Path, selected: dict, classes: list[int], records: list[dict]) -> Path:
    by_key = {r["key"]: r for r in records}
    validate_selection(selected, classes, by_key)
    if any(len(selected[str(c)]) != 10 for c in classes):
        raise ValueError("请先为每个目标类别选满 10 张。未完成的选择已保存在 selection.json。")
    reserved = {key for keys in selected.values() for key in keys}
    evaluation = [r["key"] for r in records if r["key"] not in reserved]
    if not evaluation:
        raise ValueError("排除全部支持图片后评估集为空")
    destination = output / ("export_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    destination.mkdir(parents=True, exist_ok=False)
    for k in SHOTS:
        per_class = {str(c): selected[str(c)][:k] for c in classes}
        samples = list(dict.fromkeys(key for keys in per_class.values() for key in keys))
        atomic_json(destination / f"{k}shot.json", {
            "protocol": "K ordered images per target class; full masks; union deduplicated",
            "split": "Val", "shots_per_class": k, "target_classes": classes,
            "classes": {str(c): NAMES[c] for c in range(8)}, "ignore_index": 0,
            "per_class": per_class, "samples": samples, "num_unique_images": len(samples),
            "evaluation_samples": evaluation, "reserved_support_samples": sorted(reserved),
            "class_presence": {key: [c for c in range(1, 8) if by_key[key]["counts"][c]]
                               for key in samples},
            "target_area_percent": {str(c): {key: ratio(by_key[key], c)
                                             for key in per_class[str(c)]} for c in classes},
        })
    atomic_json(destination / "evaluation.json", {
        "split": "Val", "protocol": "Fixed Val holdout excluding all 10-shot supports",
        "samples": evaluation, "excluded_samples": sorted(reserved),
    })
    return destination


class Selector:
    def __init__(self, window, args):
        import tkinter as tk
        from tkinter import ttk, messagebox
        from PIL import ImageTk

        self.tk, self.ttk, self.messagebox, self.ImageTk = tk, ttk, messagebox, ImageTk
        self.window, self.args = window, args
        self.records, self.by_key, self.visible = [], {}, []
        self.selected = {str(c): [] for c in args.classes}
        self.current, self.original, self.mask = None, None, None
        self.ready = False
        self.events = queue.Queue()
        self.class_var = tk.StringVar(value=NAMES[args.classes[0]])
        self.domain = tk.StringVar(value="全部")
        self.minimum, self.maximum = tk.StringVar(value="0"), tk.StringVar(value="100")
        self.order = tk.StringVar(value="占比从高到低")
        self.mode, self.alpha = tk.StringVar(value="仅目标叠加"), tk.DoubleVar(value=0.45)
        self.status = tk.StringVar(value="正在读取原始 mask，统计目标面积……")
        self.info, self.summary = tk.StringVar(), tk.StringVar()
        window.title("LoveDA Val · 按类别筛选 Few-shot 支持集")
        window.geometry("1450x900")
        window.minsize(1080, 700)

        top = ttk.Frame(window, padding=8)
        top.pack(fill="x")
        ttk.Label(top, text="目标类别").pack(side="left")
        self.combo(top, self.class_var, [NAMES[c] for c in args.classes], 22, self.change_class)
        self.combo(top, self.domain, ["全部", "Urban", "Rural"], 8, self.filter_records)
        ttk.Label(top, text="占整图 %").pack(side="left", padx=(10, 2))
        ttk.Entry(top, textvariable=self.minimum, width=6).pack(side="left")
        ttk.Label(top, text="至").pack(side="left")
        ttk.Entry(top, textvariable=self.maximum, width=6).pack(side="left")
        self.combo(top, self.order, ["占比从高到低", "占比从低到高", "文件名"], 16, self.filter_records)
        ttk.Button(top, text="应用筛选", command=self.filter_records).pack(side="left", padx=5)
        ttk.Button(top, text="导出 0/1/2/5/10-shot", command=self.export).pack(side="right")

        ttk.Label(window, textvariable=self.summary, padding=(8, 0)).pack(fill="x")
        body = ttk.Panedwindow(window, orient="horizontal")
        body.pack(fill="both", expand=True, padx=8, pady=8)
        left, center, right = (ttk.Frame(body) for _ in range(3))
        body.add(left, weight=0)
        body.add(center, weight=1)
        body.add(right, weight=0)
        ttk.Label(left, text="候选图片（只显示含目标的图片）").pack(anchor="w")
        self.candidates = ttk.Treeview(left, columns=("area",), show="tree headings", selectmode="browse", height=24)
        self.candidates.heading("#0", text="域 / 文件名")
        self.candidates.heading("area", text="占整图 %")
        self.candidates.column("#0", width=172)
        self.candidates.column("area", width=85, anchor="e")
        scroll = ttk.Scrollbar(left, orient="vertical", command=self.candidates.yview)
        self.candidates.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.candidates.pack(fill="both", expand=True)
        self.candidates.bind("<<TreeviewSelect>>", self.show_candidate)

        controls = ttk.Frame(center)
        controls.pack(fill="x")
        self.combo(controls, self.mode, ["仅目标叠加", "全部类别叠加", "原图"], 16, self.render)
        ttk.Label(controls, text="透明度").pack(side="left", padx=5)
        ttk.Scale(controls, from_=0, to=1, variable=self.alpha, command=self.render).pack(side="left", fill="x", expand=True)
        self.canvas = tk.Canvas(center, background="#20252b", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, pady=5)
        self.resize_job = None
        self.canvas.bind("<Configure>", self.schedule_render)
        ttk.Label(center, textvariable=self.info, justify="left").pack(anchor="w")
        nav = ttk.Frame(center)
        nav.pack(fill="x", pady=8)
        for title, command in [("上一张 ←", lambda: self.navigate(-1)), ("下一张 →", lambda: self.navigate(1)),
                               ("加入当前类别（Space）", self.add)]:
            ttk.Button(nav, text=title, command=command).pack(side="left", padx=3)

        ttk.Label(right, text="当前类别已选：前 1 / 2 / 5 / 10 张\n顺序决定各 K-shot 的支持集").pack(anchor="w")
        self.chosen = tk.Listbox(right, width=30, height=13, exportselection=False)
        self.chosen.pack(fill="x", pady=5)
        self.chosen.bind("<<ListboxSelect>>", self.show_chosen)
        for title, command in [("上移", lambda: self.move(-1)), ("下移", lambda: self.move(1)),
                               ("移除选中项", self.remove)]:
            ttk.Button(right, text=title, command=command).pack(fill="x", pady=2)
        ttk.Label(right, text="类别颜色（0 / 255 忽略区域不着色）").pack(anchor="w", pady=(20, 5))
        for cid in range(1, 8):
            color = "#%02x%02x%02x" % tuple(COLORS[cid])
            tk.Label(right, text=f"  {cid}  {NAMES[cid]}", anchor="w", bg=color, fg="black").pack(fill="x", pady=1)
        ttk.Label(right, text="选择自动保存，可关闭后继续。\n同一图片可供多个类别使用；\n导出汇总清单自动去重。\n0-shot 的支持集为空。", justify="left").pack(anchor="w", pady=15)
        ttk.Label(window, textvariable=self.status, padding=8).pack(fill="x")
        window.bind("<Left>", lambda e: self.shortcut(e, lambda: self.navigate(-1)))
        window.bind("<Right>", lambda e: self.shortcut(e, lambda: self.navigate(1)))
        window.bind("<space>", lambda e: self.shortcut(e, self.add))
        threading.Thread(target=self.scan, daemon=True).start()
        window.after(100, self.poll)

    @property
    def cid(self):
        return NAMES.index(self.class_var.get())

    def combo(self, parent, variable, values, width, callback):
        box = self.ttk.Combobox(parent, textvariable=variable, values=values, width=width, state="readonly")
        box.pack(side="left", padx=4)
        box.bind("<<ComboboxSelected>>", callback)

    def shortcut(self, event, action):
        if event.widget.winfo_class() not in ("TEntry", "TCombobox", "TScale", "TButton", "Listbox"):
            action()
            return "break"

    def scan(self):
        try:
            result = scan_dataset(self.args.data_root, lambda msg: self.events.put(("progress", msg)))
            self.events.put(("done", result))
        except Exception as exc:
            self.events.put(("error", str(exc)))

    def poll(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "progress":
                    self.status.set(value)
                elif kind == "error":
                    self.status.set("读取失败")
                    self.messagebox.showerror("数据读取失败", value)
                    return
                else:
                    self.val, self.records = value
                    self.by_key = {r["key"]: r for r in self.records}
                    saved = self.args.output / "selection.json"
                    try:
                        if saved.exists():
                            payload = json.loads(saved.read_text(encoding="utf-8"))
                            self.selected = validate_selection(payload["per_class"], self.args.classes, self.by_key)
                    except Exception as exc:
                        self.messagebox.showerror("无法恢复选择", f"{saved}\n{exc}\n请检查文件或使用新的 --output。")
                        self.status.set("恢复失败；为保护已有选择，已停止加载。")
                        return
                    self.ready = True
                    self.change_class()
                    return
        except queue.Empty:
            self.window.after(100, self.poll)

    def change_class(self, *_):
        if self.ready:
            self.refresh_chosen()
            self.filter_records()

    def filter_records(self, *_):
        if not self.ready:
            return
        try:
            lo, hi = float(self.minimum.get()), float(self.maximum.get())
            if not 0 <= lo <= hi <= 100:
                raise ValueError()
        except ValueError:
            self.messagebox.showerror("筛选范围错误", "请输入 0 至 100 内的百分比，最小值不能大于最大值。")
            return
        rows = [r for r in self.records if r["counts"][self.cid] > 0
                and lo <= ratio(r, self.cid) <= hi
                and self.domain.get() in ("全部", r["domain"])]
        if self.order.get() == "文件名":
            rows.sort(key=lambda r: r["key"])
        else:
            rows.sort(key=lambda r: ratio(r, self.cid), reverse=self.order.get() == "占比从高到低")
        self.visible = [r["key"] for r in rows]
        self.candidates.delete(*self.candidates.get_children())
        for r in rows:
            self.candidates.insert("", "end", iid=r["key"], text=r["key"], values=(f"{ratio(r, self.cid):.3f}",))
        self.status.set(f"{self.val} | 共 {len(self.records)} 张，当前筛选 {len(rows)} 张 | 保存至 {self.args.output}")
        if rows:
            self.focus_candidate(self.current if self.current in self.visible else self.visible[0])
        else:
            self.current, self.original, self.mask = None, None, None
            self.canvas.delete("all")
            self.info.set("没有符合条件的图片，请调整筛选条件。")

    def focus_candidate(self, key):
        self.candidates.selection_set(key)
        self.candidates.focus(key)
        self.candidates.see(key)
        self.load_image(key)

    def show_candidate(self, *_):
        keys = self.candidates.selection()
        if keys:
            self.load_image(keys[0])

    def show_chosen(self, *_):
        indices = self.chosen.curselection()
        if indices:
            self.load_image(self.selected[str(self.cid)][indices[0]])

    def load_image(self, key):
        try:
            if key != self.current or self.original is None:
                record = self.by_key[key]
                with Image.open(record["image"]) as im:
                    original = im.convert("RGB")
                mask = read_mask(record["mask"])
                self.original, self.mask, self.current = original, mask, key
            self.render()
        except Exception as exc:
            self.current, self.original, self.mask = None, None, None
            self.canvas.delete("all")
            self.info.set("图片读取失败")
            self.messagebox.showerror("图片读取失败", str(exc))

    def schedule_render(self, *_):
        if self.resize_job:
            self.window.after_cancel(self.resize_job)
        self.resize_job = self.window.after(80, self.render)

    def render(self, *_):
        if self.original is None:
            return
        r = self.by_key[self.current]
        w, h = self.original.size
        scale = min(max(1, self.canvas.winfo_width()) / w, max(1, self.canvas.winfo_height()) / h)
        size = max(1, int(w * scale)), max(1, int(h * scale))
        rgb = self.original.resize(size, Image.Resampling.BILINEAR)
        mask = np.asarray(Image.fromarray(self.mask).resize(size, Image.Resampling.NEAREST))
        self.photo = self.ImageTk.PhotoImage(overlay(rgb, mask, self.cid, self.alpha.get(), self.mode.get()))
        self.canvas.delete("all")
        self.canvas.create_image(self.canvas.winfo_width() // 2, self.canvas.winfo_height() // 2, image=self.photo)
        selected = self.selected[str(self.cid)]
        rank = f"已选第 {selected.index(self.current) + 1} 张" if self.current in selected else "未加入当前类别"
        self.info.set(f"{self.current} | {w} × {h} | {rank}\n"
                      f"{NAMES[self.cid]}：{r['counts'][self.cid]:,} 像素 | 占整图 {ratio(r, self.cid):.3f}%"
                      f" | 占有效区域 {ratio(r, self.cid, True):.3f}% | Ignore {r['counts'][0] / r['total'] * 100:.3f}%")

    def navigate(self, delta):
        if self.visible:
            index = self.visible.index(self.current) if self.current in self.visible else 0
            self.focus_candidate(self.visible[max(0, min(len(self.visible) - 1, index + delta))])

    def refresh_chosen(self, index=None):
        self.chosen.delete(0, "end")
        for i, key in enumerate(self.selected[str(self.cid)], 1):
            self.chosen.insert("end", f"{i:2}. {key}  {ratio(self.by_key[key], self.cid):.2f}%")
        if index is not None:
            self.chosen.selection_set(index)
        self.summary.set("    ".join(f"{NAMES[c].split()[0]} {len(self.selected[str(c)])}/10" for c in self.args.classes))

    def commit(self, updated, index=None):
        try:
            atomic_json(self.args.output / "selection.json", {
                "version": 1, "data_root": str(self.val.parent), "split": "Val",
                "protocol": "ordered images per target class", "target_classes": self.args.classes,
                "per_class": updated,
            })
        except OSError as exc:
            self.messagebox.showerror("保存失败（本次修改未生效）", str(exc))
            return
        self.selected = updated
        self.refresh_chosen(index)
        self.render()

    def add(self):
        if not self.ready or self.current is None:
            return
        keys = self.selected[str(self.cid)]
        if self.current in keys:
            return
        if len(keys) >= 10:
            self.messagebox.showinfo("已选满", "当前类别已有 10 张，请先移除一张再添加。")
            return
        if self.by_key[self.current]["counts"][self.cid] == 0:
            return
        updated = {c: list(v) for c, v in self.selected.items()}
        updated[str(self.cid)].append(self.current)
        self.commit(updated)

    def remove(self):
        indices = self.chosen.curselection()
        if indices:
            updated = {c: list(v) for c, v in self.selected.items()}
            updated[str(self.cid)].pop(indices[0])
            self.commit(updated)

    def move(self, delta):
        indices = self.chosen.curselection()
        if not indices:
            return
        i, j = indices[0], indices[0] + delta
        updated = {c: list(v) for c, v in self.selected.items()}
        keys = updated[str(self.cid)]
        if 0 <= j < len(keys):
            keys[i], keys[j] = keys[j], keys[i]
            self.commit(updated, j)

    def export(self):
        if not self.ready:
            return
        try:
            destination = export_manifests(self.args.output, self.selected, self.args.classes, self.records)
            self.messagebox.showinfo("导出完成", f"已导出至：\n{destination}\n\n包含 5 组支持集和固定评估清单。")
        except Exception as exc:
            self.messagebox.showerror("导出失败", str(exc))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "LoveDA", help="LoveDA 或 Val 目录")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/loveda_manual", help="选择记录和导出目录")
    parser.add_argument("--classes", type=int, nargs="+", default=list(range(1, 8)), help="目标类别 ID，默认 1..7")
    args = parser.parse_args()
    if len(set(args.classes)) != len(args.classes) or any(c not in range(1, 8) for c in args.classes):
        parser.error("--classes 必须为不重复的 1..7 类别 ID")
    args.data_root, args.output = args.data_root.resolve(), args.output.resolve()
    try:
        import tkinter as tk
    except ImportError:
        parser.exit(1, "需要 tkinter：Windows 请安装带 Tcl/Tk 的 Python；Ubuntu 安装 python3-tk。\n")
    try:
        window = tk.Tk()
    except tk.TclError as exc:
        parser.exit(1, f"无法创建桌面窗口：{exc}\n请在 Windows 桌面 Python 或有显示服务的 Linux 中运行。\n")
    Selector(window, args)
    window.mainloop()


if __name__ == "__main__":
    main()
