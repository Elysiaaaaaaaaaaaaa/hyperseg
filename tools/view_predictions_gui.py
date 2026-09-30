"""复赛预测掩码叠加查看器（tkinter GUI，仅依赖 Pillow + 标准库）。

功能
----
* 从预测 zip / tar.gz / 目录读取单通道掩码（像素 0..8），按固定调色板映射成彩色；
* 与 test_2.zip / 图像目录里的原图叠加，叠加比例用滑杆实时调整；
* 逐类开关（只看建筑+道路等）、Ignore 类可单独高亮；
* 滚轮缩放 / 拖拽平移 / 左右键翻页 / 搜索过滤；
* 显示当前图的类别像素占比，可导出当前叠加图。

用法
----
    python tools/view_predictions_gui.py
    python tools/view_predictions_gui.py --predictions <zip|tar.gz|目录> --images <zip|tar.gz|目录>
    python tools/view_predictions_gui.py --list                            # 只列候选包，不开界面
    python tools/view_predictions_gui.py --selftest --out <目录>     # 无界面自检 + 导出样张

约定：掩码为单通道灰度 PNG，像素值 0..8，0 为 Ignore（默认不上色）。
      调色板直接复用 tools/visualize_predictions.py 的 PALETTE，保证与既有可视化一致。
      实验目录里的预测包常以「目录 + tar.gz」两种形态并存，两种都要能被发现和读取。
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import tarfile
import zipfile
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageChops, ImageTk

# ---------------------------------------------------------------------------
# 类别定义：颜色复用 tools/visualize_predictions.py 的 PALETTE（同目录导入）
# ---------------------------------------------------------------------------
_FALLBACK_PALETTE = [
    (0, 0, 0),        # 0 ignore
    (80, 80, 80),     # 1 background
    (220, 50, 47),    # 2 building
    (38, 139, 210),   # 3 road
    (42, 161, 152),   # 4 water
    (181, 101, 29),   # 5 barren
    (67, 160, 71),    # 6 vegetation
    (238, 190, 39),   # 7 agricultural
    (155, 89, 182),   # 8 vehicle
]
try:  # 保证与既有可视化脚本一致
    from visualize_predictions import PALETTE as CLASS_COLORS  # type: ignore
except Exception:  # pragma: no cover - 单独拷贝该脚本时走兜底
    CLASS_COLORS = _FALLBACK_PALETTE

CLASS_NAMES = {
    0: "忽略 Ignore",
    1: "背景 Background",
    2: "建筑 Building",
    3: "道路 Road",
    4: "水体 Water",
    5: "裸地 Barren",
    6: "植被 Vegetation",
    7: "农用地 Agricultural",
    8: "车辆 Vehicle",
}
NUM_CLASSES = len(CLASS_COLORS)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# 数据源：zip 或目录，统一按“文件名 -> 字节”访问
# ---------------------------------------------------------------------------
class ZipSource:
    """按 basename 从 zip 中读取成员；自动剥掉公共顶层目录（如 images/）。"""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._zip = zipfile.ZipFile(self.path)
        members = {}
        for info in self._zip.infolist():
            if info.is_dir():
                continue
            name = info.filename.replace("\\", "/")
            base = name.rsplit("/", 1)[-1]
            if base:
                members.setdefault(base, name)
        self._members = members
        self._cache: dict[str, bytes] = {}
        self._cache_order: list[str] = []

    @property
    def kind(self) -> str:
        return "zip"

    def names(self) -> list[str]:
        return sorted(self._members, key=_natural_key)

    def read(self, name: str) -> bytes:
        member = self._members[name]
        if name in self._cache:
            return self._cache[name]
        data = self._zip.read(member)
        self._cache[name] = data
        self._cache_order.append(name)
        # 只保留最近 24 张，避免 1300 张原图把内存吃满
        while len(self._cache_order) > 24:
            self._cache.pop(self._cache_order.pop(0), None)
        return data

    def describe(self) -> str:
        return f"zip: {self.path}（{len(self._members)} 个条目）"


class TarSource:
    """按 basename 从 tar / tar.gz 中读取成员；自动剥掉公共顶层目录。

    实验目录里的预测包（如 `pred_test2_*_20k.tar.gz`）常用 tar.gz 形态回传，
    结构与 ZipSource 一致：内层是 `<包名>/test2_1.png` 这类路径。
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._tar = tarfile.open(self.path, "r:*")
        members = {}
        for info in self._tar.getmembers():
            if not info.isfile():
                continue
            name = info.name.replace("\\", "/")
            base = name.rsplit("/", 1)[-1]
            if base:
                members.setdefault(base, name)
        self._members = members
        self._cache: dict[str, bytes] = {}
        self._cache_order: list[str] = []

    @property
    def kind(self) -> str:
        return "tar"

    def names(self) -> list[str]:
        return sorted(self._members, key=_natural_key)

    def read(self, name: str) -> bytes:
        member = self._members[name]
        if name in self._cache:
            return self._cache[name]
        handle = self._tar.extractfile(member)
        data = handle.read() if handle is not None else b""
        self._cache[name] = data
        self._cache_order.append(name)
        # 只保留最近 24 张，避免 1300 张原图把内存吃满
        while len(self._cache_order) > 24:
            self._cache.pop(self._cache_order.pop(0), None)
        return data

    def describe(self) -> str:
        return f"tar: {self.path}（{len(self._members)} 个条目）"


class DirSource:
    """目录数据源，递归一级查 *.png。"""

    def __init__(self, path: Path):
        self.path = Path(path)
        files = {}
        for p in sorted(self.path.rglob("*.png")):
            files.setdefault(p.name, p)
        self._files = files

    @property
    def kind(self) -> str:
        return "dir"

    def names(self) -> list[str]:
        return sorted(self._files, key=_natural_key)

    def read(self, name: str) -> bytes:
        return self._files[name].read_bytes()

    def describe(self) -> str:
        return f"目录: {self.path}（{len(self._files)} 张 png）"


def open_source(path) -> object:
    if path is None:
        raise ValueError("数据源为空")
    path = Path(path)
    if path.is_dir():
        return DirSource(path)
    if path.is_file():
        if zipfile.is_zipfile(path):
            return ZipSource(path)
        if tarfile.is_tarfile(path):
            return TarSource(path)
    raise FileNotFoundError(f"不是 zip / tar.gz / 目录: {path}")


def looks_like_source(path: Path) -> bool:
    """候选包预检：目录里至少有一张 png，或文件是 zip / tar 包。"""
    path = Path(path)
    try:
        if path.is_dir():
            return any(path.glob("*.png")) or any(path.rglob("*.png"))
        if path.is_file():
            return zipfile.is_zipfile(path) or tarfile.is_tarfile(path)
    except OSError:
        return False
    return False


def _natural_key(name: str):
    """test2_2 排在 test2_10 之前（按数字段排序，而非字典序）。"""
    stem = name.rsplit(".", 1)[0]
    return tuple(int(part) if part.isdigit() else part.lower()
                 for part in re.split(r"(\d+)", stem))


def decode_image(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data))


# ---------------------------------------------------------------------------
# 叠加核心
# ---------------------------------------------------------------------------
def _luts(colors):
    """L 模式 point() 要求 256 项查找表，调色板只有 9 项，其余补零。"""
    pad = 256 - len(colors)
    lut_r = [c[0] for c in colors] + [0] * pad
    lut_g = [c[1] for c in colors] + [0] * pad
    lut_b = [c[2] for c in colors] + [0] * pad
    return lut_r, lut_g, lut_b


def colorize_mask(mask: Image.Image, colors=None) -> Image.Image:
    """单通道 0..8 掩码 -> RGB 彩色图（与 visualize_predictions.py 同逻辑）。"""
    colors = colors or CLASS_COLORS
    lut_r, lut_g, lut_b = _luts(colors)
    return Image.merge(
        "RGB",
        (mask.point(lut_r), mask.point(lut_g), mask.point(lut_b)),
    )


def compose(base: Image.Image, mask: Image.Image, alpha: float,
            visible=None, colors=None, mode: str = "overlay") -> Image.Image:
    """把彩色化掩码按 alpha 叠加到原图上。

    base   : RGB 原图
    mask   : L 掩码，取值 0..8
    alpha  : 0.0..1.0，掩码像素的混合强度
    visible: 需要上色的类别集合（None = 1..8，即默认不画 Ignore）
    mode   : overlay / image / mask
    """
    colors = colors or CLASS_COLORS
    if visible is None:
        visible = set(range(1, NUM_CLASSES))
    if mode == "image":
        return base
    colored = colorize_mask(mask, colors)
    if mode == "mask":
        return colored
    level = max(0, min(255, int(round(alpha * 255))))
    lut = [level if v in visible else 0 for v in range(256)]
    alpha_map = mask.point(lut)
    return Image.composite(colored, base, alpha_map)


def mask_stats(mask: Image.Image) -> dict:
    """统计各类别像素占比（基于直方图，O(1) 次 Python 循环）。"""
    hist = mask.histogram()[:NUM_CLASSES]
    total = sum(hist) or 1
    return {i: c / total for i, c in enumerate(hist)}


def _first_positions(mask: Image.Image, values) -> dict:
    """返回 {类别值: 首个出现坐标}。

    用 point+getbbox 在 C 层定位，避免对 1024×1024 做 Python 逐像素扫描
    （车辆这类稀疏类别也一定能取到）。
    """
    out = {}
    for val in values:
        binary = mask.point([255 if i == val else 0 for i in range(256)])
        box = binary.getbbox()
        if box is None:
            continue
        row = binary.crop((box[0], box[1], box[2], box[1] + 1))
        dx = list(row.getdata()).index(255)
        out[val] = (box[0] + dx, box[1])
    return out


# ---------------------------------------------------------------------------
# 候选路径自动发现
# ---------------------------------------------------------------------------
def discover_predictions() -> list[Path]:
    """列出候选预测掩码包（zip / tar.gz / 目录三种形态都收）。

    只按 zip 后缀找会漏掉实验目录里以「目录 + tar.gz」回传的预测结果
    （例如 `experiment/<exp>/results/predictions/pred_test2_*_20k`），
    这里改成「先按位置 glob，再用 looks_like_source 预检内容」。
    """
    out = []
    for pat in (
        "dist/**/01_预测结果/*.zip",
        "dist/**/01_预测结果/*.tar.gz",
        "experiment/**/results/predictions/*",
        "experiment/**/*test2*.zip",
        "experiment/**/*test2*.tar.gz",
        "experiment/**/*复赛*.zip",
        "runs/**/*test2*.zip",
        "runs/**/pred_*",
    ):
        out.extend(PROJECT_ROOT.glob(pat))
    return sorted({p.resolve() for p in out if looks_like_source(p)},
                  key=lambda p: p.stat().st_mtime, reverse=True)


def discover_images() -> list[Path]:
    out = []
    for rel in ("test_2.zip", "dataset/low_altitude_2026/test_2/images",
                "dataset/low_altitude_2026/test_2", "dataset/low_altitude_2026/images"):
        p = PROJECT_ROOT / rel
        if p.exists():
            out.append(p.resolve())
    return out


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
class ViewerApp:
    def __init__(self, root: tk.Tk, pred_default=None, image_default=None):
        self.root = root
        root.title("复赛预测掩码叠加查看器")
        root.geometry("1440x900")
        root.minsize(1000, 640)

        self.pred_src = None
        self.img_src = None
        self._img_names: set[str] = set()
        self.names: list[str] = []
        self.view_names: list[str] = []
        self.index = 0

        self.base: Image.Image | None = None
        self.mask: Image.Image | None = None
        self.stats: dict = {}
        self.mask_size_mismatch = False

        self.alpha = tk.DoubleVar(value=45.0)
        self.mode = tk.StringVar(value="overlay")
        self.visible = {i: tk.BooleanVar(value=(i != 0)) for i in range(NUM_CLASSES)}
        self.scale = 1.0
        self.cx = 0.0
        self.cy = 0.0
        self._photo = None
        self._drag = None
        self._render_job = None
        self._export_dir = PROJECT_ROOT / "runs" / "gui_overlays"

        self._build_ui()
        self._bind_keys()

        if pred_default:
            self._load_pred(pred_default)
        if image_default:
            self._load_images(image_default)
        self._refresh_list()
        if self.names:
            self.show_index(0)

    # ---------------- UI ----------------
    def _build_ui(self):
        style = ttk.Style()
        try:
            style.configure(".", font=("Microsoft YaHei UI", 9))
        except tk.TclError:
            pass

        top = ttk.Frame(self.root, padding=(8, 6))
        top.pack(side="top", fill="x")

        ttk.Label(top, text="预测掩码包").grid(row=0, column=0, sticky="w")
        self.pred_var = tk.StringVar()
        self.pred_combo = ttk.Combobox(top, textvariable=self.pred_var, width=58,
                                       values=[str(p) for p in discover_predictions()])
        self.pred_combo.grid(row=0, column=1, sticky="we", padx=4)
        self.pred_combo.bind("<<ComboboxSelected>>",
                             lambda e: self._load_pred(self.pred_var.get()))
        ttk.Button(top, text="浏览…", command=self._browse_pred).grid(row=0, column=2)

        ttk.Label(top, text="原图来源").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.img_var = tk.StringVar()
        self.img_combo = ttk.Combobox(top, textvariable=self.img_var, width=58,
                                      values=[str(p) for p in discover_images()])
        self.img_combo.grid(row=1, column=1, sticky="we", padx=4, pady=(4, 0))
        self.img_combo.bind("<<ComboboxSelected>>",
                            lambda e: self._load_images(self.img_var.get()))
        ttk.Button(top, text="浏览…", command=self._browse_images).grid(row=1, column=2, pady=(4, 0))

        # 叠加比例
        ctrl = ttk.Frame(top)
        ctrl.grid(row=0, column=3, rowspan=2, sticky="nsw", padx=(16, 0))
        ttk.Label(ctrl, text="叠加比例").grid(row=0, column=0, sticky="w")
        self.alpha_lbl = ttk.Label(ctrl, text="45%", width=5)
        self.alpha_lbl.grid(row=0, column=2)
        self.alpha_scale = ttk.Scale(ctrl, from_=0, to=100, orient="horizontal",
                                     length=220, variable=self.alpha,
                                     command=self._on_alpha)
        self.alpha_scale.grid(row=1, column=0, columnspan=3, sticky="we")
        mode_box = ttk.Frame(ctrl)
        mode_box.grid(row=2, column=0, columnspan=3, sticky="w", pady=(4, 0))
        for i, (val, text) in enumerate((("overlay", "叠加"), ("image", "仅原图"), ("mask", "仅掩码"))):
            ttk.Radiobutton(mode_box, text=text, value=val, variable=self.mode,
                            command=self._on_mode_change).grid(row=0, column=i, padx=(0, 8))
        ttk.Button(ctrl, text="适应窗口", command=self.fit).grid(row=3, column=0, sticky="w", pady=(4, 0))
        ttk.Button(ctrl, text="导出当前叠加图", command=self.export_current).grid(
            row=3, column=1, columnspan=2, sticky="we", pady=(4, 0))
        # 候选列表在启动时算一次；实验里经常是新包刚落地就来看，给个手动刷新
        ttk.Button(ctrl, text="刷新候选包", command=self._refresh_candidates).grid(
            row=4, column=0, columnspan=3, sticky="we", pady=(4, 0))

        top.columnconfigure(1, weight=1)

        body = ttk.Frame(self.root, padding=(8, 0))
        body.pack(side="top", fill="both", expand=True)

        # 左：列表
        left = ttk.Frame(body)
        left.pack(side="left", fill="y")
        ttk.Label(left, text="搜索（空格分隔多个关键词）").pack(anchor="w")
        self.search_var = tk.StringVar()
        entry = ttk.Entry(left, textvariable=self.search_var, width=26)
        entry.pack(fill="x")
        entry.bind("<KeyRelease>", lambda e: self._refresh_list())

        self.count_lbl = ttk.Label(left, text="0 / 0")
        self.count_lbl.pack(anchor="w", pady=(4, 2))
        list_wrap = ttk.Frame(left)
        list_wrap.pack(fill="both", expand=True)
        self.listbox = tk.Listbox(list_wrap, width=26, exportselection=False,
                                 activestyle="none")
        sb = ttk.Scrollbar(list_wrap, orient="vertical", command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=sb.set)
        self.listbox.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.listbox.bind("<<ListboxSelect>>", self._on_list_select)

        nav = ttk.Frame(left)
        nav.pack(fill="x", pady=4)
        ttk.Button(nav, text="◀ 上一张", command=lambda: self.step(-1)).pack(side="left")
        ttk.Button(nav, text="下一张 ▶", command=lambda: self.step(1)).pack(side="right")

        # 中：画布
        mid = ttk.Frame(body)
        mid.pack(side="left", fill="both", expand=True, padx=8)
        self.canvas = tk.Canvas(mid, bg="#f4f4f4", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self.schedule_render())
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", lambda e: setattr(self, "_drag", None))

        # 右：图例
        right = ttk.Frame(body, width=250)
        right.pack(side="right", fill="y")
        ttk.Label(right, text="类别（勾选控制是否上色）").pack(anchor="w")
        ttk.Button(right, text="全选", command=lambda: self._set_all(True)).pack(side="left", pady=2)
        ttk.Button(right, text="全不选", command=lambda: self._set_all(False)).pack(side="left", padx=4, pady=2)
        ttk.Button(right, text="只留前景", command=self._only_foreground).pack(side="left", pady=2)

        self.legend_rows = {}
        for cid in range(NUM_CLASSES):
            row = ttk.Frame(right)
            row.pack(fill="x", pady=1)
            color = "#%02x%02x%02x" % CLASS_COLORS[cid]
            swatch = tk.Frame(row, bg=color, width=14, height=14,
                              highlightthickness=1, highlightbackground="#888888")
            swatch.pack(side="left", padx=(0, 4))
            share = ttk.Label(row, text="", width=7, anchor="e")
            chk = ttk.Checkbutton(row, text=f"{cid} {CLASS_NAMES[cid]}",
                                  variable=self.visible[cid], command=self.schedule_render)
            chk.pack(side="left")
            share.pack(side="right")
            self.legend_rows[cid] = share

        self.status = ttk.Label(self.root, text="就绪", anchor="w", padding=(10, 4))
        self.status.pack(side="bottom", fill="x")

    def _bind_keys(self):
        self.root.bind("<Left>", lambda e: self.step(-1))
        self.root.bind("<Right>", lambda e: self.step(1))
        self.root.bind("<Prior>", lambda e: self.step(-10))
        self.root.bind("<Next>", lambda e: self.step(10))
        self.root.bind("<Home>", lambda e: self.show_index(0))
        self.root.bind("f", lambda e: self.fit())
        self.root.bind("<Control-s>", lambda e: self.export_current())

    # ---------------- 数据加载 ----------------
    def _load_pred(self, path):
        try:
            self.pred_src = open_source(path)
        except Exception as exc:
            messagebox.showerror("无法打开预测掩码包", str(exc))
            return
        self.pred_var.set(str(path))
        self._refresh_list()
        if self.names:
            self.show_index(0)

    def _load_images(self, path):
        try:
            self.img_src = open_source(path)
        except Exception as exc:
            messagebox.showerror("无法打开原图来源", str(exc))
            return
        self._img_names = set(self.img_src.names())
        self.img_var.set(str(path))

    def _refresh_candidates(self):
        """重新扫描候选预测包 / 原图来源，保留当前选中项。"""
        pred_now, img_now = self.pred_var.get(), self.img_var.get()
        preds = [str(p) for p in discover_predictions()]
        images = [str(p) for p in discover_images()]
        self.pred_combo.configure(values=preds)
        self.img_combo.configure(values=images)
        if pred_now in preds:
            self.pred_var.set(pred_now)
        if img_now in images:
            self.img_var.set(img_now)
        self.status.config(
            text=f"候选包已刷新：预测 {len(preds)} 个 / 原图 {len(images)} 个")

    def _browse_pred(self):
        path = filedialog.askopenfilename(
            title="选择预测掩码包（zip / tar.gz）",
            filetypes=[("掩码包", "*.zip *.tar.gz *.tgz *.tar"),
                       ("zip", "*.zip"), ("tar.gz", "*.tar.gz *.tgz"),
                       ("所有文件", "*.*")])
        if not path:
            path = filedialog.askdirectory(title="或选择预测掩码目录")
            if not path:
                return
        self._load_pred(path)

    def _browse_images(self):
        path = filedialog.askdirectory(title="选择原图目录")
        if not path:
            path = filedialog.askopenfilename(
                title="或选择原图包（zip / tar.gz）",
                filetypes=[("图像包", "*.zip *.tar.gz *.tgz *.tar"),
                           ("zip", "*.zip"), ("tar.gz", "*.tar.gz *.tgz"),
                           ("所有文件", "*.*")])
            if not path:
                return
        self._load_images(path)

    def _refresh_list(self):
        self.names = self.pred_src.names() if self.pred_src else []
        tokens = [t for t in self.search_var.get().lower().split() if t]
        self.view_names = [n for n in self.names if all(t in n.lower() for t in tokens)]
        self.listbox.delete(0, "end")
        for n in self.view_names:
            self.listbox.insert("end", n)
        self.count_lbl.config(text=f"{len(self.view_names)} / {len(self.names)}")
        if self.view_names:
            cur = self.current_name()
            if cur in self.view_names:
                self._highlight(cur)
            else:
                self.listbox.selection_clear(0, "end")

    def _highlight(self, name):
        try:
            pos = self.view_names.index(name)
        except ValueError:
            return
        self.listbox.selection_clear(0, "end")
        self.listbox.selection_set(pos)
        self.listbox.see(pos)

    def current_name(self):
        if not self.view_names:
            return None
        return self.view_names[max(0, min(self.index, len(self.view_names) - 1))]

    def step(self, delta):
        if not self.view_names:
            return
        self.show_index(self.index + delta)

    def show_index(self, idx):
        if not self.view_names:
            return
        self.index = max(0, min(idx, len(self.view_names) - 1))
        name = self.view_names[self.index]
        self._highlight(name)
        self.load_sample(name)
        self.fit()

    def _on_list_select(self, _event):
        sel = self.listbox.curselection()
        if not sel:
            return
        pos = int(sel[0])
        if pos == self.index:
            return
        self.index = pos
        self.load_sample(self.view_names[pos])
        self.fit()

    def load_sample(self, name):
        # 掩码
        mismatch = False
        try:
            mask = decode_image(self.pred_src.read(name)).convert("L")
        except Exception as exc:
            self.status.config(text=f"读取掩码失败：{name} — {exc}")
            self.mask = None
            return
        # 原图
        base = None
        if self.img_src is not None and name in self._img_names:
            try:
                base = decode_image(self.img_src.read(name)).convert("RGB")
            except Exception as exc:
                self.status.config(text=f"读取原图失败：{name} — {exc}")
        if base is None:
            base = Image.new("RGB", mask.size, (255, 255, 255))
        if base.size != mask.size:
            mismatch = True
            base = base.resize(mask.size, Image.Resampling.BILINEAR)
        self.base, self.mask, self.mask_size_mismatch = base, mask, mismatch
        self.stats = mask_stats(mask)
        self.scale = 0.0  # 触发 fit
        self.schedule_render()

    # ---------------- 参数回调 ----------------
    def _on_alpha(self, _value):
        # 拖动时只在文本上跟手，渲染做轻量节流
        self.alpha_lbl.config(text=f"{int(round(self.alpha.get()))}%")
        self.schedule_render()

    def _on_mode_change(self):
        self.schedule_render()

    def _set_all(self, value):
        for var in self.visible.values():
            var.set(value)
        self.schedule_render()

    def _only_foreground(self):
        for cid, var in self.visible.items():
            var.set(cid not in (0, 1))
        self.schedule_render()

    # ---------------- 渲染 ----------------
    def schedule_render(self):
        if self._render_job is not None:
            self.root.after_cancel(self._render_job)
        self._render_job = self.root.after(40, self._do_render)

    def _do_render(self):
        self._render_job = None
        self.render()

    def fit(self):
        self.scale = 0.0
        self.cx = self.cy = 0.0
        self.render()

    def _on_wheel(self, event):
        if self.base is None:
            return
        if self.scale <= 0:
            self.render()
        if self.scale <= 0:
            return
        factor = 1.25 if event.delta > 0 else 1 / 1.25
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        old_cw, old_ch = cw / self.scale, ch / self.scale
        sx = (self.cx or self.base.width / 2) - old_cw / 2 + event.x / self.scale
        sy = (self.cy or self.base.height / 2) - old_ch / 2 + event.y / self.scale
        new_scale = max(0.05, min(self.scale * factor, 16.0))
        self.scale = new_scale
        new_cw, new_ch = cw / new_scale, ch / new_scale
        self.cx = sx - new_cw / 2 + event.x / new_scale
        self.cy = sy - new_ch / 2 + event.y / new_scale
        self.render()

    def _on_press(self, event):
        self._drag = (event.x, event.y, self.cx, self.cy)

    def _on_drag(self, event):
        if self._drag is None or self.scale <= 0:
            return
        x0, y0, cx0, cy0 = self._drag
        self.cx = cx0 - (event.x - x0) / self.scale
        self.cy = cy0 - (event.y - y0) / self.scale
        self.render()

    def render(self):
        if self.base is None or self.mask is None:
            return
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        iw, ih = self.base.size

        if self.scale <= 0:  # 首帧或点“适应窗口”
            self.scale = min(cw / iw, ch / ih)
            self.cx, self.cy = iw / 2, ih / 2
        s = self.scale

        alpha = self.alpha.get() / 100.0
        visible = {cid for cid, var in self.visible.items() if var.get()}
        composed = compose(self.base, self.mask, alpha, visible, mode=self.mode.get())

        # 视口对应的源图区域（先裁剪再缩放，高倍缩放时内存恒定）
        req_w, req_h = cw / s, ch / s
        vis_w, vis_h = min(req_w, iw), min(req_h, ih)
        left = self.cx - req_w / 2
        top = self.cy - req_h / 2
        left = min(max(left, 0.0), max(iw - vis_w, 0.0))
        top = min(max(top, 0.0), max(ih - vis_h, 0.0))
        box = (int(round(left)), int(round(top)),
               int(round(left + vis_w)), int(round(top + vis_h)))
        box = (max(box[0], 0), max(box[1], 0), min(box[2], iw), min(box[3], ih))
        crop = composed.crop(box)
        disp = crop.resize((max(1, int(round(crop.width * s))),
                            max(1, int(round(crop.height * s)))),
                           Image.Resampling.LANCZOS)

        px = (box[0] - (self.cx - req_w / 2)) * s
        py = (box[1] - (self.cy - req_h / 2)) * s

        self._photo = ImageTk.PhotoImage(disp)
        self.canvas.delete("all")
        self.canvas.create_image(px, py, anchor="nw", image=self._photo)

        self.status.config(text=(
            f"{self.current_name()}  |  {iw}×{ih}  |  叠加 {int(round(alpha*100))}%  |  "
            f"{self.mode.get()}  |  上色类别 {sorted(visible) if len(visible) <= 9 else '全部'}"
            f"  |  缩放 {s*100:.0f}%"
            + ("  |  ⚠ 原图与掩码尺寸不一致，已按掩码尺寸缩放原图" if self.mask_size_mismatch else "")
        ))
        for cid, lbl in self.legend_rows.items():
            pct = self.stats.get(cid, 0.0) * 100
            lbl.config(text=f"{pct:5.1f}%")

    def export_current(self):
        if self.base is None or self.mask is None:
            return
        self._export_dir.mkdir(parents=True, exist_ok=True)
        name = self.current_name() or "sample.png"
        alpha = int(round(self.alpha.get()))
        out = self._export_dir / f"{Path(name).stem}_overlay_a{alpha}.png"
        visible = {cid for cid, var in self.visible.items() if var.get()}
        compose(self.base, self.mask, alpha / 100.0, visible,
                mode=self.mode.get()).save(out)
        self.status.config(text=f"已导出 {out}")


# ---------------------------------------------------------------------------
# 无界面自检
# ---------------------------------------------------------------------------
def run_selftest(pred, images, out_dir: Path, samples=4):
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_src = open_source(pred)
    img_src = open_source(images) if images else None
    names = [n for n in pred_src.names() if img_src is None or n in img_src.names()]
    if not names:
        raise RuntimeError("预测包与原图没有同名文件")
    picks = names[:samples]
    print(f"[selftest] 掩码源 {pred_src.describe()}")
    if img_src:
        print(f"[selftest] 原图源 {img_src.describe()}")
    print(f"[selftest] 同名样本 {len(names)} 张，取前 {len(picks)} 张")

    tiles = []
    for name in picks:
        base = decode_image(img_src.read(name)).convert("RGB") if img_src else None
        mask = decode_image(pred_src.read(name)).convert("L")
        assert base is not None and base.size == mask.size, f"{name} 尺寸不一致"
        # 数学校验：alpha=0 逐像素等于原图；alpha=1 等于调色板；alpha=0.5 是线性混合
        z = compose(base, mask, 0.0)
        assert ImageChops.difference(z, base).getbbox() is None, "alpha=0 应与原图逐像素相同"

        freqs = mask.histogram()
        present = [c for c in range(NUM_CLASSES) if freqs[c]]
        val2pos = _first_positions(mask, present)
        full = compose(base, mask, 1.0)
        mid = compose(base, mask, 0.5)
        for cid in present:
            pos = val2pos[cid]
            src = base.getpixel(pos)
            if cid == 0:  # Ignore 默认不上色
                exp_full = exp_mid = exp_zero = src
            else:
                col = CLASS_COLORS[cid]
                exp_full = col
                exp_mid = tuple(round(src[k] * 0.5 + col[k] * 0.5) for k in range(3))
                exp_zero = src
            assert full.getpixel(pos) == exp_full, f"alpha=1 时 class {cid} 颜色不符"
            assert all(abs(mid.getpixel(pos)[k] - exp_mid[k]) <= 1 for k in range(3)), \
                f"alpha=0.5 混合值不符：class {cid} got={mid.getpixel(pos)} exp={exp_mid}"
            assert z.getpixel(pos) == exp_zero, f"alpha=0 时 class {cid} 应保留原图"
        stats = mask_stats(mask)
        top = sorted(stats.items(), key=lambda kv: -kv[1])[:3]
        print(f"  {name}: {mask.size[0]}×{mask.size[1]}  "
              + "  ".join(f"{CLASS_NAMES[c]}({p*100:.1f}%)" for c, p in top))
        tiles.append(compose(base, mask, 0.45))

    side = min(t.width for t in tiles)
    cols = 2
    rows = (len(tiles) + cols - 1) // cols
    montage = Image.new("RGB", (cols * side, rows * side), (255, 255, 255))
    for i, t in enumerate(tiles):
        montage.paste(t.resize((side, side), Image.Resampling.LANCZOS),
                      ((i % cols) * side, (i // cols) * side))
    montage_path = out_dir / "selftest_montage_a45.png"
    montage.save(montage_path)
    if tiles:
        tiles[0].save(out_dir / "selftest_single_a45.png")
    print(f"[selftest] 数学校验通过；样张写入 {montage_path}")
    return montage_path


def main(argv=None):
    ap = argparse.ArgumentParser(description="复赛预测掩码叠加查看器")
    ap.add_argument("--predictions", type=Path, default=None, help="掩码 zip / tar.gz / 目录")
    ap.add_argument("--images", type=Path, default=None, help="原图 zip / tar.gz / 目录")
    ap.add_argument("--selftest", action="store_true", help="无界面自检并导出样张")
    ap.add_argument("--list", action="store_true", help="只列出候选掩码包与原图来源，不开界面")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "runs" / "gui_overlays")
    ap.add_argument("--samples", type=int, default=4)
    args = ap.parse_args(argv)

    if args.list:
        preds = discover_predictions()
        print(f"候选预测掩码包（{len(preds)} 个）:")
        for p in preds:
            try:
                print(f"  [{open_source(p).kind:<4}] {p.relative_to(PROJECT_ROOT)}")
            except Exception as exc:  # 不该发生，出现了说明预检与打开不一致
                print(f"  [????] {p.relative_to(PROJECT_ROOT)}  <打开失败: {exc}>")
        images = discover_images()
        print(f"候选原图来源（{len(images)} 个）:")
        for p in images:
            print(f"  {p.relative_to(PROJECT_ROOT)}")
        return 0

    pred = args.predictions
    images = args.images
    if pred is None:
        cands = discover_predictions()
        if not cands:
            print("未发现预测掩码包，请用 --predictions 指定", file=sys.stderr)
            return 2
        # 优先正式提交包（dist 内的那份），否则用最新
        pred = next((p for p in cands if "01_预测结果" in str(p)), cands[0])
    if images is None:
        cands = discover_images()
        images = cands[0] if cands else None

    if args.selftest:
        run_selftest(pred, images, args.out, args.samples)
        return 0

    root = tk.Tk()
    try:  # Windows 高分屏下文字更清晰
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    ViewerApp(root, pred_default=pred, image_default=images)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
