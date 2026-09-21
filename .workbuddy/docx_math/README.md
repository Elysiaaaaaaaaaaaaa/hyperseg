# docx 公式排版修复

把 docx 里用**纯文本 + Unicode 假上下标**拼出来的"公式"改写成 Word 原生公式（OMML）。

## 什么时候用

文档里的公式有以下特征，就属于本流程的处理对象：

- 公式混在中文说明段落里，被折行切断（`动态调制定义为 F′ᵢ = …。其中c是…`）。
- 用 ASCII 下划线假下标：`L_focal-CE`、`x_1`。
- 用 Unicode 上下标字符：`ᵢ`(U+1D62)、`⁻`(U+207B)、`⁴⁵`(U+2074/2075)、`ⁿ`。
  这些字符多数正文字体没有，靠字体回退渲染，字号和基线都对不齐。
- 变量是正体、函数名和变量不分（`GN`、`tanh` 应正体，`F`、`c`、`γ` 应斜体）。

## 流程

1. **先看清现状**。`unzip` 出 `word/document.xml`，统计 `<m:oMath>` 是否为 0，
   再用正则扫一遍 Unicode 数学字符（`[\u2070-\u209f\u1d2c-\u1d6a\u2032\u2211\u03b1-\u03c9]`）定位所有受影响段落。
2. **查样式与页面参数**。公式段落一般复用文档自带的 `Formula` 样式；从
   `<w:sectPr>` 取 `pgSz/pgMar` 算正文宽度（twips），后面居中/右对齐制表位要用：
   `正文宽度 = pgSz.w - pgMar.left - pgMar.right`。
3. **定点替换**。按 `w14:paraId` 唯一定位段落，整段换成新 XML；**不要全局搜索替换**，
   也不要按 `w:p` 出现序号下手。改完校验段落数、表格数、`<m:oMath>` 数量。
4. **渲染验证**。见下面"渲染链"。
5. **和修改前的 PDF 做文本 diff**，确认只有目标公式区变化。

## OMML 写法要点

```xml
<!-- 独立居中公式 + 右侧式号：靠制表位，不靠 jc -->
<w:pPr>
  <w:pStyle w:val="11"/>            <!-- Formula 样式 -->
  <w:tabs><w:tab w:val="center" w:pos="4819"/>   <!-- 正文宽/2 -->
          <w:tab w:val="right"  w:pos="9638"/></w:tabs>  <!-- 正文宽 -->
  <w:spacing w:before="160" w:after="160" w:line="240" w:lineRule="auto"/>
  <w:ind w:firstLine="0"/><w:jc w:val="left"/>   <!-- 必须显式 left，否则继承 Normal 的两端对齐 -->
</w:pPr>
<w:r><w:tab/></w:r>
<m:oMath>…</m:oMath>
<w:r><w:tab/><w:t>（1）</w:t></w:r>
```

- `m:oMath` 是 `w:p` 的直接子元素，可以夹在 `w:r` 之间（行内公式同理）。
- **斜体变量**：`<m:r>` 不写 `m:sty` 即为斜体（默认值）。
- **正体**（函数名 `tanh`/`softmax`/`GN`、数字、运算符、括号、下标里的说明文字）：
  `<m:r><m:rPr><m:sty m:val="p"/></m:rPr>…`。
- 每个 run 都带 `<w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math"/></w:rPr>`。
- 上下标用 `m:sSub` / `m:sSup`（子元素顺序：`e`，然后 `sub`/`sup`）。
- 求和用 `m:nary`，`<m:limLoc m:val="subSup"/>` 让下标落在 Σ 右下（对应 `Σᵢ`），
  `undOvr` 才是上下堆叠；`supHide` 隐藏上界时 `<m:sup/>` 留空即可。
- **数字与函数名之间要补一个显式空格 run**，否则渲染成 `0.15tanh(...)`。
- 一整个公式放在**同一个** `m:oMath` 里，数学排版引擎才会自动加运算符间距。

## 渲染链（本机 Windows）

editor_sdk 通道不可用时的替代方案：

```powershell
# 1) Word COM 导出 PDF（PowerShell 工具不回显 stdout，把结果写文件再读）
$word = New-Object -ComObject Word.Application
$word.Visible = $false; $word.DisplayAlerts = 0
$doc = $word.Documents.Open($src, $false, $true)
$doc.ExportAsFixedFormat($pdf, 17)      # 17 = wdExportFormatPDF
$doc.Close(0); $word.Quit()
```

```bash
# 2) pymupdf 出图（装在 C:\Users\86189\.workbuddy\binaries\python\envs\default）
python -c "import pymupdf; d=pymupdf.open('x.pdf'); d[2].get_pixmap(dpi=460, clip=pymupdf.Rect(60,400,560,500)).save('crop.png')"
```

用 `$doc.OMaths.Count` 可以直接确认 Word 认出了几个公式对象。

## 环境坑（已实测）

- **`editor_sdk` MCP 本地文档通道不可用**：实例会进 pool，但任何 `doc_*` 调用都报
  `document is not open`；`open_file` 只返回 "open started, chunks streaming via /stream"，
  要宿主前端消费 SSE 才真正加载。
- `doc_to_image` 报 `COS upload failed (exit 9009)` —— 服务进程 PATH 里没有 `curl`。
- **editor_sdk 解析不了非 ASCII 路径**，中文文件名一律 `file not found`。

## 本项目的实例

- 脚本：`fix_math.py`（内含本次 3 处替换的具体段落 id 与公式定义，可作模板改）。
- 效果对比：`compare_math_before_after.png`；改前/改后整份 PDF：`before.pdf` / `after.pdf`。
- 原始备份：`../backup/技术报告_公式排版前_20260920_212522.docx`。
