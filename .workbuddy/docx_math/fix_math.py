# -*- coding: utf-8 -*-
"""把技术报告里的“伪公式”（普通文本 + Unicode 上下标）重排为真正的 OMML 公式。

只做定点替换：按 w14:paraId 定位段落，整段换成新 XML，不做全局搜索替换。
输出到 OUT，不动原文件。
"""
import os
import re
import shutil
import sys
import zipfile

SRC = r"D:\myproject\hyperseg\【技术报告】无人机低空航拍图像语义分割.docx"
OUT = r"D:\myproject\hyperseg\.workbuddy\tmp_ed\report_fixed.docx"

# A4 (11906 twips) 减去左右页边距各 1134 => 正文宽度 9638
TEXT_WIDTH = 9638

CM = "Cambria Math"
WRPR = '<w:rPr><w:rFonts w:ascii="%s" w:hAnsi="%s"/></w:rPr>' % (CM, CM)


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def mplain(t):
    """直立（正体）：函数名、数字、运算符、括号。"""
    return ('<m:r><m:rPr><m:sty m:val="p"/></m:rPr>%s'
            '<m:t xml:space="preserve">%s</m:t></m:r>' % (WRPR, esc(t)))


def mvar(t):
    """斜体：数学变量。OMML 默认 italic，不写 m:sty 即为斜体。"""
    return '<m:r>%s<m:t xml:space="preserve">%s</m:t></m:r>' % (WRPR, esc(t))


def mspace():
    """二元/系数间的显式细空格，避免数字与函数名黏连。"""
    return '<m:r>%s<m:t xml:space="preserve"> </m:t></m:r>' % WRPR


def msub(base, sub):
    return '<m:sSub><m:e>%s</m:e><m:sub>%s</m:sub></m:sSub>' % (base, sub)


def msup(base, sup):
    return '<m:sSup><m:e>%s</m:e><m:sup>%s</m:sup></m:sSup>' % (base, sup)


def msum(sub, e):
    return ('<m:nary><m:naryPr><m:chr m:val="&#x2211;"/><m:limLoc m:val="subSup"/>'
            '<m:supHide m:val="1"/></m:naryPr><m:sub>%s</m:sub><m:sup/>'
            '<m:e>%s</m:e></m:nary>' % (sub, e))


def eq_para(inner, num, first=False, last=False):
    """独立的居中显示公式；num 为式号（None 则不加）。"""
    if num:
        tabs = ('<w:tabs><w:tab w:val="center" w:pos="%d"/>'
                '<w:tab w:val="right" w:pos="%d"/></w:tabs>' % (TEXT_WIDTH // 2, TEXT_WIDTH))
        content = ('<w:r><w:tab/></w:r><m:oMath>%s</m:oMath>'
                   '<w:r>%s<w:tab/><w:t>%s</w:t></w:r>' % (inner, WRPR, esc(num)))
    else:
        tabs = '<w:tabs><w:tab w:val="center" w:pos="%d"/></w:tabs>' % (TEXT_WIDTH // 2)
        content = '<w:r><w:tab/></w:r><m:oMath>%s</m:oMath>' % inner
    before = 160 if first else 100
    after = 100 if last else 160
    ppr = ('<w:pPr><w:pStyle w:val="11"/>%s'
           '<w:spacing w:before="%d" w:after="%d" w:line="240" w:lineRule="auto"/>'
           '<w:ind w:firstLine="0"/><w:jc w:val="left"/>%s</w:pPr>'
           % (tabs, before, after, WRPR))
    return '<w:p>%s%s</w:p>' % (ppr, content)


def body_para(text, style=None):
    ppr = '<w:pPr><w:pStyle w:val="%s"/></w:pPr>' % style if style else ''
    return '<w:p>%s<w:r><w:t xml:space="preserve">%s</w:t></w:r></w:p>' % (ppr, esc(text))


# --------------------------------------------------------------------------- #
# 式(1) F'_i = GN(F_i) x [1 + 0.15 tanh(gamma_i(c))] + 0.15 tanh(beta_i(c))
# --------------------------------------------------------------------------- #
F_PRIME_I = msub(msup(mvar("F"), mplain("′")), mvar("i"))
F_I = msub(mvar("F"), mvar("i"))

EQ1 = (
    F_PRIME_I
    + mplain("=")
    + mplain("GN") + mplain("(") + F_I + mplain(")")
    + mplain("×")
    + mplain("[") + mplain("1") + mplain("+") + mplain("0.15") + mspace()
    + mplain("tanh") + mplain("(") + msub(mvar("γ"), mvar("i"))
    + mplain("(") + mvar("c") + mplain(")") + mplain(")")
    + mplain("]")
    + mplain("+") + mplain("0.15") + mspace()
    + mplain("tanh") + mplain("(") + msub(mvar("β"), mvar("i"))
    + mplain("(") + mvar("c") + mplain(")") + mplain(")")
)

# --------------------------------------------------------------------------- #
# 式(2) F = sum_i softmax(g(c))_i x Resize(P_i(F'_i))
# --------------------------------------------------------------------------- #
SOFTMAX_I = msub(
    mplain("softmax") + mplain("(") + mvar("g") + mplain("(") + mvar("c") + mplain(")")
    + mplain(")"),
    mvar("i"),
)
RESIZE = (
    mplain("Resize") + mplain("(")
    + msub(mvar("P"), mvar("i")) + mplain("(") + F_PRIME_I + mplain(")")
    + mplain(")")
)

EQ2 = mvar("F") + mplain("=") + msum(mvar("i"), SOFTMAX_I + mplain("×") + RESIZE)

# --------------------------------------------------------------------------- #
# 式(3) L = 0.55 L_focal-CE + 0.30 L_Dice + 0.025 L_rare + 0.10 L_boundary
# --------------------------------------------------------------------------- #
def loss_term(weight, name):
    return mplain(weight) + msub(mvar("L"), mplain(name))


EQ3 = (
    mvar("L") + mplain("=")
    + loss_term("0.55", "focal-CE")
    + mplain("+") + loss_term("0.30", "Dice")
    + mplain("+") + loss_term("0.025", "rare")
    + mplain("+") + loss_term("0.10", "boundary")
)


def inline_pow10(exp):
    """1x10^n 的行内公式。"""
    return (mplain("1") + mplain("×")
            + msup(mplain("10"), mplain(exp)))


# --------------------------------------------------------------------------- #
PARA_23 = "04EB92C8"   # 2.3 动态调制 + 动态融合（公式混排在同一段）
PARA_24 = "4B5591EC"   # 2.4 损失函数（纯文本公式）
PARA_LR = "149E647E"   # 表格内 1x10^-5 / 1x10^-4

NEW_23 = "".join([
    body_para("动态调制定义为："),
    eq_para(EQ1, "（1）", first=True),
    body_para("其中 c 是场景向量，尺度系数限制条件参数幅度；零初始化使初始动态增量为零，"
              "初始输出为归一化特征。动态融合为："),
    eq_para(EQ2, "（2）", last=True),
])

NEW_24 = "".join([
    body_para("总损失由四项加权组成："),
    eq_para(EQ3, "（3）", first=True, last=True),
])

NEW_LR = (
    '<w:p w14:paraId="%s"><w:pPr><w:pStyle w:val="9"/></w:pPr>'
    '<w:r><w:t>骨干</w:t></w:r><m:oMath>%s</m:oMath>'
    '<w:r><w:t>；投影与HyperSeg下游模块</w:t></w:r><m:oMath>%s</m:oMath>'
    '</w:p>' % (PARA_LR, inline_pow10("-5"), inline_pow10("-4"))
)


def find_para(xml, para_id):
    """按 w14:paraId 取回整段 XML。"""
    anchors = [m.start() for m in re.finditer(r'<w:p w14:paraId="%s">' % para_id, xml)]
    if len(anchors) != 1:
        raise SystemExit("paraId %s 命中 %d 次，无法唯一定位" % (para_id, len(anchors)))
    start = anchors[0]
    end = xml.find("</w:p>", start)
    if end < 0:
        raise SystemExit("paraId %s 未找到段落结束" % para_id)
    return start, end + len("</w:p>")


def main():
    with zipfile.ZipFile(SRC) as z:
        names = [i.filename for i in z.infolist()]
        xml = z.read("word/document.xml").decode("utf-8")
        parts = [(n, z.read(n)) for n in names if n != "word/document.xml"]

    # 三处替换：先取原段落做校验，再替换（长度变化不影响后续 paraId 定位）
    for para_id, new_xml in ((PARA_23, NEW_23), (PARA_24, NEW_24), (PARA_LR, NEW_LR)):
        s, e = find_para(xml, para_id)
        old = xml[s:e]
        if "m:oMath" in old:
            raise SystemExit("%s 已包含公式，跳过以免重复处理" % para_id)
        xml = xml[:s] + new_xml + xml[e:]

    if xml.count("<m:oMath>") != 5:
        raise SystemExit("公式数量异常：%d（应为 5）" % xml.count("<m:oMath>"))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    if os.path.exists(OUT):
        os.remove(OUT)
    # 保持原有条目顺序，document.xml 原位替换，[Content_Types].xml 仍排在最前
    payload = {"word/document.xml": xml.encode("utf-8")}
    ordered = [n for n in names]
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for n in ordered:
            z.writestr(n, payload[n] if n in payload else dict(parts)[n])
    print("written:", OUT, os.path.getsize(OUT), "bytes")


if __name__ == "__main__":
    main()
