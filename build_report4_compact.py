# -*- coding: utf-8 -*-
"""
生成任务四实验报告（精简版，目标 6~8 页）。
数据来源与完整版一致：训练日志、OOF 评估、四轮提交记录。
"""
import os
import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

HERE = Path(__file__).resolve().parent
OUT = HERE / "report"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

GITHUB_URL = "https://github.com/DaoTongx/retinal-vessel-segmentation"


def set_run_font(run, ascii_font="Times New Roman", ea_font="宋体", size=None, bold=None,
                 color=None):
    run.font.name = ascii_font
    run._element.rPr.rFonts.set(qn("w:eastAsia"), ea_font)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if color is not None:
        run.font.color.rgb = color


def add_page_number_footer(section):
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    fb = OxmlElement("w:fldChar"); fb.set(qn("w:fldCharType"), "begin")
    it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = "PAGE"
    fe = OxmlElement("w:fldChar"); fe.set(qn("w:fldCharType"), "end")
    run._r.append(fb); run._r.append(it); run._r.append(fe)
    set_run_font(run, size=9)


def shade_cell(cell, hex_color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)


def set_table_borders(table):
    tblPr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement("w:%s" % edge)
        el.set(qn("w:val"), "single"); el.set(qn("w:sz"), "4")
        el.set(qn("w:space"), "0"); el.set(qn("w:color"), "808080")
        borders.append(el)
    tblPr.insert_element_before(borders, "w:shd", "w:tblLayout", "w:tblCellMar",
                                "w:tblLook", "w:tblCaption", "w:tblDescription")


class Rep:
    def __init__(self, doc):
        self.doc = doc
        self.fig_no = 0
        self.tab_no = 0

    def caption(self, text, kind="table"):
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(4 if kind == "figure" else 1)
        run = p.add_run(text)
        set_run_font(run, size=9, bold=True)

    def H1(self, text):
        p = self.doc.add_paragraph()
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(4)
        run = p.add_run(text)
        set_run_font(run, size=12, bold=True, ea_font="黑体")
        run.font.color.rgb = RGBColor(0x1F, 0x3B, 0x64)

    def H2(self, text):
        p = self.doc.add_paragraph()
        p.paragraph_format.space_before = Pt(5)
        p.paragraph_format.space_after = Pt(2)
        run = p.add_run(text)
        set_run_font(run, size=11, bold=True, ea_font="黑体")

    def P(self, text, size=10.5):
        p = self.doc.add_paragraph()
        pf = p.paragraph_format
        pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
        pf.line_spacing = 1.45
        pf.space_after = Pt(5)
        pf.first_line_indent = Cm(0.7)
        run = p.add_run(text)
        set_run_font(run, size=size)

    def FIG(self, name, caption, width_cm=14.0):
        path = FIG / name
        if not path.exists():
            print("missing:", path); return
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(3)
        p.paragraph_format.space_after = Pt(0)
        p.add_run().add_picture(str(path), width=Cm(width_cm))
        self.fig_no += 1
        self.caption(f"图 4-{self.fig_no}  {caption}", kind="figure")

    def TABLE(self, header, rows, caption, widths=None, font_size=8.5):
        self.tab_no += 1
        self.caption(f"表 4-{self.tab_no}  {caption}", kind="table")
        table = self.doc.add_table(rows=1, cols=len(header))
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.style = "Table Grid"
        table.autofit = False
        set_table_borders(table)

        hdr = table.rows[0].cells
        for i, h in enumerate(header):
            hdr[i].text = ""
            p = hdr[i].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(0)
            run = p.add_run(str(h))
            set_run_font(run, size=font_size, bold=True)
            shade_cell(hdr[i], "DCE6F1")

        for row in rows:
            cells = table.add_row().cells
            for i, v in enumerate(row):
                cells[i].text = ""
                p = cells[i].paragraphs[0]
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_after = Pt(0)
                run = p.add_run(str(v))
                set_run_font(run, size=font_size)

        if widths:
            for r in table.rows:
                for i, w in enumerate(widths):
                    r.cells[i].width = Cm(w)
        self.doc.add_paragraph().paragraph_format.space_after = Pt(2)


def build():
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(2.4)
    sec.top_margin = Cm(2.2)
    sec.bottom_margin = Cm(2.0)
    add_page_number_footer(sec)

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(10.5)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")

    r = Rep(doc)

    # ---------- 标题 ----------
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run("第 4 章  基于 U-Net 的眼底血管分割")
    set_run_font(run, size=16, bold=True, ea_font="黑体")
    run.font.color.rgb = RGBColor(0x1F, 0x3B, 0x64)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(8)
    run = p.add_run("代码仓库：" + GITHUB_URL)
    set_run_font(run, size=9)
    run.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)

    # ---------- 4.1 ----------
    r.H1("4.1  任务背景与难点")
    r.P("本任务要求从彩色眼底图像中分割出视网膜血管区域，按竞赛规定的 RLE 格式提交，"
        "以 Dice 系数评价。训练集与测试集各 20 张，分辨率均为 565×584。该任务有三个难点："
        "其一，样本极少，20 张图像远低于训练深度网络的常规需求；其二，类别极不均衡，"
        "血管像素仅占 8.63%，若只用交叉熵，模型把全部像素判为背景即可获得 91% 的像素准确率，"
        "但对 Dice 毫无贡献；其三，细结构对指标高度敏感，末梢血管常仅 1～2 像素宽，"
        "一旦漏检则会显著拉低 Dice。此外，提交格式本身存在语义陷阱，将在 4.2 节说明。")

    # ---------- 4.2 ----------
    r.H1("4.2  数据分析与提交格式")
    r.P("训练标注为单通道灰度图，仅含 0 与 255 两个值，其中血管为 255、背景为 0；"
        "血管占比在 6.03%～11.59% 之间，均值 8.63%。视野呈圆形，四角为纯黑背景，"
        "血管全部位于视野之内。数据示例如图 4-1。")
    r.TABLE(["项目", "数值"],
            [["图像数量", "训练 20 / 测试 20"],
             ["分辨率", "565 × 584"],
             ["标注取值", "0（背景）/ 255（血管）"],
             ["血管占比", "均值 8.63%（6.03%～11.59%）"]],
            "训练集数据统计", widths=[4.2, 8.6])
    r.FIG("fig4-1_data.png", "原始图像、绿通道与标注掩膜", width_cm=13.0)

    r.P("深入分析官方 segmentation_to_csv.py 后，可以确认它并非单纯的格式转换工具，"
        "而是把所有「大于 0」的像素判定为血管，再对这些像素做游程编码。由此得出两条硬约束。"
        "第一，预测图像必须满足血管=255、背景=0；若按任务描述字面理解为血管=0，"
        "脚本所编码的将是整片背景，Dice 会趋近于 0。本实验以反色掩膜构造对照实测，"
        "解码结果与真值完全互补，验证了这一判断。第二，提交图像必须为 512×512："
        "脚本内部会先双线性重采样至 512×512，尺寸不符时插值会在血管边缘产生中间值，"
        "这些中间值同样满足「大于 0」，从而被误判为血管。实测把 400×400 的掩膜输入脚本，"
        "编码像素数由 23094 膨胀到 35370，多出的 12276 个像素全部来自边界插值。")

    # ---------- 4.3 ----------
    r.H1("4.3  方法设计")
    r.P("整体流程如图 4-2，分为预处理、训练、推理与提交四个阶段。")
    r.FIG("fig4-2_pipeline.png", "整体技术流程", width_cm=11.8)
    r.P("预处理阶段提取绿通道并做 CLAHE 增强，再复制为三通道。选择绿通道是因为血红蛋白对绿光"
        "吸收最强，血管与背景的灰度差最大；选择 CLAHE 而非全局均衡化，是因为眼底图像照明不均，"
        "全局均衡化会过度拉伸视盘附近的亮区而压制周边暗区，导致细血管丢失。复制为三通道则是"
        "为了复用 ImageNet 预训练权重。")
    r.P("网络采用 U-Net，编码器为 ResNet34（ImageNet 预训练），由 segmentation-models-pytorch 实现。"
        "U-Net 的跳跃连接使解码器能访问高分辨率的浅层特征，这对细血管至关重要——细血管的位置"
        "信息在深层特征图中已被多次下采样稀释。损失函数为 BCEWithLogits 与 Dice 损失之和："
        "前者提供稳定梯度，后者直接优化区域重叠以矫正类别不平衡。训练时使用翻转、随机 90 度旋转、"
        "仿射变换、网格畸变与亮度伽马扰动等增强；其中网格畸变用于模拟眼底成像因眼球曲率造成的"
        "局部形变。增强强度需与样本量匹配，实验初期曾因使用过强的弹性形变导致收敛明显变慢，"
        "后调整为当前的仿射加网格畸变组合。")

    # ---------- 4.4 ----------
    r.H1("4.4  实验方案")
    r.P("训练配置如表 4-2。全部实验在本地 CPU 环境完成。")
    r.TABLE(["配置项", "取值", "配置项", "取值"],
            [["编码器", "ResNet34（预训练）", "优化器", "AdamW"],
             ["输入尺寸", "512 × 512", "初始学习率", "1e-3"],
             ["批大小", "4", "权重衰减", "1e-4"],
             ["训练轮次", "80（早停 25）", "学习率调度", "Warmup + 余弦"],
             ["交叉验证", "5 折", "损失函数", "BCE + Dice"],
             ["随机种子", "42（全流程固定）", "推理", "5 模型 × 8 路 TTA"]],
            "训练配置", widths=[3.0, 5.0, 3.0, 4.4])
    r.P("采用 5 折交叉验证，每折 16 张训练、4 张验证。训练结束后对每折验证集重新推理并保存概率图，"
        "汇总得到覆盖全部 20 张图像的 OOF 预测；由于每张图像的预测均来自未见过该图的模型，"
        "OOF 可无泄漏地用于阈值与后处理参数的搜索。测试阶段由 5 折模型共同推理，每个模型施加 8 路"
        "TTA（4 种旋转 × 2 种翻转）后取概率平均。为保证可复现，固定了 Python、NumPy、PyTorch 与 "
        "CUDA 的全部随机种子，设置 cuDNN 为确定性模式，并为每个 DataLoader 工作进程单独播种。")

    # ---------- 4.5 ----------
    r.H1("4.5  训练过程与结果")
    r.P("五折训练结果如表 4-3。单折 80 轮约耗时 12 分钟，五折合计约 65 分钟；测试推理约 2.4 分钟。")
    r.TABLE(["折号", "最佳 Dice", "最佳轮次", "折号", "最佳 Dice", "最佳轮次"],
            [["fold 0", "0.7823", "61", "fold 3", "0.7599", "50"],
             ["fold 1", "0.7492", "47", "fold 4", "0.7805", "50"],
             ["fold 2", "0.7762", "50", "均值", "0.7696", "±0.0129"]],
            "五折交叉验证结果", widths=[2.4, 3.0, 3.0, 2.4, 3.0, 3.0])
    r.P("损失在前 10 轮内由约 1.6 降至 0.8，验证 Dice 在前 20 轮由 0.17 升至 0.74 附近，随后进入"
        "平台期，如图 4-3 所示。五折收敛轨迹高度一致，拐点位置基本重合，说明模型对数据划分不敏感、"
        "训练过程稳定。但 fold 1 明显偏低（0.7492），比其他折低约 2 个百分点；结合后续分析可知该折"
        "验证图像包含较多细小血管，属难度较高的样本，反映的是样本难度差异而非训练不稳定。"
        "各折波动范围几乎覆盖了整个标准差，说明在 20 张样本规模下交叉验证估计本身带有不可忽略的"
        "随机性，任何小于 1 个百分点的改进都需通过 OOF 聚合确认。")
    r.FIG("fig4-3_curve.png", "五折训练损失与验证 Dice 曲线", width_cm=14.4)

    # ---------- 4.6 ----------
    r.H1("4.6  实验结果与分析")
    r.P("本任务共完成四轮实验，每轮均生成提交并在平台获得反馈，结果如表 4-4。")
    r.TABLE(["轮次", "方法配置", "CV Dice", "提交 Dice"],
            [["一", "ResNet34 五折，固定阈值 0.5", "0.7770", "0.74"],
             ["二", "EfficientNet-B4 五折，24 路 TTA", "0.7770", "0.76"],
             ["三", "ResNet34 五折 + FOV 后处理", "0.7696", "0.76"],
             ["四", "第三轮 + VascX 并集融合", "0.7844", "0.78"]],
            "四轮实验结果对比", widths=[1.4, 8.2, 3.0, 3.0])

    r.P("其一，阈值与后处理的有效性。在 OOF 上对阈值与最小连通域面积做网格搜索，阈值在 0.40～0.50 "
        "之间变化时 Dice 差异不足 0.0006。这说明模型输出的概率分布已高度极化，阈值移动几乎不改变"
        "二值化结果，性能瓶颈在于细血管的判别能力而非决策边界。实验还加入了基于亮度阈值的视野"
        "（FOV）掩膜后处理，将视野外预测强制置零。该做法在文献中常见，但实测增益精确为 0.0000："
        "统计发现模型在视野外的假阳性平均每张仅 1.4 像素，原因是训练标注在视野外本就全为 0，"
        "模型已从数据中学到了这一规律。这一负面结果说明后处理手段必须以数据为依据，"
        "不能因「文献常用」就直接采用。")

    r.P("其二，外部预训练模型的独立评估。为验证大规模眼底数据预训练能否带来突破，实验引入公开的 "
        "VascX 深度集成模型，并用其 bounds.csv 记录的视野圆心与半径把输出反向映射回原图坐标。"
        "在 20 张标注图像上的评测结果如表 4-5：VascX 平均 Dice 仅 0.7259，比本实验的 0.7701 低 4.4 "
        "个百分点，20 张中仅 1 张持平。其原因在于 VascX 平均只把 6.13% 的像素判为血管（标注为 "
        "8.63%），漏检约 30%——该模型面向管径测量等生物标志物，训练目标偏向保证主干血管的完整平滑，"
        "倾向于舍弃对比度低的末梢细血管。这说明预训练模型的收益并非自动成立，"
        "其训练目标必须与下游指标一致。")
    r.TABLE(["模型", "训练集 Dice", "预测血管占比"],
            [["本实验 U-Net", "0.7701", "0.0893"],
             ["VascX 预训练集成", "0.7259", "0.0613"],
             ["并集融合（Union）", "0.7844", "0.0965"],
             ["交集（Intersection，对照）", "0.7043", "—"]],
            "本实验模型与 VascX 的对比（标注血管占比 0.0863）",
            widths=[6.0, 3.6, 4.2])
    r.FIG("fig4-6_vascx.png", "逐图对比本实验模型与 VascX 的 Dice", width_cm=14.6)

    r.P("其三，互补融合。虽然 VascX 单独表现较差，但其漏检区域与本实验模型几乎不重叠——"
        "本实验模型漏检集中在极细的末梢分叉处，而 VascX 的漏检分布更广但相对平均。"
        "据此将两者的二值掩膜逐像素取逻辑或，再去除面积小于 20 像素的孤立连通域，训练集 Dice 达到 "
        "0.7844，比单独使用本实验模型高出 1.5 个百分点，也高于四轮中的任何单一配置。"
        "作为对照，交集策略仅为 0.7043，显著劣于任一单模型，进一步确认当前限制 Dice 的主要因素"
        "是对细血管的漏检，而非背景误检。此外，对融合结果尝试的闭运算、孔洞填充、先膨胀再融合等"
        "组合均无额外提升（最优增益仅 0.0026），说明融合已充分利用两者的互补信息。")
    r.FIG("fig4-5_pred.png", "测试集预测结果与叠加可视化", width_cm=13.0)

    r.P("其四，误差来源。综合可视化结果，主要误差有三类：末梢细血管漏检（最主要，细血管宽度仅 "
        "1～2 像素，经多次下采样后信息被显著削弱）、视盘边缘误检（视盘边缘呈亮黄色，灰度上与血管"
        "相似）、以及分叉处粘连（两分支交汇区形态复杂，有时被连成一片）。")

    # ---------- 4.7 ----------
    r.H1("4.7  创新点与不足")
    r.P("创新点主要有四。第一，对提交格式做了逆向分析并据此规避方向性错误，明确了「血管=255」"
        "与「512×512」两条硬约束，后者尤其易被忽略——尺寸不符会使编码像素数增加 50% 以上。"
        "第二，验证并否定了视野掩膜这一常见后处理，实测增益为 0，避免了无效投入。"
        "第三，发现并利用了异源模型的误差互补性，使最终 Dice 超过任一单模型；这一发现的普遍意义"
        "在于，集成学习中成员模型的独立误差模式比其绝对性能更值得关注。第四，在极小数据规模下"
        "构建了完整的可复现流程，固定全部随机源并记录环境指纹，使「代码可复现」可被实际验证。")
    r.P("不足之处在于：样本量是根本约束，交叉验证标准差达 1.3 个百分点；细血管召回仍有提升空间，"
        "且难以通过阈值或后处理解决；融合策略为固定权重的并集，未考虑不同图像上两模型的相对可靠性；"
        "受 CPU 算力限制未探索更高分辨率训练。后续可从伪标签半监督学习、640×640 分辨率训练、"
        "引入基于骨架的损失项、构建多编码器集成、以及自适应融合权重等方向改进。")

    # ---------- 4.8 小结 ----------
    r.H1("4.8  本章小结")
    r.P("本章针对 20 张标注样本条件下的眼底血管分割问题，构建了以 ResNet34 为编码器的 U-Net "
        "分割方案，并通过绿通道 CLAHE 预处理、BCE 与 Dice 组合损失、五折交叉验证与多路 TTA，"
        "使交叉验证 Dice 达到 0.7696、提交 Dice 达到 0.76。在此基础上，本章通过三项针对性分析"
        "进一步将提交分数提升至 0.78：对提交脚本的逆向分析规避了格式层面的方向性错误；"
        "在 OOF 上的对照实验否定了视野掩膜这一常见但无效的后处理；而对异源模型误差模式的分析"
        "促成了并集融合策略，使融合后的 Dice 超过任一单模型。")
    r.P("回顾整个迭代过程，四轮实验的交叉验证结果从 0.7770 到 0.7844，提升幅度有限，"
        "但提交分数由 0.74 提升至 0.78，其中相当一部分增益并非来自模型容量，而是来自对评价指标、"
        "提交格式与模型误差模式的正确理解。这一经验说明：在小样本场景下，"
        "对问题本身的细致分析往往比单纯堆叠模型或增加训练量更为有效；"
        "而任何被广泛采用的做法（如前文否定的视野掩膜），都必须以本任务数据上的实测结果为依据，"
        "不能仅凭经验直接采纳。")
    r.P("本方案的主要局限在于样本量与细血管召回率。20 张训练图像决定了模型对血管形态先验的学习"
        "必然不充分，而末梢细血管的漏检是限制 Dice 的首要因素。后续工作可围绕伪标签半监督学习、"
        "更高分辨率的训练与推理、面向细结构的损失设计以及多编码器集成等方向展开，"
        "以在现有数据条件下进一步逼近该任务的性能上限。")

    # ---------- 参考文献 ----------
    r.H1("参考文献")
    refs = [
        "[1] Ronneberger O, Fischer P, Brox T. U-Net: Convolutional Networks for Biomedical Image "
        "Segmentation[C]. MICCAI, 2015: 234-241.",
        "[2] He K, Zhang X, Ren S, et al. Deep Residual Learning for Image Recognition[C]. CVPR, 2016: 770-778.",
        "[3] Staal J, Abràmoff M D, Niemeijer M, et al. Ridge-based vessel segmentation in color images "
        "of the retina[J]. IEEE Transactions on Medical Imaging, 2004, 23(4): 501-509.",
        "[4] Vargas Quiros J D, Liefers B, van Garderen K A, et al. VascX Models: Deep Ensembles for "
        "Retinal Vascular Analysis From Color Fundus Images[J]. TVST, 2025, 14(7): 19.",
        "[5] Buslaev A, Iglovikov V I, Khvedchenya E, et al. Albumentations: Fast and Flexible Image "
        "Augmentations[J]. Information, 2020, 11(2): 125.",
        "[6] Sudre C H, Li W, Vercauteren T, et al. Generalised Dice Overlap as a Deep Learning Loss "
        "Function for Highly Unbalanced Segmentations[C]. MICCAI-DLMIA, 2017: 240-248.",
    ]
    for ref in refs:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.82)
        p.paragraph_format.first_line_indent = Cm(-0.82)
        p.paragraph_format.space_after = Pt(1)
        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
        run = p.add_run(ref)
        set_run_font(run, size=8.8)

    # ---------- 附录 ----------
    r.H1("附录  代码仓库")
    r.P("全部代码、训练脚本与报告源文件见：" + GITHUB_URL + "。主要文件包括 train_local.py"
        "（训练与推理主脚本）、vascx_batch.py（VascX 逐张推理）、make_union_submission.py"
        "（融合与最终提交）、vessel_segmentation_kaggle.ipynb（Kaggle GPU 版本）。复现命令如下：")
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.9)
    p.paragraph_format.space_after = Pt(0)
    for i, line in enumerate([
        "pip install -r requirements.txt",
        "python src/train_local.py --folds 5 --epochs 80 --size 512 --out run",
        "python src/make_union_submission.py",
    ]):
        run = p.add_run(line + ("\n" if i < 2 else ""))
        set_run_font(run, ascii_font="Consolas", size=8.8)

    out = OUT / "任务四_医学影像分割_实验报告_精简版.docx"
    doc.save(out)
    print("[docx] saved:", out)
    return out


if __name__ == "__main__":
    build()
