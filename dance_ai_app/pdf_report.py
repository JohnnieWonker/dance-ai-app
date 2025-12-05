# pdf_report_mac.py —— 专为 macOS 优化的 AI 舞蹈评估报告（单页布局）

import os
import math
import tempfile
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph,
    Image, Spacer
)
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Matplotlib 设置
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ================= 1. Mac 字体自动加载逻辑 =================
def register_mac_font():
    """
    专门适配 macOS 的字体加载逻辑。
    优先使用 'PingFang.ttc' (苹方)，其次 'STHeiti' (华文黑体)。
    """
    font_name = "MacSystemFont"
    selected_path = None

    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]

    for path in candidates:
        if os.path.exists(path):
            selected_path = path
            break

    if selected_path:
        try:
            pdfmetrics.registerFont(TTFont(font_name, selected_path, subfontIndex=0))
            print(f"✅ 成功加载 Mac 系统字体: {selected_path}")
            return font_name, selected_path
        except Exception as e:
            print(f"⚠️ 字体加载出错: {e}")
            return "Helvetica", None
    else:
        print("⚠️ 未找到常见的 Mac 中文字体。")
        return "Helvetica", None

# 执行注册
USE_FONT, FONT_PATH = register_mac_font()

# ================= 2. 样式配置 =================
COLOR_PRIMARY = colors.HexColor("#003366")
COLOR_ACCENT = colors.HexColor("#C0392B")
COLOR_BG_HEADER = colors.HexColor("#F2F4F8")
COLOR_BORDER = colors.HexColor("#D5D8DC")

styles = getSampleStyleSheet()

styles.add(ParagraphStyle(
    name="TitleCN", fontName=USE_FONT, fontSize=20, leading=24,
    textColor=COLOR_PRIMARY, alignment=1, spaceAfter=6
))
styles.add(ParagraphStyle(
    name="SubTitleCN", fontName=USE_FONT, fontSize=11, leading=14,
    textColor=colors.dimgray, alignment=1, spaceAfter=16
))
styles.add(ParagraphStyle(
    name="SectionCN", fontName=USE_FONT, fontSize=11, leading=14,
    textColor=COLOR_PRIMARY, spaceBefore=8, spaceAfter=4,
))
styles.add(ParagraphStyle(
    name="NormalCN", fontName=USE_FONT, fontSize=9, leading=12,
    textColor=colors.black
))
styles.add(ParagraphStyle(
    name="AdviceCN", fontName=USE_FONT, fontSize=9, leading=13,
    textColor=colors.black, leftIndent=10, spaceAfter=2
))

# ================= 3. 绘图函数 =================
def _make_radar_image(scores, labels, font_path):
    if not scores or not labels:
        return None

    prop = None
    if font_path and os.path.exists(font_path):
        prop = font_manager.FontProperties(fname=font_path)

    num_vars = len(scores)
    angles = [n / float(num_vars) * 2 * math.pi for n in range(num_vars)]
    angles += angles[:1]
    values = scores + scores[:1]

    fig = plt.figure(figsize=(3.2, 3.2), dpi=140)
    ax = plt.subplot(111, polar=True)

    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], labels, fontproperties=prop, size=8, color='#333333')

    ax.set_rlabel_position(0)
    plt.yticks([20, 50, 80], ["", "", ""], color="grey", size=7)
    plt.ylim(0, 100)

    ax.plot(angles, values, linewidth=2, linestyle='solid', color='#003366')
    ax.fill(angles, values, '#003366', alpha=0.15)

    ax.spines['polar'].set_visible(False)
    ax.grid(True, color='#dddddd', linestyle='--', linewidth=0.6)

    tmp_png = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    plt.tight_layout()
    plt.savefig(tmp_png, transparent=True)
    plt.close(fig)
    return tmp_png


def _make_trajectory_image(trajectory, font_path):
    """
    根据鼻子/质心轨迹生成腾空轨迹图。
    trajectory: list/array, 已经是 “随帧变化的高度值” 或原始 y 值。
    """
    if not trajectory or len(trajectory) < 2:
        return None

    prop = None
    if font_path and os.path.exists(font_path):
        prop = font_manager.FontProperties(fname=font_path)

    # 如果是原始 y（越小越高），做个反转用于直观显示
    y_vals = [-float(y) for y in trajectory]
    x_vals = list(range(len(y_vals)))

    fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=140)
    ax.plot(x_vals, y_vals, linewidth=2, color='#636EFA')
    ax.set_xlabel("Frame", fontproperties=prop, fontsize=8)
    ax.set_ylabel("Height (inv.Y)", fontproperties=prop, fontsize=8)
    ax.grid(True, linestyle="--", linewidth=0.5, color="#cccccc")

    # 标记最高点
    max_idx = int(max(range(len(y_vals)), key=lambda i: y_vals[i]))
    ax.scatter([max_idx], [y_vals[max_idx]], color="#C0392B", s=20)
    ax.annotate("Apex", xy=(max_idx, y_vals[max_idx]),
                xytext=(max_idx, y_vals[max_idx] + 0.1),
                textcoords="data", fontsize=7,
                arrowprops=dict(arrowstyle="->", color="#C0392B", linewidth=0.8),
                fontproperties=prop)

    tmp_png = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    plt.tight_layout()
    plt.savefig(tmp_png, transparent=True)
    plt.close(fig)
    return tmp_png

# ================= 4. 页眉页脚 =================
def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(COLOR_PRIMARY)
    canvas.setLineWidth(0.8)
    canvas.line(doc.leftMargin, A4[1] - 35, doc.width + doc.leftMargin, A4[1] - 35)

    canvas.setFont(USE_FONT, 8)
    canvas.setFillColor(colors.grey)
    canvas.drawString(doc.leftMargin, A4[1] - 30, "AI Dance Pro · Mac System Report")
    canvas.drawRightString(doc.width + doc.leftMargin, A4[1] - 30, "Confidential | 内部评估")

    canvas.line(doc.leftMargin, 45, doc.width + doc.leftMargin, 45)
    canvas.drawString(doc.leftMargin, 32, "Powered by Panorama AI")
    canvas.drawRightString(doc.width + doc.leftMargin, 32, f"Page {canvas.getPageNumber()}")
    canvas.restoreState()

# ================= 5. 生成主程序 =================
def generate_pdf(
    path,
    lang_pack,
    metrics,
    scores,
    advice_list,
    subject_id,
    action_cn,
    action_en,
    overall_score,
    keyframe_peak_path,
    keyframe_start_path=None,
    keyframe_end_path=None,
    trajectory=None,
    lang_code="中文",
):
    """
    兼容旧版：如果你只传了 keyframe_peak_path，其它参数会是 None。
    现在支持：
      - 雷达图
      - 腾空轨迹图
      - 三张关键帧（起势/最高点/落地）
      - 指标表
      - 建议
    全部尽量排在一页内。
    """

    doc = SimpleDocTemplate(
        path,
        pagesize=A4,
        rightMargin=1.7 * cm,
        leftMargin=1.7 * cm,
        topMargin=2.3 * cm,
        bottomMargin=2.0 * cm,
    )
    elements = []

    # ========== 标题区 ==========
    title = lang_pack.get("pdf_report_title", "舞蹈动作技术评估报告")
    elements.append(Paragraph(title, styles["TitleCN"]))

    sub = f"{action_cn} <font size=9 color='grey'>({action_en})</font>"
    elements.append(Paragraph(sub, styles["SubTitleCN"]))

    # ========== 基础信息表 ==========
    info_header = ["受试者编号", "评估日期", "综合得分 (Score)"]
    info_row = [subject_id, "2025-11-26", f"{overall_score:.1f} / 100"]
    if lang_code != "中文":
        info_header = ["Subject ID", "Date", "Overall Score"]

    info_data = [info_header, info_row]
    t_info = Table(info_data, colWidths=[5.5 * cm, 4.7 * cm, 4.7 * cm])
    t_info.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), COLOR_PRIMARY),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, -1), USE_FONT),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('FONTSIZE', (0, 1), (-1, 1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('TOPPADDING', (0, 0), (-1, 0), 6),
        ('BOTTOMPADDING', (0, 1), (-1, 1), 7),
        ('TOPPADDING', (0, 1), (-1, 1), 7),
        ('GRID', (0, 0), (-1, -1), 0.5, COLOR_BORDER),
    ]))
    elements.append(t_info)
    elements.append(Spacer(1, 8))

    # ========== 图表区：雷达 + 轨迹 ==========
    # 雷达图
    r_labels = [lang_pack["metric_labels"].get(k, k) for k in scores.keys()]
    r_values = list(scores.values())
    radar_file = _make_radar_image(r_values, r_labels, FONT_PATH)

    img_radar = Image(radar_file, width=7.0 * cm, height=7.0 * cm) if radar_file else Paragraph("No Data", styles["NormalCN"])

    # 轨迹图
    traj_file = _make_trajectory_image(trajectory, FONT_PATH) if trajectory else None
    img_traj = Image(traj_file, width=7.0 * cm, height=7.0 * cm) if traj_file else Paragraph("暂无轨迹数据", styles["NormalCN"])

    layout_data = [
        [
            Paragraph("技术维度雷达 (Radar Analysis)", styles["SectionCN"]),
            Paragraph("腾空轨迹 (Jump Trajectory)", styles["SectionCN"]),
        ],
        [img_radar, img_traj],
    ]
    t_layout = Table(layout_data, colWidths=[7.8 * cm, 7.8 * cm])
    t_layout.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    elements.append(t_layout)
    elements.append(Spacer(1, 4))

    # ========== 关键帧区：三张小图一排 ==========
    elements.append(Paragraph("关键帧序列 (Start / Peak / Landing)", styles["SectionCN"]))

    def _make_key_img(path, fallback):
        if path and os.path.exists(path):
            return Image(path, width=4.2 * cm, height=3.0 * cm, kind='proportional')
        return Paragraph(fallback, styles["NormalCN"])

    img_start = _make_key_img(keyframe_start_path, "Start 无图")
    img_peak = _make_key_img(keyframe_peak_path, "Peak 无图")
    img_end = _make_key_img(keyframe_end_path, "Landing 无图")

    key_tbl = Table(
        [[img_start, img_peak, img_end]],
        colWidths=[4.8 * cm, 4.8 * cm, 4.8 * cm],
    )
    key_tbl.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(key_tbl)
    elements.append(Spacer(1, 6))

    # ========== 技术指标表（紧凑版） ==========
    elements.append(Paragraph("详细技术指标 (Technical Metrics)", styles["SectionCN"]))

    header = ["指标名称", "测量数值", "得分", "等级"]
    if lang_code != "中文":
        header = ["Metric", "Value", "Score", "Rating"]

    metric_rows = [header]
    for k, v in metrics.items():
        label = lang_pack["metric_labels"].get(k, k)
        s_val = scores.get(k, 0)
        if s_val >= 90:
            rating = "卓越 (Excellent)"
        elif s_val >= 80:
            rating = "优秀 (Good)"
        elif s_val >= 60:
            rating = "合格 (Pass)"
        else:
            rating = "需改进 (Improve)"
        metric_rows.append([label, f"{v:.4f}", f"{s_val:.1f}", rating])

    t_metrics = Table(metric_rows, colWidths=[5.5 * cm, 3.0 * cm, 2.5 * cm, 4.6 * cm])
    t_metrics.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), USE_FONT),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BACKGROUND', (0, 0), (-1, 0), COLOR_BG_HEADER),
        ('TEXTCOLOR', (0, 0), (-1, 0), COLOR_PRIMARY),
        ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.4, COLOR_BORDER),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#FAFAFA")]),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
    ]))
    elements.append(t_metrics)
    elements.append(Spacer(1, 4))

    # ========== 建议区（最多 4 条，保证单页） ==========
    elements.append(Paragraph("AI 智能纠错建议 (Recommendations)", styles["SectionCN"]))

    if advice_list:
        # 限制最多 4 条，避免溢出到第二页
        for i, adv in enumerate(advice_list[:4]):
            p = Paragraph(f"<font color='#003366'><b>{i+1}.</b></font> {adv}", styles["AdviceCN"])
            elements.append(p)
    else:
        elements.append(Paragraph("本次评估动作标准，未发现显著错误。", styles["NormalCN"]))

    # ========== 生成 PDF ==========
    doc.build(elements, onFirstPage=header_footer, onLaterPages=header_footer)

    # 清理临时图像
    for f in [radar_file, traj_file]:
        if f and os.path.exists(f):
            os.remove(f)

    print(f"✅ Mac PDF Report Generated: {path}")
