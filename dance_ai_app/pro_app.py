# pro_app.py
# AI 舞蹈技术评估系统 Pro 版
# - 吸撩腿跃（中国古典）
# - Grand Jeté（芭蕾）
# - 多语言（中 / 韩 / 英）
# - 关键帧 + 骨架可视化 + 雷达图 + 轨迹 + CSV + PDF

import os
import tempfile
from typing import List, Dict, Tuple, Optional

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from grand_jete_model import analyze_grand_jete, detect_flight_frames
from pdf_report import generate_pdf


# ======================= 1. 多语言文本 =======================

LANGUAGES = ["中文", "한국어", "English"]

I18N: Dict[str, Dict] = {
    "中文": {
        "app_title": "AI 舞蹈技术评估系统 Pro",
        "subtitle": "吸撩腿跃 & Grand Jeté 技术分析与教学辅助",
        "sidebar_title": "🎓 教学助手",
        "subject_id": "受试者 ID（可选）",
        "language": "界面语言",
        "mode_label": "选择舞种 / 评分模式",
        "mode_xiliao": "中国古典 · 吸撩腿跃",
        "mode_ballet": "芭蕾 · Grand Jeté",
        "upload_video": "上传包含单次跳跃动作的视频（mp4 / mov / avi）",
        "processing": "正在分析视频，请稍候…",
        "section_keyframes": "📸 动作关键帧捕捉（Key Frames）",
        "section_score": "🏆 综合评分（Performance Score）",
        "section_radar": "技术维度雷达图（Radar Chart）",
        "section_traj": "腾空轨迹分析（Jump Trajectory）",
        "section_detail": "📊 技术指标与评分明细",
        "section_advice": "💡 教学建议（AI 自动生成）",
        "section_export": "📄 报告导出",
        "overall": "综合得分",
        "csv_btn": "📥 下载 CSV 数据",
        "pdf_btn": "📑 生成 PDF 报告",
        "pdf_ready": "✅ PDF 生成成功，可以下载。",
        "pdf_dl": "📥 下载 PDF 报告",
        "metric_labels": {
            "xiliao": {
                "prep_knee_angle": "起跳屈膝角 (°)",
                "flight_time": "腾空高度与持续 (s)",
                "split_angle_max": "空中横叉角度 (°)",
                "front_knee_angle": "空中前腿伸膝 (°)",
                "back_knee_min": "空中后腿伸膝 (°)",
                "pelvis_opening": "吸撩腿屈髋角 (°)",
                "torso_upright": "空中躯干稳定性 (°)",
                "landing_stability": "落地稳定性 (角度波动)",
            },
            "ballet": {
                "prep_knee_angle": "起跳屈膝角 (°)",
                "flight_time": "腾空时间 (s)",
                "split_angle_max": "空中横叉角度 (°)",
                "front_knee_angle": "空中前腿伸膝 (°)",
                "back_knee_min": "空中后腿伸膝 (°)",
                "pelvis_opening": "空中骨盆打开 (°)",
                "torso_upright": "空中躯干直立度 (°)",
                "arm_line": "空中手臂三位手线条 (°)",
            },
        },
        "action_name_xiliao_cn": "吸撩腿跃",
        "action_name_xiliao_en": "Xi-Liao Leg Leap",
        "action_name_ballet_cn": "芭蕾大跳",
        "action_name_ballet_en": "Grand Jeté",
    },
    "한국어": {
        "app_title": "AI 무용 기술 평가 시스템 Pro",
        "subtitle": "흡요퇴 점프 & Grand Jeté 기술 분석과 수업 보조",
        "sidebar_title": "🎓 수업 도우미",
        "subject_id": "피험자 ID (선택)",
        "language": "언어 선택",
        "mode_label": "무용 장르 / 평가 모드",
        "mode_xiliao": "중국 고전 · 흡요퇴 점프",
        "mode_ballet": "발레 · Grand Jeté",
        "upload_video": "단 한 번의 점프 동작이 포함된 영상을 업로드하세요 (mp4 / mov / avi)",
        "processing": "영상 분석 중입니다…",
        "section_keyframes": "📸 주요 키프레임 (Key Frames)",
        "section_score": "🏆 종합 점수 (Performance Score)",
        "section_radar": "기술 차원 레이더 차트",
        "section_traj": "체공 궤적 분석",
        "section_detail": "📊 기술 지표 및 점수",
        "section_advice": "💡 AI 피드백",
        "section_export": "📄 리포트 내보내기",
        "overall": "종합 점수",
        "csv_btn": "📥 CSV 데이터 다운로드",
        "pdf_btn": "📑 PDF 리포트 생성",
        "pdf_ready": "✅ PDF가 생성되었습니다. 다운로드할 수 있습니다.",
        "pdf_dl": "📥 PDF 리포트 다운로드",
        "metric_labels": {
            "xiliao": {
                "prep_knee_angle": "도약 준비 무릎 굴곡 각도 (°)",
                "flight_time": "체공 시간 (s)",
                "split_angle_max": "공중 다리 벌림 각도 (°)",
                "front_knee_angle": "공중 앞다리 무릎 신전 (°)",
                "back_knee_min": "공중 뒷다리 무릎 신전 (°)",
                "pelvis_opening": "흡요퇴 고관절 굴곡 각도 (°)",
                "torso_upright": "공중 상체 정렬 (°)",
                "landing_stability": "착지 안정성 (각도 변동)",
            },
            "ballet": {
                "prep_knee_angle": "도약 준비 플리에 각도 (°)",
                "flight_time": "체공 시간 (s)",
                "split_angle_max": "공중 스플릿 각도 (°)",
                "front_knee_angle": "공중 앞다리 무릎 신전 (°)",
                "back_knee_min": "공중 뒷다리 무릎 신전 (°)",
                "pelvis_opening": "공중 골반 오픈 (°)",
                "torso_upright": "공중 상체 세움 정도 (°)",
                "arm_line": "공중 팔 라인 (°)",
            },
        },
        "action_name_xiliao_cn": "흡요퇴 점프",
        "action_name_xiliao_en": "Xi-Liao Leg Leap",
        "action_name_ballet_cn": "그랑 즈떼",
        "action_name_ballet_en": "Grand Jeté",
    },
    "English": {
        "app_title": "AI Dance Technique Evaluation Pro",
        "subtitle": "Xi-Liao Leg Leap & Grand Jeté Analysis for Teaching",
        "sidebar_title": "🎓 Teaching Assistant",
        "subject_id": "Subject ID (optional)",
        "language": "Language",
        "mode_label": "Select Dance Style / Mode",
        "mode_xiliao": "Chinese Classical · Xi-Liao Leap",
        "mode_ballet": "Ballet · Grand Jeté",
        "upload_video": "Upload a video containing a single jump (mp4 / mov / avi)",
        "processing": "Analyzing video with AI…",
        "section_keyframes": "📸 Key Frames",
        "section_score": "🏆 Overall Performance Score",
        "section_radar": "Technical Dimensions (Radar Chart)",
        "section_traj": "Flight Trajectory",
        "section_detail": "📊 Technical Metrics & Scores",
        "section_advice": "💡 AI Training Suggestions",
        "section_export": "📄 Export Report",
        "overall": "Overall Score",
        "csv_btn": "📥 Download CSV Data",
        "pdf_btn": "📑 Generate PDF Report",
        "pdf_ready": "✅ PDF generated successfully.",
        "pdf_dl": "📥 Download PDF",
        "metric_labels": {
            "xiliao": {
                "prep_knee_angle": "Prep Knee Angle (°)",
                "flight_time": "Flight Time (s)",
                "split_angle_max": "Air Split Angle (°)",
                "front_knee_angle": "Front Leg Extension in Air (°)",
                "back_knee_min": "Back Leg Extension in Air (°)",
                "pelvis_opening": "Hip Flexion / Pelvis Opening (°)",
                "torso_upright": "Torso Uprightness (°)",
                "landing_stability": "Landing Stability (angle SD)",
            },
            "ballet": {
                "prep_knee_angle": "Prep Knee Angle (°)",
                "flight_time": "Flight Time (s)",
                "split_angle_max": "Max Split Angle (°)",
                "front_knee_angle": "Front Knee Extension (°)",
                "back_knee_min": "Back Knee Extension (°)",
                "pelvis_opening": "Pelvis Opening in Air (°)",
                "torso_upright": "Torso Uprightness (°)",
                "arm_line": "Arm Line in Air (°)",
            },
        },
        "action_name_xiliao_cn": "Xi-Liao Leg Leap",
        "action_name_xiliao_en": "Xi-Liao Leg Leap",
        "action_name_ballet_cn": "Grand Jeté",
        "action_name_ballet_en": "Grand Jeté",
    },
}


# ======================= 2. Streamlit 页面配置 =======================

st.set_page_config(
    page_title="AI Dance Pro",
    page_icon="💃",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main {
        background-color: #f7f9fc;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

if "lang" not in st.session_state:
    st.session_state["lang"] = "中文"

if "subject_id" not in st.session_state:
    st.session_state["subject_id"] = ""


# ======================= 3. Sidebar 选择 =======================

with st.sidebar:
    lang = st.selectbox("Language / 语言 / 언어", LANGUAGES, index=0)
    st.session_state["lang"] = lang
    TEXT = I18N[lang]

    st.title(TEXT["sidebar_title"])
    subject_id = st.text_input(TEXT["subject_id"], value=st.session_state["subject_id"])
    st.session_state["subject_id"] = subject_id

    mode_label = TEXT["mode_label"]
    mode_display = st.radio(
        mode_label,
        [TEXT["mode_xiliao"], TEXT["mode_ballet"]],
        index=0,
    )
    # 内部 key：xiliao / ballet
    mode_key = "xiliao" if mode_display == TEXT["mode_xiliao"] else "ballet"

    st.markdown("---")
    st.caption(TEXT["subtitle"])


TEXT = I18N[st.session_state["lang"]]


# ======================= 4. MediaPipe & 几何工具 =======================

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def _get_xy(landmarks, idx: int) -> Optional[np.ndarray]:
    if landmarks is None:
        return None
    try:
        lm = landmarks[idx]
    except IndexError:
        return None
    return np.array([lm.x, lm.y], dtype=float)


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8
    cos_val = np.dot(ba, bc) / denom
    cos_val = np.clip(cos_val, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_val)))


# ======================= 5. 视频处理：骨架 + 关键帧 =======================

def process_video(video_path: str) -> Tuple[List[Dict], List, float, str, Tuple[int, int, int], List[float]]:
    """
    返回：
    - frames_data: [{"image": rgb_frame, "landmarks": landmarks}, ...]
    - landmark_seq: [landmarks or None, ...]
    - fps
    - overlay_video_path: 骨架可视化视频
    - (start_idx, peak_idx, end_idx): 关键帧索引
    - nose_traj: 每帧 nose y
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    overlay_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(overlay_path, fourcc, fps, (width, height))

    frames_data: List[Dict] = []
    landmark_seq: List = []
    nose_y_list: List[float] = []

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            landmarks = results.pose_landmarks.landmark if results.pose_landmarks else None

            draw_frame = frame_rgb.copy()
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    draw_frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(66, 135, 245), thickness=2, circle_radius=2),
                )

            # 写入骨架视频
            writer.write(cv2.cvtColor(draw_frame, cv2.COLOR_RGB2BGR))

            frames_data.append({"image": draw_frame, "landmarks": landmarks})
            landmark_seq.append(landmarks)

            # nose y 轨迹
            if landmarks is not None:
                nose = landmarks[0]
                nose_y_list.append(nose.y)
            else:
                nose_y_list.append(np.nan)

    cap.release()
    writer.release()

    # 使用 grand_jete_model 中的腾空检测来找关键帧
    f_start, f_end = detect_flight_frames(landmark_seq, fps)
    
    # === 1. 峰值帧 (Peak) - 以右脚踝最高点为准 ===
    peak_idx = f_start
    min_right_ankle_y = 1.0
    
    # 遍历腾空段 [f_start, f_end] 寻找右脚踝y最小（最高）的帧
    for i in range(f_start, f_end + 1):
        lm = landmark_seq[i]
        if lm is None:
            continue
        
        # 尝试获取右脚踝 y 坐标
        try:
            r_ankle_y = lm[RIGHT_ANKLE].y
        except IndexError:
            continue
            
        if r_ankle_y < min_right_ankle_y:
            min_right_ankle_y = r_ankle_y
            peak_idx = i

    # === 2. 起始帧 (Start) - 预备帧，确保有人体 ===
    
    # 从 f_start 往前找 2 帧，但不能超过 0
    start_search_idx = max(0, f_start - 2) 
    start_idx = -1 # 初始化为 -1 (未找到)
    
    # 确保选取帧有人体骨架
    for i in range(start_search_idx, f_start + 1):
        if landmark_seq[i] is not None:
            start_idx = i
            break # 找到第一个有人体的预备帧
    
    # 如果找不到，就用 f_start
    if start_idx == -1:
        start_idx = f_start

    # === 3. 落地帧 (End) - 寻找平稳落地帧 ===
    
    # 从 f_end 往后搜索 0.5s 后的帧 (例如 30fps -> 15帧)
    end_search_range = range(f_end, min(len(frames_data) - 1, f_end + int(fps * 0.5)))
    
    end_idx = f_end # 默认使用腾空结束帧
    
    # 在搜索范围内，寻找膝盖角度接近 180° 的帧（表示直立）
    for i in end_search_range:
        lm = landmark_seq[i]
        if lm is None:
            continue
            
        # 检查左右膝盖，哪个更直就用哪个
        # 注意：这里需要 _get_xy 函数，但由于它在外面，我们直接使用 lm 索引
        # 我们使用 LEFT_HIP, LEFT_KNEE, LEFT_ANKLE 等常数
        
        # 简化版：只需确保关键点存在即可调用 _angle
        try:
            # 确保 _get_xy 能在全局/函数外部被访问，我们直接调用全局的 _get_xy
            
            # 由于 _get_xy 依赖 landmarks 是 mediapipe 对象，这里需要确保它能被正确调用
            
            # 为了避免引入新的依赖问题，我们假设 _get_xy 可以在这里被调用，
            # 并且它返回的是 numpy 数组或 None。如果返回 None，_angle 内部的 np.linalg.norm 会报错。
            # 为了安全，我们用 try/except 保护。
            
            # 使用局部函数来获取 numpy 数组，确保它们不是 None
            def get_coords(lm, p1, p2, p3):
                a = _get_xy(lm, p1)
                b = _get_xy(lm, p2)
                c = _get_xy(lm, p3)
                if None in (a, b, c):
                    raise ValueError("Missing coordinates")
                return a, b, c
            
            # 左膝角度
            a_l, b_l, c_l = get_coords(lm, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)
            l_angle = _angle(a_l, b_l, c_l)
            
            # 右膝角度
            a_r, b_r, c_r = get_coords(lm, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)
            r_angle = _angle(a_r, b_r, c_r)

            # 认为膝盖角度 > 170 度是平稳直立落地（膝盖伸直）
            if max(l_angle, r_angle) > 170.0:
                end_idx = i
                break
                
        except (IndexError, ValueError):
            # 关键点缺失或 get_coords 抛出错误时，跳过此帧
            continue
    
    # 如果平稳落地帧没找到，则使用 f_end 后 2 帧作为兜底
    if end_idx == f_end:
        end_idx = min(len(frames_data) - 1, f_end + 2)
    # 腾空时间计算
    flight_duration = max(0.0, (f_end - f_start + 1) / fps)

    # 动作验证：如果腾空时间太短，则认为不是目标动作
    # 经验值：一次明显的跳跃动作至少持续 0.20 秒
    MIN_FLIGHT_TIME_SECONDS = 0.20
    
    # 【新增强制检查】计算腾空段的垂直位移
    
    # 使用鼻子 y 坐标（y 越小越高）
    if nose_y_list and f_start < len(nose_y_list) and f_end < len(nose_y_list):
        # 腾空前的鼻子高度（约等于 f_start 时的 y）
        y_start = nose_y_list[f_start]
        # 腾空时的最高点（y 最小）
        y_min = np.nanmin(nose_y_list[f_start:f_end+1])
        
        # 垂直位移：(y_start - y_min)。位移必须是正数，且要大于阈值。
        # 阈值设定：0.05（在归一化坐标系中，鼻子必须至少上升 5% 的视频高度）
        MIN_VERTICAL_RISE = 0.05 
        vertical_rise = max(0.0, y_start - y_min)
    else:
        vertical_rise = 0.0 # 无法计算，默认为 0.0
    
    # 最终动作有效性检查：必须有足够的腾空时间和垂直位移
    if (flight_duration < MIN_FLIGHT_TIME_SECONDS or
        vertical_rise < MIN_VERTICAL_RISE):
        # 如果不是有效的跳跃，清空 landmark_seq，迫使后续评分失败
        landmark_seq = []
        
    return frames_data, landmark_seq, fps, overlay_path, (start_idx, peak_idx, end_idx), nose_y_list

# ======================= 6. 吸撩腿跃评分模型（简版） =======================

# 关键点索引
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12


def analyze_xiliao(landmark_seq: List, fps: float, is_left_lead: bool = True) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    吸撩腿跃：简单 5 维指标 + 评分
    """
    n = len(landmark_seq)
    if n == 0:
        return {}, {}

    # 体空时间：复用 detect_flight_frames
    f_start, f_end = detect_flight_frames(landmark_seq, fps)
    flight_time = max(0.0, (f_end - f_start + 1) / fps)

    # 在腾空段找到 peak
    peak_idx = f_start
    min_nose = 1.0
    for i in range(f_start, f_end + 1):
        lm = landmark_seq[i]
        if lm is None:
            continue
        y = lm[0].y
        if y < min_nose:
            min_nose = y
            peak_idx = i

    # 选左腿为前腿
    def get_xy(lm, idx):
        if lm is None:
            return None
        try:
            p = lm[idx]
        except IndexError:
            return None
        return np.array([p.x, p.y], dtype=float)

    # 1) 空中横叉角度（以骨盆中心为顶点，左右踝为端点）
    # 我给你改成了以每侧髋部向下的线与该侧腿的夹角之和
    split_angles = []

    for i in range(f_start, f_end + 1):
        lm = landmark_seq[i]
        hl = get_xy(lm, LEFT_HIP)
        hr = get_xy(lm, RIGHT_HIP)
        la = get_xy(lm, LEFT_ANKLE)
        ra = get_xy(lm, RIGHT_ANKLE)

        if any(p is None for p in [hl, hr, la, ra]):
            print("Skipping as the critical point is missed (hips or ankle).")
            continue

        hl_down = hl + np.array([0, 1.0])
        hr_down = hr + np.array([0, 1.0])
        left_angle  = _angle(la, hl, hl_down)
        right_angle = _angle(ra, hr, hr_down)
        split_angles.append(left_angle + right_angle)

    split_angle = float(max(split_angles)) if split_angles else 0.0



    # 2) 吸撩腿屈髋角（peak 帧：躯干与前腿的夹角）
    lm_peak = landmark_seq[peak_idx]
    hip = get_xy(lm_peak, LEFT_HIP)
    knee = get_xy(lm_peak, LEFT_KNEE)
    shoulder = get_xy(lm_peak, LEFT_SHOULDER)
    hip_flex = 0.0
    if all(p is not None for p in [hip, knee, shoulder]):
        # 躯干向量：hip -> shoulder，腿向量：hip -> knee
        hip_flex = _angle(shoulder, hip, knee)

    # 3) 躯干直立度（peak 帧：左右肩中点 -> 骨盆中点 与 垂直线夹角）
    ls = get_xy(lm_peak, LEFT_SHOULDER)
    rs = get_xy(lm_peak, RIGHT_SHOULDER)
    if all(p is not None for p in [ls, rs, hip, shoulder]):
        torso_top = (ls + rs) / 2.0
        pelvis = hip  # 近似
        v = torso_top - pelvis
        # 与竖直方向 (0,-1) 的夹角，越小说明越直
        up = np.array([0.0, -1.0])
        denom = np.linalg.norm(v) * np.linalg.norm(up) + 1e-8
        cos_val = np.dot(v, up) / denom
        cos_val = np.clip(cos_val, -1.0, 1.0)
        torso_upright = float(np.degrees(np.arccos(cos_val)))
    else:
        torso_upright = 90.0

    # 4) 落地稳定性（落地前后几帧膝关节角度的标准差）
    landing_frames = range(max(0, f_end - 5), f_end + 1)
    knee_angles = []
    for i in landing_frames:
        lm = landmark_seq[i]
        hl = get_xy(lm, LEFT_HIP)
        kl = get_xy(lm, LEFT_KNEE)
        al = get_xy(lm, LEFT_ANKLE)
        if any(p is None for p in [hl, kl, al]):
            continue
        knee_angles.append(_angle(hl, kl, al))
    landing_stab = float(np.std(knee_angles)) if knee_angles else 0.0

    # ... (analyze_xiliao 函数内，所有指标计算完毕) ...

    metrics = {
        "split_angle": split_angle,
        "flight_time": flight_time,
        "hip_flex": hip_flex,
        "torso_upright": torso_upright,
        "landing_stability": landing_stab,
    }

# ... (analyze_xiliao 函数内，所有指标计算完毕) ...

    # 找到腾空段的帧索引
    f_start, f_end = detect_flight_frames(landmark_seq, fps)
    
    # === 1. 定义辅助函数（安全获取膝盖角度） ===
    def get_knee_angle(lm, hip_idx, knee_idx, ankle_idx):
        # ... (此辅助函数保持不变) ...
        try:
            # 使用全局定义的 _get_xy 来获取坐标
            h = _get_xy(lm, hip_idx)
            k = _get_xy(lm, knee_idx)
            a = _get_xy(lm, ankle_idx)
            # 使用列表检查 None
            if any(p is None for p in [h, k, a]):
                return 0.0  # 关键点缺失返回 0 度
            return _angle(h, k, a)
        except Exception:
            return 0.0
            
    # === 2. 计算峰值帧膝盖角度 (定义 front_knee 和 back_knee) ===
    # lm_peak 已在函数上部定义
    l_knee_angle = get_knee_angle(lm_peak, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)
    r_knee_angle = get_knee_angle(lm_peak, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)
    
    # 根据传入的 is_left_lead 确定前后腿
    if is_left_lead:
        front_knee = l_knee_angle
        back_knee = r_knee_angle
        front_knee_idx = LEFT_KNEE
        back_knee_idx = RIGHT_KNEE
    else:
        front_knee = r_knee_angle
        back_knee = l_knee_angle
        front_knee_idx = RIGHT_KNEE
        back_knee_idx = LEFT_KNEE

    # === 3. 引入动态特征检查：吸腿最小屈膝角度 ===
    # 我们只检查腾空段的前半段（例如：从 f_start 到 peak_idx）
    min_front_knee_angle_during_flight = 180.0
    
    # 遍历腾空段的前半段
    # 注意：如果 peak_idx 和 f_start 相同，这个循环不会执行。
    # 我们可以稍微扩大搜索范围到腾空前的几帧，但这里我们只专注于腾空段。
    
    search_end = min(peak_idx + int((f_end - peak_idx) / 2), f_end) # 检查到峰值点和结束点之间
    
    for i in range(f_start, search_end + 1):
        lm = landmark_seq[i]
        if lm is None:
            continue
        
        # 提取前导腿的膝盖角度
        if is_left_lead:
            hip_idx, knee_idx, ankle_idx = LEFT_HIP, LEFT_KNEE, LEFT_ANKLE
        else:
            hip_idx, knee_idx, ankle_idx = RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE
            
        current_knee_angle = get_knee_angle(lm, hip_idx, knee_idx, ankle_idx)
        
        if current_knee_angle > 0.0: # 确保关键点存在
            min_front_knee_angle_during_flight = min(min_front_knee_angle_during_flight, current_knee_angle)

    # === 4. 重新汇总指标到 metrics 字典（包含膝盖角度和动态特征） ===
    metrics = {
        # 1 起跳屈膝角
        "prep_knee_angle": min_front_knee_angle_during_flight,
        # 2 腾空高度与持续
        "flight_time": flight_time,
        # 3 空中横叉角度
        "split_angle_max": split_angle,
        # 4 前腿伸膝线条
        "front_knee_angle": front_knee,
        # 5 后腿伸膝线条
        "back_knee_min": back_knee,
        # 6 吸撩腿屈髋 / 骨盆打开
        "pelvis_opening": hip_flex,
        # 7 空中躯干稳定性
        "torso_upright": torso_upright,
        # 8 落地稳定性
        "landing_stability": landing_stab,
    }

    # === 5. 动作验证 (防止非目标动作得分) ===
    MIN_FLIGHT_TIME = 0.20
    MIN_SPLIT_ANGLE = 120.0
    MAX_TORSO_ANGLE = 35.0
    MIN_KNEE_STRAIGHT = 160.0
    MAX_PREP_KNEE_ANGLE = 120.0 # 核心要求：腾空过程中前腿必须先大幅屈膝（角度要小）

    if (flight_time < MIN_FLIGHT_TIME or 
        split_angle < MIN_SPLIT_ANGLE or
        torso_upright > MAX_TORSO_ANGLE or
        front_knee < MIN_KNEE_STRAIGHT or
        back_knee < MIN_KNEE_STRAIGHT or
        min_front_knee_angle_during_flight > MAX_PREP_KNEE_ANGLE): # <-- 新增的动态检查
        
        # 如果判定为无效动作，直接返回 0 分
        invalid_metrics = {k: 0.0 for k in metrics.keys()}
        invalid_scores = {k: 0.0 for k in metrics.keys()}
        
        # 保留真实指标值，便于调试
        invalid_metrics.update(metrics) # 直接更新整个 metrics 字典
        return invalid_metrics, invalid_scores
        
    # -------- 更专业的 8 维评分（0-100） --------
    scores = {}

    # 1) 起跳屈膝 prep_knee_angle：70 ~ 130，中间最好（沿用你原来的区间）
    prep = min_front_knee_angle_during_flight
    if prep > 130:
        s_prep = 60.0
    elif prep > 110:
        # 110–130: 70–80
        s_prep = 70.0 + (130 - prep) / (130 - 110) * 10.0
    elif prep > 90:
        # 90–110: 80–95
        s_prep = 80.0 + (110 - prep) / (110 - 90) * 15.0
    elif prep >= 70:
        # 70–90: 95–100
        s_prep = 95.0 + (90 - prep) / (90 - 70) * 5.0
    else:
        # <70: 太深，略扣一点
        s_prep = 90.0
    scores["prep_knee_angle"] = float(np.clip(s_prep, 0, 100))

    # 2) 腾空时间 flight_time：0.28 ~ 0.50+（保留你原来的区间）
    ft = flight_time
    if ft < 0.28:
        s_ft = 55.0
    elif ft < 0.38:
        # 0.28–0.38: 70–85
        s_ft = 70.0 + (ft - 0.28) / (0.38 - 0.28) * 15.0
    elif ft <= 0.50:
        # 0.38–0.50: 85–100
        s_ft = 85.0 + (ft - 0.38) / (0.50 - 0.38) * 15.0
    else:
        s_ft = 100.0
    scores["flight_time"] = float(np.clip(s_ft, 0, 100))

    # 3) 空中横叉 split_angle_max：120 ~ 200+
    sa = split_angle
    if sa < 120:
        s_sa = 50.0
    elif sa < 160:
        # 120–160: 70–90
        s_sa = 70.0 + (sa - 120) / (160 - 120) * 20.0
    elif sa < 180:
        # 160–180: 90–98
        s_sa = 90.0 + (sa - 160) / (180 - 160) * 8.0
    else:
        s_sa = 100.0
    scores["split_angle_max"] = float(np.clip(s_sa, 0, 100))

    # 4) 前腿伸膝 front_knee_angle：150 ~ 180 (越大越好)
    fk = front_knee
    if fk < 150:
        s_fk = 60.0
    elif fk < 165:
        # 150–165: 70–85
        s_fk = 70.0 + (fk - 150) / (165 - 150) * 15.0
    elif fk <= 175:
        # 165–175: 85–95
        s_fk = 85.0 + (fk - 165) / (175 - 165) * 10.0
    else:
        s_fk = 100.0
    scores["front_knee_angle"] = float(np.clip(s_fk, 0, 100))

    # 5) 后腿伸膝 back_knee_min：145 ~ 180 (略宽松)
    bk = back_knee
    if bk < 145:
        s_bk = 60.0
    elif bk < 160:
        # 145–160: 70–85
        s_bk = 70.0 + (bk - 145) / (160 - 145) * 15.0
    elif bk <= 175:
        # 160–175: 85–95
        s_bk = 85.0 + (bk - 160) / (175 - 160) * 10.0
    else:
        s_bk = 100.0
    scores["back_knee_min"] = float(np.clip(s_bk, 0, 100))

    # 6) 吸撩腿屈髋 / 骨盆打开 pelvis_opening：60 ~ 120+
    hf = hip_flex
    if hf < 60:
        s_hf = 55.0
    elif hf < 80:
        # 60–80: 70–85
        s_hf = 70.0 + (hf - 60) / (80 - 60) * 15.0
    elif hf <= 120:
        # 80–120: 85–100
        s_hf = 85.0 + (hf - 80) / (120 - 80) * 15.0
    else:
        s_hf = 100.0
    scores["pelvis_opening"] = float(np.clip(s_hf, 0, 100))

    # 7) 躯干直立 torso_upright：0 ~ 35 (越小越好)
    tu = torso_upright
    if tu >= 35:
        s_tu = 60.0
    elif tu >= 25:
        # 25–35: 70–80
        s_tu = 70.0 + (35 - tu) / (35 - 25) * 10.0
    elif tu >= 10:
        # 10–25: 80–95
        s_tu = 80.0 + (25 - tu) / (25 - 10) * 15.0
    else:
        # <10: 95–100
        s_tu = 95.0 + (10 - tu) / 10.0 * 5.0
    scores["torso_upright"] = float(np.clip(s_tu, 0, 100))

    # 8) 落地稳定性 landing_stability：std 3 ~ 10 (越小越好)
    ls_val = landing_stab
    if ls_val >= 10:
        s_ls = 60.0
    elif ls_val >= 6:
        # 6–10: 70–85
        s_ls = 70.0 + (10 - ls_val) / (10 - 6) * 15.0
    elif ls_val >= 3:
        # 3–6: 85–95
        s_ls = 85.0 + (6 - ls_val) / (6 - 3) * 10.0
    else:
        # <3: 95–100
        s_ls = 100.0
    scores["landing_stability"] = float(np.clip(s_ls, 0, 100))

    return metrics, scores


# ======================= 7. 规则型 AI 建议 =======================

def generate_advice(mode_key: str, scores: Dict[str, float], lang: str) -> List[str]:
    adv: List[str] = []

    if mode_key == "xiliao":
        # 统一 8 维 key
        prep = scores.get("prep_knee_angle", 0)
        ft = scores.get("flight_time", 0)
        sa = scores.get("split_angle_max", 0)
        fk = scores.get("front_knee_angle", 0)
        bk = scores.get("back_knee_min", 0)
        hf = scores.get("pelvis_opening", 0)
        tu = scores.get("torso_upright", 0)
        ls_val = scores.get("landing_stability", 0)

        # 1) 空中横叉
        if sa < 80:
            if lang == "中文":
                adv.append("空中横叉角度偏小，可加强前后腿劈叉柔韧与跳跃配合训练（压腿 + 原地小跳 / 组合跳跃）。")
            elif lang == "한국어":
                adv.append("공중 다리 벌림 각도가 부족합니다. 전후 스플릿 유연성과 점프를 함께 훈련하세요.")
            else:
                adv.append("Air split angle is limited. Work on flexibility and power for front and back splits with jump drills.")

        # 2) 腾空时间
        if ft < 80:
            if lang == "中文":
                adv.append("腾空时间略短，可通过加深屈膝预备、增强下肢推蹬力量来提升体空感。")
            elif lang == "한국어":
                adv.append("체공 시간이 다소 짧습니다. 깊은 플리에와 하체 추진력을 통해 체공감을 높이세요.")
            else:
                adv.append("Flight time is slightly short. Use deeper plié and stronger push-off to increase airtime.")

        # 3) 吸撩腿屈髋
        if hf < 80:
            if lang == "中文":
                adv.append("吸撩腿屈髋角度不足，可增加前腿主动抬腿、摆腿和腹股沟力量训练。")
            elif lang == "한국어":
                adv.append("흡요퇴 굴곡 각도가 부족합니다. 앞다리 능동 리프트와 고관절·코어 근력을 강화하세요.")
            else:
                adv.append("Hip flexion is limited. Strengthen active leg lifts and hip flexor/core conditioning.")

        # 4) 躯干直立
        if tu < 70:  # 分数高，不用提醒
            pass
        elif tu < 85:
            if lang == "中文":
                adv.append("空中躯干略有前倾/后仰，建议在跳跃练习中加入上身控制与核心稳定训练。")
            elif lang == "한국어":
                adv.append("공중에서 상체가 약간 흔들립니다. 점프 중 상체 컨트롤과 코어 안정성을 훈련하세요.")
            else:
                adv.append("Torso alignment in the air can be more stable. Focus on core engagement during jumps.")
        else:
            if lang == "中文":
                adv.append("空中躯干稳定性较弱，可结合平衡练习与慢速分解跳，专注上身不晃动。")
            elif lang == "한국어":
                adv.append("공중 상체 정렬이 많이 흐트러집니다. 균형 훈련과 슬로우 점프 분해 연습을 병행하세요.")
            else:
                adv.append("Torso stability is weak in the air. Combine balance work with slow-motion jump breakdowns.")

        # 5) 落地稳定
        if ls_val < 80:
            if lang == "中文":
                adv.append("落地时膝关节控制略不稳定，可增加单脚缓冲、蹲跃和下肢力量训练，避免冲击性伤害。")
            elif lang == "한국어":
                adv.append("착지 시 무릎 컨트롤이 다소 불안정합니다. 한발 착지와 하체 근력 훈련으로 충격을 줄이세요.")
            else:
                adv.append("Landing stability can be improved. Practice single-leg landings and lower-body strength to reduce impact.")

        # 6) 前腿伸膝
        if fk < 80:
            if lang == "中文":
                adv.append("前腿伸膝线条不够干净，可针对性练习“绷脚 + 直膝”的连续摆腿与控腿。")
            elif lang == "한국어":
                adv.append("앞다리 무릎 라인이 다소 흐립니다. 포인과 무릎 신전을 동시에 유지하는 다리 스윙을 반복 연습하세요.")
            else:
                adv.append("Front knee line is not fully extended. Drill repeated leg swings focusing on knee extension and pointed foot.")

        # 7) 后腿伸膝
        if bk < 80:
            if lang == "中文":
                adv.append("后腿略显拖腿，建议在扶把练习中强化后腿主动伸展与髋关节打开。")
            elif lang == "한국어":
                adv.append("뒷다리가 약간 끌리는 느낌입니다. 바에서 뒷다리 신전과 고관절 오픈을 강화하세요.")
            else:
                adv.append("Back leg tends to drag. Strengthen active extension and hip opening for the back leg at the barre.")

        # 8) 起跳屈膝
        if prep < 75:
            if lang == "中文":
                adv.append("起跳屈膝过浅，腾空高度受限，可适当加深助跳 plié 并保持脚底推地。")
            elif lang == "한국어":
                adv.append("도약 준비 플리에가 얕아 체공이 제한됩니다. 적절히 더 깊게 앉아 지면을 밀어 올리세요.")
            else:
                adv.append("Prep knee bend is too shallow, which limits height. Try a slightly deeper plié with strong push-off.")
        elif prep > 120:
            if lang == "中文":
                adv.append("起跳屈膝过深，容易导致起跳迟缓，可控制屈膝角度在适中范围，提升反弹感。")
            elif lang == "한국어":
                adv.append("도약 준비에서 무릎을 너무 깊게 굽혀 동작이 무거워질 수 있습니다. 적당한 깊이에서 탄성을 살려보세요.")
            else:
                adv.append("Prep knee bend is too deep, making the jump heavy. Aim for a moderate depth with more rebound.")

        # 如果一句建议都没有，就给总体性鼓励
        if not adv:
            if lang == "中文":
                adv.append("本次吸撩腿跃在技术与控制上都较为均衡，可进一步在上身线条、视线与音乐表现上深化舞台效果。")
            elif lang == "한국어":
                adv.append("이번 흡요퇴 점프는 전반적으로 균형 잡힌 기술 수행을 보입니다. 상체 라인과 시선, 음악 표현을 더 살려보세요.")
            else:
                adv.append("Your Xi-Liao leap is technically well-balanced. Next, focus on upper-body lines, eye focus, and musicality.")

    return adv

# ======================= 8. 页面主体 =======================

st.title(TEXT["app_title"])
st.caption(TEXT["subtitle"])
st.markdown("")

uploaded_file = st.file_uploader(
    TEXT["upload_video"],
    type=["mp4", "mov", "avi"],
)

if not uploaded_file:
    st.info("👆 请在上方上传包含单次跳跃动作的视频。")
    st.stop()

# 保存上传视频到临时文件
with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
    tmp.write(uploaded_file.read())
    video_path = tmp.name

with st.spinner(TEXT["processing"]):
    frames_data, landmark_seq, fps, overlay_path, (start_idx, peak_idx, end_idx), nose_traj = process_video(
        video_path
    )

    # 评分：根据模式切换
    
    # 确保 is_left_lead_value 在此作用域内被定义，我们默认左腿主导
    is_left_lead_value = st.session_state.get("is_left_lead", True) 

    if mode_key == "ballet":
        metrics, scores = analyze_grand_jete(landmark_seq, fps, is_left_lead=is_left_lead_value)
    else:
        # 修复 IndentationError 和 NameError
        metrics, scores = analyze_xiliao(landmark_seq, fps, is_left_lead=is_left_lead_value) 
        
    # 综合得分
    if scores:
        overall_score = float(np.mean(list(scores.values())))
    else:
        overall_score = 0.0

# ======================= 9. 展示：关键帧 =======================

st.markdown(f"### {TEXT['section_keyframes']}")

c1, c2, c3 = st.columns(3)

start_img = frames_data[start_idx]["image"]
peak_img = frames_data[peak_idx]["image"]
end_img = frames_data[end_idx]["image"]

with c1:
    st.image(start_img, caption="Start / 起势", use_container_width=True)
with c2:
    st.image(peak_img, caption="Peak / 最高点", use_container_width=True)
with c3:
    st.image(end_img, caption="Landing / 落地", use_container_width=True)

st.markdown("---")

# ======================= 10. 综合评分 + 雷达 =======================

st.markdown(f"### {TEXT['section_score']}")

metric_label_map = TEXT["metric_labels"][mode_key]

# 先单独展示综合得分
st.metric(TEXT["overall"], f"{overall_score:.1f}")
st.markdown("")

# 再分行展示各维度分数（每行最多 4 个）
score_items = list(scores.items())
row_size = 4

for row_start in range(0, len(score_items), row_size):
    row_items = score_items[row_start: row_start + row_size]
    cols = st.columns(len(row_items))
    for (key, val), col in zip(row_items, cols):
        label = metric_label_map.get(key, key)
        with col:
            st.metric(label, f"{val:.1f}")

st.markdown("---")

c_radar, c_traj = st.columns(2)

with c_radar:
    st.subheader(TEXT["section_radar"])
    labels = [metric_label_map.get(k, k) for k in scores.keys()]
    values = list(scores.values())
    if values:
        fig_radar = go.Figure(
            data=go.Scatterpolar(
                r=values + [values[0]],
                theta=labels + [labels[0]],
                fill="toself",
                line_color="#003366",
                fillcolor="rgba(0,51,102,0.3)",
            )
        )
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10),
            height=360,
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    else:
        st.info("暂无可视化数据。")

with c_traj:
    st.subheader(TEXT["section_traj"])
    traj_df = pd.DataFrame(
        {
            "Frame": list(range(len(nose_traj))),
            "HeightInv": [-y if not np.isnan(y) else np.nan for y in nose_traj],
        }
    )
    fig_line = px.line(traj_df, x="Frame", y="HeightInv")
    fig_line.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )
    st.plotly_chart(fig_line, use_container_width=True)

st.markdown("---")

# ======================= 11. 明细表格 =======================

st.markdown(f"### {TEXT['section_detail']}")

detail_rows = []
for k, v in metrics.items():
    label = metric_label_map.get(k, k)
    s_val = scores.get(k, 0.0)
    detail_rows.append(
        {
            "指标 / Metric": label,
            "测量值 / Value": f"{v:.4f}",
            "得分 / Score": f"{s_val:.1f}",
        }
    )

if detail_rows:
    detail_df = pd.DataFrame(detail_rows)
    st.dataframe(detail_df, use_container_width=True, hide_index=True)

st.markdown("---")

# ======================= 12. 建议 + 导出 =======================

c_adv, c_export = st.columns([2, 1])

with c_adv:
    st.subheader(TEXT["section_advice"])
    advice_list = generate_advice(mode_key, scores, st.session_state["lang"])
    if advice_list:
        for adv in advice_list:
            st.info(adv)
    else:
        st.info("暂无建议。")

with c_export:
    st.subheader(TEXT["section_export"])
    st.write("")

    # CSV 导出 (保持不变)
    csv_df = pd.DataFrame([{"metric": k, "score": v} for k, v in scores.items()])
    csv_bytes = csv_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        TEXT["csv_btn"],
        data=csv_bytes,
        file_name="dance_scores.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # PDF 报告 (已修复：包含三个关键帧和轨迹)
    if st.button(TEXT["pdf_btn"], type="primary", use_container_width=True):
        
        # 1. 封装关键帧图像数据
        key_frames_imgs = {
            "start": start_img,
            "peak": peak_img,
            "end": end_img,
        }
        temp_paths = {}
        temp_files_to_delete = []

        # 辅助函数：保存图像到临时文件
        def save_temp_image(img):
            if img is not None:
                # 使用 tempfile 确保文件名唯一且安全
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                path = temp_file.name
                temp_file.close() # 必须先关闭文件句柄才能写入
                
                # 使用 cv2.imwrite 保存 RGB 图像 (需要转回 BGR)
                cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                temp_files_to_delete.append(path)
                return path
            return None
        
        # 保存所有关键帧
        temp_paths["start"] = save_temp_image(key_frames_imgs["start"])
        temp_paths["peak"] = save_temp_image(key_frames_imgs["peak"])
        temp_paths["end"] = save_temp_image(key_frames_imgs["end"])

        # 2. 准备 PDF 文件路径
        pdf_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        pdf_path = pdf_temp_file.name
        pdf_temp_file.close()
        temp_files_to_delete.append(pdf_path) # 即使成功下载也要清理

        # 3. 动作名称
        if mode_key == "ballet":
            action_cn = TEXT["action_name_ballet_cn"]
            action_en = TEXT["action_name_ballet_en"]
        else:
            action_cn = TEXT["action_name_xiliao_cn"]
            action_en = TEXT["action_name_xiliao_en"]

        # 4. 调用 generate_pdf 函数 (传入全部三张图和轨迹)
        with st.spinner(TEXT["processing"].replace("分析视频", "生成报告")):
            
            # --- 基本信息获取 (确保模式同步) ---
            # 注意：此处使用 mode_key（"xiliao" 或 "ballet"），而非 session_state["mode"]
            current_mode = st.session_state.get("mode_key", "xiliao") 
            subject_id = st.session_state.get("subject_id", "") or "N/A"
            # overall_score 使用之前计算的综合得分

            # --- 调用 PDF 函数 ---
            generate_pdf(
                pdf_path,
                dict(TEXT),     # 语言包
                metrics,
                scores,
                advice_list,    
                subject_id,
                action_cn,      
                action_en,      
                overall_score,
                temp_paths["peak"],      # Peak 帧
                temp_paths["start"],     # Start 帧
                temp_paths["end"],       # End 帧
                nose_traj,               # 腾空轨迹 (作为列表传入)
                lang_code=st.session_state["lang"],
            )

        # 5. 读取 PDF 内容并提供下载按钮
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        st.success(TEXT["pdf_ready"])
        st.download_button(
            TEXT["pdf_dl"],
            data=pdf_bytes,
            file_name=f"{subject_id}_dance_report.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="final_pdf_download"
        )
        
        # 6. 清理临时文件
        for path in temp_files_to_delete:
            if os.path.exists(path):
                os.unlink(path)

st.markdown("---")

st.markdown("✅ 当前为 Pro 版界面：支持中国古典吸撩腿跃 & 芭蕾 Grand Jeté 双模态评估。")
