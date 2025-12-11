"""
grand_jete_model.py
升级版：8 个 AI 技术指标
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np

# -------- MediaPipe Pose 关键点索引（v0.10.x）---------
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28


@dataclass
class GrandJeteMetrics:
    flight_time: float          # 体空时间 (s)
    split_angle_max: float      # 空中最大横叉角 (°)
    pelvis_opening: float       # 空中骨盆打开 / 屈髋角 (°，前腿髋部)
    back_knee_min: float        # 空中后腿伸膝角 (°，越接近 180 越好)
    prep_knee_angle: float      # 起跳屈膝角 (°，起跳前几帧前腿膝角平均)

    front_knee_angle: float     # 空中前腿伸膝角 (°，峰值帧)
    torso_upright: float        # 空中躯干直立度：与垂直线的夹角 (°，越小越直)
    arm_line: float             # 空中手臂线条：左右肘角平均 (°，越接近 180 越直)



# ----------------- 小工具函数 -----------------

def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """三点夹角，单位：度"""
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cos_val = np.dot(ba, bc) / denom
    cos_val = np.clip(cos_val, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_val)))


def _get_xy(landmarks, idx: int) -> Optional[np.ndarray]:
    if landmarks is None:
        return None
    try:
        lm = landmarks[idx]
    except IndexError:
        return None
    return np.array([lm.x, lm.y], dtype=float)


def _is_valid_triplet(pts) -> bool:
    return all(p is not None for p in pts)


def _trunk_angle(landmarks) -> Optional[float]:
    """
    NEW: 躯干相对垂直方向的倾斜角（度数，0 = 完全竖直）。
    计算方法：骨盆中心 -> 肩部中心向量与垂直方向的夹角。
    """
    lh = _get_xy(landmarks, LEFT_HIP)
    rh = _get_xy(landmarks, RIGHT_HIP)
    ls = _get_xy(landmarks, LEFT_SHOULDER)
    rs = _get_xy(landmarks, RIGHT_SHOULDER)
    if any(p is None for p in [lh, rh, ls, rs]):
        return None
    pelvis = (lh + rh) / 2.0
    shoulder = (ls + rs) / 2.0
    v = shoulder - pelvis  # 从骨盆指向肩部
    # 垂直方向向量（向上）
    vertical = np.array([0.0, -1.0])
    denom = (np.linalg.norm(v) * np.linalg.norm(vertical) + 1e-8)
    cos_val = np.dot(v, vertical) / denom
    cos_val = np.clip(cos_val, -1.0, 1.0)
    ang = float(np.degrees(np.arccos(cos_val)))  # 与垂直的夹角
    return ang


# ----------------- 1. 检测起跳 / 落地，计算体空时间 -----------------

def detect_flight_frames(landmark_seq: List, fps: float) -> Tuple[int, int]:
    """
    粗略检测“腾空段”：两脚踝 y 值显著高于起跳基线时的连续帧。
    返回：(start_frame_idx, end_frame_idx)，若失败则返回 (0, len-1)
    """
    n = len(landmark_seq)
    if n < 3:
        return 0, max(n - 1, 0)

    # 用前三帧的脚踝 y 平均作为“地面基线”
    y_list = []
    for i in range(min(5, n)):
        lank = _get_xy(landmark_seq[i], LEFT_ANKLE)
        rank = _get_xy(landmark_seq[i], RIGHT_ANKLE)
        if lank is not None:
            y_list.append(lank[1])
        if rank is not None:
            y_list.append(rank[1])
    if not y_list:
        return 0, max(n - 1, 0)
    ground_y = float(np.median(y_list))

    threshold = 0.04  # 根据你的视频分辨率可适当微调
    airborne = []
    for i, lm in enumerate(landmark_seq):
        lank = _get_xy(lm, LEFT_ANKLE)
        rank = _get_xy(lm, RIGHT_ANKLE)
        if lank is None or rank is None:
            airborne.append(False)
            continue
        high_enough = (ground_y - lank[1] > threshold) and (ground_y - rank[1] > threshold)
        airborne.append(high_enough)

    best_start = 0
    best_end = n - 1
    cur_start = None
    for i, flag in enumerate(airborne):
        if flag and cur_start is None:
            cur_start = i
        elif not flag and cur_start is not None:
            cur_len = i - cur_start
            best_len = best_end - best_start
            if cur_len > best_len:
                best_start, best_end = cur_start, i - 1
            cur_start = None
    if cur_start is not None:
        cur_len = n - cur_start
        best_len = best_end - best_start
        if cur_len > best_len:
            best_start, best_end = cur_start, n - 1

    return best_start, best_end


# ----------------- 2. 计算各技术指标 -----------------

def compute_metrics_for_grand_jete(
    landmark_seq: List,
    fps: float,
    is_left_lead: bool = True,
) -> GrandJeteMetrics:
    """
    统一 8 个指标的“原始测量值”，供芭蕾使用：
    1) flight_time        腾空时间
    2) split_angle_max    空中横叉角
    3) pelvis_opening     空中骨盆打开 / 屈髋角（前腿肩-髋-膝夹角）
    4) back_knee_min      空中后腿伸膝
    5) prep_knee_angle    起跳屈膝（起跳前几帧前腿膝角平均）
    6) front_knee_angle   空中前腿伸膝（峰值帧）
    7) torso_upright      空中躯干直立度（峰值帧，越小越直）
    8) arm_line           空中手臂三位手线条（左右肘角平均）
    """
    n = len(landmark_seq)
    if n == 0:
        return GrandJeteMetrics(
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0
        )

    # 1) 腾空段 + 腾空时间
    f_start, f_end = detect_flight_frames(landmark_seq, fps)
    flight_time = max(0.0, (f_end - f_start + 1) / fps)

    # 2) 前腿 / 后腿
    if is_left_lead:
        front_hip, front_knee, front_ankle = LEFT_HIP, LEFT_KNEE, LEFT_ANKLE
        back_hip, back_knee, back_ankle = RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE
        front_shoulder = LEFT_SHOULDER
    else:
        front_hip, front_knee, front_ankle = RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE
        back_hip, back_knee, back_ankle = LEFT_HIP, LEFT_KNEE, LEFT_ANKLE
        front_shoulder = RIGHT_SHOULDER

    # 3) 峰值帧（nose y 最小）
    peak_idx = f_start
    min_nose_y = 1.0
    for i in range(f_start, f_end + 1):
        nose = _get_xy(landmark_seq[i], NOSE)
        if nose is None:
            continue
        if nose[1] < min_nose_y:
            min_nose_y = nose[1]
            peak_idx = i

    # 4) 空中横叉角：两侧髋向下的竖线 与 腿的夹角之和
    split_angles = []
    for i in range(f_start, f_end + 1):
        hip_l = _get_xy(landmark_seq[i], LEFT_HIP)
        hip_r = _get_xy(landmark_seq[i], RIGHT_HIP)
        f_ank = _get_xy(landmark_seq[i], front_ankle)
        b_ank = _get_xy(landmark_seq[i], back_ankle)
        if any(p is None for p in [hip_l, hip_r, f_ank, b_ank]):
            continue

        hip_l_down = hip_l + np.array([0.0, 1.0])
        hip_r_down = hip_r + np.array([0.0, 1.0])

        if is_left_lead:
            front_hip_xy, front_down = hip_l, hip_l_down
            back_hip_xy,  back_down  = hip_r, hip_r_down
        else:
            front_hip_xy, front_down = hip_r, hip_r_down
            back_hip_xy,  back_down  = hip_l, hip_l_down

        front_angle = _angle(f_ank, front_hip_xy, front_down)
        back_angle  = _angle(b_ank, back_hip_xy, back_down)
        split_angles.append(front_angle + back_angle)

    split_angle_max = float(max(split_angles)) if split_angles else 0.0

    # 5) 后腿伸膝：腾空段内后腿膝角最小值（越接近 180 越好）
    back_knee_angles = []
    for i in range(f_start, f_end + 1):
        h = _get_xy(landmark_seq[i], back_hip)
        k = _get_xy(landmark_seq[i], back_knee)
        a = _get_xy(landmark_seq[i], back_ankle)
        if not _is_valid_triplet((h, k, a)):
            continue
        back_knee_angles.append(_angle(h, k, a))
    back_knee_min = float(min(back_knee_angles)) if back_knee_angles else 0.0

    # 6) 起跳屈膝：腾空开始前几帧前腿膝角平均
    prep_frames = range(max(0, f_start - 4), f_start)
    prep_angles = []
    for i in prep_frames:
        h = _get_xy(landmark_seq[i], front_hip)
        k = _get_xy(landmark_seq[i], front_knee)
        a = _get_xy(landmark_seq[i], front_ankle)
        if not _is_valid_triplet((h, k, a)):
            continue
        prep_angles.append(_angle(h, k, a))
    prep_knee_angle = float(np.mean(prep_angles)) if prep_angles else 0.0

    # 7) 峰值帧的前腿屈髋（pelvis_opening）与前后腿伸膝
    lm_peak = landmark_seq[peak_idx]

    fh = _get_xy(lm_peak, front_hip)
    fk = _get_xy(lm_peak, front_knee)
    fa = _get_xy(lm_peak, front_ankle)
    fs = _get_xy(lm_peak, front_shoulder)

    bh = _get_xy(lm_peak, back_hip)
    bk = _get_xy(lm_peak, back_knee)
    ba = _get_xy(lm_peak, back_ankle)

    # pelvis_opening：前腿肩-髋-膝夹角（其实就是“屈髋角”）
    if all(p is not None for p in [fh, fk, fs]):
        pelvis_opening = _angle(fs, fh, fk)
    else:
        pelvis_opening = 0.0

    # front / back 膝角：峰值帧
    front_knee_angle = _angle(fh, fk, fa) if all(p is not None for p in [fh, fk, fa]) else 0.0
    # 用 peak 帧的后腿膝角补充一份（与 back_knee_min 相互印证）
    back_knee_peak = _angle(bh, bk, ba) if all(p is not None for p in [bh, bk, ba]) else back_knee_min
    if back_knee_min == 0.0:
        back_knee_min = back_knee_peak

    # 8) 空中躯干直立度：峰值帧与垂直线夹角
    t_ang = _trunk_angle(lm_peak)
    torso_upright = float(t_ang) if t_ang is not None else 0.0

    # 9) 空中手臂线条：左右肘角平均
    ls = _get_xy(lm_peak, LEFT_SHOULDER)
    le = _get_xy(lm_peak, LEFT_ELBOW)
    lw = _get_xy(lm_peak, LEFT_WRIST)
    rs = _get_xy(lm_peak, RIGHT_SHOULDER)
    re = _get_xy(lm_peak, RIGHT_ELBOW)
    rw = _get_xy(lm_peak, RIGHT_WRIST)

    elbow_angles = []
    if all(p is not None for p in [ls, le, lw]):
        elbow_angles.append(_angle(ls, le, lw))
    if all(p is not None for p in [rs, re, rw]):
        elbow_angles.append(_angle(rs, re, rw))

    arm_line = float(np.mean(elbow_angles)) if elbow_angles else 0.0

    return GrandJeteMetrics(
        flight_time=flight_time,
        split_angle_max=split_angle_max,
        pelvis_opening=pelvis_opening,
        back_knee_min=back_knee_min,
        prep_knee_angle=prep_knee_angle,
        front_knee_angle=front_knee_angle,
        torso_upright=torso_upright,
        arm_line=arm_line,
    )

# ----------------- 3. 将指标转换为 0–100 分 -----------------

def score_grand_jete(metrics: GrandJeteMetrics) -> Dict[str, float]:
    """Grand Jeté 8 指标 → 0–100 分"""

    m = metrics
    scores: Dict[str, float] = {}

    # 1) flight_time：0.45~0.70s → 60~100
    ft = m.flight_time
    if ft <= 0.45:
        s_ft = 50.0
    elif ft >= 0.70:
        s_ft = 100.0
    else:
        s_ft = 60.0 + (ft - 0.45) / (0.70 - 0.45) * 40.0
    scores["flight_time"] = float(np.clip(s_ft, 0, 100))

    # 2) split_angle_max：140~180+ (越大越好)
    sa = m.split_angle_max
    if sa < 140:
        s_sa = 50.0
    elif sa < 160:
        s_sa = 60.0 + (sa - 140) / 20.0 * 15.0   # 140–160: 60–75
    elif sa < 180:
        s_sa = 75.0 + (sa - 160) / 20.0 * 15.0   # 160–180: 75–90
    else:
        s_sa = 100.0
    scores["split_angle_max"] = float(np.clip(s_sa, 0, 100))

    # 3) pelvis_opening（前腿屈髋 / 骨盆打开）：90~150 (越大越好)
    po = m.pelvis_opening
    if po < 90:
        s_po = 60.0
    elif po < 120:
        s_po = 70.0 + (po - 90) / 30.0 * 15.0    # 90–120: 70–85
    elif po <= 150:
        s_po = 85.0 + (po - 120) / 30.0 * 15.0   # 120–150: 85–100
    else:
        s_po = 100.0
    scores["pelvis_opening"] = float(np.clip(s_po, 0, 100))

    # 4) back_knee_min：145~180 (越大越好)
    bk = m.back_knee_min
    if bk < 145:
        s_bk = 60.0
    elif bk < 160:
        s_bk = 70.0 + (bk - 145) / 15.0 * 15.0   # 145–160: 70–85
    elif bk <= 175:
        s_bk = 85.0 + (bk - 160) / 15.0 * 10.0   # 160–175: 85–95
    else:
        s_bk = 100.0
    scores["back_knee_min"] = float(np.clip(s_bk, 0, 100))

    # 5) prep_knee_angle：起跳屈膝，100° 左右最佳
    pk = m.prep_knee_angle
    if pk >= 140:
        s_pk = 55.0
    elif pk <= 60:
        s_pk = 80.0
    else:
        diff = abs(pk - 100.0)
        s_pk = 95.0 - diff * 0.5
    scores["prep_knee_angle"] = float(np.clip(s_pk, 0, 100))

    # 6) front_knee_angle：150~180 (越大越好)
    fk = m.front_knee_angle
    if fk < 150:
        s_fk = 60.0
    elif fk < 165:
        s_fk = 70.0 + (fk - 150) / 15.0 * 15.0   # 150–165: 70–85
    elif fk <= 175:
        s_fk = 85.0 + (fk - 165) / 10.0 * 10.0   # 165–175: 85–95
    else:
        s_fk = 100.0
    scores["front_knee_angle"] = float(np.clip(s_fk, 0, 100))

    # 7) torso_upright：0~35°（越小越直）
    tu = m.torso_upright
    if tu >= 35:
        s_tu = 60.0
    elif tu >= 25:
        s_tu = 70.0 + (35 - tu) / 10.0 * 10.0    # 25–35: 70–80
    elif tu >= 10:
        s_tu = 80.0 + (25 - tu) / 15.0 * 15.0   # 10–25: 80–95
    else:
        s_tu = 95.0 + (10 - tu) / 10.0 * 5.0    # <10: 95–100
    scores["torso_upright"] = float(np.clip(s_tu, 0, 100))

    # 8) arm_line：肘角 150~180 (越接近 180 越好)
    al = m.arm_line
    if al == 0.0:
        s_al = 50.0
    elif al < 150:
        s_al = 60.0
    elif al < 165:
        s_al = 70.0 + (al - 150) / 15.0 * 15.0   # 150–165: 70–85
    elif al <= 175:
        s_al = 85.0 + (al - 165) / 10.0 * 10.0   # 165–175: 85–95
    else:
        s_al = 100.0
    scores["arm_line"] = float(np.clip(s_al, 0, 100))

    return scores



def analyze_grand_jete(
    landmark_seq: List,
    fps: float,
    is_left_lead: bool = True,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    供 app.py 调用的总入口：
    返回：metrics_dict, scores_dict
    """
    m = compute_metrics_for_grand_jete(landmark_seq, fps, is_left_lead=is_left_lead)
    scores = score_grand_jete(m)

    metrics_dict = {
        "flight_time": m.flight_time,
        "split_angle_max": m.split_angle_max,
        "pelvis_opening": m.pelvis_opening,
        "back_knee_min": m.back_knee_min,
        "prep_knee_angle": m.prep_knee_angle,
        "front_knee_angle": m.front_knee_angle,
        "torso_upright": m.torso_upright,
        "arm_line": m.arm_line,
    }
    return metrics_dict, scores

