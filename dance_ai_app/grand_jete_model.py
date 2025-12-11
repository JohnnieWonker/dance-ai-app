"""
grand_jete_model.py
升级版：8 个 AI 技术指标
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np

# -------- MediaPipe Pose 关键点索引（v0.10.x）---------
NOSE = 0
LEFT_SHOULDER = 11   # NEW
RIGHT_SHOULDER = 12  # NEW
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
    back_knee_min: float        # 后腿最小膝角 (°)
    pelvis_opening: float       # 骨盆打开程度 (°，Event4 - Event3)
    prep_knee_angle: float      # 助跳阶段前腿膝角 (°)

    trunk_lean_std: float       # NEW 腾空阶段躯干倾斜角标准差 (°)
    landing_knee_flexion: float # NEW 落地瞬间膝角（两膝平均，°）
    landing_trunk_lean: float   # NEW 落地瞬间躯干倾斜角 (°)


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
    landmark_seq: 每帧的 pose_landmarks.landmark 列表
    fps: 视频帧率
    is_left_lead: 是否左腿在前（默认 True，右脚助跑 → 左腿前叉）
    """
    n = len(landmark_seq)
    if n == 0:
        return GrandJeteMetrics(0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0)

    # 1) 体空时间
    f_start, f_end = detect_flight_frames(landmark_seq, fps)
    flight_time = max(0.0, (f_end - f_start + 1) / fps)

    # 2) 确定前腿 / 后腿 index
    if is_left_lead:
        front_hip, front_knee, front_ankle = LEFT_HIP, LEFT_KNEE, LEFT_ANKLE
        back_hip, back_knee, back_ankle = RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE
    else:
        front_hip, front_knee, front_ankle = RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE
        back_hip, back_knee, back_ankle = LEFT_HIP, LEFT_KNEE, LEFT_ANKLE

    # 3) 在腾空段中找到“最高点”(nose y 最小) ≈ Event3
    peak_idx = f_start
    min_nose_y = 1.0
    for i in range(f_start, f_end + 1):
        nose = _get_xy(landmark_seq[i], NOSE)
        if nose is None:
            continue
        if nose[1] < min_nose_y:
            min_nose_y = nose[1]
            peak_idx = i

    event3_idx = peak_idx
    event4_idx = min(f_end, peak_idx + max(1, int(0.2 * (f_end - f_start + 1))))

    # 4) 空中最大横叉角
    split_angles = []
    # 4) 空中最大横叉角（按你的想法：每侧髋部向下的线，与该侧腿的夹角之和）
    split_angles = []
    for i in range(f_start, f_end + 1):
        hip_l = _get_xy(landmark_seq[i], LEFT_HIP)
        hip_r = _get_xy(landmark_seq[i], RIGHT_HIP)
        f_ank = _get_xy(landmark_seq[i], front_ankle)
        b_ank = _get_xy(landmark_seq[i], back_ankle)
        if any(p is None for p in [hip_l, hip_r, f_ank, b_ank]):
            continue

        # 从每个髋部往“图片正下方”延伸一小段
        hip_l_down = hip_l + np.array([0.0, 1.0])
        hip_r_down = hip_r + np.array([0.0, 1.0])

        # 前腿 / 后腿各用自己那一侧的髋部和“向下”的点
        if is_left_lead:
            # 左腿在前
            front_hip_xy, front_down = hip_l, hip_l_down
            back_hip_xy,  back_down  = hip_r, hip_r_down
        else:
            # 右腿在前
            front_hip_xy, front_down = hip_r, hip_r_down
            back_hip_xy,  back_down  = hip_l, hip_l_down

        # 前腿和“向下”的夹角
        front_angle = _angle(f_ank, front_hip_xy, front_down)
        # 后腿和“向下”的夹角
        back_angle  = _angle(b_ank, back_hip_xy, back_down)

        # 两条腿角度相加，表示这个瞬间的“横叉打开程度”
        split_angles.append(front_angle + back_angle)

    split_angle_max = float(max(split_angles)) if split_angles else 0.0


    # 5) 后腿最小膝角
    back_knee_angles = []
    for i in range(f_start, f_end + 1):
        h = _get_xy(landmark_seq[i], back_hip)
        k = _get_xy(landmark_seq[i], back_knee)
        a = _get_xy(landmark_seq[i], back_ankle)
        if not _is_valid_triplet((h, k, a)):
            continue
        back_knee_angles.append(_angle(h, k, a))
    back_knee_min = float(min(back_knee_angles)) if back_knee_angles else 0.0

    # 6) 骨盆打开程度：Event3 vs Event4
    def pelvis_angle(idx: int) -> Optional[float]:
        hl = _get_xy(landmark_seq[idx], LEFT_HIP)
        hr = _get_xy(landmark_seq[idx], RIGHT_HIP)
        if hl is None or hr is None:
            return None
        v = hr - hl
        return float(np.degrees(np.arctan2(v[1], v[0])))

    ang3 = pelvis_angle(event3_idx)
    ang4 = pelvis_angle(event4_idx)
    pelvis_opening = 0.0
    if ang3 is not None and ang4 is not None:
        pelvis_opening = float(ang4 - ang3)

    # 7) 助跳阶段前腿膝角
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

    # 8) NEW: 躯干倾斜稳定性（腾空段）
    trunk_angles = []
    for i in range(f_start, f_end + 1):
        ang = _trunk_angle(landmark_seq[i])
        if ang is not None:
            trunk_angles.append(ang)
    trunk_lean_std = float(np.std(trunk_angles)) if trunk_angles else 0.0

    # 9) NEW: 落地瞬间膝角（两膝平均）
    def knee_angle_for_side(landmarks, hip_idx, knee_idx, ankle_idx) -> Optional[float]:
        h = _get_xy(landmarks, hip_idx)
        k = _get_xy(landmarks, knee_idx)
        a = _get_xy(landmarks, ankle_idx)
        if not _is_valid_triplet((h, k, a)):
            return None
        return _angle(h, k, a)

    lk = knee_angle_for_side(landmark_seq[f_end], LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)
    rk = knee_angle_for_side(landmark_seq[f_end], RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)
    if lk is not None and rk is not None:
        landing_knee_flexion = float((lk + rk) / 2.0)
    elif lk is not None:
        landing_knee_flexion = float(lk)
    elif rk is not None:
        landing_knee_flexion = float(rk)
    else:
        landing_knee_flexion = 0.0

    # 10) NEW: 落地瞬间躯干倾斜角
    lt_ang = _trunk_angle(landmark_seq[f_end])
    landing_trunk_lean = float(lt_ang) if lt_ang is not None else 0.0

    return GrandJeteMetrics(
        flight_time=flight_time,
        split_angle_max=split_angle_max,
        back_knee_min=back_knee_min,
        pelvis_opening=pelvis_opening,
        prep_knee_angle=prep_knee_angle,
        trunk_lean_std=trunk_lean_std,
        landing_knee_flexion=landing_knee_flexion,
        landing_trunk_lean=landing_trunk_lean,
    )


# ----------------- 3. 将指标转换为 0–100 分 -----------------

def score_grand_jete(metrics: GrandJeteMetrics) -> Dict[str, float]:
    """按照论文趋势与专业经验，将 8 个指标映射为 0–100 分"""

    m = metrics
    scores: Dict[str, float] = {}

    # 1) 体空时间：0.45~0.70s 映射到 60~100
    ft = m.flight_time
    if ft <= 0.45:
        s_ft = 50.0
    elif ft >= 0.70:
        s_ft = 100.0
    else:
        s_ft = 60.0 + (ft - 0.45) / (0.70 - 0.45) * 40.0
    scores["flight_time"] = float(np.clip(s_ft, 0, 100))

    # 2) 空中横叉角
    sa = m.split_angle_max
    if sa <= 140:
        s_sa = 50.0
    elif sa >= 180:
        s_sa = 100.0
    else:
        s_sa = 65.0 + (sa - 140) / (180 - 140) * 30.0
    scores["split_angle_max"] = float(np.clip(s_sa, 0, 100))

    # 3) 后腿膝角（越小越好）
    bk = m.back_knee_min
    if bk >= 120:
        s_bk = 50.0
    elif bk <= 60:
        s_bk = 100.0
    else:
        s_bk = 70.0 + (120 - bk) / (120 - 60) * 25.0
    scores["back_knee_min"] = float(np.clip(s_bk, 0, 100))

    # 4) 骨盆打开程度
    po = m.pelvis_opening
    if po <= -5:
        s_po = 50.0
    elif po >= 10:
        s_po = 100.0
    else:
        s_po = 70.0 + (po + 5) / (10 + 5) * 25.0
    scores["pelvis_opening"] = float(np.clip(s_po, 0, 100))

    # 5) 助跳前腿膝角（适度弯曲）
    pk = m.prep_knee_angle
    if pk >= 140:
        s_pk = 55.0
    elif pk <= 60:
        s_pk = 80.0
    else:
        diff = abs(pk - 100.0)
        s_pk = 95.0 - diff * 0.5
    scores["prep_knee_angle"] = float(np.clip(s_pk, 0, 100))

    # 6) NEW 躯干稳定性（std 越小越好）
    tls = m.trunk_lean_std
    if tls >= 20.0:
        s_tls = 50.0
    elif tls <= 5.0:
        s_tls = 100.0
    else:
        s_tls = 100.0 - (tls - 5.0) / (20.0 - 5.0) * 50.0
    scores["trunk_lean_std"] = float(np.clip(s_tls, 0, 100))

    # 7) NEW 落地膝角（约 90~110° 最佳）
    lkf = m.landing_knee_flexion
    if lkf == 0.0:
        s_lkf = 50.0
    else:
        diff = abs(lkf - 100.0)  # 以 100° 为理想
        s_lkf = 95.0 - diff * 0.7
    scores["landing_knee_flexion"] = float(np.clip(s_lkf, 0, 100))

    # 8) NEW 落地躯干倾斜（越直越好）
    ltl = abs(m.landing_trunk_lean)
    if ltl >= 30.0:
        s_ltl = 50.0
    elif ltl <= 5.0:
        s_ltl = 100.0
    else:
        s_ltl = 100.0 - (ltl - 5.0) / (30.0 - 5.0) * 50.0
    scores["landing_trunk_lean"] = float(np.clip(s_ltl, 0, 100))

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
        "back_knee_min": m.back_knee_min,
        "pelvis_opening": m.pelvis_opening,
        "prep_knee_angle": m.prep_knee_angle,
        "trunk_lean_std": m.trunk_lean_std,
        "landing_knee_flexion": m.landing_knee_flexion,
        "landing_trunk_lean": m.landing_trunk_lean,
    }
    return metrics_dict, scores
