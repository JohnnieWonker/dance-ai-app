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


def compute_arm_line(landmarks) -> Tuple[float, Dict[str, float]]:
    if landmarks is None:
        return 50.0, {}

    ls = _get_xy(landmarks, LEFT_SHOULDER)
    rs = _get_xy(landmarks, RIGHT_SHOULDER)
    le = _get_xy(landmarks, LEFT_ELBOW)
    re = _get_xy(landmarks, RIGHT_ELBOW)
    lw = _get_xy(landmarks, LEFT_WRIST)
    rw = _get_xy(landmarks, RIGHT_WRIST)
    lh = _get_xy(landmarks, LEFT_HIP)
    rh = _get_xy(landmarks, RIGHT_HIP)
    torso_center = None
    
    if lh is not None and rh is not None:
        torso_center = (lh + rh) / 2.0
    elif ls is not None and rs is not None:
        torso_center = (ls + rs) / 2.0

    def safe_angle(a: Optional[np.ndarray],
                   b: Optional[np.ndarray],
                   c: Optional[np.ndarray]) -> Optional[float]:
        if a is None or b is None or c is None:
            return None
        return _angle(a, b, c)

    theta1_left  = safe_angle(torso_center, ls, le)
    theta1_right = safe_angle(torso_center, rs, re)
    theta2_left  = safe_angle(ls, le, lw)
    theta2_right = safe_angle(rs, re, rw)
    H_left = None
    
    if le is not None and lw is not None:
        H_left = float(le[1] - lw[1])

    H_right = None
    if re is not None and rw is not None:
        H_right = float(re[1] - rw[1])

    S_parts = []
    
    if (theta1_left is not None) and (theta1_right is not None):
        S_parts.append(abs(theta1_left - theta1_right))
    if (theta2_left is not None) and (theta2_right is not None):
        S_parts.append(abs(theta2_left - theta2_right))
    if S_parts:
        symmetry_S = float(sum(S_parts) / len(S_parts)) 
    else:
        symmetry_S = 0.0

    def score_theta1(theta: Optional[float]) -> float:
        if theta is None:
            return 60.0
        mid = 55.0
        tol = 30.0 
        d = abs(theta - mid)
        if d >= tol:
            return 60.0
        return 100.0 - (d / tol) * 40.0  # 100 → 60

    def score_theta2(theta: Optional[float]) -> float:
        if theta is None:
            return 60.0
        mid = 165.0
        tol = 25.0 
        d = abs(theta - mid)
        if d >= tol:
            return 60.0
        return 100.0 - (d / tol) * 40.0

    def score_H(H: Optional[float]) -> float:
        if H is None:
            return 60.0

        H_clamp = float(np.clip(H, -0.1, 0.1))

        if H_clamp < -0.02: 
            return 60.0
        if H_clamp < 0.0:     # 略低于肘: 70~85
            return 70.0 + (H_clamp + 0.02) / 0.02 * 15.0
        if H_clamp <= 0.05:   # 理想区间: 85~100
            return 85.0 + (H_clamp / 0.05) * 15.0
            
        return 100.0 - (H_clamp - 0.05) / 0.05 * 10.0

    def score_symmetry(S: float) -> float:
        S_clamp = float(max(0.0, min(S, 40.0)))
        if S_clamp <= 10.0:
            return 100.0
        if S_clamp <= 20.0:
            # 10–20: 100 → 85
            return 100.0 - (S_clamp - 10.0) / 10.0 * 15.0
        # 20–40: 85 → 60
        return 85.0 - (S_clamp - 20.0) / 20.0 * 25.0

    def mean_valid(values: List[Optional[float]]) -> Optional[float]:
        valid = [v for v in values if v is not None]
        if not valid:
            return None
        return float(sum(valid) / len(valid))

    theta1_mean = mean_valid([theta1_left, theta1_right])
    theta2_mean = mean_valid([theta2_left, theta2_right])
    H_mean      = mean_valid([H_left, H_right])
    s1 = score_theta1(theta1_mean)
    s2 = score_theta2(theta2_mean)
    s3 = score_H(H_mean)
    s4 = score_symmetry(symmetry_S)
    arm_line_score = 0.25 * (s1 + s2 + s3 + s4)

    details = {
        "theta1_left":  float(theta1_left)  if theta1_left  is not None else 0.0,
        "theta1_right": float(theta1_right) if theta1_right is not None else 0.0,
        "theta2_left":  float(theta2_left)  if theta2_left  is not None else 0.0,
        "theta2_right": float(theta2_right) if theta2_right is not None else 0.0,
        "H_left":       float(H_left)       if H_left       is not None else 0.0,
        "H_right":      float(H_right)      if H_right      is not None else 0.0,
        "symmetry_S":   float(symmetry_S),
        "score_theta1": float(s1),
        "score_theta2": float(s2),
        "score_H":      float(s3),
        "score_symmetry": float(s4),
    }

    return float(arm_line_score), details

def pelvis_opening_angle(lm) -> float:
    LH = _get_xy(lm, LEFT_HIP)
    RH = _get_xy(lm, RIGHT_HIP)
    LS = _get_xy(lm, LEFT_SHOULDER)
    RS = _get_xy(lm, RIGHT_SHOULDER)

    if any(p is None for p in [LH, RH, LS, RS]):
        return 0.0

    pelvis_vec = RH - LH
    shoulder_vec = RS - LS  
    denom = (np.linalg.norm(pelvis_vec) *
             np.linalg.norm(shoulder_vec) + 1e-8)
    cos_val = np.dot(pelvis_vec, shoulder_vec) / denom
    cos_val = np.clip(cos_val, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_val))
    
    return float(angle)



def auto_detect_lead_leg(landmark_seq: List, f_start: int, f_end: int, peak_idx: int) -> Optional[bool]:
    """
    自动判断哪条腿在前（lead leg）。
    逻辑：
    1) 用腾空段开始和结束时的髋部中心判断运动方向（向右 / 向左）。
    2) 在峰值帧比较左右脚踝的 x 坐标：
       - 若向右移动，则 x 更大的那条腿视为前腿；
       - 若向左移动，则 x 更小的那条腿视为前腿。
    3) 若方向不明显或脚踝缺失，则退化为比较膝盖的 x。
    返回:
        True  -> 左腿在前
        False -> 右腿在前
        None  -> 判断失败（回退到调用者给的 is_left_lead）
    """
    n = len(landmark_seq)
    if n == 0:
        return None

    def hip_center_x(idx: int) -> Optional[float]:
        if idx < 0 or idx >= n:
            return None
        lm = landmark_seq[idx]
        lh = _get_xy(lm, LEFT_HIP)
        rh = _get_xy(lm, RIGHT_HIP)
        if lh is not None and rh is not None:
            return float((lh[0] + rh[0]) / 2.0)
        if lh is not None:
            return float(lh[0])
        if rh is not None:
            return float(rh[0])
        return None

    # 1) 估计水平运动方向（髋部中心从腾空开始到结束的 x 变化）
    x_start = hip_center_x(f_start)
    x_end = hip_center_x(f_end)
    move_dir = 0.0
    if x_start is not None and x_end is not None:
        move_dir = x_end - x_start  # >0: 向右, <0: 向左

    # 2) 峰值帧的脚踝 / 膝盖位置
    lm_peak = landmark_seq[peak_idx]
    la = _get_xy(lm_peak, LEFT_ANKLE)
    ra = _get_xy(lm_peak, RIGHT_ANKLE)
    lk = _get_xy(lm_peak, LEFT_KNEE)
    rk = _get_xy(lm_peak, RIGHT_KNEE)

    def x_or_none(p):
        return None if p is None else float(p[0])

    la_x = x_or_none(la)
    ra_x = x_or_none(ra)
    lk_x = x_or_none(lk)
    rk_x = x_or_none(rk)

    # 3) 判定逻辑
    def decide_front(using_ankle: bool = True) -> Optional[bool]:
        lx = la_x if using_ankle else lk_x
        rx = ra_x if using_ankle else rk_x
        if lx is None or rx is None:
            return None
        if abs(move_dir) >= 0.005:
            # 有明显运动方向
            if move_dir > 0:   # 向右
                return lx > rx
            else:              # 向左
                return lx < rx
        else:
            # 方向不明显：就把 x 更大的那条腿视作“前腿”
            return lx > rx

    front_is_left = decide_front(using_ankle=True)
    if front_is_left is None:
        front_is_left = decide_front(using_ankle=False)

    return front_is_left



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
    3) pelvis_opening     骨盆-肩带夹角 (°，越小越对齐)
    4) back_knee_min      空中后腿伸膝 (峰值帧膝角)
    5) prep_knee_angle    起跳屈膝（起跳前几帧前腿膝角平均）
    6) front_knee_angle   空中前腿伸膝（峰值帧）
    7) torso_upright      空中躯干直立度（峰值帧，越小越直）
    8) arm_line           空中手臂三位手线条综合得分（0–100）
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

    # 2) 峰值帧（nose y 最小）
    peak_idx = f_start
    min_nose_y = 1.0
    for i in range(f_start, f_end + 1):
        nose = _get_xy(landmark_seq[i], NOSE)
        if nose is None:
            continue
        if nose[1] < min_nose_y:
            min_nose_y = nose[1]
            peak_idx = i

    # 3) 自动识别前后腿（如果失败，再用 is_left_lead 作为兜底）
    auto_lead = auto_detect_lead_leg(landmark_seq, f_start, f_end, peak_idx)
    if auto_lead is not None:
        is_left_lead = auto_lead

    if is_left_lead:
        front_hip, front_knee, front_ankle = LEFT_HIP, LEFT_KNEE, LEFT_ANKLE
        back_hip,  back_knee,  back_ankle  = RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE
        front_shoulder = LEFT_SHOULDER
    else:
        front_hip, front_knee, front_ankle = RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE
        back_hip,  back_knee,  back_ankle  = LEFT_HIP, LEFT_KNEE, LEFT_ANKLE
        front_shoulder = RIGHT_SHOULDER

    # 4) 空中横叉角：两侧髋向下的竖线 与 两腿的夹角之和（整个腾空段）
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

    # 5) 后腿伸膝：**只看峰值帧的后腿膝角**
    lm_peak = landmark_seq[peak_idx]
    bh = _get_xy(lm_peak, back_hip)
    bk = _get_xy(lm_peak, back_knee)
    ba = _get_xy(lm_peak, back_ankle)
    if all(p is not None for p in [bh, bk, ba]):
        back_knee_min = _angle(bh, bk, ba)
    else:
        back_knee_min = 0.0

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

    # 7) 峰值帧：骨盆-肩带夹角 / 前腿膝角 / 躯干直立度
    fh = _get_xy(lm_peak, front_hip)
    fk = _get_xy(lm_peak, front_knee)
    fa = _get_xy(lm_peak, front_ankle)

    pelvis_opening = pelvis_opening_angle(lm_peak)
    front_knee_angle = _angle(fh, fk, fa) if all(p is not None for p in [fh, fk, fa]) else 0.0

    t_ang = _trunk_angle(lm_peak)
    torso_upright = float(t_ang) if t_ang is not None else 0.0

    # 8) 空中手臂线条：峰值帧三位手综合得分 (0–100)
    arm_line_score, _ = compute_arm_line(lm_peak)

    return GrandJeteMetrics(
        flight_time=flight_time,
        split_angle_max=split_angle_max,
        pelvis_opening=pelvis_opening,
        back_knee_min=back_knee_min,
        prep_knee_angle=prep_knee_angle,
        front_knee_angle=front_knee_angle,
        torso_upright=torso_upright,
        arm_line=arm_line_score,
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

    # 3) pelvis_opening：骨盆-肩带夹角 (°)，越小越好
    po = m.pelvis_opening
    # 一般正常训练下，0~40° 已经覆盖大多数情况：
    # 0–10°：几乎完全平行，接近满分
    # 10–25°：轻微错位，略扣
    # 25–40°：明显错位，分数中等
    if po >= 40:
        s_po = 60.0
    elif po >= 25:
        # 25–40: 70–80
        s_po = 70.0 + (40.0 - po) / (40.0 - 25.0) * 10.0
    elif po >= 10:
        # 10–25: 80–95
        s_po = 80.0 + (25.0 - po) / (25.0 - 10.0) * 15.0
    else:
        # 0–10: 95–100
        s_po = 95.0 + (10.0 - po) / 10.0 * 5.0
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
    s_al = float(np.clip(al, 0, 100))
    scores["arm_line"] = s_al

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

