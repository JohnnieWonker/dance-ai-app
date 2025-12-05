from typing import Dict, List

LANGUAGES = ["中文", "한국어", "English"]

I18N: Dict[str, Dict] = {
    # =======================
    # 中文
    # =======================
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

        # -------- 中 国 古 典 舞 8 项 + 芭 蕾 8 项 --------
        "metric_labels": {
            "xiliao": {
                "split_angle": "空中横叉角度 (°)",
                "flight_time": "腾空时间 (s)",
                "hip_flex": "吸撩腿屈髋角 (°)",
                "torso_upright": "空中躯干直立度 (°)",
                "landing_stability": "落地稳定性 (角度波动)",
                "front_knee_angle": "前腿伸膝角 (°)",
                "back_knee_angle": "后腿伸膝角 (°)",
                "min_prep_knee_angle": "起跳屈膝角 (°)",
            },
            "ballet": {
                "flight_time": "体空时间 (s)",
                "split_angle_max": "空中最大横叉角 (°)",
                "back_knee_min": "后腿最小膝角 (°)",
                "pelvis_opening": "骨盆打开程度 (°)",
                "prep_knee_angle": "助跳前腿膝角 (°)",
                "trunk_lean_std": "空中躯干稳定性 (° 标准差)",
                "landing_knee_flexion": "落地膝关节控制 (°)",
                "landing_trunk_lean": "落地躯干倾斜角 (°)",
            },
        },

        "action_name_xiliao_cn": "吸撩腿跃",
        "action_name_xiliao_en": "Xi-Liao Leg Leap",

        "action_name_ballet_cn": "芭蕾大跳 · Grand Jeté",
        "action_name_ballet_en": "Grand Jeté",
    },

    # =======================
    # 韩文
    # =======================
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
        "pdf_ready": "✅ PDF가 생성되었습니다.",
        "pdf_dl": "📥 PDF 리포트 다운로드",

        # -------- 韩 文 吸 撩 腿 跃 8 项 + 芭 蕾 8 项 --------
        "metric_labels": {
            "xiliao": {
                "split_angle": "공중 다리 벌림 각도 (°)",
                "flight_time": "체공 시간 (s)",
                "hip_flex": "흡요퇴 굴곡 각도 (°)",
                "torso_upright": "공중 상체 정렬 (°)",
                "landing_stability": "착지 안정성 (각도 변동)",
                "front_knee_angle": "앞다리 무릎 각도 (°)",
                "back_knee_angle": "뒷다리 무릎 각도 (°)",
                "min_prep_knee_angle": "도약 준비 무릎 각도 (°)",
            },
            "ballet": {
                "flight_time": "체공 시간 (s)",
                "split_angle_max": "공중 스플릿 최대 각도 (°)",
                "back_knee_min": "뒷다리 최소 무릎 각도 (°)",
                "pelvis_opening": "골반 열림 변화 (°)",
                "prep_knee_angle": "도약 준비 앞다리 무릎 각도 (°)",
                "trunk_lean_std": "공중 상체 안정성 (° 표준편차)",
                "landing_knee_flexion": "착지 무릎 사용 (°)",
                "landing_trunk_lean": "착지 시 상체 기울기 (°)",
            },
        },

        "action_name_xiliao_cn": "흡요퇴 점프",
        "action_name_xiliao_en": "Xi-Liao Leg Leap",

        "action_name_ballet_cn": "그랑 즈떼 (Grand Jeté)",
        "action_name_ballet_en": "Grand Jeté",
    },

    # =======================
    # 英文
    # =======================
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

        # -------- 英 文 吸 撩 腿 跃 8 项 + 芭 蕾 8 项 --------
        "metric_labels": {
            "xiliao": {
                "split_angle": "Air Split Angle (°)",
                "flight_time": "Flight Time (s)",
                "hip_flex": "Hip Flexion (°)",
                "torso_upright": "Torso Uprightness (°)",
                "landing_stability": "Landing Stability (angle SD)",
                "front_knee_angle": "Front Knee Angle (°)",
                "back_knee_angle": "Back Knee Angle (°)",
                "min_prep_knee_angle": "Prep Knee Angle (°)",
            },
            "ballet": {
                "flight_time": "Flight Time (s)",
                "split_angle_max": "Max Air Split Angle (°)",
                "back_knee_min": "Minimum Back Knee Angle (°)",
                "pelvis_opening": "Pelvic Opening Change (°)",
                "prep_knee_angle": "Prep Front Knee Angle (°)",
                "trunk_lean_std": "Torso Stability in Air (° SD)",
                "landing_knee_flexion": "Landing Knee Control (°)",
                "landing_trunk_lean": "Landing Torso Lean (°)",
            },
        },

        "action_name_xiliao_cn": "Xi-Liao Leg Leap",
        "action_name_xiliao_en": "Xi-Liao Leg Leap",

        "action_name_ballet_cn": "Grand Jeté",
        "action_name_ballet_en": "Grand Jeté",
    },
}
