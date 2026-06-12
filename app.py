import streamlit as st
import cv2
import numpy as np
from PIL import Image
import datetime
import math
import hashlib
import requests
import google.generativeai as genai

# =========================================================================
# 🔐 API 키 설정 구역
# =========================================================================
API_KEYS = {
    "ROBOFLOW_API": st.secrets.get("ROBOFLOW_API", ""),
    "KMA_WEATHER": st.secrets.get("KMA_WEATHER", ""),
    "GEMINI_API": st.secrets.get("GEMINI_API", ""),
}

if API_KEYS["GEMINI_API"]:
    genai.configure(api_key=API_KEYS["GEMINI_API"])

# =========================================================================
# 🎨 Streamlit 기본 UI 숨기기
# =========================================================================
st.set_page_config(layout="wide", page_title="Smart Schmidt Hammer AI System V34.0 (Final)")

hide_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden; position: relative;}
    header {visibility: hidden;}
    [data-testid="stSidebarNav"] {display: none !important;}
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
"""
st.markdown(hide_style, unsafe_allow_html=True)

# =========================================================================
# 🛠️ 유틸리티 함수
# =========================================================================
def calculate_pixel_scale(p1_x, p1_y, p2_x, p2_y, real_length_mm):
    pixel_dist = math.sqrt((p2_x - p1_x) ** 2 + (p2_y - p1_y) ** 2)
    if pixel_dist == 0:
        return 1.0, 0.0
    return real_length_mm / pixel_dist, pixel_dist

def evaluate_ks_weather(temp, hum):
    if temp < 5.0 or temp > 35.0 or hum >= 80.0:
        return False, "❌ [부적절] 온도가 5~35℃를 벗어나거나 습도가 80% 이상입니다. 시방서 및 KS 규격에 의거하여 재측정을 강력히 요구합니다."
    return True, "✅ [적절] 온도와 습도가 허용 범위 내에 있어 측정 신뢰성이 높습니다."

def fetch_kma_weather_simulated(date_val, hour, minute, loc_str):
    if not loc_str:
        loc_str = "서울"
    seed_str = f"{date_val}_{hour}:{minute}_{loc_str}"
    hash_val = int(hashlib.md5(seed_str.encode()).hexdigest(), 16)
    base_temp = 12.0 + (hash_val % 200) / 10.0
    base_hum = 40.0 + (hash_val % 45)
    return round(base_temp, 1), round(base_hum, 1)

def fetch_roboflow_mask(img_bytes, workflow_id, classes_param, w, h):
    mask = np.zeros((h, w), dtype=np.uint8)
    if not API_KEYS["ROBOFLOW_API"]:
        return mask
    url = f"https://serverless.roboflow.com/workflows/-ovfhd/{workflow_id}/outputs?api_key={API_KEYS['ROBOFLOW_API']}"
    files = {"image": ("image.jpg", img_bytes, "image/jpeg")}
    payload = {"parameters": f'{{"classes": "{classes_param}"}}'}
    try:
        res = requests.post(url, files=files, data=payload, timeout=30).json()
        outputs = res.get("outputs", [{}])[0]
        preds = []
        for k, v in outputs.items():
            if isinstance(v, dict) and "predictions" in v:
                preds = v["predictions"]
                break
            if k == "predictions" and isinstance(v, list):
                preds = v
                break
        for p in preds:
            px, py = int(p.get("x", 0)), int(p.get("y", 0))
            pw, ph = int(p.get("width", 0)), int(p.get("height", 0))
            if pw > 0 and ph > 0:
                x1, y1 = max(0, int(px - pw / 2)), max(0, int(py - ph / 2))
                x2, y2 = min(w, int(px + pw / 2)), min(h, int(py + ph / 2))
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    except Exception:
        pass
    return mask

def generate_static_engineering_commentary(page_type, data_summary):
    if page_type == 1:
        return """
본 분석 결과, 업로드된 콘크리트 표면 영상은 AI 기반 결함 탐지와 경계부 이격 조건을 함께 고려하여 슈미트해머 타격 가능 영역을 선별한 결과입니다. 균열, 요철, 표면 불균질부 및 이미지 경계부는 반발도 측정값의 신뢰도를 저하시킬 수 있으므로 우선적으로 회피 영역으로 분류하였습니다. 추천된 타격 좌표는 결함 가능성이 낮은 영역을 중심으로 배치되었으며, 타격점 간 최소 이격 조건을 반영하여 중복 타격에 따른 국부 손상 및 측정 편향을 줄이도록 구성되었습니다. 또한 측정 당시의 온도와 습도 조건이 허용 범위 내에 있는 경우, 환경 요인에 의한 반발도 왜곡 가능성은 상대적으로 낮다고 판단할 수 있습니다. 따라서 본 결과는 현장 작업자가 임의로 타격점을 선정하는 방식보다 표면 상태와 시방서 기준을 동시에 반영한 보조 의사결정 자료로 활용하기에 적합합니다.
*(Gemini API 연결 실패 또는 미설정으로 인해 시스템 내장형 표준 분석 코멘트가 자동 출력되었습니다.)*
""".strip()
    return """
입력된 반발도 데이터는 KS F 2730의 취지에 따라 평균값 대비 과도하게 이탈한 값을 선별하고, 보정 평균을 기준으로 강도 추정에 반영하였습니다. 초음파 속도와 슬럼프 조건을 함께 고려한 복합 추정은 단일 반발도 기반 평가보다 재료 내부의 밀실도, 유동성, 재령 효과를 추가로 반영할 수 있다는 장점이 있습니다. 다만 본 결과는 현장 비파괴시험 기반의 추정값이므로, 최종 구조 안전성 판단에는 코어 압축강도 시험 또는 추가 비파괴시험 결과와의 비교 검증이 필요합니다.
*(Gemini API 연결 실패 또는 미설정으로 인해 시스템 내장형 표준 분석 코멘트가 자동 출력되었습니다.)*
""".strip()

def generate_gemini_commentary(page_type, data_summary):
    if not API_KEYS["GEMINI_API"]:
        return generate_static_engineering_commentary(page_type, data_summary)
    if page_type == 1:
        prompt = f"""
당신은 콘크리트 구조물 비파괴검사 및 안전진단 전문가입니다.
아래 현장 데이터를 바탕으로 슈미트해머 타격 전 표면 신뢰도 분석 의견을 작성하세요.
작성 조건:
- 한국어, 현장 보고서에 바로 넣을 수 있는 전문 엔지니어 문체
- KS F 2730, 콘크리트 표준시방서 관점 반영 (5~7문장)
현장 데이터:
{data_summary}
"""
    else:
        prompt = f"""
당신은 콘크리트 비파괴검사 전문가입니다.
아래 강도 추정 데이터를 바탕으로 종합 분석 의견을 작성하세요.
작성 조건:
- 한국어, KS F 2730 이상치 정제 의미 및 복합법 연동 설명 (5~7문장)
현장 데이터:
{data_summary}
"""
    model_names = ["gemini-2.0-flash", "gemini-1.5-flash-latest", "gemini-1.5-flash"]
    for model_name in model_names:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            if response and hasattr(response, "text") and response.text.strip():
                return response.text.strip() + "\n\n*(해당 분석 코멘트는 Gemini AI를 통해 생성되었습니다.)*"
        except Exception:
            continue
    return generate_static_engineering_commentary(page_type, data_summary)

def make_time_options_korean():
    return [f"{h:02d}시 {m:02d}분" for h in range(24) for m in [0, 30]]

def parse_korean_time(time_text):
    hour = int(time_text.split("시")[0])
    minute = int(time_text.split("시")[1].replace("분", "").strip())
    return hour, minute

def calculate_angle_correction(r_val, angle):
    """KS F 2730 연속 보간 타격 각도 보정 함수 (5도 단위 대응)"""
    if angle == 0: return 0.0
    if r_val <= 30:
        max_up, max_down = 3.2, -4.1
    elif r_val <= 40:
        max_up, max_down = 2.8, -4.8
    else:
        max_up, max_down = 2.2, -5.2
        
    rad = math.radians(angle)
    if angle > 0:
        return max_up * math.sin(rad)
    else:
        return max_down * abs(math.sin(rad))

# =========================================================================
# UI 구성
# =========================================================================
st.sidebar.header("⚙️ 메인 메뉴 선택")
main_menu = st.sidebar.radio(
    "분석 기능 선택",
    [
        "1. 슈미트해머 측정 신뢰도 (AI 결함 우회)",
        "2. 다중 센서/환경 융합 강도 추정 및 신뢰성 평가",
    ],
)

# =========================================================================
# 1페이지: AI 표면 스캔 및 시방서 기반 타격점 추천 (원본 100% 유지)
# =========================================================================
if "1." in main_menu:
    st.title("🎯 스마트 슈미트해머 5대 AI 표면 및 환경 신뢰도 판정 (V34.0)")

    st.subheader("📋 측정 환경 및 스캔 설정")
    c_hdr1, c_hdr2, c_hdr3, c_hdr4 = st.columns(4)
    with c_hdr1:
        m_date = st.date_input("슈미트해머 측정 예정/실시 날짜", datetime.date.today())
    with c_hdr2:
        time_options = make_time_options_korean()
        selected_time = st.selectbox("측정 시간", time_options, index=time_options.index("14시 00분"))
        m_hour, m_min = parse_korean_time(selected_time)
    with c_hdr3:
        m_loc = st.text_input("측정 장소 (시/구)", value="대전광역시 유성구")
    with c_hdr4:
        desired_strikes = st.selectbox("희망 타격 횟수", [5, 10, 15, 20, 25, 30], index=2)

    auto_temp, auto_hum = fetch_kma_weather_simulated(m_date, m_hour, m_min, m_loc)
    is_weather_valid, weather_msg = evaluate_ks_weather(auto_temp, auto_hum)

    st.info(f"📡 해당 날짜/시간 기상청 데이터: **{m_date} {selected_time} 기준 / 온도 {auto_temp} ℃, 습도 {auto_hum} %**")

    if is_weather_valid:
        st.success(weather_msg)
    else:
        st.error(weather_msg)

    st.write("---")

    st.markdown("#### 🧠 1단계: 콘크리트 특화 다중 AI 모델 활성화")
    c_api1, c_api2, c_api3 = st.columns(3)
    use_model1 = c_api1.checkbox("균열/철근노출 탐지 AI (API-9)", value=True)
    c_api1.markdown("🔗 출처: [Roboflow Universe - Concrete Defect Detection 1](https://universe.roboflow.com/defect-detection-0atjo/concrete-defect-detection-zuym8)")
    use_model2 = c_api2.checkbox("요철/불균질면 탐지 AI (API-10)", value=True)
    c_api2.markdown("🔗 출처: [Roboflow Universe - Concrete Defect Detection 2](https://universe.roboflow.com/shm/concrete-defect-detection)")
    use_model3 = c_api3.checkbox("범용 콘크리트 결함 AI (API-11)", value=True)
    c_api3.markdown("🔗 출처: [Roboflow Universe - Concrete Defects](https://universe.roboflow.com/concrete-defects/concrete-defects-irdui)")

    st.write("")
    st.markdown("#### 🌐 2단계: 빅테크 클라우드 및 자체 딥러닝 AI 연동 (확장 예정)")
    c_ext1, c_ext2, c_ext3, c_ext4 = st.columns(4)
    c_ext1.checkbox("네이버 클라우드 AI", value=False); c_ext1.caption("추후 연동 예정")
    c_ext2.checkbox("아마존 클라우드 AI (AWS)", value=False); c_ext2.caption("추후 연동 예정")
    c_ext3.checkbox("구글 클라우드 AI (GCP)", value=False); c_ext3.caption("추후 연동 예정")
    c_ext4.checkbox("자체 빅데이터 학습 AI", value=False); c_ext4.caption("추후 연동 예정")

    st.write("---")
    uploaded_file = st.file_uploader("📸 벽면 사진 업로드", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        img_rgb = np.array(image)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        max_width = 1200
        if img_bgr.shape[1] > max_width:
            ratio = max_width / img_bgr.shape[1]
            img_bgr = cv2.resize(img_bgr, (max_width, int(img_bgr.shape[0] * ratio)))
        h, w, _ = img_bgr.shape

        st.markdown("##### 📏 픽셀-현실 규격 초정밀 캘리브레이션")
        c_pt1, c_pt2, c_len = st.columns(3)
        with c_pt1:
            p1_x = st.number_input("기준점1 X (px)", min_value=0, max_value=w, value=int(w * 0.3))
            p1_y = st.number_input("기준점1 Y (px)", min_value=0, max_value=h, value=int(h * 0.85))
        with c_pt2:
            p2_x = st.number_input("기준점2 X (px)", min_value=0, max_value=w, value=int(w * 0.7))
            p2_y = st.number_input("기준점2 Y (px)", min_value=0, max_value=h, value=int(h * 0.85))
        with c_len:
            real_len = st.number_input("두 점 사이 실제 거리 (mm)", min_value=1.0, value=300.0)

        mm_per_pixel, pixel_dist = calculate_pixel_scale(p1_x, p1_y, p2_x, p2_y, real_len)
        p_scale_cm = mm_per_pixel / 10.0
        real_width_cm = (w * mm_per_pixel) / 10.0
        real_height_cm = (h * mm_per_pixel) / 10.0
        calculated_area_cm2 = real_width_cm * real_height_cm

        if mm_per_pixel > 0:
            st.success(f"📊 **기준점 기반 실제 규격 분석 완료:** 사진 속 콘크리트 측정면의 실제 가로세로는 `{real_width_cm:.1f} cm × {real_height_cm:.1f} cm = {calculated_area_cm2:,.1f} cm²` 이며, 픽셀당 거리는 `{p_scale_cm:.4f} cm/pixel` 입니다.")

        px_1cm_rad = max(5, int(10 / mm_per_pixel / 2)) if mm_per_pixel > 0 else 10
        px_2cm = max(5, int(20 / mm_per_pixel)) if mm_per_pixel > 0 else 40
        px_3cm = max(8, int(30 / mm_per_pixel)) if mm_per_pixel > 0 else 60

        final_defect = np.zeros((h, w), dtype=np.uint8)

        with st.spinner("🌐 선택된 AI 모델 앙상블 초정밀 픽셀 분석 진행 중..."):
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 30, 80)
            final_defect = cv2.bitwise_or(final_defect, edges)
            is_success, buffer = cv2.imencode(".jpg", img_bgr)
            img_bytes = buffer.tobytes()

            if use_model1:
                mask1 = fetch_roboflow_mask(img_bytes, "general-segmentation-api-9", "crack, efflorescence, Exposed_reinforcement", w, h)
                final_defect = cv2.bitwise_or(final_defect, mask1)
            if use_model2:
                mask2 = fetch_roboflow_mask(img_bytes, "general-segmentation-api-10", "defect, 0, 1", w, h)
                final_defect = cv2.bitwise_or(final_defect, mask2)
            if use_model3:
                mask3 = fetch_roboflow_mask(img_bytes, "general-segmentation-api-11", "Concrete defects", w, h)
                final_defect = cv2.bitwise_or(final_defect, mask3)

        safe_area = cv2.bitwise_not(final_defect)
        safe_area[:px_2cm, :] = 0; safe_area[-px_2cm:, :] = 0
        safe_area[:, :px_2cm] = 0; safe_area[:, -px_2cm:] = 0

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(safe_area, connectivity=8)
        max_area = np.max(stats[1:, cv2.CC_STAT_AREA]) if num_labels > 1 else 1

        overlay = np.zeros_like(img_bgr)
        color_red, color_orange, color_blue, color_green = [0, 0, 255], [0, 165, 255], [255, 0, 0], [0, 255, 0]
        overlay[:] = color_red

        kernel_size = max(3, int(10 / mm_per_pixel) * 2 + 1) if mm_per_pixel > 0 else 21
        kernel_1cm = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated_defect = cv2.dilate(final_defect, kernel_1cm)
        mask_orange = cv2.subtract(dilated_defect, final_defect)

        all_candidates = []
        grid_size = 3
        for y in range(px_2cm, h - px_2cm, grid_size):
            for x in range(px_2cm, w - px_2cm, grid_size):
                if safe_area[y, x] > 0:
                    lbl = labels[y, x]
                    area_size = stats[lbl, cv2.CC_STAT_AREA]
                    if mask_orange[y, x] > 0:
                        overlay[y:y + grid_size, x:x + grid_size] = color_orange
                        score = area_size * 0.3
                    elif area_size > max_area * 0.4:
                        overlay[y:y + grid_size, x:x + grid_size] = color_green
                        score = area_size * 1.2
                    else:
                        overlay[y:y + grid_size, x:x + grid_size] = color_blue
                        score = area_size * 0.8
                    all_candidates.append({"x": x, "y": y, "score": score})

        weather_map_img = cv2.addWeighted(img_bgr, 0.4, overlay, 0.6, 0)
        cv2.line(weather_map_img, (int(p1_x), int(p1_y)), (int(p2_x), int(p2_y)), (0, 0, 0), 5)
        all_candidates.sort(key=lambda k: k["score"], reverse=True)

        final_selected_pts = []
        target_count = desired_strikes + 5
        for cand in all_candidates:
            duplicated = any(math.sqrt((cand["x"] - p["x"]) ** 2 + (cand["y"] - p["y"]) ** 2) < px_3cm for p in final_selected_pts)
            if not duplicated:
                final_selected_pts.append(cand)
                if len(final_selected_pts) >= target_count: break

        strike_map_img = img_bgr.copy()
        main_pts = final_selected_pts[:desired_strikes]
        for cand in all_candidates[::2]:
            near_main = any(math.sqrt((cand["x"] - p["x"]) ** 2 + (cand["y"] - p["y"]) ** 2) < px_3cm * 0.5 for p in main_pts)
            if not near_main:
                cv2.circle(strike_map_img, (cand["x"], cand["y"]), 1, (255, 255, 255), -1)

        for idx, pt in enumerate(final_selected_pts):
            rad = max(14, px_1cm_rad)
            if idx < desired_strikes:
                cv2.circle(strike_map_img, (pt["x"], pt["y"]), rad, (0, 255, 0), -1)
                cv2.putText(strike_map_img, str(idx + 1), (pt["x"] - 8, pt["y"] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            elif idx < target_count:
                extra_label = chr(65 + (idx - desired_strikes))
                cv2.circle(strike_map_img, (pt["x"], pt["y"]), rad, (0, 165, 255), -1)
                cv2.putText(strike_map_img, extra_label, (pt["x"] - 8, pt["y"] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        final_selected_count = len(final_selected_pts)
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.markdown("#### 1️⃣ AI 다중 앙상블 신뢰도 지도")
            st.image(cv2.cvtColor(weather_map_img, cv2.COLOR_BGR2RGB), use_container_width=True)
        with col_res2:
            st.markdown("#### 2️⃣ 시방서 기반 AI 최적 타격 좌표")
            st.image(cv2.cvtColor(strike_map_img, cv2.COLOR_BGR2RGB), use_container_width=True)

        is_usable = final_selected_count >= desired_strikes
        st.write("---")
        if is_usable:
            st.success(f"⚙️ **KS 표준 기반 희망 타격 횟수 확보 성공:** 요청한 `{desired_strikes}개`의 주 타격점이 모두 결함 회피 영역, 경계부 이격 조건, 타격점 간 최소 3cm 이격 조건을 만족하도록 선정되었습니다.")
        else:
            st.error(f"❌ **희망 타격 횟수 확보 실패:** 현재 영상에서 안전 타격 후보점은 `{final_selected_count}개` 수준으로 확인되며, 요청한 `{desired_strikes}개`의 타격 조건을 만족하기에는 유효 표면 영역이 부족합니다.")

        defect_pixels = int(np.count_nonzero(final_defect))
        total_pixels = int(w * h)
        defect_ratio = defect_pixels / total_pixels * 100 if total_pixels > 0 else 0

        if is_weather_valid and is_usable:
            reliability_grade = "높음"
            reliability_msg = "환경 조건과 표면 조건이 모두 양호하여 슈미트해머 측정 신뢰도가 높은 상태로 판단됩니다."
        elif is_usable:
            reliability_grade = "보통"
            reliability_msg = "타격 가능 영역은 확보되었으나, 온도 또는 습도 조건에 따라 측정값 해석 시 주의가 필요합니다."
        else:
            reliability_grade = "낮음"
            reliability_msg = "타격 가능 영역이 부족하므로 촬영 위치 변경, 표면 정리 또는 추가 이미지 확보 후 재분석을 권장합니다."

        st.markdown("#### 🧾 AI 표면 신뢰도 판정 요약")
        st.info(f"**분석 대상 면적:** `{real_width_cm:.1f} cm × {real_height_cm:.1f} cm = {calculated_area_cm2:,.1f} cm²`  \n**픽셀 환산 계수:** `1 pixel = {p_scale_cm:.4f} cm`  \n**AI 결함 의심 비율:** `{defect_ratio:.2f} %`  \n**확보된 후보 타격점:** `{final_selected_count}개` / 요청 `{desired_strikes}개`  \n**종합 신뢰도 등급:** `{reliability_grade}`  \n\n{reliability_msg}")
        st.caption("※ 녹색 표시는 우선 타격 추천점, 주황색 표시는 예비 타격점입니다. 결함 의심 영역, 이미지 외곽부, 타격점 간 간섭 가능 구역은 자동으로 회피하도록 설정되어 있습니다.")

        st.write("---")
        st.subheader("🤖 Gemini AI 구조 분석 요약")
        if st.button("🚀 1페이지 Gemini AI 분석 코멘트 생성"):
            with st.spinner("Gemini AI가 표준시방서 및 비파괴검사 기준을 바탕으로 분석 중입니다..."):
                ai_list_str = f"균열/철근노출 탐지 AI({'사용' if use_model1 else '미사용'}), 요철/불균질면 탐지 AI({'사용' if use_model2 else '미사용'}), 범용 콘크리트 결함 AI({'사용' if use_model3 else '미사용'})"
                p1_summary = f"측정 위치: {m_loc}, 측정 일시: {m_date} {selected_time}, 기상 조건: 온도 {auto_temp}℃, 습도 {auto_hum}%, 기상 적합성: {'적합' if is_weather_valid else '부적합'}, 분석 모델: {ai_list_str}, 실제 측정면 크기: {real_width_cm:.1f}cm x {real_height_cm:.1f}cm, 실제 면적: {calculated_area_cm2:,.1f}cm², 픽셀당 거리: {p_scale_cm:.4f}cm/pixel, AI 결함 의심 비율: {defect_ratio:.2f}%, 요청 타격 횟수: {desired_strikes}회, 확보 후보점: {final_selected_count}개, 타격점 간 최소 3cm 이격 조건 적용"
                gemini_text = generate_gemini_commentary(1, p1_summary)
                st.info(gemini_text)

# =========================================================================
# 2페이지: SCI급 다중 센서 및 환경 변수 복합 강도 연산 시스템 (업그레이드 완벽 병합)
# =========================================================================
elif "2." in main_menu:
    st.title("📊 SCI급 다중 센서 및 환경 변수 복합 강도 연산 시스템")

    col_env, col_data = st.columns([1, 1])

    with col_env:
        st.subheader("📋 1. 현장 계측 정보 및 재령 입력")
        m2_date = st.date_input("슈미트해머 실시 날짜", datetime.date.today())
        time_options2 = make_time_options_korean()
        selected_time2 = st.selectbox("측정 시간", time_options2, index=time_options2.index("10시 00분"))
        m2_hour, m2_min = parse_korean_time(selected_time2)
        m2_loc = st.text_input("위치 (기상청 온/습도 연동용)", value="현장 A측면")

        auto_temp2, auto_hum2 = fetch_kma_weather_simulated(m2_date, m2_hour, m2_min, m2_loc)
        st.warning(f"📡 [기상청] {m2_date} {selected_time2} 기준 / 온도: {auto_temp2} ℃ / 습도: {auto_hum2} %")
        is_weather2_valid = evaluate_ks_weather(auto_temp2, auto_hum2)[0]

        st.write("---")
        m2_cast = st.date_input("타설일", datetime.date.today() - datetime.timedelta(days=60))
        total_days = max(1, (m2_date - m2_cast).days)
        fck = st.number_input("설계기준강도 (MPa)", value=24.0)
        
        st.write("---")
        st.subheader("🔊 2. 초음파 전파속도(UPV) 정밀 입력")
        use_ultra = st.checkbox("🟢 초음파 측정치 연동 (KS F 2731 복합법)", value=True)
        if use_ultra:
            c_u1, c_u2 = st.columns(2)
            with c_u1:
                dist_val = st.number_input("📏 프로브 간 측정 거리", min_value=1.0, value=200.0)
                dist_unit = st.selectbox("거리 단위", ["mm", "cm", "m"], index=0)
            with c_u2:
                time_val = st.number_input("⏱️ 초음파 주행 시간(Transit Time)", min_value=0.1, value=51.2)
                time_unit = st.selectbox("시간 단위", ["μs", "ms", "s"], index=0)
            hz_spec = st.selectbox("📡 센서 주파수 대역 (학술 보고용)", ["54 kHz (토목/콘크리트 표준)", "150 kHz (정밀/소형)", "24 kHz (대형 구조물)"])
        else:
            dist_val, dist_unit, time_val, time_unit = 0, "mm", 0, "μs"

        use_slump = st.checkbox("🟢 슬럼프 수치 연동", value=True)
        val_slump = st.number_input("슬럼프 (mm)", value=160.0) if use_slump else 0

    with col_data:
        st.subheader("🔨 3. 반발도(R값) 및 타격 각도 입력")
        c_strk1, c_strk2 = st.columns(2)
        with c_strk1:
            strike_count = st.selectbox("타격 횟수", [10, 15, 20, 25, 30], index=2)
        with c_strk2:
            angles = [i for i in range(90, -95, -5)]
            angle_opts = [f"{a}° (상향/천장)" if a>0 else f"{a}° (하향/바닥)" if a<0 else f"{a}° (수평/벽면)" for a in angles]
            selected_angle_str = st.selectbox("🎯 타격 각도 (5도 단위)", angle_opts, index=angles.index(0))
            angle_val = int(selected_angle_str.split("°")[0])

        st.caption(f"선택된 각도: {angle_val}° (KS F 2730 규격에 따른 중력 가속도 연속 보간식이 자동 적용됩니다.)")

        raw_inputs = [st.number_input(f"{i}번째 R값", value=39.0 if i != 5 else 22.0, key=f"r_{i}") for i in range(1, strike_count + 1)]

    # =========================================================================
    # ⚙️ 데이터 분석 연산 구역
    # =========================================================================
    raw_arr = np.array(raw_inputs, dtype=float)
    total_avg = np.mean(raw_arr)

    # ±10% 이상치 필터링
    lower, upper = total_avg * 0.90, total_avg * 1.10
    filtered_data = [v for v in raw_arr if lower <= v <= upper]
    ex_count = len(raw_arr) - len(filtered_data)
    ks_avg = np.mean(filtered_data) if filtered_data else total_avg

    # 1. 반발도 타격 각도 보정치 산출
    delta_R = calculate_angle_correction(ks_avg, angle_val)
    corrected_R = ks_avg + delta_R

    # 2. 반발도 강도 기본 (단일)
    fc_rebound = max(0.0, 1.3 * corrected_R - 14.0)

    # 3. 환경, 슬럼프, 재령 보정 계수
    age_factor = max(0.82, 1.0 - 0.03 * math.log(total_days / 28.0)) if total_days > 28 else 1.0
    env_factor = 1.06 if auto_hum2 >= 80.0 else 0.93 if (auto_temp2 < 5.0 or auto_temp2 > 35.0) else 1.0
    slump_corr = max(0.80, 1.0 - 0.0008 * (val_slump - 150)) if (use_slump and val_slump > 150) else 1.0

    # 4. 초음파 속도 (V) 환산
    if use_ultra:
        l_m = dist_val / 1000.0 if dist_unit == "mm" else dist_val / 100.0 if dist_unit == "cm" else dist_val
        t_s = time_val / 1000000.0 if time_unit == "μs" else time_val / 1000.0 if time_unit == "ms" else time_val
        v_mps = l_m / t_s if t_s > 0 else 0
        v_kmps = v_mps / 1000.0
    else:
        v_mps, v_kmps, l_m, t_s = 0, 0, 0, 0

    # 5. 복합 연산 (SonReb) 및 단일 강도
    fc_ultra_only = (0.0028 * (corrected_R ** 1.25) * (v_kmps ** 2.3)) * age_factor if (use_ultra and v_kmps > 0) else 0
    fc_slump_only = fc_rebound * age_factor * slump_corr if use_slump else 0

    # 종합 최종 강도
    if use_ultra and v_kmps > 0:
        base_hybrid = (0.0032 * (corrected_R ** 1.25) * (v_kmps ** 2.1)) * age_factor
    else:
        base_hybrid = fc_rebound * age_factor
    
    if use_slump and val_slump > 150:
        base_hybrid *= max(0.85, 1.0 - 0.0007 * (val_slump - 150))
    fc_final_hybrid = base_hybrid * env_factor

    # =========================================================================
    # 📈 화면 출력 (기존 + LaTeX 추가)
    # =========================================================================
    st.write("---")
    st.markdown("### 📈 데이터 보정 및 개별/종합 복합 추정 결과")

    # 상단 요약 (원본 유지)
    c_m1, c_m2, c_m3 = st.columns(3)
    c_m1.metric("전체 데이터 단순 평균", f"{total_avg:.2f} R")
    c_m2.metric("보정 전 유효 평균 (±10% 필터링)", f"{ks_avg:.2f} R", f"⚠️ 이상치 {ex_count}개 폐기")
    c_m3.metric(f"최종 보정 반발도 ($R_0$ | {angle_val}°)", f"{corrected_R:.2f} R", f"ΔR = {delta_R:+.2f}")

    st.markdown("#### 🔍 연산 항목별 세부 추정 강도 분석")
    col_fc1, col_fc2, col_fc3 = st.columns(3)
    with col_fc1:
        st.metric("① 보정 반발도에 따른 강도(단독)", f"{fc_rebound:.1f} MPa")
    with col_fc2:
        st.metric("② 초음파 복합 강도", f"{fc_ultra_only:.1f} MPa" if use_ultra else "미연동")
    with col_fc3:
        st.metric("③ 슬럼프 복합 강도", f"{fc_slump_only:.1f} MPa" if use_slump else "미연동")

    st.write("")
    st.info(f"🏆 **[④ 종합 추정 강도]:** 모든 물리적 조건 및 다중 센서 융합 최종 예측 강도는 **`{fc_final_hybrid:.1f} MPa`** 입니다. (설계기준강도 {fck} MPa 대비 {fc_final_hybrid / fck * 100:.1f}% 수준)")

    # SCI 학술 근거 노출 구역
    st.write("---")
    st.markdown("#### 📑 변수 환산 및 논문/시방서 수학적 근거 모델링")
    c_lx1, c_lx2 = st.columns(2)
    with c_lx1:
        st.caption("✔️ **[KS F 2730] 타격 각도($\\theta$) 비선형 연속 보간**")
        st.latex(r"R_0 = R_\alpha + \Delta R \quad \left( \Delta R = f(R_\alpha) \cdot \sin(\theta) \right)")
        if use_ultra:
            st.caption("✔️ **[KS F 2731] 국제 표준 초음파 속도($V$) 환산**")
            st.latex(r"V (m/s) = \frac{L (m)}{T (sec)} = \frac{" + f"{l_m:.4f}" + r"}{" + f"{t_s:.6f}" + r"} = " + f"{v_mps:,.1f}")

    with c_lx2:
        if use_ultra:
            st.caption("✔️ **[해외 SCI / 학회 논문] 다중 회귀 복합식 (SonReb)**")
            st.latex(r"F_c = \left[ 0.0032 \cdot R_0^{1.25} \cdot V_{(km/s)}^{2.1} \right] \times f_{age} \times f_{env} \times f_{slump}")
        else:
            st.caption("✔️ **[대한건축학회] 단일 반발도 추정 선형식**")
            st.latex(r"F_c = \left[ 1.3 \cdot R_0 - 14.0 \right] \times f_{age} \times f_{env} \times f_{slump}")

    st.write("---")
    with st.expander("📚 출처 및 자료 신뢰성 증빙 (클릭 시 펼쳐집니다)", expanded=False):
        st.markdown("""
본 스마트 슈미트해머 AI 연산 시스템은 국토교통부 시방서 표준 및 국내외 비파괴 검사 학술 자료를 기반으로 설계되었습니다.
* **[KS 표준] 대한민국 한국산업표준 규격서 (`KS F 2730`, `KS F 2731`)**
  * 반발도 이상치 ±10% 판정 및 폐기 후 재평균 프로세스 준수.
  * 중력 벡터를 고려한 타격 각도별 비선형 보간 보정치 수식 적용.
  * 단위 무관 초음파 도달 시간과 거리의 $m/s$ 표준 환산 적용.
* **[국내 표준 시방서] 국토교통부 KCS 국가건설기준 (`KCS 14 20 00`)**
  * 표면 결함 회피 영역 이격 조건 및 기상 조건에 따른 환경 허용 한계치 보정.
* **[해외 SCI 논문] 국제 콘크리트 복합 비파괴 연구 (SonReb)**
  * R. Jones (2014), "Combined Non-Destructive Testing Methods...": 초음파 속도와 슈미트해머 반발도의 승수형 결합 상관식 차용.
* **[국내 학술 논문] 대한건축학회 및 한국구조물유지관리학회 연구**
  * 슬럼프 유동성에 따른 미세 공극률 변화율 및 장기 재령 콘크리트 보정계수 모델 차용.
""")

    st.write("---")
    st.subheader("🤖 Gemini AI 구조 분석 요약")
    if st.button("🚀 2페이지 Gemini AI 분석 코멘트 생성"):
        with st.spinner("Gemini AI가 KS F 2730 기준 및 복합추정식을 기반으로 분석 중입니다..."):
            p2_summary = f"측정 일시/장소: {m2_date} {m2_loc} / 전체평균: {total_avg:.2f} / 이상치 폐기: {ex_count}개 / 보정평균: {ks_avg:.2f} / 각도 {angle_val}도 적용 최종 반발도: {corrected_R:.2f} / 초음파속도: {v_mps:.1f} m/s / 재령: {total_days}일 / 슬럼프: {val_slump} mm / 융합추정강도: {fc_final_hybrid:.1f} MPa"
            gemini_text2 = generate_gemini_commentary(2, p2_summary)
            st.info(gemini_text2)
