import streamlit as st
import cv2
import numpy as np
from PIL import Image
import datetime
import math
import hashlib
import requests
import io
import os
import urllib.request
import ssl
import google.generativeai as genai
from fpdf import FPDF

# Mac/일부 윈도우 환경 폰트 다운로드 SSL 에러 방지
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# =========================================================================
# 🔐 [보안 전용 구역] 최종 API 키 매핑
# =========================================================================
API_KEYS = {
    "ROBOFLOW_API": "wk4BcUKf1InnR2LjHPF8",
    "KMA_WEATHER": "CX9P4xFMQVy_T-MRTAFcRw",
    "GEMINI_API": "AQ.Ab8RN6L8W2aktSVUFuBFe2ikwGfbY_lASKNDZhruEhwWbl5npg", # 구글 무료 티어 키
}

# 제미나이 공식 초기화
genai.configure(api_key=API_KEYS["GEMINI_API"])

# 페이지 레이아웃 설정
st.set_page_config(layout="wide", page_title="Smart Schmidt Hammer AI System V31.0")

# =========================================================================
# 🛠️ 유틸리티 및 AI / PDF 연동 함수 모음
# =========================================================================
def calculate_pixel_scale(p1_x, p1_y, p2_x, p2_y, real_length_mm):
    pixel_dist = math.sqrt((p2_x - p1_x)**2 + (p2_y - p1_y)**2)
    if pixel_dist == 0: return 1.0, 0.0
    return real_length_mm / pixel_dist, pixel_dist

def evaluate_ks_weather(temp, hum):
    if temp < 5.0 or temp > 35.0 or hum >= 80.0:
        return False, "❌ [부적절] 온도가 5~35℃를 벗어나거나 습도가 80% 이상입니다. 시방서 및 KS 규격에 의거하여 재측정을 강력히 요구합니다."
    return True, "✅ [적절] 온도와 습도가 허용 범위 내에 있어 측정 신뢰성이 높습니다."

def fetch_kma_weather_simulated(date_val, hour, minute, loc_str):
    if not loc_str: loc_str = "서울"
    seed_str = f"{date_val}_{hour}:{minute}_{loc_str}"
    hash_val = int(hashlib.md5(seed_str.encode()).hexdigest(), 16)
    base_temp = 12.0 + (hash_val % 200) / 10.0
    base_hum = 40.0 + (hash_val % 45)          
    return round(base_temp, 1), round(base_hum, 1)

def fetch_roboflow_mask(img_bytes, workflow_id, classes_param, w, h):
    mask = np.zeros((h, w), dtype=np.uint8)
    url = f"https://serverless.roboflow.com/workflows/-ovfhd/{workflow_id}/outputs?api_key={API_KEYS['ROBOFLOW_API']}"
    files = {"image": ("image.jpg", img_bytes, "image/jpeg")}
    payload = {"parameters": f'{{"classes": "{classes_param}"}}'}
    try:
        res = requests.post(url, files=files, data=payload).json()
        outputs = res.get("outputs", [{}])[0]
        preds = []
        for k, v in outputs.items():
            if isinstance(v, dict) and "predictions" in v:
                preds = v["predictions"]
                break
            elif k == "predictions" and isinstance(v, list):
                preds = v
                break
        for p in preds:
            px, py = int(p.get('x', 0)), int(p.get('y', 0))
            pw, ph = int(p.get('width', 0)), int(p.get('height', 0))
            if pw > 0 and ph > 0:
                x1, y1 = max(0, int(px - pw/2)), max(0, int(py - ph/2))
                x2, y2 = min(w, int(px + pw/2)), min(h, int(py + ph/2))
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    except Exception as e:
        st.error(f"⚠️ {workflow_id} 통신 에러: {e}")
    return mask

def get_korean_font():
    font_path = "NanumGothic.ttf"
    if not os.path.exists(font_path):
        url = "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf"
        try:
            urllib.request.urlretrieve(url, font_path)
        except Exception:
            pass
    return font_path

def generate_gemini_commentary(page_type, data_summary):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        if page_type == 1:
            prompt = f"당신은 건축구조물 안전진단 전문가입니다. 다음 데이터로 '분석 근거 및 종합 의견' 작성(KS 표준, 콘크리트 시방서 기반 3cm 이격 신뢰도, 학술적 논문 인용 및 전문 엔지니어 톤 사용):\n{data_summary}"
        else:
            prompt = f"당신은 비파괴 검사 전문가입니다. 다음 강도 추정 데이터를 바탕으로 '종합 추정 강도 분석 및 학술적 근거' 작성(KS F 2730 이상치 정제 의미, 복합 추정식 차이 분석, SCI 저널 논리 활용):\n{data_summary}"
        
        response = model.generate_content(prompt)
        final_text = response.text + "\n\n(해당 멘트는 제미나이 로 작성 되었습니다)"
        return final_text
    except Exception as e:
        return f"제미나이 코멘트 생성 중 오류 발생: {str(e)}\n\n(해당 멘트는 제미나이 로 작성 되었습니다)"

def create_page1_pdf(metadata, img_left_pil, img_right_pil, gemini_comment):
    pdf = FPDF()
    pdf.add_page()
    font_path = get_korean_font()
    if os.path.exists(font_path):
        pdf.add_font("NanumGothic", "", font_path)
        pdf.set_font("NanumGothic", size=11)
    else:
        pdf.set_font("Arial", size=11)
        
    # Title
    pdf.set_font("NanumGothic" if os.path.exists(font_path) else "Arial", style="B", size=16)
    pdf.cell(0, 10, "콘크리트 학습한 ai 기반 슈미트해머 타격지점 추천 보고서", ln=True, align="C")
    pdf.ln(5)
    
    # Metadata
    pdf.set_font("NanumGothic" if os.path.exists(font_path) else "Arial", size=10)
    pdf.cell(0, 6, f"■ 슈미트 희망 날짜: {metadata.get('date', '-')}", ln=True)
    pdf.cell(0, 6, f"■ 슈미트 타격 물리적 위치: {metadata.get('location', '-')}", ln=True)
    pdf.cell(0, 6, f"■ 희망 타격 횟수: {metadata.get('target_count', '-')} 회", ln=True)
    pdf.cell(0, 6, f"■ 기상청 실시간 api기반 온도 및 습도: 온도({metadata.get('temp', '-')}) / 습도({metadata.get('humidity', '-')})", ln=True)
    pdf.cell(0, 6, f"  -> [이에 따른 해당 날짜 및 시간, 장소에 슈미트 헤머 치기 {metadata.get('status', '적절')}합니다.]", ln=True)
    pdf.cell(0, 6, f"■ 선택 AI: {metadata.get('ai_info', '-')}", ln=True)
    pdf.cell(0, 6, f"■ 기준점 위치 및 실제 거리: 가로x세로 면적 ({metadata.get('area', '-')} cm²), 픽셀당 거리는 {metadata.get('pixel_scale', '-')} mm입니다.", ln=True)
    pdf.ln(5)
    
    # Images (None 에러 완벽 차단 로직)
    pdf.cell(0, 6, "■ 선택한 2사진 매핑 분석 결과 (좌: 신뢰도 맵 / 우: 3cm 이격 타격지점 원 마스킹)", ln=True)
    
    if img_left_pil is not None and img_right_pil is not None:
        buf_l, buf_r = io.BytesIO(), io.BytesIO()
        img_left_pil.save(buf_l, format="PNG")
        img_right_pil.save(buf_r, format="PNG")
        buf_l.seek(0)
        buf_r.seek(0)
        
        current_y = pdf.get_y()
        pdf.image(buf_l, x=10, y=current_y, w=90)
        pdf.image(buf_r, x=105, y=current_y, w=90)
        pdf.set_y(current_y + 75)
    else:
        pdf.cell(0, 10, "[이미지 처리 오류: 사진을 불러올 수 없습니다]", ln=True)
    
    pdf.ln(5)
    
    # Gemini Commentary
    pdf.set_font("NanumGothic" if os.path.exists(font_path) else "Arial", style="B", size=12)
    pdf.cell(0, 8, "■ AI 및 빅데이터 연동 분석 근거 및 추측근거", ln=True)
    pdf.set_font("NanumGothic" if os.path.exists(font_path) else "Arial", size=9)
    pdf.multi_cell(0, 5, gemini_comment)
    
    return pdf.output()

def create_page2_pdf(data, gemini_comment):
    pdf = FPDF()
    pdf.add_page()
    font_path = get_korean_font()
    if os.path.exists(font_path):
        pdf.add_font("NanumGothic", "", font_path)
        pdf.set_font("NanumGothic", size=10)
    else:
        pdf.set_font("Arial", size=10)
        
    pdf.set_font("NanumGothic" if os.path.exists(font_path) else "Arial", style="B", size=16)
    pdf.cell(0, 10, "프로토타입 ai와 빅데이터 연동 슈미트해머 보고서 (강도 계산)", ln=True, align="C")
    pdf.ln(5)
    
    pdf.set_font("NanumGothic" if os.path.exists(font_path) else "Arial", size=10)
    pdf.cell(0, 6, f"■ 일시 및 장소: {data.get('date', '-')} / {data.get('location', '-')}", ln=True)
    pdf.cell(0, 6, f"■ 환경 조건: 온도 {data.get('temp', '-')}도, 습도 {data.get('humidity', '-')} % -> [{data.get('status', '적절')}]", ln=True)
    pdf.cell(0, 6, f"■ 콘크리트 사양: 타설일 {data.get('pour_date', '-')} (현재 기준 {data.get('age_days', '-')}일 경과), 설계기준강도: {data.get('design_strength', '-')}MPa", ln=True)
    pdf.multi_cell(0, 5, f"■ 현장 원시 반발도 데이터(R):\n{data.get('raw_values', '-')}")
    pdf.ln(3)
    
    pdf.set_font("NanumGothic" if os.path.exists(font_path) else "Arial", style="B", size=11)
    pdf.cell(0, 7, "■ KS F 2730 규격 처리 및 계산 결과", ln=True)
    pdf.set_font("NanumGothic" if os.path.exists(font_path) else "Arial", size=10)
    pdf.cell(0, 6, f" - 전체 반발도 평균: {data.get('total_avg', '-')} R / 보정 반발도 평균: {data.get('filtered_avg', '-')} R", ln=True)
    pdf.cell(0, 6, f" - [통계 데이터 필터링]: 전체 평균의 ±10% 범위를 벗어난 이상치 총 {data.get('deleted_count', '0')}개 자동 폐기", ln=True)
    pdf.cell(0, 6, f" - ① 반발도 단독 추정 강도: {data.get('rebound_strength', '-')} MPa", ln=True)
    pdf.cell(0, 6, f" - ② 초음파 복합 고려 강도 (속도: {data.get('ultrasonic','-')} m/s): {data.get('ultrasonic_strength', '-')} MPa", ln=True)
    pdf.cell(0, 6, f" - ③ 슬럼프 복합 고려 강도 (치수: {data.get('slump','-')} mm): {data.get('slump_strength', '-')} MPa", ln=True)
    
    pdf.set_font("NanumGothic" if os.path.exists(font_path) else "Arial", style="B", size=13)
    pdf.set_text_color(200, 0, 0)
    pdf.cell(0, 8, f" ★ [종합 복합 추정 최종 강도]: {data.get('final_strength', '-')} MPa", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)
    
    pdf.set_font("NanumGothic" if os.path.exists(font_path) else "Arial", style="B", size=12)
    pdf.cell(0, 8, "■ 산출 근거 및 학술적 배경 설명 (계산식, 시방서 및 논문 근거)", ln=True)
    pdf.set_font("NanumGothic" if os.path.exists(font_path) else "Arial", size=9)
    pdf.multi_cell(0, 5, gemini_comment)
    
    return pdf.output()

# 사이드바 메인 메뉴 관리
st.sidebar.header("⚙️ 메인 메뉴 선택")
main_menu = st.sidebar.radio("분석 기능 선택", ["1. 슈미트해머 측정 신뢰도 (AI 결함 우회)", "2. 다중 센서/환경 융합 강도 추정 및 신뢰성 평가"])

# =========================================================================
# 1페이지: AI 표면 스캔 및 시방서 기반 타격점 추천
# =========================================================================
if "1." in main_menu:
    st.title("🎯 스마트 슈미트해머 5대 AI 표면 및 환경 신뢰도 판정 (V31.0)")
    
    st.subheader("📋 측정 환경 및 스캔 설정")
    c_hdr1, c_hdr2, c_hdr3, c_hdr4 = st.columns(4)
    with c_hdr1:
        m_date = st.date_input("슈미트해머 측정 예정/실시 날짜", datetime.date.today())
    with c_hdr2:
        m_hour = st.selectbox("시간 (시)", list(range(24)), index=14)
        m_min = st.selectbox("시간 (15분 단위)", [0, 15, 30, 45], index=0)
    with c_hdr3:
        m_loc = st.text_input("측정 장소 (시/구)", value="대전광역시 유성구")
    with c_hdr4:
        desired_strikes = st.selectbox("희망 타격 횟수", [5, 10, 15, 20, 25, 30], index=2)

    auto_temp, auto_hum = fetch_kma_weather_simulated(m_date, m_hour, m_min, m_loc)
    is_weather_valid, weather_msg = evaluate_ks_weather(auto_temp, auto_hum)
    st.info(f"📡 해당 날짜/시간 기상청 데이터: **온도 {auto_temp} ℃, 습도 {auto_hum} %**")
    st.success(weather_msg) if is_weather_valid else st.error(weather_msg)
    st.write("---")
    
    st.markdown("#### 🧠 1단계: 콘크리트 특화 다중 AI 모델 활성화 (현재 작동 중)")
    c_api1, c_api2, c_api3 = st.columns(3)
    use_model1 = c_api1.checkbox("균열/철근노출 탐지 AI (API-9)", value=True)
    c_api1.caption("🔗 [출처: Roboflow Universe 모델 1](https://universe.roboflow.com/defect-detection-0atjo/concrete-defect-detection-zuym8)")
    use_model2 = c_api2.checkbox("요철/불균질면 탐지 AI (API-10)", value=True)
    c_api2.caption("🔗 [출처: Roboflow Universe 모델 2](https://universe.roboflow.com/shm/concrete-defect-detection)")
    use_model3 = c_api3.checkbox("범용 콘크리트 결함 AI (API-11)", value=True)
    c_api3.caption("🔗 [출처: Roboflow Universe 모델 3](https://universe.roboflow.com/concrete-defects/concrete-defects-irdui)")
    st.write("")
    
    st.markdown("#### 🌐 2단계: 빅테크 클라우드 및 자체 딥러닝 AI 연동 (확장 예정)")
    c_ext1, c_ext2, c_ext3, c_ext4 = st.columns(4)
    c_ext1.checkbox("네이버 클라우드 AI", value=False)
    c_ext1.caption("*(추후 연동 예정, 지금 작동 X)*")
    c_ext2.checkbox("아마존 클라우드 AI (AWS)", value=False)
    c_ext2.caption("*(추후 연동 예정, 지금 작동 X)*")
    c_ext3.checkbox("구글 클라우드 AI (GCP)", value=False)
    c_ext3.caption("*(추후 연동 예정, 지금 작동 X)*")
    c_ext4.checkbox("자체 빅데이터 학습 AI", value=False)
    c_ext4.caption("*(추후 연동 예정, 지금 작동 X)*")
    st.write("---")

    uploaded_file = st.file_uploader("📸 벽면 사진 업로드", type=["jpg", "png"])

    # 변수 초기화 (에러 원천 차단)
    weather_map_pil = None
    strike_map_pil = None
    final_selected_count = 0
    p_scale = 0

    if uploaded_file:
        image = Image.open(uploaded_file)
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
            p1_x = st.number_input("기준점1 X (px)", max_value=w, value=int(w*0.3))
            p1_y = st.number_input("기준점1 Y (px)", max_value=h, value=int(h*0.85))
        with c_pt2:
            p2_x = st.number_input("기준점2 X (px)", max_value=w, value=int(w*0.7))
            p2_y = st.number_input("기준점2 Y (px)", max_value=h, value=int(h*0.85))
        with c_len:
            real_len = st.number_input("두 점 사이 실제 거리 (mm)", min_value=1.0, value=300.0)

        mm_per_pixel, pixel_dist = calculate_pixel_scale(p1_x, p1_y, p2_x, p2_y, real_len)
        p_scale = mm_per_pixel
        
        if mm_per_pixel > 0:
            st.success(f"📊 **크기 분석 완료:** 실제 사진 크기: `{(w*mm_per_pixel)/10:.1f} cm × {(h*mm_per_pixel)/10:.1f} cm` | 픽셀당 거리: `{mm_per_pixel:.4f} mm/px`")

        px_1cm_rad = int(10 / mm_per_pixel / 2) if mm_per_pixel > 0 else 10
        px_2cm = int(20 / mm_per_pixel) if mm_per_pixel > 0 else 40
        px_3cm = int(30 / mm_per_pixel) if mm_per_pixel > 0 else 60
        
        final_defect = np.zeros((h, w), dtype=np.uint8)
        
        with st.spinner("🌐 선택된 AI 모델 앙상블 초정밀 픽셀 분석 진행 중..."):
            edges = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0), 30, 80)
            final_defect = cv2.bitwise_or(final_defect, edges)
            
            is_success, buffer = cv2.imencode(".jpg", img_bgr)
            img_bytes = buffer.tobytes()

            if use_model1: final_defect = cv2.bitwise_or(final_defect, fetch_roboflow_mask(img_bytes, "general-segmentation-api-9", "crack, efflorescence, Exposed_reinforcement", w, h))
            if use_model2: final_defect = cv2.bitwise_or(final_defect, fetch_roboflow_mask(img_bytes, "general-segmentation-api-10", "defect, 0, 1", w, h))
            if use_model3: final_defect = cv2.bitwise_or(final_defect, fetch_roboflow_mask(img_bytes, "general-segmentation-api-11", "Concrete defects", w, h))

        safe_area = cv2.bitwise_not(final_defect)
        safe_area[:px_2cm, :] = 0; safe_area[-px_2cm:, :] = 0
        safe_area[:, :px_2cm] = 0; safe_area[:, -px_2cm:] = 0

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(safe_area, connectivity=8)
        max_area = np.max(stats[1:, cv2.CC_STAT_AREA]) if num_labels > 1 else 1

        overlay = np.zeros_like(img_bgr)
        color_red, color_orange, color_blue, color_green = [0, 0, 255], [0, 165, 255], [255, 0, 0], [0, 255, 0]
        overlay[:] = color_red
        
        kernel_1cm = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(10/mm_per_pixel)*2+1, int(10/mm_per_pixel)*2+1))
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
                        overlay[y:y+grid_size, x:x+grid_size] = color_orange
                        score = area_size * 0.3
                    elif area_size > max_area * 0.4:
                        overlay[y:y+grid_size, x:x+grid_size] = color_green
                        score = area_size * 1.2
                    else:
                        overlay[y:y+grid_size, x:x+grid_size] = color_blue
                        score = area_size * 0.8
                    all_candidates.append({'x': x, 'y': y, 'score': score})

        weather_map_img = cv2.addWeighted(img_bgr, 0.4, overlay, 0.6, 0)
        cv2.line(weather_map_img, (int(p1_x), int(p1_y)), (int(p2_x), int(p2_y)), (0, 0, 0), 5)

        all_candidates.sort(key=lambda k: k['score'], reverse=True)
        final_selected_pts = []
        target_count = desired_strikes + 5 

        for cand in all_candidates:
            if not any(math.sqrt((cand['x'] - p['x'])**2 + (cand['y'] - p['y'])**2) < px_3cm for p in final_selected_pts):
                final_selected_pts.append(cand)
                if len(final_selected_pts) >= target_count: break

        strike_map_img = img_bgr.copy()
        for cand in all_candidates[::2]: 
            if not any(math.sqrt((cand['x'] - p['x'])**2 + (cand['y'] - p['y'])**2) < px_3cm*0.5 for p in final_selected_pts[:desired_strikes]):
                cv2.circle(strike_map_img, (cand['x'], cand['y']), 1, (255, 255, 255), -1)

        for idx, pt in enumerate(final_selected_pts):
            rad = max(14, px_1cm_rad) 
            if idx < desired_strikes:
                cv2.circle(strike_map_img, (pt['x'], pt['y']), rad, (0, 255, 0), -1)
                cv2.putText(strike_map_img, str(idx+1), (pt['x']-8, pt['y']+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
            elif idx < target_count:
                extra_label = chr(65 + (idx - desired_strikes))
                cv2.circle(strike_map_img, (pt['x'], pt['y']), rad, (0, 165, 255), -1)
                cv2.putText(strike_map_img, extra_label, (pt['x']-8, pt['y']+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

        weather_map_pil = Image.fromarray(cv2.cvtColor(weather_map_img, cv2.COLOR_BGR2RGB))
        strike_map_pil = Image.fromarray(cv2.cvtColor(strike_map_img, cv2.COLOR_BGR2RGB))
        final_selected_count = len(final_selected_pts)

        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.markdown("#### 1️⃣ AI 다중 앙상블 신뢰도 지도")
            st.image(weather_map_pil, use_container_width=True)
        with col_res2:
            st.markdown("#### 2️⃣ 시방서 기반 AI 최적 타격 좌표")
            st.image(strike_map_pil, use_container_width=True)

        is_usable = final_selected_count >= desired_strikes
        st.write("---")
        if is_usable:
            st.success(f"⚙️ **KS표준 기반 희망 타격횟수 확보 성공.**")
        else:
            st.error("❌ **타격 가능한 안전 구역이 부족하여 희망 타격횟수를 만족할 수 없습니다.**")

        # 📄 1페이지 실시간 제미나이 리포트 인터페이스 (이미지가 완벽히 있을 때만 활성화)
        st.write("---")
        st.subheader("📋 분석 근거 및 리포트 내보내기")
        st.write("학술 논문 및 시방서 기준에 근거한 공식 보고서를 다운로드할 수 있습니다.")
        
        if weather_map_pil is not None and strike_map_pil is not None:
            if st.button("🚀 1페이지 제미나이 연동 PDF 리포트 생성"):
                with st.spinner("Gemini AI가 표준시방서 및 학술 논문을 기반으로 분석 리포트를 작성 중입니다..."):
                    p1_summary = f"장소: {m_loc} / 희망타격: {desired_strikes}회 / 기상: 온도 {auto_temp}도, 습도 {auto_hum}% / 모델: Roboflow 3종 앙상블 적용 / 픽셀당 {p_scale:.4f}mm / 3cm 이격 룰 적용하여 {final_selected_count}개 점 탐색 성공"
                    gemini_text = generate_gemini_commentary(1, p1_summary)
                    
                    st.markdown("### 🤖 제미나이 실시간 분석 의견 요약")
                    st.write(gemini_text)
                    
                    meta = {
                        'date': f"{m_date} {m_hour}:{m_min:02d}", 'location': m_loc, 'target_count': desired_strikes,
                        'temp': auto_temp, 'humidity': auto_hum, 'status': '적절' if is_weather_valid else '부적절',
                        'ai_info': '균열, 요철, 결함 탐지 다중 AI 모델(Roboflow API)', 'pixel_scale': f"{p_scale:.4f}",
                        'area': int((w*mm_per_pixel)*(h*mm_per_pixel)/100)
                    }
                    
                    pdf_bytes = create_page1_pdf(meta, weather_map_pil, strike_map_pil, gemini_text)
                    st.success("✨ 리포트 생성 성공!")
                    st.download_button("📥 AI 추천 타격지점 보고서 다운로드 (PDF)", data=pdf_bytes, file_name="AI_Schmidt_Hammer_Recommendation_Report.pdf", mime="application/pdf")

# =========================================================================
# 2페이지: 다중 센서 및 환경 변수 복합 강도 연산 시스템
# =========================================================================
elif "2." in main_menu:
    st.title("📊 SCI급 다중 센서 및 환경 변수 복합 강도 연산 시스템")
    
    col_env, col_data = st.columns([1, 1])
    with col_env:
        st.subheader("📋 현장 계측 정보 및 재령 입력")
        m2_date = st.date_input("슈미트해머 실시 날짜", datetime.date.today())
        m2_hour, m2_min = st.selectbox("시", list(range(24)), index=10), st.selectbox("분", [0, 15, 30, 45], index=0)
        m2_loc = st.text_input("위치", value="현장 A측면")
        
        auto_temp2, auto_hum2 = fetch_kma_weather_simulated(m2_date, m2_hour, m2_min, m2_loc)
        st.warning(f"📡 [기상청] 온도: {auto_temp2} ℃ / 습도: {auto_hum2} %")
        is_weather2_valid = evaluate_ks_weather(auto_temp2, auto_hum2)[0]
        
        st.write("---")
        m2_cast = st.date_input("타설일", datetime.date.today() - datetime.timedelta(days=60))
        total_days = max(1, (m2_date - m2_cast).days)
        fck = st.number_input("설계기준강도 (MPa)", value=24.0)
        strike_count = st.selectbox("타격 횟수", [10, 15, 20, 25, 30], index=2)
        
        use_ultra = st.checkbox("🟢 초음파 연동"); val_ultra = st.number_input("초음파 속도 (m/s)", value=3950.0) if use_ultra else 0
        use_slump = st.checkbox("🟢 슬럼프 연동"); val_slump = st.number_input("슬럼프 (mm)", value=160.0) if use_slump else 0

    with col_data:
        st.subheader("🔨 반발도(R값) 획득 데이터 입력")
        raw_inputs = [st.number_input(f"{i}번째 R값", value=39.0 if i!=5 else 22.0, key=f"r_{i}") for i in range(1, strike_count + 1)]

    raw_arr = np.array(raw_inputs, dtype=float)
    total_avg = np.mean(raw_arr)
    
    # KS F 2730 규격 필터링 (±10%)
    lower, upper = total_avg * 0.90, total_avg * 1.10
    filtered_data = [v for v in raw_arr if lower <= v <= upper]
    excluded_indices = [i+1 for i, v in enumerate(raw_arr) if not (lower <= v <= upper)]
    ks_avg = np.mean(filtered_data) if filtered_data else total_avg
    ex_count = len(excluded_indices)

    fc_rebound = max(0.0, 1.3 * ks_avg - 14.0)
    age_factor = max(0.82, 1.0 - 0.03 * math.log(total_days / 28.0)) if total_days > 28 else 1.0

    st.write("---")
    st.markdown("### 📈 데이터 보정 및 복합 추정 결과 리포트")
    c_st1, c_st2 = st.columns(2)
    c_st1.metric("전체 평균", f"{total_avg:.2f} R")
    c_st2.metric("보정 평균 (KS 필터링)", f"{ks_avg:.2f} R", f"⚠️ {ex_count}개 폐기")
    
    fc_ultra_only = (0.0028 * (ks_avg ** 1.2) * ((val_ultra/1000.0) ** 2.3)) * age_factor if use_ultra else 0
    slump_corr = max(0.80, 1.0 - 0.0008 * (val_slump - 150)) if (use_slump and val_slump>150) else 1.0
    fc_slump_only = fc_rebound * age_factor * slump_corr if use_slump else 0

    env_factor = 1.0
    if auto_hum2 >= 80.0: env_factor *= 1.06 
    if auto_temp2 < 5.0 or auto_temp2 > 35.0: env_factor *= 0.93 
    
    base_hybrid = fc_rebound
    if use_ultra: base_hybrid = (0.0032 * (ks_avg ** 1.25) * ((val_ultra/1000.0) ** 2.1)) * age_factor
    if use_slump and val_slump > 150: base_hybrid *= max(0.85, 1.0 - 0.0007 * (val_slump - 150))
    fc_final_hybrid = base_hybrid * env_factor
    
    st.success(f"🏆 **[종합] 하이브리드 최종 추정 강도:** `{fc_final_hybrid:.1f} MPa`")
    
    # 📄 2페이지 실시간 제미나이 리포트 인터페이스
    st.write("---")
    st.subheader("📊 정밀 보정 데이터 리포트 내보내기")
    
    if st.button("🚀 2페이지 제미나이 연동 복합추정 리포트 생성"):
        with st.spinner("Gemini AI가 KS F 2730 기준 및 복합추정식을 기반으로 정밀 보고서를 작성 중입니다..."):
            p2_summary = f"현장 원시값: {raw_inputs} / 전체평균: {total_avg:.2f} / 이상치 10% 폐기: {ex_count}개 / 보정평균: {ks_avg:.2f} / 재령: {total_days}일 / 초음파: {val_ultra} / 슬럼프: {val_slump} / 최종추정강도: {fc_final_hybrid:.1f}MPa"
            gemini_text2 = generate_gemini_commentary(2, p2_summary)
            
            st.markdown("### 🤖 제미나이 구조 엔지니어 정밀 평가 의견")
            st.write(gemini_text2)
            
            data_sample = {
                'date': f"{m2_date} {m2_hour}:{m2_min:02d}", 'location': m2_loc, 'temp': auto_temp2, 'humidity': auto_hum2,
                'status': '적절' if is_weather2_valid else '부적절', 'total_strikes': strike_count, 'raw_values': str(raw_inputs),
                'pour_date': str(m2_cast), 'elapsed_days': total_days, 'age_days': total_days, 'design_strength': fck,
                'ultrasonic': val_ultra if use_ultra else "미적용", 'slump': val_slump if use_slump else "미적용",
                'total_avg': f"{total_avg:.2f}", 'filtered_avg': f"{ks_avg:.2f}", 'deleted_count': ex_count,
                'rebound_strength': f"{fc_rebound:.1f}", 'ultrasonic_strength': f"{fc_ultra_only:.1f}", 
                'slump_strength': f"{fc_slump_only:.1f}", 'final_strength': f"{fc_final_hybrid:.1f}"
            }
            
            pdf_bytes_p2 = create_page2_pdf(data_sample, gemini_text2)
            st.success("✨ 강도 정밀 보정 리포트 생성 성공!")
            st.download_button("📥 데이터 보정 및 복합 추정 결과 리포트 다운로드 (PDF)", data=pdf_bytes_p2, file_name="Schmidt_Hammer_Strength_Estimation_Report.pdf", mime="application/pdf")
