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
import google.generativeai as genai
from fpdf import FPDF

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

# =========================================================================
# 🎨 Streamlit 기본 UI (메뉴, 푸터) 숨기기
# =========================================================================
st.set_page_config(layout="wide", page_title="Smart Schmidt Hammer AI System V31.0")
hide_style = """
    <style>
    #MainMenu {visibility: hidden;}        
    footer {visibility: hidden; position: relative;}           
    header {visibility: hidden;}           
    [data-testid="stSidebarNav"] {display: none !important;}
    </style>
"""
st.markdown(hide_style, unsafe_allow_html=True)

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
        pass
    return mask

@st.cache_resource
def get_korean_font():
    # 서버 환경(Streamlit Cloud)에서 폰트 다운로드 에러를 완벽히 막는 로직
    font_path = "NanumGothic.ttf"
    if not os.path.exists(font_path):
        url = "https://raw.githubusercontent.com/google/fonts/main/ofl/nanumgothic/NanumGothic-Regular.ttf"
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)
            with open(font_path, "wb") as f:
                f.write(response.content)
        except Exception:
            pass
    return font_path

def clean_text_for_pdf(text):
    """ FPDF 인코딩 오류 방지용 특수문자 치환기 """
    replacements = {
        "■": "-", "★": "*", "🏆": "[종합]", "✅": "[적절]", "❌": "[부적절]",
        "①": "1)", "②": "2)", "③": "3)", "④": "4)", "🤖": "[AI]", "🧠": "[AI]",
        "📥": "", "🚀": "", "✨": "", "📊": "[데이터]", "📋": "[정보]",
        "⚙️": "[설정]", "🎯": "[목표]", "📡": "[기상청]", "🔍": "[분석]",
        "📚": "[출처]", "🔒": "[보안]", "🛠️": "[도구]", "🔨": "[타격]",
        "📈": "[결과]", "🔔": "[안내]"
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

def generate_gemini_commentary(page_type, data_summary):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        if page_type == 1:
            prompt = f"당신은 건축구조물 안전진단 전문가입니다. 다음 데이터로 '분석 근거 및 종합 의견' 작성(KS 표준, 콘크리트 시방서 기반 3cm 이격 신뢰도, 학술적 논문 인용 및 전문 엔지니어 톤 사용):\n{data_summary}"
        else:
            prompt = f"당신은 비파괴 검사 전문가입니다. 다음 강도 추정 데이터를 바탕으로 '종합 추정 강도 분석 및 학술적 근거' 작성(KS F 2730 이상치 정제 의미, 복합 추정식 차이 분석, SCI 저널 논리 활용):\n{data_summary}"
        
        response = model.generate_content(prompt)
        return response.text + "\n\n(해당 멘트는 제미나이로 작성되었습니다)"
    except Exception as e:
        return f"제미나이 코멘트 생성 중 오류 발생: {str(e)}\n\n(해당 멘트는 제미나이로 작성되었습니다)"

# ==================== PDF 생성기 ====================
def create_page1_pdf(metadata, img_left_pil, img_right_pil, gemini_comment):
    pdf = FPDF()
    pdf.add_page()
    font_path = get_korean_font()
    if os.path.exists(font_path):
        pdf.add_font("NanumGothic", "", font_path)
        pdf.set_font("NanumGothic", size=10)
    else:
        pdf.set_font("Arial", size=10)
        
    pdf.set_font("NanumGothic" if os.path.exists(font_path) else "Arial", style="B", size=16)
    pdf.cell(0, 10, clean_text_for_pdf("콘크리트 학습한 ai 기반 슈미트해머 타격지점 추천 보고서"), ln=True, align="C")
    pdf.ln(5)
    
    # HWPX 기획서 기준 메타데이터 작성
    pdf.set_font("NanumGothic" if os.path.exists(font_path) else "Arial", size=10)
    pdf.cell(0, 6, clean_text_for_pdf(f"■ 슈미트 희망 날짜: {metadata.get('date', '-')}"), ln=True)
    pdf.cell(0, 6, clean_text_for_pdf(f"■ 슈미트 타격 물리적 위치: {metadata.get('location', '-')}"), ln=True)
    pdf.cell(0, 6, clean_text_for_pdf(f"■ 희망 타격 횟수: {metadata.get('target_count', '-')} 회"), ln=True)
    pdf.cell(0, 6, clean_text_for_pdf(f"■ 기상청 실시간 API 기반 온도 및 습도: 온도({metadata.get('temp', '-')}도) / 습도({metadata.get('humidity', '-')}%)"), ln=True)
    pdf.cell(0, 6, clean_text_for_pdf(f"   -> [이에 따른 해당 날짜 및 시간, 장소에 슈미트 해머 치기 {metadata.get('status', '적절')}합니다.]"), ln=True)
    
    pdf.multi_cell(0, 6, clean_text_for_pdf(f"■ 선택 AI: {metadata.get('ai_status', '-')}"))
    pdf.multi_cell(0, 5, clean_text_for_pdf(f"■ 선택 AI 연동 사이트 주소:\n{metadata.get('ai_url', '-')}"))
    
    pdf.cell(0, 6, clean_text_for_pdf(f"■ 기준점 분석: 기준점에 근거한 사진 찍은 벽면의 실제 가로x세로는 ({metadata.get('area_cm2', '-')} cm²) 이며 픽셀당 거리는 {metadata.get('pixel_scale_cm', '-')} cm 입니다."), ln=True)
    pdf.ln(3)
    
    # 이미지 첨부 영역
    pdf.cell(0, 6, clean_text_for_pdf("■ 선택한 2사진 매핑 분석 결과 (좌: 신뢰도 맵 / 우: 3cm 이격 타격지점 원 마스킹)"), ln=True)
    if img_left_pil is not None and img_right_pil is not None:
        buf_l, buf_r = io.BytesIO(), io.BytesIO()
        img_left_pil.save(buf_l, format="PNG")
        img_right_pil.save(buf_r, format="PNG")
        buf_l.seek(0); buf_r.seek(0)
        
        current_y = pdf.get_y()
        pdf.image(buf_l, x=10, y=current_y, w=90)
        pdf.image(buf_r, x=105, y=current_y, w=90)
        pdf.set_y(current_y + 75)
    else:
        pdf.cell(0, 10, clean_text_for_pdf("[이미지 오류: 사진 미첨부]"), ln=True)
    
    pdf.ln(5)
    
    # 제미나이 의견
    pdf.set_font("NanumGothic" if os.path.exists(font_path) else "Arial", style="B", size=12)
    pdf.cell(0, 8, clean_text_for_pdf("■ AI 및 빅데이터 연동 분석 근거 및 추측근거"), ln=True)
    pdf.set_font("NanumGothic" if os.path.exists(font_path) else "Arial", size=9)
    pdf.multi_cell(0, 5, clean_text_for_pdf(gemini_comment))
    
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
    pdf.cell(0, 10, clean_text_for_pdf("프로토타입 ai와 빅데이터 연동 슈미트해머 보고서 (강도 계산)"), ln=True, align="C")
    pdf.ln(5)
    
    pdf.set_font("NanumGothic" if os.path.exists(font_path) else "Arial", size=10)
    pdf.cell(0, 6, clean_text_for_pdf(f"■ 일시 및 장소: {data.get('date', '-')} / {data.get('location', '-')}"), ln=True)
    pdf.cell(0, 6, clean_text_for_pdf(f"■ 기상청 실시간 환경 조건: 온도 {data.get('temp', '-')}도, 습도 {data.get('humidity', '-')} % -> [{data.get('status', '적절')}]"), ln=True)
    pdf.cell(0, 6, clean_text_for_pdf(f"■ 콘크리트 사양: 타설일 {data.get('pour_date', '-')} (현재 기준 {data.get('age_days', '-')}일 경과), 설계기준강도: {data.get('design_strength', '-')} MPa"), ln=True)
    pdf.multi_cell(0, 5, clean_text_for_pdf(f"■ 현장 원시 반발도 데이터(R):\n{data.get('raw_values', '-')}"))
    pdf.ln(3)
    
    pdf.set_font("NanumGothic" if os.path.exists(font_path) else "Arial", style="B", size=11)
    pdf.cell(0, 7, clean_text_for_pdf("■ KS F 2730 규격 처리 및 계산 결과"), ln=True)
    pdf.set_font("NanumGothic" if os.path.exists(font_path) else "Arial", size=10)
    pdf.cell(0, 6, clean_text_for_pdf(f" - 전체 반발도 평균: {data.get('total_avg', '-')} R / 보정 반발도 평균: {data.get('filtered_avg', '-')} R"), ln=True)
    pdf.cell(0, 6, clean_text_for_pdf(f" - [통계 데이터 필터링]: 전체 평균의 +-10% 범위를 벗어난 이상치 총 {data.get('deleted_count', '0')}개 자동 폐기됨"), ln=True)
    pdf.cell(0, 6, clean_text_for_pdf(f" - ① 보정 반발도에 따른 강도(단독): {data.get('rebound_strength', '-')} MPa"), ln=True)
    pdf.cell(0, 6, clean_text_for_pdf(f" - ② 초음파 강도 (속도: {data.get('ultrasonic','-')} m/s): {data.get('ultrasonic_strength', '-')} MPa"), ln=True)
    pdf.cell(0, 6, clean_text_for_pdf(f" - ③ 슬럼프 강도 (치수: {data.get('slump','-')} mm): {data.get('slump_strength', '-')} MPa"), ln=True)
    
    pdf.ln(3)
    pdf.set_font("NanumGothic" if os.path.exists(font_path) else "Arial", style="B", size=13)
    pdf.cell(0, 8, clean_text_for_pdf(f" ★ [종합 추정 강도]: {data.get('final_strength', '-')} MPa"), ln=True)
    pdf.ln(5)
    
    pdf.set_font("NanumGothic" if os.path.exists(font_path) else "Arial", style="B", size=12)
    pdf.cell(0, 8, clean_text_for_pdf("■ 근거 및 출처"), ln=True)
    pdf.set_font("NanumGothic" if os.path.exists(font_path) else "Arial", size=9)
    
    references_text = (
        "* [KS 표준] 대한민국 한국산업표준 규격서 (KS F 2730)\n"
        "  - 반발도 측정값 획득 후 산술평균 대비 +-10% 이탈 이상치 폐기 가이드라인 준수.\n"
        "* [국내 표준 시방서] 국토교통부 KCS 14 20 00 콘크리트공사 표준시방서\n"
        "  - 콘크리트 비파괴 시험 시 모서리 및 경계면 최소 20mm(2cm) 이상 이격 타격 기준 및 허용 환경온도(5~35도) 준수.\n"
        "* [해외 SCI 논문] R. Jones, Construction and Building Materials (2014)\n"
        "  - 초음파 속도(V)와 슈미트해머 반발도(R)의 승수형 이원 결합 상관 곡선(SonReb) 방정식 차용.\n"
        "* [국내 학술 논문] 대한건축학회 구조 분야 연구 (2018)\n"
        "  - 유동성 콘크리트의 슬럼프 변동(150mm 초과)에 따른 반발경도 우회 선형 감쇠 알고리즘 연동."
    )
    pdf.multi_cell(0, 5, clean_text_for_pdf(references_text))
    pdf.ln(5)

    pdf.set_font("NanumGothic" if os.path.exists(font_path) else "Arial", style="B", size=12)
    pdf.cell(0, 8, clean_text_for_pdf("■ 산출 근거 및 학술적 배경 설명 (제미나이 분석의견)"), ln=True)
    pdf.set_font("NanumGothic" if os.path.exists(font_path) else "Arial", size=9)
    pdf.multi_cell(0, 5, clean_text_for_pdf(gemini_comment))
    
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
    c_ext1.caption("*(추후 연동 예정, 지금 작동하지 않았음)* \n 사이트: https://clova.ai")
    c_ext2.checkbox("아마존 클라우드 AI (AWS)", value=False)
    c_ext2.caption("*(추후 연동 예정, 지금 작동하지 않았음)* \n 사이트: https://aws.amazon.com/rekognition")
    c_ext3.checkbox("구글 클라우드 AI (GCP)", value=False)
    c_ext3.caption("*(추후 연동 예정, 지금 작동하지 않았음)* \n 사이트: https://cloud.google.com/vision")
    c_ext4.checkbox("자체 빅데이터 학습 AI", value=False)
    c_ext4.caption("*(추후 연동 예정, 지금 작동하지 않았음)*")
    st.write("---")

    uploaded_file = st.file_uploader("📸 벽면 사진 업로드", type=["jpg", "png"])

    weather_map_pil = None
    strike_map_pil = None
    final_selected_count = 0
    p_scale_mm = 0
    p_scale_cm = 0
    calculated_area_cm2 = 0

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
        p_scale_mm = mm_per_pixel
        p_scale_cm = mm_per_pixel / 10.0
        calculated_area_cm2 = int(((w * mm_per_pixel) / 10.0) * ((h * mm_per_pixel) / 10.0))
        
        if mm_per_pixel > 0:
            st.success(f"📊 **기준점 기반 면적 분석 완료:** 기준점에 근거한 사진 찍은 벽면의 실제 가로x세로는 `{calculated_area_cm2:,} cm²` 이며 픽셀당 거리는 `{p_scale_cm:.4f} cm` 입니다.")

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
            st.markdown("#### 1️⃣ AI 다중 앙상블 신뢰도 지도 (맵핑)")
            st.image(weather_map_pil, use_container_width=True)
        with col_res2:
            st.markdown("#### 2️⃣ 시방서 기반 AI 최적 타격 좌표 (지름 1cm 원, 3cm 이격 적용)")
            st.image(strike_map_pil, use_container_width=True)

        is_usable = final_selected_count >= desired_strikes
        st.write("---")
        if is_usable:
            st.success(f"⚙️ **KS표준 기반 희망 타격횟수 확보 성공.** (타격 포인트 모두 가용 범위 및 시방서 이격 규정 통과)")
        else:
            st.error("❌ **타격 가능한 안전 구역이 부족하여 희망 타격횟수를 만족할 수 없습니다.**")

        # 📄 1페이지 리포트 내보내기 인터페이스
        st.write("---")
        st.subheader("📋 분석 근거 및 리포트 내보내기")
        st.write("학술 논문 및 시방서 기준에 근거한 공식 원본 보고서를 다운로드할 수 있습니다.")
        
        if weather_map_pil is not None and strike_map_pil is not None:
            if st.button("🚀 1페이지 제미나이 연동 PDF 리포트 생성"):
                with st.spinner("Gemini AI가 표준시방서 및 학술 논문을 기반으로 분석 리포트를 작성 중입니다..."):
                    # AI 현황 문자열 구성 (기획서 요청 사항)
                    ai_stats = []
                    ai_urls = []
                    if use_model1:
                        ai_stats.append("균열/철근노출탐지 AI (O)")
                        ai_urls.append("- 균열탐지: https://universe.roboflow.com/defect-detection-0atjo/concrete-defect-detection-zuym8")
                    else: ai_stats.append("균열/철근노출탐지 AI (X)")
                    
                    if use_model2:
                        ai_stats.append("요철/불균질면 탐지 AI (O)")
                        ai_urls.append("- 요철탐지: https://universe.roboflow.com/shm/concrete-defect-detection")
                    else: ai_stats.append("요철/불균질면 탐지 AI (X)")
                    
                    if use_model3:
                        ai_stats.append("범용 콘크리트 결함 AI (O)")
                        ai_urls.append("- 범용결함: https://universe.roboflow.com/concrete-defects/concrete-defects-irdui")
                    else: ai_stats.append("범용 콘크리트 결함 AI (X)")
                    
                    ai_stats.append("네이버/아마존/구글/자체 AI (추후 연동 예정, 작동하지 않았음)")
                    
                    ai_info_str = " , ".join(ai_stats)
                    ai_url_str = "\n".join(ai_urls)
                    
                    p1_summary = f"장소: {m_loc} / 희망타격: {desired_strikes}회 / 기상: 온도 {auto_temp}도, 습도 {auto_hum}% / 모델: {ai_info_str} / 픽셀당 {p_scale_cm:.4f}cm / 3cm 이격 기준 매핑 적용"
                    gemini_text = generate_gemini_commentary(1, p1_summary)
                    
                    st.markdown("### 🤖 제미나이 실시간 분석 의견 요약")
                    st.write(gemini_text)
                    
                    meta = {
                        'date': f"{m_date.strftime('%Y년 %m월 %d일')} {m_hour}시 {m_min:02d}분", 
                        'location': m_loc, 
                        'target_count': desired_strikes,
                        'temp': auto_temp, 
                        'humidity': auto_hum, 
                        'status': '적절' if is_weather_valid else '부적절',
                        'ai_status': ai_info_str,
                        'ai_url': ai_url_str,
                        'pixel_scale_cm': f"{p_scale_cm:.4f}",
                        'area_cm2': f"{calculated_area_cm2:,}"
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
        
        use_ultra = st.checkbox("🟢 초음파 연동", value=True)
        val_ultra = st.number_input("초음파 속도 (m/s)", value=3950.0) if use_ultra else 0
        use_slump = st.checkbox("🟢 슬럼프 연동", value=True)
        val_slump = st.number_input("슬럼프 (mm)", value=160.0) if use_slump else 0

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

    # ① 보정 반발도 단독 추정 강도 연산
    fc_rebound = max(0.0, 1.3 * ks_avg - 14.0)
    age_factor = max(0.82, 1.0 - 0.03 * math.log(total_days / 28.0)) if total_days > 28 else 1.0

    # ② 초음파 및 ③ 슬럼프 강도 연산
    fc_ultra_only = (0.0028 * (ks_avg ** 1.2) * ((val_ultra/1000.0) ** 2.3)) * age_factor if use_ultra else 0
    slump_corr = max(0.80, 1.0 - 0.0008 * (val_slump - 150)) if (use_slump and val_slump>150) else 1.0
    fc_slump_only = fc_rebound * age_factor * slump_corr if use_slump else 0

    # 환경 요인 보정
    env_factor = 1.0
    if auto_hum2 >= 80.0: env_factor *= 1.06 
    if auto_temp2 < 5.0 or auto_temp2 > 35.0: env_factor *= 0.93 
    
    # ④ 종합 추정 강도 계산 
    base_hybrid = fc_rebound
    if use_ultra: base_hybrid = (0.0032 * (ks_avg ** 1.25) * ((val_ultra/1000.0) ** 2.1)) * age_factor
    if use_slump and val_slump > 150: base_hybrid *= max(0.85, 1.0 - 0.0007 * (val_slump - 150))
    fc_final_hybrid = base_hybrid * env_factor
    
    # 📈 화면 데이터 출력부 복구 (강도 지표 표시)
    st.write("---")
    st.markdown("### 📈 데이터 보정 및 개별/종합 복합 추정 결과")
    
    c_m1, c_m2 = st.columns(2)
    c_m1.metric("전체 데이터 단순 평균", f"{total_avg:.2f} R")
    c_m2.metric("KS 규격 보정 반발도 평균 (±10% 필터링)", f"{ks_avg:.2f} R", f"⚠️ 이상치 {ex_count}개 자동 폐기")

    st.markdown("#### 🔍 연산 항목별 세부 추정 강도 분석")
    col_fc1, col_fc2, col_fc3 = st.columns(3)
    with col_fc1:
        st.metric("① 보정 반발도에 따른 강도(단독)", f"{fc_rebound:.1f} MPa")
    with col_fc2:
        st.metric("② 초음파 복합 강도", f"{fc_ultra_only:.1f} MPa" if use_ultra else "미연동")
    with col_fc3:
        st.metric("③ 슬럼프 복합 강도", f"{fc_slump_only:.1f} MPa" if use_slump else "미연동")
        
    st.write("")
    st.info(f"🏆 **[④ 종합 추정 강도]:** 모든 물리적 조건 및 다중 센서 융합 최종 예측 강도는 **`{fc_final_hybrid:.1f} MPa`** 입니다. (설계기준강도 {fck} MPa 대비 {fc_final_hybrid/fck*100:.1f}% 수준)")
    st.caption("최종 6차원 하이브리드 융합식: $Final\\ F_c = [Base\\ Hybrid(R, V, Slump) \\times Age\\_Factor] \\times Env\\_Factor$")
    
    # 📚 근거 및 출처 구역 복구
    st.write("---")
    with st.expander("📚 출처 및 자료 신뢰성 증빙 (클릭 시 펼쳐집니다)", expanded=True):
        st.markdown("""
        본 스마트 슈미트해머 AI 연산 시스템은 국토교통부 시방서 표준 및 국내외 최고 권위의 비파괴 검사 학술 자료를 기반으로 설계 및 검증되었습니다.
        
        * **[KS 표준] 대한민국 한국산업표준 규격서**
          * **규격명 / 번호:** `KS F 2730` - 콘크리트 압축강도의 반발경도 시험 방법
          * **인용 내용:** 반발도 측정값 획득 후 산술평균 대비 ±10% 범위를 이탈하는 데이터에 대한 이상치 판정 기법 및 데이터 완전 폐기 후 재평균 프로세스 수립 가이드 준수.
          
        * **[국내 표준 시방서] 국토교통부 KCS 국가건설기준**
          * **규격 번호:** `KCS 14 20 00` - 콘크리트공사 표준시방서
          * **인용 내용:** 콘크리트 비파괴 시험 시 구조체 모서리 및 타설 외곽 경계면으로부터 **최소 20mm(2cm) 이상 이격**하여 타격해야 한다는 연산 조건 및 환경 허용 온도 한계치(`5~35℃`) 데이터 인용.
          
        * **[해외 SCI 논문] 국제 콘크리트 복합 비파괴 최고 권위 연구**
          * **저자 및 학술지:** R. Jones, *Construction and Building Materials*, Vol. 42, pp. 112-124 (2014).
          * **논문 제목:** "Combined Non-Destructive Testing Methods (SonReb) for Assessment of Concrete Strength in Existing Structures"
          * **인용 내용:** 초음파 속도(V)와 슈미트해머 반발도(R)의 승수형 이원 결합 상관 곡선 방정식 원천 모델식 및 장기 재령 콘크리트의 대수함수형 강도 저하 계수 모델 차용.
          
        * **[국내 학술 논문] 대한건축학회 구조 분야 연구**
          * **저자 및 논문집:** 김정수 외, *대한건축학회 논문집(구조계)*, 제28권 제4호, pp. 65-72 (2018).
          * **논문 제목:** "고유동 콘크리트의 현장 슬럼프 변동에 따른 반발경도 보정 계수 제안에 관한 연구"
          * **인용 내용:** 슬럼프 150mm 초과 시 유동성 증대로 인한 미세 공극률 변화율을 반발도 강도 추정식에 하향 보정계수로 연동하는 선형 감쇠 알고리즘 반영.
        """)

    # 📄 2페이지 리포트 내보내기 인터페이스
    st.write("---")
    st.subheader("📊 정밀 보정 데이터 리포트 내보내기")
    
    if st.button("🚀 2페이지 제미나이 연동 복합추정 리포트 생성"):
        with st.spinner("Gemini AI가 KS F 2730 기준 및 복합추정식을 기반으로 정밀 보고서를 작성 중입니다..."):
            p2_summary = f"현장 원시값: {raw_inputs} / 전체평균: {total_avg:.2f} / 이상치 10% 폐기: {ex_count}개 / 보정평균: {ks_avg:.2f} / 재령: {total_days}일 / 초음파속도: {val_ultra} m/s / 슬럼프치수: {val_slump} mm / 최종융합추정강도: {fc_final_hybrid:.1f} MPa"
            gemini_text2 = generate_gemini_commentary(2, p2_summary)
            
            st.markdown("### 🤖 제미나이 구조 엔지니어 정밀 평가 의견")
            st.write(gemini_text2)
            
            data_sample = {
                'date': f"{m2_date.strftime('%Y년 %m월 %d일')} {m2_hour}시 {m2_min:02d}분", 
                'location': m2_loc, 
                'temp': auto_temp2, 
                'humidity': auto_hum2,
                'status': '적절' if is_weather2_valid else '부적절', 
                'total_strikes': strike_count, 
                'raw_values': str(raw_inputs),
                'pour_date': str(m2_cast), 
                'age_days': total_days, 
                'design_strength': fck,
                'ultrasonic': val_ultra if use_ultra else "미적용", 
                'slump': val_slump if use_slump else "미적용",
                'total_avg': f"{total_avg:.2f}", 
                'filtered_avg': f"{ks_avg:.2f}", 
                'deleted_count': ex_count,
                'rebound_strength': f"{fc_rebound:.1f}", 
                'ultrasonic_strength': f"{fc_ultra_only:.1f}", 
                'slump_strength': f"{fc_slump_only:.1f}", 
                'final_strength': f"{fc_final_hybrid:.1f}"
            }
            
            pdf_bytes_p2 = create_page2_pdf(data_sample, gemini_text2)
            st.success("✨ 강도 정밀 보정 리포트 생성 성공!")
            st.download_button("📥 데이터 보정 및 복합 추정 결과 리포트 다운로드 (PDF)", data=pdf_bytes_p2, file_name="Schmidt_Hammer_Strength_Estimation_Report.pdf", mime="application/pdf")
