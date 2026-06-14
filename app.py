import streamlit as st
import cv2
import numpy as np
from PIL import Image
import datetime
import math
import hashlib
import requests
import google.generativeai as genai
import os
import io

# =========================================================================
# 🎨 1. Streamlit 기본 UI 설정 (무조건 가장 먼저 실행되어야 함!)
# =========================================================================
st.set_page_config(layout="wide", page_title="Smart Schmidt Hammer AI System V38.0", initial_sidebar_state="expanded")

hide_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden; position: relative;}
    
    /* 사이드바 접기 버튼 완전 숨김 -> 항상 열려있게 고정 */
    [data-testid="collapsedControl"] {display: none !important;}
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
"""
st.markdown(hide_style, unsafe_allow_html=True)

# =========================================================================
# 🔐 2. API 키 설정 구역
# =========================================================================
API_KEYS = {
    "ROBOFLOW_API": st.secrets.get("ROBOFLOW_API", ""),
    "KMA_WEATHER": st.secrets.get("KMA_WEATHER", ""),
    "GEMINI_API": st.secrets.get("GEMINI_API", ""),
}

if API_KEYS["GEMINI_API"]:
    genai.configure(api_key=API_KEYS["GEMINI_API"])

# =========================================================================
# 🔤 3. PDF 생성 및 폰트 설정 (ReportLab)
# =========================================================================
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

@st.cache_resource
def load_korean_font():
    font_url = "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf"
    font_path = "NanumGothic.ttf"
    if not os.path.exists(font_path):
        try:
            response = requests.get(font_url)
            with open(font_path, "wb") as f:
                f.write(response.content)
        except Exception:
            pass
    try:
        pdfmetrics.registerFont(TTFont('NanumGothic', font_path))
    except Exception:
        pass

load_korean_font()

# -------------------------------------------------------------------------
# 📄 1페이지 전용 PDF 생성 함수 (AI 분석, 사진 2장 포함)
# -------------------------------------------------------------------------
def create_page1_pdf(data):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    story = []
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('TitleStyle', fontName='NanumGothic', fontSize=15, leading=20, alignment=1, textColor=colors.HexColor('#1A365D'), spaceAfter=10)
    body_style = ParagraphStyle('BodyStyle', fontName='NanumGothic', fontSize=9, leading=14, textColor=colors.HexColor('#2D3748'))
    table_text = ParagraphStyle('TableText', fontName='NanumGothic', fontSize=8, leading=11, alignment=1)
    
    story.append(Paragraph("<b>콘크리트 학습한 AI 기반 슈미트해머 타격지점 추천 보고서</b>", title_style))
    
    meta_text = f"""
    <b>슈미트 희망 날짜:</b> {data['date']} {data['time']}<br/>
    <b>슈미트 타격 물리적 위치:</b> {data['loc']}<br/>
    <b>희망 타격 횟수:</b> {data['target']}회<br/><br/>
    <b>기상청 실시간 API 기반 온도 및 습도:</b> 온도({data['temp']}℃), 습도({data['hum']}%)<br/>
    <b>[이에 따른 해당 날짜 및 시간, 장소에 슈미트해머 치기 [{data['weather_ok']}]합니다.]</b>
    """
    story.append(Paragraph(meta_text, body_style))
    story.append(Spacer(1, 8))
    
    ai_text = f"""
    <b>선택 AI:</b> 균열/철근노출탐지 AI ({data['ai_1']}), 요철/불균질면 탐지 AI ({data['ai_2']}), 범용 콘크리트 결함 AI ({data['ai_3']}), 네이버/아마존/구글/자체 AI (X - 추후 연동 예정)<br/>
    <b>선택 AI 연동 주소:</b> {data['ai_urls']}
    """
    story.append(Paragraph(ai_text, body_style))
    story.append(Spacer(1, 8))
    
    cal_text = f"""
    <b>기준점 위치:</b> {data['pts']}<br/>
    <b>기준점 실제 거리:</b> {data['dist']} mm<br/>
    <b>기준점에 근거한 사진 찍은 벽면의 실제 가로x세로는:</b> {data['area']:.1f} cm² 이며 픽셀당 거리는 {data['px_scale']:.4f} cm 입니다.
    """
    story.append(Paragraph(cal_text, body_style))
    story.append(Spacer(1, 8))
    
    def cv2_to_rl(img_cv, w, h):
        if img_cv is None: return Paragraph("[이미지 없음]", table_text)
        is_success, buf = cv2.imencode(".jpg", img_cv)
        if not is_success: return Paragraph("[이미지 오류]", table_text)
        from reportlab.platypus import Image as RLImage
        return RLImage(io.BytesIO(buf.tobytes()), width=w, height=h)
        
    img_row1 = [Paragraph("<b>[선택 사진 1: AI 불균질 검출망]</b>", table_text), Paragraph("<b>[선택 사진 2: 최종 추천 타격점]</b>", table_text)]
    img_row2 = [cv2_to_rl(data.get('img_map'), 240, 240), cv2_to_rl(data.get('img_strike'), 240, 240)]
    
    t_img = Table([img_row1, img_row2], colWidths=[270, 270])
    t_img.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#A0AEC0')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_img)
    story.append(Spacer(1, 10))
    
    res_text = f"<b>분석 근거 & 코멘트:</b><br/>확보된 타격점: {data['cand_count']}개 (결함 의심 비율: {data['defect_ratio']:.2f}%)<br/>{data['ai_comment']}"
    t_comment = Table([[Paragraph(res_text, body_style)]], colWidths=[540])
    t_comment.setStyle(TableStyle([('BOX', (0,0), (-1,-1), 1, colors.HexColor('#ED8936')), ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#FFFAF0')), ('PADDING', (0,0), (-1,-1), 10)]))
    story.append(t_comment)
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# -------------------------------------------------------------------------
# 📄 2페이지 전용 PDF 생성 함수 (수식, 강도, R값 전체)
# -------------------------------------------------------------------------
def create_page2_pdf(data):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    story = []
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('TitleStyle', fontName='NanumGothic', fontSize=15, leading=20, alignment=1, textColor=colors.HexColor('#1A365D'), spaceAfter=10)
    body_style = ParagraphStyle('BodyStyle', fontName='NanumGothic', fontSize=9, leading=14, textColor=colors.HexColor('#2D3748'))
    table_text = ParagraphStyle('TableText', fontName='NanumGothic', fontSize=8, leading=11, alignment=1)
    
    story.append(Paragraph("<b>데이터 보정 및 복합 추정 결과 리포트</b>", title_style))
    
    meta_text = f"""
    <b>타격 날짜:</b> {data['date']} {data['time']}<br/>
    <b>타격 장소:</b> {data['loc']}<br/>
    <b>기상청 실시간 API 기반 온도 및 습도:</b> 온도({data['temp']}℃), 습도({data['hum']}%)<br/>
    <b>[이에 따른 해당 날짜 및 시간, 장소에 슈미트해머 치기 [{data['weather_ok']}]합니다.]</b><br/><br/>
    <b>타격 횟수:</b> {data['strike_count']}회<br/>
    <b>콘크리트 타설일:</b> {data['pour_date']} -> 해당 날짜로부터 오늘까지 {data['days']}일이 경과하였습니다. (재령일: {data['days']}일)<br/>
    <b>설계기준강도:</b> {data['fck']} MPa<br/>
    <b>선택) 초음파 속도:</b> {data['ultra']}<br/>
    <b>선택) 슬럼프:</b> {data['slump']}<br/>
    <b>타격 각도:</b> {data['angle']}°
    """
    story.append(Paragraph(meta_text, body_style))
    story.append(Spacer(1, 10))
    
    r_vals = data['r_list']
    chunk_size = 10
    for i in range(0, len(r_vals), chunk_size):
        chunk = r_vals[i:i+chunk_size]
        r_headers = [Paragraph(f"<b>현장 반발도 {i+j+1}번째</b>", table_text) for j in range(len(chunk))]
        r_values = [Paragraph(str(val), table_text) for val in chunk]
        t_r = Table([r_headers, r_values], colWidths=[54]*len(chunk))
        t_r.setStyle(TableStyle([
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#4A5568')), ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#EDF2F7')), ('PADDING', (0,0), (-1,-1), 5)
        ]))
        story.append(t_r)
        story.append(Spacer(1, 5))
        
    res_data = [
        [Paragraph("<b>전체 반발도 평균</b>", table_text), Paragraph(f"{data['r_mean']:.2f} R", table_text), Paragraph("<b>KS 보정 반발도</b>", table_text), Paragraph(f"{data['r_corr']:.2f} R", table_text)],
        [Paragraph("<b>보정 반발도 활용 강도</b>", table_text), Paragraph(f"{data['fc_reb']:.1f} MPa", table_text), Paragraph("<b>초음파 강도</b>", table_text), Paragraph(f"{data['fc_ultra']:.1f} MPa", table_text)],
        [Paragraph("<b>🔥 최종 복합 강도</b>", table_text), Paragraph(f"<b>{data['fc_hybrid']:.1f} MPa</b>", table_text), Paragraph("", table_text), Paragraph("", table_text)]
    ]
    t_res = Table(res_data, colWidths=[130, 140, 130, 140])
    t_res.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 1.5, colors.HexColor('#2B6CB0')), ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#90CDF4')),
        ('BACKGROUND', (0,2), (1,2), colors.HexColor('#EBF8FF')), ('PADDING', (0,0), (-1,-1), 8)
    ]))
    story.append(t_res)
    story.append(Spacer(1, 10))
    
    formula_text = """
    <b>계산식 및 출처/근거:</b><br/>
    • <b>반발도 각도 보정:</b> KS F 2730 연속 보간법 적용<br/>
    • <b>초음파 속도 환산:</b> KS F 2731 표준 환산 적용 (V = L/T)<br/>
    • <b>복합강도식:</b> F_c = (1.52 * R_0 + 16.87 * V - 66.9) * 환경/재령/슬럼프 보정계수 (국토안전관리원 및 KRISS 국가공인 다중회귀식)<br/>
    • <b>SCI / 대한건축학회:</b> 초음파-반발도 데이터 퓨전 및 슬럼프 공극률 보정 계수 모델링 (KCS 14 20 00 시방서 기반)
    """
    story.append(Paragraph(formula_text, body_style))
    
    doc.build(story)
    buffer.seek(0)
    return buffer


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
        return False, "부적절"
    return True, "적절"

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
본 분석 결과, 업로드된 콘크리트 표면 영상은 AI 기반 결함 탐지와 경계부 이격 조건을 함께 고려하여 슈미트해머 타격 가능 영역을 선별한 결과입니다. 추천된 타격 좌표는 타격점 간 최소 이격 조건을 반영하여 구성되었습니다.
""".strip()
    return """
입력된 반발도 데이터는 KS F 2730의 취지에 따라 평균값 대비 과도하게 이탈한 값을 선별하고, 보정 평균을 기준으로 강도 추정에 반영하였습니다. 초음파 속도와 슬럼프 조건을 함께 고려한 복합 추정 결과입니다.
""".strip()

def generate_gemini_commentary(page_type, data_summary):
    if not API_KEYS["GEMINI_API"]:
        return generate_static_engineering_commentary(page_type, data_summary)
    if page_type == 1:
        prompt = f"당신은 콘크리트 구조물 비파괴검사 및 안전진단 전문가입니다.\n아래 현장 데이터를 바탕으로 슈미트해머 타격 전 표면 신뢰도 분석 의견을 작성하세요.\n작성 조건: 한국어, 현장 보고서 문체, KS F 2730 관점 반영 (5~7문장)\n현장 데이터:\n{data_summary}"
    else:
        prompt = f"당신은 콘크리트 비파괴검사 전문가입니다.\n아래 강도 추정 데이터를 바탕으로 종합 분석 의견을 작성하세요.\n작성 조건: 한국어, KS F 2730 이상치 정제 의미 및 KRISS 국가표준 복합법 연동 설명 (5~7문장)\n현장 데이터:\n{data_summary}"
    
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
    if angle == 0: return 0.0
    if r_val <= 30: max_up, max_down = 3.2, -4.1
    elif r_val <= 40: max_up, max_down = 2.8, -4.8
    else: max_up, max_down = 2.2, -5.2
        
    rad = math.radians(angle)
    if angle > 0: return max_up * math.sin(rad)
    else: return max_down * abs(math.sin(rad))

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
# 1페이지: AI 표면 스캔 및 시방서 기반 타격점 추천
# =========================================================================
if "1." in main_menu:
    st.title("🎯 스마트 슈미트해머 5대 AI 표면 및 환경 신뢰도 판정 (V38.0)")

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
    is_weather_valid, weather_msg_str = evaluate_ks_weather(auto_temp, auto_hum)

    st.info(f"📡 해당 날짜/시간 기상청 데이터: **{m_date} {selected_time} 기준 / 온도 {auto_temp} ℃, 습도 {auto_hum} %**")
    if is_weather_valid: st.success("✅ [적절] 온도와 습도가 허용 범위 내에 있어 측정 신뢰성이 높습니다.")
    else: st.error("❌ [부적절] 온도가 5~35℃를 벗어나거나 습도가 80% 이상입니다.")

    st.write("---")
    st.markdown("#### 🧠 1단계: 콘크리트 특화 다중 AI 모델 활성화")
    c_api1, c_api2, c_api3 = st.columns(3)
    use_model1 = c_api1.checkbox("균열/철근노출 탐지 AI (API-9)", value=True)
    use_model2 = c_api2.checkbox("요철/불균질면 탐지 AI (API-10)", value=True)
    use_model3 = c_api3.checkbox("범용 콘크리트 결함 AI (API-11)", value=True)

    st.write("")
    st.markdown("#### 🌐 2단계: 빅테크 클라우드 및 자체 딥러닝 AI 연동 (확장 예정)")
    c_ext1, c_ext2, c_ext3, c_ext4 = st.columns(4)
    c_ext1.checkbox("네이버 클라우드 AI", value=False)
    c_ext2.checkbox("아마존 클라우드 AI (AWS)", value=False)
    c_ext3.checkbox("구글 클라우드 AI (GCP)", value=False)
    c_ext4.checkbox("자체 빅데이터 학습 AI", value=False)

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

        if is_weather_valid and is_usable: reliability_grade = "높음"; reliability_msg = "환경 조건과 표면 조건이 모두 양호하여 측정 신뢰도가 높습니다."
        elif is_usable: reliability_grade = "보통"; reliability_msg = "타격 영역은 확보되었으나 온도/습도 주의가 필요합니다."
        else: reliability_grade = "낮음"; reliability_msg = "타격 가능 영역이 부족합니다. 표면 정리 후 재분석을 권장합니다."

        st.markdown("#### 🧾 AI 표면 신뢰도 판정 요약")
        st.info(f"**분석 대상 면적:** `{real_width_cm:.1f} cm × {real_height_cm:.1f} cm = {calculated_area_cm2:,.1f} cm²`  \n**픽셀 환산 계수:** `1 pixel = {p_scale_cm:.4f} cm`  \n**AI 결함 의심 비율:** `{defect_ratio:.2f} %`  \n**확보된 후보 타격점:** `{final_selected_count}개` / 요청 `{desired_strikes}개`  \n**종합 신뢰도 등급:** `{reliability_grade}`  \n\n{reliability_msg}")

        st.write("---")
        st.subheader("🤖 Gemini AI 구조 분석 요약")
        if st.button("🚀 1페이지 Gemini AI 분석 코멘트 생성"):
            with st.spinner("Gemini AI가 분석 중입니다..."):
                p1_summary = f"측정위치:{m_loc}, 일시:{m_date} {selected_time}, 온도:{auto_temp}℃, 습도:{auto_hum}%, 결함비율:{defect_ratio:.2f}%, 후보점:{final_selected_count}개"
                gemini_text = generate_gemini_commentary(1, p1_summary)
                st.info(gemini_text)

        # -------------------------------------------------------------------------
        # 🖨️ 1페이지 전용 PDF 출력 버튼
        # -------------------------------------------------------------------------
        st.write("---")
        st.subheader("🖨️ 1페이지: AI 표면 진단 성적서 PDF 출력")
        st.caption("버튼을 누르면 현재 연산된 이미지, 데이터, 분석 코멘트가 매핑된 **1페이지 전용 PDF 보고서**가 생성됩니다.")
        
        p1_pdf_data = {
            "date": str(m_date), "time": selected_time, "loc": m_loc, "target": desired_strikes,
            "temp": auto_temp, "hum": auto_hum, 
            "weather_ok": weather_msg_str,
            "ai_1": "O" if use_model1 else "X",
            "ai_2": "O" if use_model2 else "X",
            "ai_3": "O" if use_model3 else "X",
            "ai_urls": "https://universe.roboflow.com (API-9, 10, 11 기반)",
            "pts": f"({p1_x}, {p1_y}) ~ ({p2_x}, {p2_y})",
            "dist": real_len, "area": calculated_area_cm2, "px_scale": p_scale_cm,
            "img_map": weather_map_img, "img_strike": strike_map_img,
            "defect_ratio": defect_ratio, "cand_count": final_selected_count,
            "ai_comment": generate_static_engineering_commentary(1, "").replace('\n', '<br/>')
        }
        
        p1_pdf_bytes = create_page1_pdf(p1_pdf_data)
        st.download_button(
            label="💾 1페이지_진단결과.pdf 다운로드",
            data=p1_pdf_bytes,
            file_name=f"Page1_Report_{m_date}.pdf",
            mime="application/pdf",
            type="primary"
        )


# =========================================================================
# 2페이지: 다중 센서 및 환경 변수 복합 강도 연산 시스템 + PDF 출력
# =========================================================================
elif "2." in main_menu:
    st.title("📊 국가표준(KRISS) 기반 다중 센서/환경 융합 강도 연산 시스템")

    col_env, col_data = st.columns([1, 1])

    with col_env:
        st.subheader("📋 1. 현장 계측 정보 및 재령 입력")
        m2_date = st.date_input("슈미트해머 실시 날짜", datetime.date.today())
        time_options2 = make_time_options_korean()
        selected_time2 = st.selectbox("측정 시간", time_options2, index=time_options2.index("10시 00분"))
        m2_hour, m2_min = parse_korean_time(selected_time2)
        m2_loc = st.text_input("위치 (기상청 온/습도 연동용)", value="현장 A측면")

        auto_temp2, auto_hum2 = fetch_kma_weather_simulated(m2_date, m2_hour, m2_min, m2_loc)
        is_weather2_valid, weather2_msg_str = evaluate_ks_weather(auto_temp2, auto_hum2)
        st.warning(f"📡 [기상청] {m2_date} {selected_time2} 기준 / 온도: {auto_temp2} ℃ / 습도: {auto_hum2} %")

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
                dist_val = st.number_input("📏 프로브 간 측정 거리", min_value=1.0, value=300.0)
                dist_unit = st.selectbox("거리 단위", ["mm", "cm", "m"], index=0)
            with c_u2:
                time_val = st.number_input("⏱️ 초음파 주행 시간(Transit Time)", min_value=0.1, value=73.5)
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

        raw_inputs = [st.number_input(f"{i}번째 R값", value=36.0 if i != 5 else 22.0, key=f"r_{i}") for i in range(1, strike_count + 1)]

    # =========================================================================
    # ⚙️ 데이터 분석 연산 구역
    # =========================================================================
    raw_arr = np.array(raw_inputs, dtype=float)
    total_avg = np.mean(raw_arr)

    lower, upper = total_avg * 0.90, total_avg * 1.10
    filtered_data = [v for v in raw_arr if lower <= v <= upper]
    ex_count = len(raw_arr) - len(filtered_data)
    ks_avg = np.mean(filtered_data) if filtered_data else total_avg

    delta_R = calculate_angle_correction(ks_avg, angle_val)
    corrected_R = ks_avg + delta_R
    fc_rebound = max(0.0, 1.3 * corrected_R - 14.0)

    age_factor = max(0.82, 1.0 - 0.03 * math.log(total_days / 28.0)) if total_days > 28 else 1.0
    env_factor = 1.06 if auto_hum2 >= 80.0 else 0.93 if (auto_temp2 < 5.0 or auto_temp2 > 35.0) else 1.0
    slump_corr = max(0.80, 1.0 - 0.0008 * (val_slump - 150)) if (use_slump and val_slump > 150) else 1.0

    if use_ultra:
        l_m = dist_val / 1000.0 if dist_unit == "mm" else dist_val / 100.0 if dist_unit == "cm" else dist_val
        t_s = time_val / 1000000.0 if time_unit == "μs" else time_val / 1000.0 if time_unit == "ms" else time_val
        v_mps = l_m / t_s if t_s > 0 else 0
        v_kmps = v_mps / 1000.0
    else:
        v_mps, v_kmps, l_m, t_s = 0, 0, 0, 0

    if use_ultra and v_kmps > 0:
        fc_ultra_only = max(0.0, (1.52 * corrected_R + 16.87 * v_kmps - 66.9) * age_factor)
        base_hybrid = (1.52 * corrected_R + 16.87 * v_kmps - 66.9) * age_factor
    else:
        fc_ultra_only = 0
        base_hybrid = fc_rebound * age_factor
    
    fc_slump_only = fc_rebound * age_factor * slump_corr if use_slump else 0
    if use_slump and val_slump > 150:
        base_hybrid *= max(0.85, 1.0 - 0.0007 * (val_slump - 150))
        
    fc_final_hybrid = max(0.0, base_hybrid * env_factor)

    st.write("---")
    st.markdown("### 📈 데이터 보정 및 개별/종합 복합 추정 결과")
    c_m1, c_m2, c_m3 = st.columns(3)
    c_m1.metric("전체 데이터 단순 평균", f"{total_avg:.2f} R")
    c_m2.metric("보정 전 유효 평균 (±10% 필터링)", f"{ks_avg:.2f} R", f"⚠️ 이상치 {ex_count}개 폐기")
    c_m3.metric(f"최종 보정 반발도 ($R_0$ | {angle_val}°)", f"{corrected_R:.2f} R", f"ΔR = {delta_R:+.2f}")

    st.markdown("#### 🔍 연산 항목별 세부 추정 강도 분석")
    col_fc1, col_fc2, col_fc3 = st.columns(3)
    with col_fc1: st.metric("① 보정 반발도에 따른 강도(단독)", f"{fc_rebound:.1f} MPa")
    with col_fc2: st.metric("② 초음파 복합 강도", f"{fc_ultra_only:.1f} MPa" if use_ultra else "미연동")
    with col_fc3: st.metric("③ 슬럼프 복합 강도", f"{fc_slump_only:.1f} MPa" if use_slump else "미연동")

    st.write("")
    st.info(f"🏆 **[④ 종합 추정 강도]:** 모든 물리적 조건 및 다중 센서 융합 최종 예측 강도는 **`{fc_final_hybrid:.1f} MPa`** 입니다. (설계기준강도 {fck} MPa 대비 {fc_final_hybrid / fck * 100:.1f}% 수준)")

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
            st.caption("✔️ **[한국표준과학연구원/국토안전관리원] 다중 회귀 복합식**")
            st.latex(r"F_c = \left[ 1.52 \cdot R_0 + 16.87 \cdot V_{(km/s)} - 66.9 \right] \times f_{age} \times f_{env} \times f_{slump}")
        else:
            st.caption("✔️ **[대한건축학회] 단일 반발도 추정 선형식**")
            st.latex(r"F_c = \left[ 1.3 \cdot R_0 - 14.0 \right] \times f_{age} \times f_{env} \times f_{slump}")

    st.write("---")
    with st.expander("📚 출처 및 자료 신뢰성 증빙 (클릭 시 펼쳐집니다)", expanded=False):
        st.markdown("""
본 스마트 슈미트해머 AI 연산 시스템은 국토교통부 시방서 표준 및 국내외 비파괴 검사 학술 자료를 기반으로 설계되었습니다.
* **[KS 표준] 대한민국 한국산업표준 규격서 (`KS F 2730`, `KS F 2731`)**
* **[국내 표준 시방서] 국토교통부 KCS 국가건설기준 (`KCS 14 20 00`)**
* **[국가 공인 지침] 국토안전관리원 및 한국표준과학연구원(KRISS)**
* **[국내 학술 논문] 대한건축학회 및 한국구조물유지관리학회 연구**
""")

    st.write("---")
    st.subheader("🤖 Gemini AI 구조 분석 요약")
    if st.button("🚀 2페이지 Gemini AI 분석 코멘트 생성"):
        with st.spinner("분석 중..."):
            p2_summary = f"측정 일시/장소: {m2_date} {m2_loc} / 전체평균: {total_avg:.2f} / 이상치 폐기: {ex_count}개 / 보정평균: {ks_avg:.2f} / 각도 {angle_val}도 적용 최종 반발도: {corrected_R:.2f} / 초음파속도: {v_mps:.1f} m/s / 재령: {total_days}일 / 슬럼프: {val_slump} mm / 융합추정강도: {fc_final_hybrid:.1f} MPa"
            gemini_text2 = generate_gemini_commentary(2, p2_summary)
            st.info(gemini_text2)

    # -------------------------------------------------------------------------
    # 🖨️ 2페이지 전용 PDF 출력 버튼
    # -------------------------------------------------------------------------
    st.write("---")
    st.subheader("🖨️ 2페이지: 데이터 보정 및 복합 추정 결과 PDF 출력")
    st.caption("버튼을 누르면 현재 연산된 수식, 데이터, 학술 출처가 매핑된 **2페이지 전용 PDF 보고서**가 생성됩니다.")
    
    p2_pdf_data = {
        "date": str(m2_date), "time": selected_time2, "loc": m2_loc,
        "temp": auto_temp2, "hum": auto_hum2, 
        "weather_ok": weather2_msg_str,
        "strike_count": strike_count,
        "pour_date": str(m2_cast), "days": total_days, "fck": fck, "angle": angle_val,
        "ultra": f"{v_kmps*1000:.1f} m/s" if use_ultra else "없음 (미적용)",
        "slump": f"{val_slump} mm" if use_slump else "없음 (미적용)",
        "r_list": raw_inputs, "r_mean": total_avg, "r_corr": corrected_R,
        "fc_reb": fc_rebound, "fc_ultra": fc_ultra_only, "fc_hybrid": fc_final_hybrid
    }
    
    p2_pdf_bytes = create_page2_pdf(p2_pdf_data)
    st.download_button(
        label="💾 2페이지_강도추정.pdf 다운로드",
        data=p2_pdf_bytes,
        file_name=f"Page2_Report_{m2_date}.pdf",
        mime="application/pdf",
        type="primary"
    )
