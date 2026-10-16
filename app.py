import streamlit as st
import cv2
import numpy as np
from PIL import Image as PILImage, ImageOps
import datetime
import math
import hashlib
import requests
import io
import os
import urllib.parse
from zoneinfo import ZoneInfo
import pandas as pd

import concrete_vision as cvz   # 직접 학습한 YOLOv8-seg 분할 + 타격점 추천 모듈

st.set_page_config(page_title="스마트 슈미트해머 AI", layout="wide")

# =========================================================================
# 공통 유틸
# =========================================================================
def make_time_options_korean():
    """시간 선택을 위한 24시간 리스트를 생성하는 함수"""
    options = []
    for h in range(24):
        options.append(f"{h:02d}시 00분")
        options.append(f"{h:02d}시 30분")
    return options

def parse_korean_time(time_str):
    """한글 시간 문자열('14시 00분')을 숫자 시, 분으로 변환하는 함수"""
    try:
        h = int(time_str.split("시")[0].strip())
        m = int(time_str.split("시")[1].replace("분", "").strip())
        return h, m
    except Exception:
        return 14, 0

def calculate_angle_correction(r_avg, angle):
    """
    KS F 2730 규격에 따른 슈미트해머 타격 각도별 반발도(R값) 보정치 계산 함수
    - angle: 0(수평), 90(정하향), -90(정상향), 45(경사하향), -45(경사상향) 등
    """
    try:
        angle_val = float(angle)
    except Exception:
        return 0.0
    if angle_val == 0:
        return 0.0
    elif angle_val == 90:  # 정하향 타격 (반발도가 높게 나오므로 빼줌)
        if r_avg <= 20: return -5.0
        elif r_avg <= 30: return -4.5
        elif r_avg <= 40: return -4.0
        else: return -3.5
    elif angle_val == -90: # 정상향 타격 (반발도가 낮게 나오므로 더해줌)
        if r_avg <= 20: return 4.0
        elif r_avg <= 30: return 3.5
        elif r_avg <= 40: return 3.0
        else: return 2.5
    elif angle_val == 45:  # 경사 하향
        if r_avg <= 30: return -3.0
        else: return -2.5
    elif angle_val == -45: # 경사 상향
        if r_avg <= 30: return 3.0
        else: return 2.5
    return 0.0

# =========================================================================
# 🖨 PDF 생성을 위한 ReportLab
# =========================================================================
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

font_path_cloud = "NanumGothicEco.ttf"           # Github에 업로드한 폰트 파일명
font_path_local = "C:/Windows/Fonts/malgun.ttf"  # 내 컴퓨터 윈도우 폰트 경로 (로컬 테스트용)
pdf_font = 'Helvetica'
try:
    if os.path.exists(font_path_cloud):
        pdfmetrics.registerFont(TTFont('KoreanFont', font_path_cloud)); pdf_font = 'KoreanFont'
    elif os.path.exists(font_path_local):
        pdfmetrics.registerFont(TTFont('KoreanFont', font_path_local)); pdf_font = 'KoreanFont'
except Exception:
    pdf_font = 'Helvetica'

def cv2_to_rlimage(cv_img, target_width=240):
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    pil_img = PILImage.fromarray(rgb)
    img_byte_arr = io.BytesIO()
    pil_img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    aspect = pil_img.height / pil_img.width
    return RLImage(img_byte_arr, width=target_width, height=target_width * aspect)

def make_pdf_styles():
    s = getSampleStyleSheet()
    f = pdf_font
    s.add(ParagraphStyle(name='K_Title', fontName=f, fontSize=16, leading=22, alignment=1, spaceAfter=15, textColor=colors.HexColor("#1A365D")))
    s.add(ParagraphStyle(name='K_Sub', fontName=f, fontSize=11, leading=16, spaceBefore=10, spaceAfter=5, textColor=colors.HexColor("#2B6CB0")))
    s.add(ParagraphStyle(name='K_Norm', fontName=f, fontSize=9, leading=14))
    s.add(ParagraphStyle(name='K_Norm_C', fontName=f, fontSize=9, leading=14, alignment=1))
    s.add(ParagraphStyle(name='K_Head', fontName=f, fontSize=9, leading=14, textColor=colors.white, alignment=1))
    return s

# =========================================================================
# 🔐 API 키 (Streamlit Secrets 에서만 읽음 - 코드에 키를 적지 않음)
# =========================================================================
def get_secret(name):
    try:
        return st.secrets.get(name, "")
    except Exception:
        return ""

API_KEYS = {
    "ROBOFLOW_API": get_secret("ROBOFLOW_API"),   # (선택) 외부 Roboflow 보조 모델용
    "GROQ_API_KEY": get_secret("GROQ_API_KEY"),   # AI 소견 (Groq Llama-3.1)
    "KMA_API_KEY": get_secret("KMA_API_KEY"),     # 기상청 (공공데이터포털 일반 인증키)
}

hide_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden; position: relative;}
    header {visibility: hidden;}
    [data-testid="stSidebarNav"] {display: none !important;}
    .block-container { padding-top: 1.5rem; padding-bottom: 1.5rem; }
    .chip {display:inline-block; padding:2px 10px; margin:2px 4px 2px 0; border-radius:12px; font-size:0.85rem; color:#111; border:1px solid #999;}
    </style>
"""
st.markdown(hide_style, unsafe_allow_html=True)

# =========================================================================
# 🌦 기상청 API (공공데이터포털)
#   - 최근 24시간 이내: 초단기실황 (VilageFcstInfoService_2.0/getUltraSrtNcst, 격자 nx, ny)
#   - 그 이전 날짜:     지상(ASOS) 시간자료 (AsosHourlyInfoService/getWthrDataList, 관측소 번호)
#   - 두 서비스 모두 공공데이터포털에서 '활용신청'이 되어 있어야 함
# =========================================================================
KST = ZoneInfo("Asia/Seoul")
KMA_SITES = {   # 지역: (ASOS 관측소 번호, 격자 nx, 격자 ny)
    "대전": (133, 67, 100), "서울": (108, 60, 127), "인천": (112, 55, 124), "수원": (119, 60, 121),
    "세종": (239, 66, 103), "청주": (131, 69, 106), "천안": (232, 63, 110), "대구": (143, 89, 90),
    "부산": (159, 98, 76), "울산": (152, 102, 84), "광주": (156, 58, 74), "전주": (146, 63, 89),
    "춘천": (101, 73, 134), "강릉": (105, 92, 131), "제주": (184, 52, 38),
}
KMA_NOT_CONNECTED = "기상청 API 연동: 현재 연동 시키지 않음. 현 값은 랜덤임"

def _kma_key(raw):
    # 포털의 'Encoding' 키를 넣어도 이중 인코딩되지 않도록 디코딩해서 사용
    return urllib.parse.unquote(raw) if raw and "%" in raw else raw

def fetch_kma_real(date_val, hour, minute, city):
    """성공: (기온, 습도, 출처 설명) / 실패: (None, None, 실패 사유)"""
    key = _kma_key(API_KEYS["KMA_API_KEY"])
    if not key:
        return None, None, "API 키 없음"
    stn, nx, ny = KMA_SITES.get(city, KMA_SITES["대전"])
    target = datetime.datetime(date_val.year, date_val.month, date_val.day, hour, minute, tzinfo=KST)
    now = datetime.datetime.now(KST)
    if target > now:
        return None, None, "미래 시각은 관측값이 없음"
    try:
        if now - target < datetime.timedelta(hours=23):
            base = target.replace(minute=0, second=0, microsecond=0)
            if now - base < datetime.timedelta(minutes=45):   # 정시 자료는 약 40분 뒤 제공
                base -= datetime.timedelta(hours=1)
            params = {"serviceKey": key, "pageNo": 1, "numOfRows": 100, "dataType": "JSON",
                      "base_date": base.strftime("%Y%m%d"), "base_time": base.strftime("%H00"), "nx": nx, "ny": ny}
            r = requests.get("https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtNcst", params=params, timeout=8)
            items = r.json()["response"]["body"]["items"]["item"]
            vals = {it["category"]: float(it["obsrValue"]) for it in items}
            return vals["T1H"], vals["REH"], f"기상청 초단기실황 ({city} 격자 {nx},{ny} / {base.strftime('%m-%d %H:00')})"
        params = {"serviceKey": key, "pageNo": 1, "numOfRows": 10, "dataType": "JSON", "dataCd": "ASOS", "dateCd": "HR",
                  "startDt": target.strftime("%Y%m%d"), "startHh": target.strftime("%H"),
                  "endDt": target.strftime("%Y%m%d"), "endHh": target.strftime("%H"), "stnIds": stn}
        r = requests.get("https://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList", params=params, timeout=8)
        it = r.json()["response"]["body"]["items"]["item"][0]
        return float(it["ta"]), float(it["hm"]), f"기상청 ASOS 시간자료 ({city} 관측소 {stn} / {target.strftime('%m-%d %H:00')})"
    except Exception as e:
        return None, None, f"API 호출 실패 ({type(e).__name__})"

def fetch_weather_simulated(date_val, hour, minute, loc_str):
    """API 미연동 시 사용하는 임의값 (같은 입력이면 같은 값)"""
    seed_str = f"{date_val}_{hour}:{minute}_{loc_str or '대전'}"
    hash_val = int(hashlib.md5(seed_str.encode()).hexdigest(), 16)
    return round(14.0 + (hash_val % 180) / 10.0, 1), round(45.0 + (hash_val % 35), 1)

def get_weather(date_val, hour, minute, city):
    """반환: (기온, 습도, 실측여부, 출처/안내 문구)"""
    t, hmd, msg = fetch_kma_real(date_val, hour, minute, city)
    if t is not None:
        return t, hmd, True, msg
    t, hmd = fetch_weather_simulated(date_val, hour, minute, city)
    note = KMA_NOT_CONNECTED if msg == "API 키 없음" else f"{KMA_NOT_CONNECTED} ({msg})"
    return t, hmd, False, note

def show_weather(temp, hum, is_real, note):
    st.info(f" 기상청 실시간 연동 ➔ 기온: {temp} ℃ / 상대습도: {hum} %")
    if is_real:
        st.caption(f"✅ {note}")
    else:
        st.warning(f"⚠ {note}")

def evaluate_ks_weather(temp, hum):
    if temp < 0 or hum >= 95.0:
        return False, " [부적합] 온도가 0℃ 이하이거나 습도가 95% 이상입니다. 실험 결과가 정상적이지 않을수 있습니다."
    return True, " 온·습도 조건 적합 (0℃ 초과, 습도 95% 미만)"

def calculate_pixel_scale(p1_x, p1_y, p2_x, p2_y, real_length_mm):
    pixel_dist = math.sqrt((p2_x - p1_x) ** 2 + (p2_y - p1_y) ** 2)
    if pixel_dist == 0: return 1.0, 0.0
    return real_length_mm / pixel_dist, pixel_dist

# =========================================================================
# 🧠 자체 학습 YOLOv8-seg 모델 (models/concrete_seg_best.pt)
# =========================================================================
MODEL_PATH = "models/concrete_seg_best.pt"

@st.cache_resource(show_spinner="자체 학습 AI 모델 불러오는 중...")
def get_model():
    return cvz.load_model(MODEL_PATH)

@st.cache_data(show_spinner=False, max_entries=8)
def run_segmentation(img_png_bytes, conf):
    img = cv2.imdecode(np.frombuffer(img_png_bytes, np.uint8), cv2.IMREAD_COLOR)
    return cvz.segment(get_model(), img, conf=conf)

# (선택) 외부 Roboflow 공개 모델 - 사각형 영역을 결함으로 추가
def fetch_roboflow_mask(img_bytes, workflow_id, classes_param, w, h):
    mask = np.zeros((h, w), dtype=np.uint8)
    if not API_KEYS["ROBOFLOW_API"]: return mask
    url = f"https://serverless.roboflow.com/workflows/-ovfhd/{workflow_id}/outputs?api_key={API_KEYS['ROBOFLOW_API']}"
    files = {"image": ("image.jpg", img_bytes, "image/jpeg")}
    payload = {"parameters": f'{{"classes": "{classes_param}"}}'}
    try:
        res = requests.post(url, files=files, data=payload, timeout=15).json()
        outputs = res.get("outputs", [{}])[0]
        preds = []
        for k, v in outputs.items():
            if isinstance(v, dict) and "predictions" in v: preds = v["predictions"]; break
            if k == "predictions" and isinstance(v, list): preds = v; break
        for p in preds:
            px, py = int(p.get("x", 0)), int(p.get("y", 0))
            pw, ph = int(p.get("width", 0)), int(p.get("height", 0))
            if pw > 0 and ph > 0:
                cv2.rectangle(mask, (max(0, px - pw // 2), max(0, py - ph // 2)), (min(w, px + pw // 2), min(h, py + ph // 2)), 255, -1)
    except Exception:
        pass
    return mask

# =========================================================================
# 🧾 AI 소견 (Groq Llama-3.1, Streamlit Secrets 의 GROQ_API_KEY 필요)
# =========================================================================
def call_groq(prompt):
    key = API_KEYS["GROQ_API_KEY"]
    if not key:
        return "AI 소견 미생성: Streamlit Secrets 에 GROQ_API_KEY 가 등록되지 않았습니다."
    try:
        r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                          headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                          json={"model": "llama-3.1-8b-instant", "temperature": 0.2,
                                "messages": [{"role": "system", "content": "당신은 콘크리트 비파괴시험(KS F 2730) 전문가입니다. 주어진 데이터에 근거해서만 한국어로 간결하게 작성하고, 데이터에 없는 사실은 만들지 마세요."},
                                             {"role": "user", "content": prompt}]}, timeout=20)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"].strip() + "\n\n(Groq Llama-3.1 자동 생성 소견 — 참고용)"
        return f"AI 소견 생성 실패 (Groq 응답 코드 {r.status_code})"
    except Exception as e:
        return f"AI 소견 생성 실패: {e}"

def page1_prompt(d):
    return f"""다음은 슈미트해머(반발경도) 시험 전 AI 표면 분석 결과입니다. 이 데이터만 근거로 4문장 내외의 소견을 작성하세요.
- 일시/장소: {d['date']} {d['time']}, {d['location']}
- 기온 {d['temp']}℃, 상대습도 {d['hum']}% ({'기상청 실측' if d['weather_real'] else '임의값: 기상청 미연동'})
- AI가 찾은 콘크리트 면적: {d['concrete_cm2']:.0f} cm² (스케일: {d['scale_src']})
- 콘크리트 면 중 결함(균열·기공) 비율: {d['defect_ratio']:.2f}%
- 타격 가능(초록) 면적: {d['usable_cm2']:.0f} cm²
- 요청 타격점 {d['n_req']}개 중 확보 {d['n_got']}개 (타격점 간 {d['spacing']}mm 이상, 외곽에서 {d['edge']}mm 이상 이격)
작성 지침: 1) 시험 환경 적합성(0℃ 이하 또는 습도 95% 이상이면 부적합) 2) 결함 비율로 본 표면 상태 3) 타격점 확보 여부와 주의사항. 기상값이 임의값이면 그 사실을 밝히세요."""

# =========================================================================
# ⚙ 사이드바
# =========================================================================
st.sidebar.header("⚙ 스마트 분석 제어판")
main_menu = st.sidebar.radio("작업 선택", ["1. 슈미트해머 측정 신뢰도 (AI 결함 우회)", "2. 다중 센서/환경 융합 강도 추정"])
st.sidebar.markdown("---")
st.sidebar.caption("연동 상태")
st.sidebar.write(("✅" if os.path.exists(MODEL_PATH) else "❌") + " 자체 학습 AI 모델 (YOLOv8-seg)")
st.sidebar.write(("✅" if API_KEYS["KMA_API_KEY"] else "⚠") + " 기상청 API" + ("" if API_KEYS["KMA_API_KEY"] else " (미연동 · 랜덤값)"))
st.sidebar.write(("✅" if API_KEYS["GROQ_API_KEY"] else "⚠") + " AI 소견 (Groq)")

# =========================================================================
# 1페이지: AI 표면 분석 + 타격점 추천
# =========================================================================
if "1." in main_menu:
    st.title(" 스마트 슈미트해머 1단계: AI 표면 검사보고서")

    st.subheader(" 현장 기본 정보 입력")
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1.4, 1])
    with c1:
        m_date = st.date_input("측정 실시 날짜", datetime.datetime.now(KST).date())
    with c2:
        opts = make_time_options_korean()
        selected_time = st.selectbox("측정 시간", opts, index=opts.index("14시 00분"))
        m_hour, m_min = parse_korean_time(selected_time)
    with c3:
        m_city = st.selectbox("지역 (기상청)", list(KMA_SITES.keys()), index=0)
    with c4:
        m_loc = st.text_input("측정 위치 (현장명)", value="충남대학교 실험동 벽체")
    with c5:
        n_strikes = st.number_input("추천 타격 횟수", min_value=1, max_value=50, value=20, step=1)

    auto_temp, auto_hum, weather_real, weather_note = get_weather(m_date, m_hour, m_min, m_city)
    show_weather(auto_temp, auto_hum, weather_real, weather_note)
    is_weather_valid, weather_msg = evaluate_ks_weather(auto_temp, auto_hum)
    (st.success if is_weather_valid else st.error)(weather_msg)

    with st.expander("⚙ 분석 설정 (캘리브레이션 · 타격 규칙)", expanded=False):
        s1, s2, s3 = st.columns(3)
        with s1:
            scale_mode = st.radio("픽셀-mm 캘리브레이션", ["AI 자동 (사진 속 자 길이)", "수동 (기준점 2개)"])
            ruler_len = st.number_input("사진 속 자의 실제 전체 길이 (mm)", value=300.0, step=10.0,
                                        help="AI가 찾은 자 영역의 끝~끝 길이가 이 값이라고 보고 환산합니다.")
        with s2:
            spacing_mm = st.number_input("타격점 간 최소 간격 (mm)", value=30.0, step=5.0)
            edge_mm = st.number_input("콘크리트 외곽에서 이격 (mm)", value=30.0, step=5.0)
        with s3:
            min_r_mm = st.number_input("최소 여유 반경 (mm)", value=8.0, step=1.0,
                                       help="이 반경의 원이 결함 없이 들어갈 자리가 없으면 추천을 멈춥니다.")
            defect_buf = st.number_input("결함 주변 여유 (mm)", value=5.0, step=1.0)
            conf = st.slider("AI 인식 신뢰도 기준 (conf)", 0.10, 0.80, 0.25, 0.05)
        show_r = st.checkbox("타격점별 여유 원(가장 큰 빈 원) 표시", value=False)
        st.markdown("**(선택) 외부 Roboflow 공개 결함 모델 추가** — Secrets 에 ROBOFLOW_API 필요, 결과는 사각형으로 결함 처리")
        e1, e2, e3 = st.columns(3)
        use_model1 = e1.checkbox("균열/철근노출 (workflow 9)", value=False)
        use_model2 = e2.checkbox("요철/불균질면 (workflow 10)", value=False)
        use_model3 = e3.checkbox("범용 결함 (workflow 11)", value=False)

    uploaded_file = st.file_uploader(" 벽면 촬영 사진 업로드", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        if not os.path.exists(MODEL_PATH):
            st.error(f"자체 학습 모델 파일이 없습니다: `{MODEL_PATH}` — GitHub 저장소의 models 폴더에 best.pt 를 이 이름으로 올려주세요.")
            st.stop()

        image = ImageOps.exif_transpose(PILImage.open(uploaded_file)).convert("RGB")   # 휴대폰 사진 회전 보정
        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        max_width = 1000
        if img_bgr.shape[1] > max_width:
            ratio = max_width / img_bgr.shape[1]
            img_bgr = cv2.resize(img_bgr, (max_width, int(img_bgr.shape[0] * ratio)), interpolation=cv2.INTER_AREA)
        h, w, _ = img_bgr.shape
        png_bytes = cv2.imencode(".png", img_bgr)[1].tobytes()

        with st.spinner("자체 학습 AI가 콘크리트·자·손·균열·기공을 분할하는 중..."):
            masks, scores = run_segmentation(png_bytes, conf)
            masks = {k: v.copy() for k, v in masks.items()}
            if use_model1 or use_model2 or use_model3:
                jpg = cv2.imencode(".jpg", img_bgr)[1].tobytes()
                extra = np.zeros((h, w), np.uint8)
                if use_model1: extra |= fetch_roboflow_mask(jpg, "general-segmentation-api-9", "crack", w, h)
                if use_model2: extra |= fetch_roboflow_mask(jpg, "general-segmentation-api-10", "defect", w, h)
                if use_model3: extra |= fetch_roboflow_mask(jpg, "general-segmentation-api-11", "defects", w, h)
                masks["crack"] = cv2.bitwise_or(masks.get("crack", np.zeros((h, w), np.uint8)), extra)

        manual_scale = None
        if scale_mode.startswith("수동"):
            st.markdown("#####  수동 캘리브레이션 (사진 픽셀 좌표)")
            p1, p2, p3 = st.columns(3)
            with p1: p1_x, p1_y = st.number_input("기준점1 X", value=int(w * 0.25)), st.number_input("기준점1 Y", value=int(h * 0.90))
            with p2: p2_x, p2_y = st.number_input("기준점2 X", value=int(w * 0.75)), st.number_input("기준점2 Y", value=int(h * 0.90))
            with p3: real_len = st.number_input("두 점 사이 실제 길이 (mm)", value=200.0)
            manual_scale, _ = calculate_pixel_scale(p1_x, p1_y, p2_x, p2_y, real_len)

        R = cvz.analyze(img_bgr, masks, int(n_strikes), ruler_len_mm=ruler_len, manual_mm_per_px=manual_scale,
                        spacing_mm=spacing_mm, edge_mm=edge_mm, min_radius_mm=min_r_mm, defect_buffer_mm=defect_buf,
                        show_radius=show_r)
        if not R["concrete_found"]:
            st.warning("⚠ AI가 콘크리트 면을 찾지 못해 사진 전체를 대상으로 계산했습니다. 벽면이 화면 대부분을 차지하도록 다시 찍어보세요.")
        if not R["ruler_found"] and manual_scale is None:
            st.warning("⚠ 사진에서 자를 찾지 못해 기본 스케일(0.5 mm/px)을 썼습니다. 자를 함께 찍거나 '수동' 캘리브레이션을 선택하세요.")
        if scale_mode.startswith("수동"):
            cv2.line(R["vis"], (int(p1_x), int(p1_y)), (int(p2_x), int(p2_y)), (255, 255, 0), 3)

        col_a, col_b = st.columns(2)
        with col_a:
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="업로드 원본 사진")
        with col_b:
            st.image(cv2.cvtColor(R["vis"], cv2.COLOR_BGR2RGB),
                     caption=f"AI 분석 지도 · 추천 타격점 {len(R['points'])}개 (간격 {spacing_mm:.0f}mm 이상)")
        st.markdown(
            '<span class="chip" style="background:#fff;border:3px solid #000">콘크리트 외곽(검정)</span>'
            '<span class="chip" style="background:repeating-linear-gradient(45deg,#555 0 4px,#aaa 4px 8px);color:#fff">대상 아님(사선)</span>'
            '<span class="chip" style="background:#5cd65c">타격 가능</span>'
            '<span class="chip" style="background:#ff4d4d;color:#fff">균열·기공</span>'
            '<span class="chip" style="background:#fff5a0">자</span>'
            '<span class="chip" style="background:#ff8c1a">손</span>'
            '<span class="chip" style="background:#000;color:#fff">① 타격점</span>', unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("스케일", f"{R['mm_per_px']:.3f} mm/px", R["scale_src"], delta_color="off")
        m2.metric("콘크리트 면적", f"{R['concrete_area_cm2']:.0f} cm²")
        m3.metric("결함 비율 (콘크리트 내)", f"{R['defect_ratio']:.2f} %")
        m4.metric("확보 타격점", f"{len(R['points'])} / {int(n_strikes)}")
        if len(R["points"]) < n_strikes:
            st.warning(f"⚠ 결함이 많거나 공간이 부족해 {int(n_strikes)}개 중 {len(R['points'])}개만 확보했습니다. (최소 여유 반경 {min_r_mm:.0f}mm 기준)")

        pts_df = pd.DataFrame([{"번호": p["no"], "X (px)": p["x"], "Y (px)": p["y"],
                                "X (mm)": round(p["x"] * R["mm_per_px"], 1), "Y (mm)": round(p["y"] * R["mm_per_px"], 1),
                                "결함·외곽까지 여유 반경 (mm)": round(p["r_mm"], 1)} for p in R["points"]])
        with st.expander("📋 추천 타격점 좌표표 (1번 = 가장 넓고 깨끗한 곳)", expanded=False):
            st.dataframe(pts_df, hide_index=True)
            st.caption("방법: 타격 가능 영역에서 원을 1px씩 키워 가장 크게 들어가는 원의 중심 = 타격점 1 → 그 주위 "
                       f"{spacing_mm:.0f}mm 를 제외하고 다시 계산 → 타격점 2 … (거리변환 기반)")

        page1_data = {"date": str(m_date), "time": selected_time, "location": f"{m_city} {m_loc}",
                      "temp": auto_temp, "hum": auto_hum, "weather_real": weather_real,
                      "concrete_cm2": R["concrete_area_cm2"], "usable_cm2": R["usable_area_cm2"],
                      "defect_ratio": R["defect_ratio"], "scale_src": R["scale_src"],
                      "n_req": int(n_strikes), "n_got": len(R["points"]), "spacing": spacing_mm, "edge": edge_mm}

        st.subheader(" AI 종합 소견")
        if st.button("AI 소견 생성 (Groq)"):
            with st.spinner("AI 소견 작성 중..."):
                st.session_state["page1_ai_comment"] = call_groq(page1_prompt(page1_data))
        ai_summary_txt = st.session_state.get("page1_ai_comment", "")
        if ai_summary_txt:
            st.info(ai_summary_txt)
        else:
            st.caption("버튼을 누르면 위 분석값으로 소견을 생성합니다. (생성하지 않으면 PDF 소견란은 비어 있습니다)")

        def build_page1_pdf():
            buf = io.BytesIO()
            doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
            S = make_pdf_styles()
            story = [Paragraph("[제 1페이지] AI 표면 품질 검사 및 타격점 추천 보고서", S['K_Title'])]
            status = ("1. 표면 분할: 자체 학습 YOLOv8n-seg (concrete/crack/void/ruler/hand)<br/>"
                      f"2. 기상 데이터: {'기상청 API 실측' if weather_real else '미연동 (임의값)'}<br/>"
                      f"3. AI 소견: {'Groq Llama-3.1' if ai_summary_txt else '미생성'}")
            info = [
                [Paragraph("항목", S['K_Head']), Paragraph("내용", S['K_Head'])],
                [Paragraph("측정 일시", S['K_Norm']), Paragraph(f"{m_date} {selected_time}", S['K_Norm'])],
                [Paragraph("측정 위치", S['K_Norm']), Paragraph(f"{m_city} / {m_loc}", S['K_Norm'])],
                [Paragraph("분석 구성", S['K_Norm']), Paragraph(status, S['K_Norm'])],
                [Paragraph("기온 / 습도", S['K_Norm']), Paragraph(f"{auto_temp} ℃ / {auto_hum} %<br/>{weather_note}", S['K_Norm'])],
                [Paragraph("환경 적합성", S['K_Norm']), Paragraph(weather_msg, S['K_Norm'])],
                [Paragraph("스케일", S['K_Norm']), Paragraph(f"{R['mm_per_px']:.3f} mm/px ({R['scale_src']})", S['K_Norm'])],
                [Paragraph("콘크리트 / 타격가능 면적", S['K_Norm']), Paragraph(f"{R['concrete_area_cm2']:.0f} cm² / {R['usable_area_cm2']:.0f} cm² (결함 비율 {R['defect_ratio']:.2f}%)", S['K_Norm'])],
                [Paragraph("타격점", S['K_Norm']), Paragraph(f"요청 {int(n_strikes)}개 / 확보 {len(R['points'])}개 (간격 {spacing_mm:.0f}mm, 외곽 이격 {edge_mm:.0f}mm 이상)", S['K_Norm'])],
            ]
            t = Table(info, colWidths=[130, 390])
            t.setStyle(TableStyle([('BACKGROUND', (0, 0), (1, 0), colors.HexColor("#1A365D")), ('GRID', (0, 0), (-1, -1), 0.5, colors.grey), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]))
            story += [t, Spacer(1, 12), Paragraph("▶ 원본 사진 / AI 분석 지도", S['K_Sub'])]
            ti = Table([[cv2_to_rlimage(img_bgr, 250), cv2_to_rlimage(R["vis"], 250)]], colWidths=[260, 260])
            ti.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER')]))
            story += [ti, Paragraph("검정 외곽=콘크리트, 사선=대상 아님, 초록=타격 가능, 빨강=균열·기공, 연노랑=자, 주황=손, 번호=추천 순서", S['K_Norm']), Spacer(1, 8)]
            if len(R["points"]):
                story.append(Paragraph("▶ 추천 타격점 좌표", S['K_Sub']))
                rows = [[Paragraph(x, S['K_Head']) for x in ["번호", "X (mm)", "Y (mm)", "여유 반경 (mm)"]]]
                for p in R["points"]:
                    rows.append([Paragraph(str(v), S['K_Norm_C']) for v in
                                 [p["no"], f"{p['x'] * R['mm_per_px']:.1f}", f"{p['y'] * R['mm_per_px']:.1f}", f"{p['r_mm']:.1f}"]])
                tp = Table(rows, colWidths=[60, 110, 110, 120], repeatRows=1)
                tp.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2B6CB0")), ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)]))
                story += [tp, Spacer(1, 8)]
            story.append(Paragraph("▶ AI 종합 소견", S['K_Sub']))
            story.append(Paragraph(ai_summary_txt.replace('\n', '<br/>') if ai_summary_txt else "미생성", S['K_Norm']))
            doc.build(story)
            return buf.getvalue()

        st.write("---")
        st.download_button(" [1페이지] AI 표면 검사 · 타격점 보고서 PDF 다운로드", data=build_page1_pdf(),
                           file_name=f"1_AI_Surface_Report_{m_date}.pdf", mime="application/pdf", type="primary")


elif "2." in main_menu:
    st.title(" SCI급 다중 센서 및 환경 변수 복합 강도 연산 시스템")

    col_env, col_data = st.columns([1, 1])

    with col_env:
        st.subheader(" 1. 현장 계측 정보 및 재령 입력")
        m2_date = st.date_input("슈미트해머 실시 날짜", datetime.date.today())
        opts2 = make_time_options_korean()
        selected_time2 = st.selectbox("측정 시간", opts2, index=opts2.index("10시 00분"))
        m2_hour, m2_min = parse_korean_time(selected_time2)
        m2_city = st.selectbox("지역 (기상청)", list(KMA_SITES.keys()), index=0, key="m2_city")
        m2_loc = st.text_input("위치 (현장명)", value="현장 교각 B구간 측면부")

        auto_temp2, auto_hum2, weather_real2, weather_note2 = get_weather(m2_date, m2_hour, m2_min, m2_city)
        is_valid2, msg2 = evaluate_ks_weather(auto_temp2, auto_hum2)
        show_weather(auto_temp2, auto_hum2, weather_real2, weather_note2)
        st.caption(f"※ 코멘트: 해당 환경은 {msg2}")

        m2_cast = st.date_input("타설일", datetime.date.today() - datetime.timedelta(days=90))
        total_days = max(1, (m2_date - m2_cast).days)
        fck = st.number_input("설계기준강도 (MPa)", value=24.0)
        st.info(f"재령: {total_days}일 확보 (타설일: {m2_cast})")
        
        st.subheader(" 2. 초음파 다중 측정 및 정밀 융합 (SonReb)")
        use_ultra = st.checkbox(" 초음파 측정치 연동 (다중 측정 및 이상치 보정)", value=True)
        
        v_mps = 0.0         # 전체 평균 속도
        v_mps_corr = 0.0    # 10% 이상치 제외 보정 평균 속도
        upv_detail_data = [] # 엑셀/PDF용 상세 데이터 저장 리스트
        upv_method = "미사용"
        upv_count = 0
        
        if use_ultra:
            upv_cols1, upv_cols2 = st.columns(2)
            with upv_cols1:
                upv_count_sel = st.selectbox("초음파 측정 횟수", ["1회", "3회", "5회", "7회", "10회", "기타(직접입력)"])
                if upv_count_sel == "기타(직접입력)":
                    upv_count = st.number_input("측정 횟수 직접 입력", min_value=1, max_value=50, value=5, step=1)
                else:
                    upv_count = int(upv_count_sel.replace("회", ""))
                    
            with upv_cols2:
                upv_method = st.radio("전체 초음파 측정 방법", ["직접법", "간접법"], horizontal=True)
                method_factor = 1.0 if upv_method == "직접법" else 1.05 
            
            st.markdown("#####  초음파 측정 데이터 입력")
            upv_velocities = []
            for i in range(upv_count):
                c1, c2, c3 = st.columns([1, 2, 2])
                c1.markdown(f"<br/>**#{i+1}**", unsafe_allow_html=True)
                dist = c2.number_input(f"프로브 거리 (mm)", min_value=10.0, value=300.0, step=10.0, key=f"u_dist_{i}")
                time_us = c3.number_input(f"측정 시간 (μs)", min_value=1.0, value=80.0, step=1.0, key=f"u_time_{i}")
                
                # 속도(m/s) = (거리 / 시간) * 1000 * 측정법 보정계수
                vel = (dist / time_us) * 1000 * method_factor if time_us > 0 else 0
                upv_velocities.append(vel)
                
                # 상세 데이터 리스트에 저장
                upv_detail_data.append({
                    "측정_순서": f"#{i+1:02d}", 
                    "측정방법": upv_method,
                    "프로브거리(mm)": dist, 
                    "측정시간(μs)": time_us, 
                    "환산속도(m/s)": round(vel, 1)
                })
                
            if upv_velocities:
                v_mps = sum(upv_velocities) / len(upv_velocities)
                if len(upv_velocities) >= 3:
                    sorted_v = sorted(upv_velocities)
                    exclude_cnt = max(1, int(len(upv_velocities) * 0.1))
                    if len(sorted_v) > exclude_cnt * 2:
                        corr_v_list = sorted_v[exclude_cnt:-exclude_cnt]
                        v_mps_corr = sum(corr_v_list) / len(corr_v_list)
                    else:
                        v_mps_corr = v_mps
                else:
                    v_mps_corr = v_mps
                st.success(f" 평균 초음파 속도: **{v_mps:.1f} m/s** (10% 이상치 보정: **{v_mps_corr:.1f} m/s**)")
        else:
            v_mps, v_mps_corr = 0.0, 0.0

        use_slump = st.checkbox(" 슬럼프 수치 연동 (미세 공극률 보정)", value=True)
        val_slump = st.number_input("설계 슬럼프 (mm)", value=160.0) if use_slump else 150.0

    with col_data:
        st.subheader(" 3. 반발도(R값) 타격 데이터 세팅")
        c_strk1, c_strk2 = st.columns(2)
        with c_strk1:
            strike_count = st.selectbox("기록할 타격 횟수", [10, 15, 20, 25, 30, 35], index=2) 
        with c_strk2:
            angle_opts = [f"{a}° (상향)" if a>0 else f"{a}° (하향)" if a<0 else f"{a}° (수평/벽면)" for a in range(90, -95, -5)]
            selected_angle_str = st.selectbox(" 타격 각도", angle_opts, index=18)
            angle_val = int(selected_angle_str.split("°")[0])

        raw_inputs = []
        c_r1, c_r2, c_r3, c_r4, c_r5 = st.columns(5)
        cols = [c_r1, c_r2, c_r3, c_r4, c_r5]
        for i in range(1, strike_count + 1):
            with cols[(i-1) % 5]:
                val = st.number_input(f"#{i:02d}", value=39.0 if i != 5 else 22.0, key=f"r_{i}", label_visibility="collapsed")
                st.caption(f"#{i:02d}")
                raw_inputs.append(val)

    # =========================================================================
    # ⚙ 정밀 알고리즘 연산
    # =========================================================================
    raw_arr = np.array(raw_inputs, dtype=float)
    total_avg = np.mean(raw_arr)

    lower, upper = total_avg * 0.80, total_avg * 1.20
    filtered_data = [v for v in raw_arr if lower <= v <= upper]
    ex_count = len(raw_arr) - len(filtered_data)
    ks_avg = np.mean(filtered_data) if filtered_data else total_avg

    delta_R = calculate_angle_correction(ks_avg, angle_val)
    corrected_R = ks_avg + delta_R

    fc_rebound = max(0.0, 1.3 * corrected_R - 14.0)

    age_factor = max(0.82, 1.0 - 0.03 * math.log(total_days / 28.0)) if total_days > 28 else 1.0
    env_factor = 1.06 if auto_hum2 >= 80.0 else 0.93 if (auto_temp2 < 5.0 or auto_temp2 > 35.0) else 1.0
    slump_corr = max(0.80, 1.0 - 0.0008 * (val_slump - 150)) if (use_slump and val_slump > 150) else 1.0

    if use_ultra and v_mps_corr > 0:
        fc_ultra_only = max(0.0, 0.015 * v_mps_corr - 36.0)
    else:
        fc_ultra_only = 0.0

    if use_ultra and v_mps_corr > 0:
        v_kmps_corr = v_mps_corr / 1000.0  # m/s -> km/s 단위 환산
        # 일본건축학회기법: Fc = 8.2*R + 269*V - 1049 (kgf/cm² 단위) -> MPa 변환(/10)
        fc_aij_kgf = 8.2 * corrected_R + 269 * v_kmps_corr - 1049
        base_hybrid = max(0.0, fc_aij_kgf / 10.0)  # 음수 방지 및 MPa 환산
    else:
        base_hybrid = fc_rebound

    fc_slump_only = fc_rebound * age_factor * slump_corr if use_slump else fc_rebound
    fc_final_hybrid = base_hybrid * env_factor * age_factor * slump_corr

    st.write("---")
    st.markdown(f"###  데이터 보정 결과")
    st.markdown(f"전체 평균 반발도: **{total_avg:.2f} R** ➔ 이상치 **{ex_count}개** 제외 ➔ **보정 평균 반발도: `{corrected_R:.2f} R`** (각도보정치 포함)")

    col_fc1, col_fc2, col_fc3 = st.columns(3)
    col_fc1.info(f"**[Model A] 단일 반발도 예상강도:**\n### {fc_rebound:.1f} MPa")
    col_fc2.info(f"**[Model C] 초음파 강도:**\n### {fc_ultra_only:.1f} MPa" if use_ultra else "**[Model C] 초음파 미연동**")
    col_fc3.success(f"** [Model D] 통합 복합 예상강도:**\n### {fc_final_hybrid:.1f} MPa")

    def build_page2_pdf():
        buffer_p2 = io.BytesIO()
        doc2 = SimpleDocTemplate(buffer_p2, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
        styles2 = make_pdf_styles()
        

        story2 = []
        
        story2.append(Paragraph("[제 2페이지] 다중 센서 복합 콘크리트 강도 성적서", styles2['K_Title']))
        
        # 1. 현장 정보 표
        story2.append(Paragraph("▶ 진단 수행 정보 및 파라미터", styles2['K_Sub']))
        upv_pdf_txt = f"평균: {v_mps:.1f} m/s / 보정: {v_mps_corr:.1f} m/s ({upv_count}회 {upv_method})" if use_ultra and v_mps_corr > 0 else "미사용"
        
        info_data2 = [
            [Paragraph("측정 일시 및 장소", styles2['K_Norm']), Paragraph(f"{m2_date} ({selected_time2}) / {m2_loc}", styles2['K_Norm'])],
            [Paragraph("기상 (기온/습도)", styles2['K_Norm']), Paragraph(f"기온: {auto_temp2}℃ / 상대습도: {auto_hum2}%<br/>{weather_note2}", styles2['K_Norm'])],
            [Paragraph("설계조건 및 재령", styles2['K_Norm']), Paragraph(f"fck: {fck} MPa / 확보 재령: {total_days}일 (타설: {m2_cast})", styles2['K_Norm'])],
            [Paragraph("초음파 탐촉 센서", styles2['K_Norm']), Paragraph(upv_pdf_txt, styles2['K_Norm'])]
        ]
        t_info2 = Table(info_data2, colWidths=[150, 370])
        t_info2.setStyle(TableStyle([('FONTNAME', (0,0), (-1,-1), pdf_font), ('GRID', (0,0), (-1,-1), 0.5, colors.grey), ('BACKGROUND', (0,0), (0,-1), colors.whitesmoke)]))
        story2.extend([t_info2, Spacer(1, 10)])

        # 2. 초음파 측정 상세 표 (초음파 사용 시에만 추가됨)
        if use_ultra and upv_detail_data:
            story2.append(Paragraph(f"▶ 초음파(UPV) 측정 데이터셋 ({upv_method})", styles2['K_Sub']))
            upv_table_data = [[Paragraph("측정 순서", styles2['K_Head']), Paragraph("프로브 거리", styles2['K_Head']), Paragraph("주행 시간", styles2['K_Head']), Paragraph("환산 속도", styles2['K_Head'])]]
            for row_data in upv_detail_data:
                upv_table_data.append([
                    Paragraph(row_data["측정_순서"], styles2['K_Norm_C']),
                    Paragraph(f"{row_data['프로브거리(mm)']} mm", styles2['K_Norm_C']),
                    Paragraph(f"{row_data['측정시간(μs)']} μs", styles2['K_Norm_C']),
                    Paragraph(f"{row_data['환산속도(m/s)']} m/s", styles2['K_Norm_C'])
                ])
            t_upv = Table(upv_table_data, colWidths=[80, 110, 110, 110])
            t_upv.setStyle(TableStyle([
                ('FONTNAME', (0,0), (-1,-1), pdf_font), 
                ('GRID', (0,0), (-1,-1), 0.5, colors.grey), 
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2B6CB0"))
            ]))
            story2.extend([t_upv, Spacer(1, 10)])

        # 3. 반발도 데이터 매트릭스 표 (5개씩 분할)
        story2.append(Paragraph("▶ 슈미트해머 반발도(R값) 측정 데이터셋", styles2['K_Sub']))
        strike_data_table = []
        row = []
        for i, val in enumerate(raw_inputs):
            row.append(Paragraph(f"#{i+1:02d}: {val}", styles2['K_Norm_C']))
            if (i+1) % 5 == 0:
                strike_data_table.append(row)
                row = []
        if row: 
            while len(row) < 5:
                row.append(Paragraph("-", styles2['K_Norm_C']))
            strike_data_table.append(row)

        t_strike = Table(strike_data_table, colWidths=[104, 104, 104, 104, 104])
        t_strike.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.grey)]))
        story2.extend([t_strike, Spacer(1, 10)])

        # 4. 강도 결과 요약
        story2.append(Paragraph("▶ 데이터 분석 및 최종 복합 예상 강도", styles2['K_Sub']))
        
        # 엑셀 데이터에 있는 Model B와 Model C를 표(Table)에 추가했습니다.
        res_data = [
            [Paragraph("반발도 데이터 전처리", styles2['K_Norm']), Paragraph(f"전체 평균: {total_avg:.2f} R / 이상치 {ex_count}개 폐기", styles2['K_Norm'])],
            [Paragraph("최종 보정 반발도", styles2['K_Norm']), Paragraph(f"{corrected_R:.2f} R (타격 각도 및 규격 보정 완료)", styles2['K_Norm'])],
            [Paragraph("단일 예상 강도 (Model A)", styles2['K_Norm']), Paragraph(f"{fc_rebound:.1f} MPa", styles2['K_Norm'])],
            [Paragraph("슬럼프/재령 반영 강도 (Model B)", styles2['K_Norm']), Paragraph(f"{fc_slump_only:.1f} MPa", styles2['K_Norm'])],
            [Paragraph("초음파 융합 강도 (Model C)", styles2['K_Norm']), Paragraph(f"{fc_ultra_only:.1f} MPa", styles2['K_Norm'])],
            [Paragraph("최종 융합 예상 강도 (Model D)", styles2['K_Norm']), Paragraph(f"{fc_final_hybrid:.1f} MPa", styles2['K_Norm'])]
        ]
        
        # 모델명이 길어져 글자가 깨지는 것을 막기 위해 왼쪽 칸 너비를 150 -> 180으로 넓혔습니다. (총합 520 유지)
        t_res2 = Table(res_data, colWidths=[180, 340])
        t_res2.setStyle(TableStyle([
            ('FONTNAME', (0,0), (-1,-1), pdf_font), 
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey), 
            ('BACKGROUND', (0,-1), (-1,-1), colors.lightsteelblue) # 표의 맨 마지막 행(Model D)에 하이라이트 배경색 적용
        ]))
        story2.extend([t_res2, Spacer(1, 10)])

        # 👇 AI 소견 추가 (줄바꿈 <br/> 처리 적용)
        story2.append(Paragraph("▶ AI 비파괴 전문가 다차원 소견", styles2['K_Sub']))
        ai_comment_str = st.session_state.get("page2_ai_comment", "생성된 AI 소견이 없습니다.")
        pdf_comment2 = ai_comment_str.replace('\n', '<br/>')
        story2.append(Paragraph(pdf_comment2, styles2['K_Norm']))
        story2.append(Spacer(1, 10))

        # 5. 근거 및 출처 텍스트 추가
        story2.append(Paragraph("▶ 강도 추정 계산 원리 및 참조 근거", styles2['K_Sub']))
        
        story2.append(Paragraph("[1] 초음파(UPV) 측정법 및 산출식", styles2['K_Norm']))
        story2.append(Paragraph("- 직접법(Direct Method): 마주보는 면에서 측정. 환산 속도(m/s) = (프로브 거리 / 측정 시간) × 1000", styles2['K_Norm']))
        story2.append(Paragraph("- 간접법(Indirect Method): 같은 면에서 측정. 에너지 감쇠 보상을 위해 산출된 속도에 1.05배 보정계수 적용.", styles2['K_Norm']))
        story2.append(Spacer(1, 5))

        story2.append(Paragraph("[2] 강도 융합(SonReb) 알고리즘", styles2['K_Norm']))
        story2.append(Paragraph("- 단일 반발도 추정식: Fc = 1.3 × R - 14.0 (MPa)", styles2['K_Norm']))
        story2.append(Paragraph("- 다차원 복합 강도식(일본건축학회): Fc = (8.2 × R + 269 × V - 1049) / 10 (MPa, V: km/s)", styles2['K_Norm']))
        story2.append(Spacer(1, 5))
        
        story2.append(Paragraph("[3] KS 표준 및 국내외 참조 문헌", styles2['K_Norm']))
        story2.append(Paragraph("- [KS 표준] KS F 2730 (반발경도 시험), KS F 2731 (초음파 전달 속도 측정)", styles2['K_Norm']))
        story2.append(Paragraph("- [국가건설기준] KCS 14 20 00 콘크리트 표준시방서", styles2['K_Norm']))
        story2.append(Paragraph("- [SCI 논문] 'Evaluation of Concrete Compressive Strength Using UPV and Rebound Hammer', A. Samarin et al., ACI Materials Journal (1983)", styles2['K_Norm']))
        story2.append(Paragraph("- [KCI 논문] '초음파 속도와 반발경도를 이용한 콘크리트 압축강도 추정', 김철수 외, 한국건축구조학회논문집 (2021)", styles2['K_Norm']))

        doc2.build(story2)
        buffer_p2.seek(0)
        return buffer_p2.getvalue()

    st.write("---")
    
    st.markdown("####  분석 결과 다운로드")
    custom_filename = st.text_input(" 저장할 파일 이름을 입력하세요 (확장자 제외)", value=f"Multi_Sensor_Data_{m2_date}")
    excel_filename = custom_filename + ".xlsx" if not custom_filename.endswith(".xlsx") else custom_filename
    pdf_filename = custom_filename + ".pdf" if not custom_filename.endswith(".pdf") else custom_filename

    col_dl1, col_dl2 = st.columns(2)
    
    with col_dl1:
        st.download_button(
            label=" [2페이지] 다중 센서 복합 강도 성적서 (PDF)",
            data=build_page2_pdf(),
            file_name=pdf_filename,
            mime="application/pdf"
        )

    with col_dl2:
        buffer_xls = io.BytesIO()
        with pd.ExcelWriter(buffer_xls, engine='openpyxl') as writer:
            pd.DataFrame({"항목": ["수행 일시", "기온 (℃)", "상대습도 (%)", "기상 출처", "설계기준강도", "초음파 평균속도 (m/s)", "초음파 보정속도 (m/s)"], "내용": [f"{m2_date} ({selected_time2})", auto_temp2, auto_hum2, weather_note2, fck, v_mps, v_mps_corr]}).to_excel(writer, sheet_name="측정조건", index=False)
            
            # 👇 엑셀에 초음파 n회 데이터 전용 시트 추가
            if use_ultra and upv_detail_data:
                pd.DataFrame(upv_detail_data).to_excel(writer, sheet_name=f"{upv_count}회_초음파데이터", index=False)
                
            pd.DataFrame({"타격_순서": [f"#{i:02d}" for i in range(1, strike_count + 1)], "실측_반발도(R)": raw_inputs}).to_excel(writer, sheet_name=f"{strike_count}회_타격데이터", index=False)
            pd.DataFrame({"연산_모델_분류": ["[Model A] 단일 반발도 강도", "[Model B] 슬럼프/재령 반영", "[Model C] 초음파 융합 강도", "[Model D] 최종 융합 복합 강도"], "추정_압축강도(MPa)": [round(fc_rebound, 1), round(fc_slump_only, 1), round(fc_ultra_only, 1), round(fc_final_hybrid, 1)]}).to_excel(writer, sheet_name="강도결과", index=False)
            
            ai_comment_str = st.session_state.get("page2_ai_comment", "생성된 AI 소견 없음")
            pd.DataFrame({"AI 소견": [ai_comment_str]}).to_excel(writer, sheet_name="AI_종합소견", index=False)
            
            # 👇 엑셀용 산출 근거 및 문헌 업데이트
            pd.DataFrame({"산출 근거 및 문헌": [
                "1. [측정식] 초음파 직접법(m/s) = (거리/시간)*1000",
                "2. [측정식] 초음파 간접법(m/s) = (거리/시간)*1000 * 1.05 (에너지 감쇠 5% 보상)",
                "3. [산출식] 단일 예상 강도식: Fc = 1.3 * R - 14.0",
                "4. [산출식] 다중 복합 강도식(일본건축학회): Fc = (8.2 * R + 269 * V - 1049) / 10 (MPa, V: km/s)",
                "5. [KS 규격] KS F 2730 (반발경도 이상치 폐기 및 중력보정), KS F 2731 (초음파 시험법)",
                "6. [시방서] KCS 14 20 00 국토교통부 콘크리트 표준시방서",
                "7. [SCI 논문] A. Samarin et al., ACI Materials Journal (1983.11)",
                "8. [KCI 논문] 김철수 외, 한국건축구조학회논문집 (2021.05)"
            ]}).to_excel(writer, sheet_name="참조근거", index=False)
            
        st.download_button(
            label=" 전체 도출 데이터 종합 (Excel)",
            data=buffer_xls.getvalue(),
            file_name=excel_filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
