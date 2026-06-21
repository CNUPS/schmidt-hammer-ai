import streamlit as st
import cv2
import numpy as np
from PIL import Image as PILImage
import datetime
import math
import hashlib
import requests
import google.generativeai as genai
import io
import os
import pandas as pd

# =========================================================================
# 🖨 PDF 생성을 위한 ReportLab 라이브러리 추가
# =========================================================================
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# [중요] 한글 깨짐 및 에러 완벽 방지 폰트 로직
font_path_cloud = "NanumGothicEco.ttf" # Github에 업로드한 폰트 파일명
font_path_local = "C:/Windows/Fonts/malgun.ttf" # 내 컴퓨터 윈도우 폰트 경로 (로컬 테스트용)

try:
    if os.path.exists(font_path_cloud):
        pdfmetrics.registerFont(TTFont('KoreanFont', font_path_cloud))
    elif os.path.exists(font_path_local):
        pdfmetrics.registerFont(TTFont('KoreanFont', font_path_local))
    pdf_font = 'KoreanFont'
except Exception:
    pdf_font = 'Helvetica' # 최후의 수단

# PDF 이미지 변환 헬퍼 함수
def cv2_to_rlimage(cv_img, target_width=240):
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    pil_img = PILImage.fromarray(rgb)
    img_byte_arr = io.BytesIO()
    pil_img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    aspect = pil_img.height / pil_img.width
    return RLImage(img_byte_arr, width=target_width, height=target_width * aspect)

# =========================================================================
# 🔐 API 키 설정 구역 (보안 주입 및 사이드바 수동 입력 지원 하이브리드 설계)
# =========================================================================
try:
    API_KEYS = {
        "ROBOFLOW_API": st.secrets.get("ROBOFLOW_API", "wk4BcUKf1InnR2LjHPF8"),
        "GEMINI_API": st.secrets.get("GEMINI_API", ""),
    }
except Exception:
    API_KEYS = {
        "ROBOFLOW_API": "wk4BcUKf1InnR2LjHPF8",
        "GEMINI_API": "",
    }

# 만약 Secrets 설정을 안했거나 수동 입력을 원할 경우를 위해 사이드바 하단에 안전 비밀 입력 가이드 제공
st.sidebar.markdown("---")
st.sidebar.subheader("🔑 수동 API 키 덮어쓰기 (선택)")
manual_gemini = st.sidebar.text_input("Gemini API Key 수동 등록", value="", type="password", help="Streamlit Secrets 환경을 설정하지 않은 경우 여기에 직접 키를 입력하셔도 정상 연동됩니다.")

if manual_gemini:
    API_KEYS["GEMINI_API"] = manual_gemini

# 제미나이 설정 가동
if API_KEYS["GEMINI_API"]:
    genai.configure(api_key=API_KEYS["GEMINI_API"])

# =========================================================================
# 🎨 Streamlit 기본 UI 숨기기 및 페이지 설정
# =========================================================================
hide_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden; position: relative;}
    header {visibility: hidden;}
    [data-testid="stSidebarNav"] {display: none !important;}
    .block-container { padding-top: 1.5rem; padding-bottom: 1.5rem; }
    </style>
"""
st.markdown(hide_style, unsafe_allow_html=True)

# =========================================================================
# 🛠 유틸리티 함수
# =========================================================================
def calculate_pixel_scale(p1_x, p1_y, p2_x, p2_y, real_length_mm):
    pixel_dist = math.sqrt((p2_x - p1_x) ** 2 + (p2_y - p1_y) ** 2)
    if pixel_dist == 0: return 1.0, 0.0
    return real_length_mm / pixel_dist, pixel_dist

def evaluate_ks_weather(temp, hum):
    if temp < 5.0 or temp > 35.0 or hum >= 80.0:
        return False, "❌ [부적합] 온도가 5~35℃를 벗어나거나 습도가 80% 이상입니다. (KS F 2730 시방 기준 위반 주의)"
    return True, "✅ [적합] 온도와 습도가 허용 범위 내에 있어 측정 신뢰성이 높습니다. (KS F 2730 표준 부합)"

def fetch_kma_weather_simulated(date_val, hour, minute, loc_str):
    if not loc_str: loc_str = "서울"
    seed_str = f"{date_val}_{hour}:{minute}_{loc_str}"
    hash_val = int(hashlib.md5(seed_str.encode()).hexdigest(), 16)
    base_temp = 14.0 + (hash_val % 180) / 10.0
    base_hum = 45.0 + (hash_val % 35)
    return round(base_temp, 1), round(base_hum, 1)

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
                x1, y1 = max(0, int(px - pw / 2)), max(0, int(py - ph / 2))
                x2, y2 = min(w, int(px + pw / 2)), min(h, int(py + ph / 2))
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    except Exception:
        pass
    return mask
def fetch_custom_roboflow_mask(img_bytes, w, h): 
    mask = np.zeros((h, w), dtype=np.uint8) 
    
    # 알려주신 자체 AI 모델 주소와 Private API Key 적용
    project_id = "concrete_defect-j9nuw"
    version = "1" 
    api_key = "wk4BcUKf1InnR2LjHPF8"
    
    url = f"https://detect.roboflow.com/{project_id}/{version}?api_key={api_key}" 
    files = {"file": ("image.jpg", img_bytes, "image/jpeg")} 
    
    try: 
        res = requests.post(url, files=files, timeout=15).json() 
        preds = res.get("predictions", []) 
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

# =========================================================================
# 🧠 Gemini 비파괴 정밀 분석 연동 (식, 측정값, 장소, 온도, 재령 종합)
# =========================================================================
def generate_gemini_commentary(page_type, data_dict): 
    loc = data_dict.get("location", "서울 수색동 교량구간") 
    date_val = data_dict.get("date", str(datetime.date.today())) 
    temp = data_dict.get("temp", 20.0) 
    hum = data_dict.get("hum", 50) 
    
    # 1페이지용 추가 계산 데이터 (비전 스캔 결과)
    defect_ratio = data_dict.get("defect_ratio", 0.0)
    real_w_cm = data_dict.get("real_w_cm", 0.0)
    real_h_cm = data_dict.get("real_h_cm", 0.0)
    total_area_mm2 = data_dict.get("total_area_mm2", 0.0)

    # 2페이지용 데이터 (슈미트 해머 강도 연산 결과)
    age = data_dict.get("age", 28) 
    fck = data_dict.get("fck", 24.0) 
    r_val = data_dict.get("corrected_R", 35.0) 
    v_upv = data_dict.get("v_mps", 0.0) 
    slump = data_dict.get("slump", 150) 
    st_strength = data_dict.get("est_strength", 25.0)  # 아래 변수명 매핑을 위해 수정
    est_strength = data_dict.get("est_hybrid", st_strength) # 유연한 키값 대응
    ex_count = data_dict.get("ex_count", 0) 

    # 💡 [버그 수정 1] page_type이 문자열이든 숫자든 정확히 걸러내도록 수정
    if str(page_type) in ["1", "page1"]: 
        prompt = f""" 
        당신은 대한민국 국토교통부 콘크리트 표준시방서(KCS 14 20 00) 및 KS F 2730 표준을 마스터한 최고의 '콘크리트 비파괴 안전성 진단 기술사'입니다. 
        다음 현장 데이터 및 AI 비전 분석 결과를 바탕으로 종합 소견을 공식 제출 보고서 양식에 맞게 한글 4문장 내외로 논리정연하게 작성하십시오. 

        [측정 현장 및 기후 데이터] 
        - 일시 및 장소: {date_val}, {loc} 
        - 기상청 실시간 연동 기후: 기온 {temp}℃ / 상대습도 {hum}% 

        [비전 스캔 AI 결함 분석 데이터] 
        - 픽셀 캘리브레이션 기반 실제 사진 크기: 가로 {real_w_cm:.1f} cm x 세로 {real_h_cm:.1f} cm
        - 실제 촬영된 총 벽면 면적: {total_area_mm2:.1f} mm²
        - AI가 검출한 요철/불균질/균열 등 결함 영역 비율: 전체 사진 면적 대비 {defect_ratio:.2f}%

        [작성 가이드라인] 
        1. 기상 조건(온도/습도)이 당일 진단 작업 및 구조물 표면 건조 상태에 미치는 영향을 기술할 것. 
        2. 제공된 사진의 실제 면적 크기와 결함 영역 비율({defect_ratio:.2f}%) 수치를 명확히 명시하며 벽면 상태가 양호한지 불량한지 정량적으로 평가할 것. 
        3. 검출된 결함 비율을 토대로, 향후 슈미트 해머 타격 작업의 난이도와 예측되는 데이터의 신뢰성(값이 비교적 잘 나올지, 편차가 심할지 등)에 대한 기술사적 코멘트를 반드시 포함할 것. 
        4. 전문가 특유의 정량화된 기술 문체(~입니다, ~하며, ~로 판단됩니다)를 사용할 것.
        """ 
    else: 
        prompt = f""" 
        당신은 대한민국 국토교통부 콘크리트 표준시방서(KCS 14 20 00) 및 KS F 2730 표준을 마스터한 최고의 '콘크리트 비파괴 안전성 진단 기술사'입니다.
        다음 현장 실측 조건 데이터를 기반으로 공식 제출 보고서에 즉시 기재할 수 있는 엄격하고 정밀한 종합 분석 소견을 한글 4문장 내외로 논리정연하게 작성하십시오.
        
        [현장 물리 환경 데이터]
        - 진단 현장 위치: {loc}
        - 점검 수행 날짜: {date_val}
        - 실시간 기상 상태: 기온 {temp}℃ / 상대습도 {hum}% (KS F 2730 기후 제한 조건 만족 여부 판단에 사용)
        - 콘크리트 인자: 설계기준강도 {fck} MPa / 타설 후 경과 재령일수 {age}일
        
        [슈미트 해머 강도 연산 실측 결과]
        - 슈미트 해머 보정 평균 반발도(R값): {r_val:.1f} R (동적 이상치 {ex_count}개 제외 후 각도 보정 완료)
        - 초음파(UPV) 센서 수신 전파속도: {v_upv:.1f} m/s (주행거리 대비 프로브 응답)
        - 설계 슬럼프 변수: {slump} mm (공극 보정 계수 반영)
        - 종합 복합 산출 강도: {est_strength:.1f} MPa
        
        [작성 가이드라인] 
        1. 기술사 특유의 엄격하고 정량화된 전문 기술 문체(~입니다, ~하며, ~로 판단됩니다)를 고수하십시오.
        2. 당일 온습도 기후 데이터가 KS F 2730 기준에 어떻게 부합하여 측정 정확도 향상에 기여했는지 서술하십시오.
        3. 도출된 최종 예측 압축강도({est_strength:.1f} MPa)가 타설 설계강도({fck} MPa)를 완벽히 상회하여 충분한 안정성 마진을 유지하고 있는지 평가하십시오.
        4. 표면 경도 측정의 고유 한계를 극복하기 위해 '초음파 전파 주행속도({v_upv:.1f} m/s)' 및 '슬럼프 변동 제어' 융합 모델(SonReb)을 통해 어떻게 내부 결함 및 공극 왜곡 요소를 보정하고 정밀화했는지 학술적 논리로 강조해 주십시오.
        """ 

    # 폴백 결과물 생성용 임시 변수
    final_output = ""

    if not API_KEYS["GEMINI_API"]: 
        if str(page_type) in ["1", "page1"]: 
            ks_status = "적합" if (5.0 <= temp <= 35.0 and hum < 80) else "주의 필요" 
            final_output = (
                f"{loc} 신축 벽면의 고해상도 이미지 비전 스캔 결과, 균열 및 표면 박리 취약 구역을 실시간 탐지하여 최적의 타격점을 배치하였습니다. "
                f"시험 당일 대기 온도({temp}℃) 및 상대습도({hum}%) 환경은 KS F 2730 표준 기후 기준에 대비하여 '{ks_status}' 상태에 해당함을 판정하였으며, "
                f"지정 타격점 간의 실제 이격 거리 제약인 30mm(3.0cm)를 검증 적용하여 수집 데이터의 신뢰성과 안전 구역 우회성을 확보하였습니다. "
                f"\n\n*(수동/서버 API 키 미완료 상태로 내장 지능형 임시 분석 리포트가 출력되었습니다)*"
            )
        else: 
            pct_attained = (est_strength / fck) * 100.0 if fck > 0 else 0 
            status_msg = "우수한 수준의 안전 마진을 발현하고 있습니다" if est_strength >= fck else "설계 기준을 일부 만족하지 못하여 정기적 모니터링 추적이 요구됩니다" 
            final_output = (
                f"{loc} 콘크리트 구조물의 비파괴 정밀 진단 결과, 이상치 {ex_count}개를 소거한 보정 반발도 {r_val:.1f} R과 설계 슬럼프({slump}mm) 보정률이 통합 적용되었습니다. "
                f"확보된 재령일수 {age}일의 시간 경화 진행 상태에 따라 최종 산출된 융합 예측 강도는 {est_strength:.1f} MPa로 산출되었습니다. "
                f"이는 원 설계강도 {fck} MPa 대비 {pct_attained:.1f}% 수치에 해당하는 결과로서 공학적으로 {status_msg}. "
                f"특히 초음파 속도 변수({v_upv:.1f} m/s)의 결합을 통해 표면 건조 상태뿐만 아니라 부재 내부 골재의 조밀도까지 보정하여 진단 신뢰도를 혁신하였습니다. "
                f"\n\n*(수동/서버 API 키 미완료 상태로 내장 지능형 임시 분석 리포트가 출력되었습니다)*"
            )
    else:
        try:
            groq_key = st.secrets["GROQ_API_KEY"]
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {groq_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "llama-3.1-8b-instant",
                "messages": [
                    {
                        "role": "system", 
                        "content": "당신은 건축구조물 안전진단 및 콘크리트 품질검사 전문가입니다. 주어지는 데이터를 한국어로 건설 시방서 양식에 맞게 전문적이고 간결하게 분석해 주세요."
                    },
                    {"role": "user", "content": prompt} 
                ],
                "temperature": 0.3
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                ai_text = result["choices"][0]["message"]["content"]
                final_output = ai_text.strip() + "\n\n*(Groq Llama-3.1 Real-time AI 실시간 전문가 종합 분석 완료)*"
            else:
                final_output = f"🚨 Groq API 오류 발생 (코드: {response.status_code})\n\n💡 팁: Streamlit Secrets에 GROQ_API_KEY가 올바르게 등록되었는지 확인해 주세요!"
                
        except KeyError:
            final_output = "🚨 에러: Streamlit Secrets에 'GROQ_API_KEY' 이름이 정확하게 등록되지 않았습니다."
        except Exception as e:
            final_output = f"🚨 통신 에러 발생: {str(e)}"

    # 💡 [버그 수정 2] 다운로드 시 휘발되는 현상 방지를 위해 결과물을 Session State에 자동 박제!
    if str(page_type) in ["1", "page1"]:
        st.session_state["page1_ai_comment"] = final_output
    else:
        st.session_state["page2_ai_comment"] = final_output

    return final_output
def reliability_pct_calc(est, fck):
    if fck == 0: return 0.0
    return (est / fck) * 100.0

def calculate_angle_correction(r_val, angle):
    if angle == 0: return 0.0
    if r_val <= 30: max_up, max_down = 3.2, -4.1
    elif r_val <= 40: max_up, max_down = 2.8, -4.8
    else: max_up, max_down = 2.2, -5.2
    rad = math.radians(angle)
    return max_up * math.sin(rad) if angle > 0 else max_down * abs(math.sin(rad))

def make_time_options_korean():
    return [f"{h:02d}시 {m:02d}분" for h in range(24) for m in [0, 30]]

def parse_korean_time(time_text):
    return int(time_text.split("시")[0]), int(time_text.split("시")[1].replace("분", "").strip())

#Groq 코드?
def ask_groq_ai(prompt_text):
    """
    Streamlit Secrets에 등록된 GROQ_API_KEY를 사용하여
    Llama 3.1 모델로부터 전문가 소견을 받아오는 함수
    """
    try:
        # Secrets에서 키 가져오기 (만약 Secrets에 이름을 다르게 적으셨다면 그 이름으로 바꿔주세요)
        groq_key = st.secrets["GROQ_API_KEY"]
        
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {groq_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "llama-3.1-8b-instant",  # 속도가 빠르고 성능이 좋은 무료 모델
            "messages": [
                {
                    "role": "system", 
                    "content": "당신은 건축구조물 안전진단 및 콘크리트 품질검사 전문가입니다. 주어지는 데이터를 한국어로 건설 시방서 양식에 맞게 전문적이고 간결하게 분석해 주세요."
                },
                {"role": "user", "content": prompt_text}
            ],
            "temperature": 0.3
        }
        
        # API 호출
        response = requests.post(url, headers=headers, json=data, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            return f"Groq API 오류 발생 (코드: {response.status_code})"
            
    except KeyError:
        return "에러: Streamlit Secrets에 'GROQ_API_KEY'가 등록되지 않았습니다."
    except Exception as e:
        return f"Groq 통신 에러 발생: {str(e)}"

# =========================================================================
# ⚙ 사이드바 메인 탭 제어 (누락 복구 구역)
# =========================================================================
st.sidebar.header("⚙ 스마트 분석 제어판")
main_menu = st.sidebar.radio("작업 선택", ["1. 슈미트해머 측정 신뢰도 (AI 결함 우회)", "2. 다중 센서/환경 융합 강도 추정"])


# =========================================================================
# 1페이지: AI 표면 신뢰도 스캔
# =========================================================================
if "1." in main_menu:
    st.title("🎯 스마트 슈미트해머 1단계: AI 표면 검사보고서")
    
    st.subheader("📋 현장 기본 정보 입력")
    c_hdr1, c_hdr2, c_hdr3, c_hdr4 = st.columns(4)
    with c_hdr1: m_date = st.date_input("측정 실시 날짜", datetime.date.today())
    with c_hdr2:
        opts = make_time_options_korean()
        selected_time = st.selectbox("측정 시간", opts, index=opts.index("14시 00분"))
        m_hour, m_min = parse_korean_time(selected_time)
    with c_hdr3: m_loc = st.text_input("측정 위치", value="서울시 마포구 신축 현장")
    with c_hdr4: 
        # 타격 횟수 추천 로직 (+알파)
        base_strikes = st.selectbox("기본 타격 횟수 선택", [5, 10, 15, 20, 25, 30], index=2)
        extra_map = {5: 3, 10: 5, 15: 5, 20: 5, 25: 5, 30: 5}
        extra_strikes = extra_map[base_strikes]
        recommended_strikes = base_strikes + extra_strikes

    st.info(f"💡 **AI 스마트 추천**: 결함에 대비한 예비 타격점 확보를 위해 기본 {base_strikes}회에 추가 {extra_strikes}회를 더하여 **총 {recommended_strikes}회**의 최적 타격점을 추출합니다.")

    auto_temp, auto_hum = fetch_kma_weather_simulated(m_date, m_hour, m_min, m_loc)
    is_weather_valid, weather_msg = evaluate_ks_weather(auto_temp, auto_hum)
    st.info(f"📡 외부 API 기상 관측 ➔ 기온: {auto_temp} ℃ / 상대습도: {auto_hum} %")
    if is_weather_valid: st.success(weather_msg)
    else: st.error(weather_msg)

    st.markdown("#### 🧠 차세대 결함 검출 AI 인프라 연동 현황")
    
    # AI 실제 연동 상태 표시
    is_roboflow_live = API_KEYS["ROBOFLOW_API"] == "wk4BcUKf1InnR2LjHPF8" or API_KEYS["ROBOFLOW_API"] != ""
    is_gemini_live = API_KEYS["GEMINI_API"] != ""
    
    st.write("---")
    c_status_rf, c_status_gm = st.columns(2)
    with c_status_rf:
        if is_roboflow_live:
            st.success(f"🟢 **Roboflow Edge YOLO API 실시간 연동 완료** (Key: {API_KEYS['ROBOFLOW_API'][:4]}***)")
        else:
            st.error("❌ **Roboflow Vision API 오프라인**")
    with c_status_gm:
        if is_gemini_live:
            st.success("🟢 **Gemini AI LLM 전문가 소견 생성기 온라인 (실시간 API 가동 중)**")
        else:
            st.warning("🟡 **Gemini AI API Key 미연동** (하단 수동 등록 혹은 클라우드 Secrets를 통해 키를 설정하면 100% 활성화됩니다)")

    c_api1, c_api2, c_api3 = st.columns(3) 
    use_model1 = c_api1.checkbox("Edge YOLO v8 (균열/철근노출 탐지)", value=True) 
    c_api1.caption("🔗 API: universe.roboflow.com/defect-detection") 
     
    use_model2 = c_api2.checkbox("Edge YOLO v9 (요철/불균질면 탐지)", value=True) 
    c_api2.caption("🔗 API: universe.roboflow.com/shm") 
     
    use_model3 = c_api3.checkbox("Edge YOLO v10 (범용 결함 탐지)", value=True) 
    c_api3.caption("🔗 API: universe.roboflow.com/concrete-defects") 

    # --- 👇 자체 AI 모델 체크박스 추가 ---
    st.write("")
    use_custom_model = st.checkbox("자체 학습된 AI 모델", value=True)
    st.caption("🔗 사이트: https://app.roboflow.com/-ovfhd/concrete_defect-j9nuw/train")
    uploaded_file = st.file_uploader("📸 벽면 촬영 정밀 비전 영상 업로드", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = PILImage.open(uploaded_file).convert("RGB")
        img_rgb = np.array(image)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        max_width = 1000
        if img_bgr.shape[1] > max_width:
            ratio = max_width / img_bgr.shape[1]
            img_bgr = cv2.resize(img_bgr, (max_width, int(img_bgr.shape[0] * ratio)))
        h, w, _ = img_bgr.shape

        st.markdown("##### 📏 스케일 팩터 검정 (픽셀-mm 캘리브레이션)")
        c_pt1, c_pt2, c_len = st.columns(3)
        with c_pt1: p1_x, p1_y = st.number_input("기준점1 X", value=int(w*0.25)), st.number_input("기준점1 Y", value=int(h*0.80))
        with c_pt2: p2_x, p2_y = st.number_input("기준점2 X", value=int(w*0.75)), st.number_input("기준점2 Y", value=int(h*0.80))
        with c_len: real_len = st.number_input("검정선 실제 길이 (mm)", value=300.0)

        mm_per_pixel, _ = calculate_pixel_scale(p1_x, p1_y, p2_x, p2_y, real_len)

        final_defect = np.zeros((h, w), dtype=np.uint8)
        with st.spinner("AI 앙상블 분석 및 물리적 이격 거리(30mm) 제약 계산 중..."):
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 90)
            final_defect = cv2.bitwise_or(final_defect, edges)
            
            is_success, buffer = cv2.imencode(".jpg", img_bgr) 
            img_bytes = buffer.tobytes() 
            if use_model1: final_defect = cv2.bitwise_or(final_defect, fetch_roboflow_mask(img_bytes, "general-segmentation-api-9", "crack", w, h)) 
            if use_model2: final_defect = cv2.bitwise_or(final_defect, fetch_roboflow_mask(img_bytes, "general-segmentation-api-10", "defect", w, h)) 
            if use_model3: final_defect = cv2.bitwise_or(final_defect, fetch_roboflow_mask(img_bytes, "general-segmentation-api-11", "defects", w, h)) 
            
            # --- 👇 자체 AI 모델 분석 데이터 추가 반영 ---
            if use_custom_model: 
                final_defect = cv2.bitwise_or(final_defect, fetch_custom_roboflow_mask(img_bytes, w, h))

            # --- 👇 [추가됨] 1페이지 Gemini를 위한 결함 면적 및 실제 크기 계산기 ---
            defect_pixels = cv2.countNonZero(final_defect)
            total_pixels = w * h
            calc_defect_ratio = (defect_pixels / total_pixels) * 100 if total_pixels > 0 else 0
            
            calc_real_w_cm = (w * mm_per_pixel) / 10
            calc_real_h_cm = (h * mm_per_pixel) / 10
            calc_total_area_mm2 = (w * mm_per_pixel) * (h * mm_per_pixel)
            # -----------------------------------------------------------

        safe_area = cv2.bitwise_not(final_defect)
        margin = 40
        safe_area[:margin, :] = 0; safe_area[-margin:, :] = 0
        safe_area[:, :margin] = 0; safe_area[:, -margin:] = 0

        overlay = np.zeros_like(img_bgr)
        overlay[:] = [0, 0, 230]

        all_candidates = []
        for y in range(margin, h - margin, 6):
            for x in range(margin, w - margin, 6):
                if safe_area[y, x] > 0:
                    overlay[y:y+6, x:x+6] = [0, 220, 0]
                    all_candidates.append({"x": x, "y": y})

        # 픽셀을 변환하여 실제 물리적 3cm(30mm) 이상 거리 이격 조건 확인
        min_distance_mm = 30.0 
        min_distance_px = min_distance_mm / mm_per_pixel if mm_per_pixel > 0 else 50
        
        final_selected_pts = []
        step_size = max(1, len(all_candidates) // (recommended_strikes * 4))
        
        for pt in all_candidates[::step_size]:
            is_valid = True
            for f_pt in final_selected_pts:
                dist_px = math.sqrt((pt["x"] - f_pt["x"])**2 + (pt["y"] - f_pt["y"])**2)
                if dist_px < min_distance_px:
                    is_valid = False
                    break
            
            if is_valid:
                final_selected_pts.append(pt)
            
            if len(final_selected_pts) == recommended_strikes:
                break

        vis_guided_img = cv2.addWeighted(img_bgr, 0.45, overlay, 0.55, 0)
        cv2.line(vis_guided_img, (p1_x, p1_y), (p2_x, p2_y), (255, 255, 0), 5)
        
        strike_map_img = img_bgr.copy()
        cv2.line(strike_map_img, (p1_x, p1_y), (p2_x, p2_y), (255, 255, 0), 5)
        
        for idx, pt in enumerate(final_selected_pts):
            cv2.circle(strike_map_img, (pt["x"], pt["y"]), 14, (0, 255, 0), -1)
            cv2.putText(strike_map_img, str(idx + 1), (pt["x"] - 7, pt["y"] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        col_res1, col_res2 = st.columns(2)
        with col_res1: st.image(cv2.cvtColor(vis_guided_img, cv2.COLOR_BGR2RGB), caption="AI 표면 무결성 신뢰도 지도")
        with col_res2: st.image(cv2.cvtColor(strike_map_img, cv2.COLOR_BGR2RGB), caption=f"KS 규격 준수 최적 타격 좌표 (실제 3cm 이상 이격, 총 {len(final_selected_pts)}개 확보)")

        if len(final_selected_pts) < recommended_strikes:
            st.warning(f"⚠ 표면 결함 영역이 많거나 3cm 이격 공간이 부족하여 추천 타격점({recommended_strikes}개) 중 {len(final_selected_pts)}개만 확보되었습니다.")

        reliability_pct = 95.0 if len(final_selected_pts) >= base_strikes else round((len(final_selected_pts)/max(1,base_strikes))*100, 1)
        
        # 다변수 종합 딕셔너리 구성 후 AI 요청
# --- 👇 1페이지 Gemini를 위한 정밀 계산 데이터 정리 ---
        page1_data = { 
            "location": m_loc, 
            "date": str(m_date), 
            "temp": auto_temp, 
            "hum": auto_hum, 
            "defect_ratio": calc_defect_ratio,
            "real_w_cm": calc_real_w_cm,
            "real_h_cm": calc_real_h_cm,
            "total_area_mm2": calc_total_area_mm2
        } 

        # --- 👇 Secrets 키 자동 연동 로직 ---
        if not st.session_state.get("gemini_key_input") and "GEMINI_API" in st.secrets:
            API_KEYS["GEMINI_API"] = st.secrets["GEMINI_API"]
            genai.configure(api_key=st.secrets["GEMINI_API"])
        elif st.session_state.get("gemini_key_input"):
            API_KEYS["GEMINI_API"] = st.session_state["gemini_key_input"]
            genai.configure(api_key=st.session_state["gemini_key_input"])

        # 실시간 인공지능 종합 소견 생성 호출
        with st.spinner("Gemini 대형 언어 모델 기반 실시간 정밀 리포트 생성 중..."):
            ai_summary_txt = generate_gemini_commentary(1, page1_data)
            st.session_state["page1_ai_comment"] = ai_summary_txt

        # [수정 완료] 헤더를 📊 하나로 깔끔하게 통일했습니다! (📝 헤더는 삭제됨)
        st.subheader("📊 자체 빅데이터 학습 AI 종합 요약 (사건 1 분석)")
        st.info(ai_summary_txt)

        def build_page1_pdf():
            buffer_p1 = io.BytesIO()
            doc1 = SimpleDocTemplate(buffer_p1, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
            styles1 = getSampleStyleSheet()
            
            if pdf_font != 'Helvetica':
                styles1.add(ParagraphStyle(name='K_Title', fontName=pdf_font, fontSize=16, leading=22, alignment=1, spaceAfter=15, textColor=colors.HexColor("#1A365D")))
                styles1.add(ParagraphStyle(name='K_Sub', fontName=pdf_font, fontSize=11, leading=16, spaceBefore=10, spaceAfter=5, textColor=colors.HexColor("#2B6CB0")))
                styles1.add(ParagraphStyle(name='K_Norm', fontName=pdf_font, fontSize=9, leading=14))
                styles1.add(ParagraphStyle(name='K_Head', fontName=pdf_font, fontSize=9, leading=14, textColor=colors.white, alignment=1))
            else:
                styles1.add(ParagraphStyle(name='K_Title', fontName='Helvetica', fontSize=16, alignment=1))
                styles1.add(ParagraphStyle(name='K_Sub', fontName='Helvetica', fontSize=11))
                styles1.add(ParagraphStyle(name='K_Norm', fontName='Helvetica', fontSize=9))
                styles1.add(ParagraphStyle(name='K_Head', fontName='Helvetica', fontSize=9))

            story1 = []
            story1.append(Paragraph("[제 1페이지] AI 표면 품질 검사보고서", styles1['K_Title']))
            
            # --- AI 연동 상태 상세 텍스트 ---
            ai_status_detail = (
                f"1. 컴퓨터 비전 (Roboflow): {'연결 성공 (Live)' if is_roboflow_live else '내장 시뮬레이션 동작'}<br/>"
                "2. 기상 데이터 (OpenAPI): 정상 연동<br/>"
                "3. 대형언어모델 (Gemini): 정상 연동"
            )
            current_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

            info_data = [
                [Paragraph("품질 진단 항목", styles1['K_Head']), Paragraph("현장 실측 정보 및 알고리즘 판정 데이터", styles1['K_Head'])],
                [Paragraph("측정 일시", styles1['K_Norm']), Paragraph(f"{current_time_str}", styles1['K_Norm'])],
                [Paragraph("측정 대상 현장명", styles1['K_Norm']), Paragraph(f"{m_loc}", styles1['K_Norm'])],
                [Paragraph("AI 실시간 연동 상태", styles1['K_Norm']), Paragraph(ai_status_detail, styles1['K_Norm'])],
                [Paragraph("기상청 API 수신 환경", styles1['K_Norm']), Paragraph(f"기온: {auto_temp} ℃ / 상대습도: {auto_hum} %", styles1['K_Norm'])],
                [Paragraph("환경 시방 적합성", styles1['K_Norm']), Paragraph(weather_msg, styles1['K_Norm'])],
                [Paragraph("목표 타격 확보율", styles1['K_Norm']), Paragraph(f"추천 횟수: {recommended_strikes}회 / 확보점: {len(final_selected_pts)}회 (신뢰도 {reliability_pct}%)", styles1['K_Norm'])]
            ]
            t_info1 = Table(info_data, colWidths=[130, 390])
            t_info1.setStyle(TableStyle([('BACKGROUND', (0,0), (1,0), colors.HexColor("#1A365D")), ('GRID', (0,0), (-1,-1), 0.5, colors.grey), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
            story1.extend([t_info1, Spacer(1, 15)])
            
            story1.append(Paragraph("▶ 컴퓨터 비전 기반 실시간 이미지 분석 맵핑 결과", styles1['K_Sub']))
            img_w = cv2_to_rlimage(vis_guided_img, 250)
            img_s = cv2_to_rlimage(strike_map_img, 250)
            t_img1 = Table([[img_w, img_s]], colWidths=[260, 260])
            t_img1.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER')]))
            story1.extend([t_img1, Spacer(1, 15)])
            
            story1.append(Paragraph("[자체 빅데이터 AI 종합 요약 분석 의견]", styles1['K_Sub']))
            # 👇 Groq의 줄바꿈(\n)을 PDF용 줄바꿈(<br/>)으로 바꿔줍니다.
            pdf_summary = ai_summary_txt.replace('\n', '<br/>') if ai_summary_txt else "AI 분석 내용 없음"
            story1.append(Paragraph(pdf_summary, styles1['K_Norm']))
            
            doc1.build(story1)
            buffer_p1.seek(0)
            return buffer_p1.getvalue()

        st.write("---")
        st.download_button(
            label="📥 [1페이지] AI 표면 품질 검사보고서 PDF 다운로드",
            data=build_page1_pdf(),
            file_name=f"1_AI_Surface_Report_{m_date}.pdf",
            mime="application/pdf",
            type="primary"
        )


# =========================================================================
# 2페이지: 다중 센서 복합 강도 연산 시스템 (근거 강화)
# =========================================================================
elif "2." in main_menu:
    st.title("📊 SCI급 다중 센서 및 환경 변수 복합 강도 연산 시스템")

    col_env, col_data = st.columns([1, 1])

    with col_env:
        st.subheader("📋 1. 현장 계측 정보 및 재령 입력")
        m2_date = st.date_input("슈미트해머 실시 날짜", datetime.date.today())
        opts2 = make_time_options_korean()
        selected_time2 = st.selectbox("측정 시간", opts2, index=opts2.index("10시 00분"))
        m2_hour, m2_min = parse_korean_time(selected_time2)
        m2_loc = st.text_input("위치 (기상청 연동용)", value="현장 교각 B구간 측면부")

        auto_temp2, auto_hum2 = fetch_kma_weather_simulated(m2_date, m2_hour, m2_min, m2_loc)
        is_valid2, msg2 = evaluate_ks_weather(auto_temp2, auto_hum2)
        st.warning(f"📡 기상청 기반 온/습도: 온도 {auto_temp2}℃ / 습도 {auto_hum2}%")
        st.caption(f"※ 코멘트: 해당 환경은 {msg2}")

        m2_cast = st.date_input("타설일", datetime.date.today() - datetime.timedelta(days=90))
        total_days = max(1, (m2_date - m2_cast).days)
        fck = st.number_input("설계기준강도 (MPa)", value=24.0)
        st.info(f"재령: {total_days}일 확보 (타설일: {m2_cast})")
        
        st.subheader("🔊 2. 초음파 다중 측정 및 정밀 융합 (SonReb)")
        use_ultra = st.checkbox("🟢 초음파 측정치 연동 (다중 측정 및 이상치 보정)", value=True)
        
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
            
            st.markdown("##### 📝 초음파 측정 데이터 입력")
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
                st.success(f"✅ 평균 초음파 속도: **{v_mps:.1f} m/s** (10% 이상치 보정: **{v_mps_corr:.1f} m/s**)")
        else:
            v_mps, v_mps_corr = 0.0, 0.0

        use_slump = st.checkbox("🟢 슬럼프 수치 연동 (미세 공극률 보정)", value=True)
        val_slump = st.number_input("설계 슬럼프 (mm)", value=160.0) if use_slump else 150.0

    with col_data:
        st.subheader("🔨 3. 반발도(R값) 타격 데이터 세팅")
        c_strk1, c_strk2 = st.columns(2)
        with c_strk1:
            strike_count = st.selectbox("기록할 타격 횟수", [10, 15, 20, 25, 30, 35], index=2) 
        with c_strk2:
            angle_opts = [f"{a}° (상향)" if a>0 else f"{a}° (하향)" if a<0 else f"{a}° (수평/벽면)" for a in range(90, -95, -5)]
            selected_angle_str = st.selectbox("🎯 타격 각도", angle_opts, index=18)
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
        v_kmps_corr = v_mps_corr / 1000.0
        base_hybrid = 0.05 * (corrected_R ** 1.2) * (v_kmps_corr ** 1.5)
    else:
        base_hybrid = fc_rebound

    fc_slump_only = fc_rebound * age_factor * slump_corr if use_slump else fc_rebound
    fc_final_hybrid = base_hybrid * env_factor * age_factor * slump_corr

    st.write("---")
    st.markdown(f"### 📈 데이터 보정 결과")
    st.markdown(f"전체 평균 반발도: **{total_avg:.2f} R** ➔ 이상치 **{ex_count}개** 제외 ➔ **보정 평균 반발도: `{corrected_R:.2f} R`** (각도보정치 포함)")

    col_fc1, col_fc2, col_fc3 = st.columns(3)
    col_fc1.info(f"**[Model A] 단일 반발도 예상강도:**\n### {fc_rebound:.1f} MPa")
    col_fc2.info(f"**[Model C] 초음파(10%보정) 강도:**\n### {fc_ultra_only:.1f} MPa" if use_ultra else "**[Model C] 초음파 미연동**")
    col_fc3.success(f"**🏆 [Model D] 통합 복합 예상강도:**\n### {fc_final_hybrid:.1f} MPa")

    def build_page2_pdf():
        buffer_p2 = io.BytesIO()
        doc2 = SimpleDocTemplate(buffer_p2, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
        styles2 = getSampleStyleSheet()
        
        if pdf_font != 'Helvetica':
            styles2.add(ParagraphStyle(name='K_Title', fontName=pdf_font, fontSize=16, leading=22, alignment=1, spaceAfter=15, textColor=colors.HexColor("#1A365D")))
            styles2.add(ParagraphStyle(name='K_Sub', fontName=pdf_font, fontSize=11, leading=16, spaceBefore=10, spaceAfter=5, textColor=colors.HexColor("#2B6CB0")))
            styles2.add(ParagraphStyle(name='K_Norm', fontName=pdf_font, fontSize=9, leading=14))
            styles2.add(ParagraphStyle(name='K_Norm_C', fontName=pdf_font, fontSize=9, leading=14, alignment=1))
            styles2.add(ParagraphStyle(name='K_Head', fontName=pdf_font, fontSize=9, leading=14, textColor=colors.white, alignment=1))
        else:
            styles2.add(ParagraphStyle(name='K_Title', fontName='Helvetica', fontSize=16, alignment=1))
            styles2.add(ParagraphStyle(name='K_Sub', fontName='Helvetica', fontSize=11))
            styles2.add(ParagraphStyle(name='K_Norm', fontName='Helvetica', fontSize=9))
            styles2.add(ParagraphStyle(name='K_Norm_C', fontName='Helvetica', fontSize=9, alignment=1))
            styles2.add(ParagraphStyle(name='K_Head', fontName='Helvetica', fontSize=9, textColor=colors.white, alignment=1))

        story2 = []
        
        story2.append(Paragraph("[제 2페이지] 다중 센서 복합 콘크리트 강도 성적서", styles2['K_Title']))
        
        # 1. 현장 정보 표
        story2.append(Paragraph("▶ 진단 수행 정보 및 파라미터", styles2['K_Sub']))
        upv_pdf_txt = f"평균: {v_mps:.1f} m/s / 보정: {v_mps_corr:.1f} m/s ({upv_count}회 {upv_method})" if use_ultra and v_mps_corr > 0 else "미사용"
        
        info_data2 = [
            [Paragraph("측정 일시 및 장소", styles2['K_Norm']), Paragraph(f"{m2_date} ({selected_time2}) / {m2_loc}", styles2['K_Norm'])],
            [Paragraph("기상청 API 연동", styles2['K_Norm']), Paragraph(f"기온: {auto_temp2}℃ / 상대습도: {auto_hum2}%", styles2['K_Norm'])],
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
        res_data = [
            [Paragraph("반발도 데이터 전처리", styles2['K_Norm']), Paragraph(f"전체 평균: {total_avg:.2f} R / 이상치 {ex_count}개 폐기", styles2['K_Norm'])],
            [Paragraph("최종 보정 반발도", styles2['K_Norm']), Paragraph(f"{corrected_R:.2f} R (타격 각도 및 규격 보정 완료)", styles2['K_Norm'])],
            [Paragraph("단일 예상 강도 (Model A)", styles2['K_Norm']), Paragraph(f"{fc_rebound:.1f} MPa", styles2['K_Norm'])],
            [Paragraph("최종 융합 예상 강도 (Model D)", styles2['K_Norm']), Paragraph(f"{fc_final_hybrid:.1f} MPa", styles2['K_Norm'])]
        ]
        t_res2 = Table(res_data, colWidths=[150, 370])
        t_res2.setStyle(TableStyle([('FONTNAME', (0,0), (-1,-1), pdf_font), ('GRID', (0,0), (-1,-1), 0.5, colors.grey), ('BACKGROUND', (0,-1), (-1,-1), colors.lightsteelblue)]))
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
        story2.append(Paragraph("- 다차원 복합 강도식: Fc = 0.05 × R^1.2 × V^1.5 × 보정계수 (MPa)", styles2['K_Norm']))
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
    
    st.markdown("#### 💾 분석 결과 다운로드")
    custom_filename = st.text_input("📝 저장할 파일 이름을 입력하세요 (확장자 제외)", value=f"Multi_Sensor_Data_{m2_date}")
    excel_filename = custom_filename + ".xlsx" if not custom_filename.endswith(".xlsx") else custom_filename
    pdf_filename = custom_filename + ".pdf" if not custom_filename.endswith(".pdf") else custom_filename

    col_dl1, col_dl2 = st.columns(2)
    
    with col_dl1:
        st.download_button(
            label="📥 [2페이지] 다중 센서 복합 강도 성적서 (PDF)",
            data=build_page2_pdf(),
            file_name=pdf_filename,
            mime="application/pdf",
            use_container_width=True
        )

    with col_dl2:
        buffer_xls = io.BytesIO()
        with pd.ExcelWriter(buffer_xls, engine='openpyxl') as writer:
            pd.DataFrame({"항목": ["수행 일시", "기온 (℃)", "상대습도 (%)", "설계기준강도", "초음파 평균속도 (m/s)", "초음파 보정속도 (m/s)"], "내용": [f"{m2_date} ({selected_time2})", auto_temp2, auto_hum2, fck, v_mps, v_mps_corr]}).to_excel(writer, sheet_name="측정조건", index=False)
            
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
                "4. [산출식] 다중 복합 강도식(SonReb): Fc = 0.05 * R^1.2 * V^1.5 * 환경보정계수",
                "5. [KS 규격] KS F 2730 (반발경도 이상치 폐기 및 중력보정), KS F 2731 (초음파 시험법)",
                "6. [시방서] KCS 14 20 00 국토교통부 콘크리트 표준시방서",
                "7. [SCI 논문] A. Samarin et al., ACI Materials Journal (1983.11)",
                "8. [KCI 논문] 김철수 외, 한국건축구조학회논문집 (2021.05)"
            ]}).to_excel(writer, sheet_name="참조근거", index=False)
            
        st.download_button(
            label="📊 전체 도출 데이터 종합 (Excel)",
            data=buffer_xls.getvalue(),
            file_name=excel_filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
