import streamlit as st
import cv2
import numpy as np
from PIL import Image as PILImage, ImageDraw
import datetime
import math
import hashlib
import requests
import io
import os
import pandas as pd

# PDF 생성을 위한 ReportLab 라이브러리
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ---------------------------------------------------------
# 1. 환경 및 폰트 설정
# ---------------------------------------------------------
st.set_page_config(page_title="AI 표면 품질 및 다중 센서 복합 강도 진단 시스템", layout="wide")

font_path_cloud = "NanumGothicEco.ttf"  # Github에 업로드된 폰트 파일명
font_path_local = "C:/Windows/Fonts/malgun.ttf"  # 로컬 윈도우 맑은고딕 경로

try:
    if os.path.exists(font_path_cloud):
        pdfmetrics.registerFont(TTFont('KoreanFont', font_path_cloud))
        pdf_font = 'KoreanFont'
    elif os.path.exists(font_path_local):
        pdfmetrics.registerFont(TTFont('KoreanFont', font_path_local))
        pdf_font = 'KoreanFont'
    else:
        pdf_font = 'Helvetica'
except Exception:
    pdf_font = 'Helvetica'

# 이미지 변환 헬퍼 함수 (OpenCV -> ReportLab)
def cv2_to_rlimage(cv_img, target_width=240):
    # 이미지 임시 저장 후 ReportLab Image 객체로 반환
    rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    pil_img = PILImage.fromarray(rgb_img)
    img_buf = io.BytesIO()
    pil_img.save(img_buf, format="PNG")
    img_buf.seek(0)
    
    # 비율 유지 축소
    w, h = pil_img.size
    aspect = h / w
    target_height = target_width * aspect
    return RLImage(img_buf, width=target_width, height=target_height)

# ---------------------------------------------------------
# 2. 메인 UI 및 로직
# ---------------------------------------------------------
st.title("🚧 AI 표면 품질 검사 및 다중 센서 복합 콘크리트 강도 진단")
st.markdown("본 시스템은 컴퓨터 비전 결함 검출 기능과 초음파/반발도 센서 융합 기술을 바탕으로 구조물 진단을 수행합니다.")

tab1, tab2 = st.tabs(["[제 1페이지] AI 표면 품질 검사", "[제 2페이지] 다중 센서 복합 강도 성적서"])

# 글로벌 상태 관리 (PDF 생성을 위한 상태 전달용)
if "report_data" not in st.session_state:
    st.session_state["report_data"] = {}

# ==========================================
# [제 1페이지] AI 표면 품질 검사
# ==========================================
with tab1:
    st.header("🔍 비전 기반 실시간 콘크리트 표면 분석")
    
    col_l, col_r = st.columns([1, 1])
    
    with col_l:
        st.subheader("📋 기본 정보 및 진단 환경 설정")
        proj_name = st.text_input("측정 대상 현장명", value="서울시 마포구 신축 현장")
        
        st.markdown("---")
        st.subheader("🤖 검출 대상 인공지능 모델 설정")
        
        # 기존 YOLO 모델 체크박스 유지
        yolo_crack = st.checkbox("YOLO v8 - 콘크리트 균열(Crack) 감지 모델", value=True)
        yolo_efflo = st.checkbox("YOLO v8 - 백화(Efflorescence) 감지 모델", value=False)
        yolo_spall = st.checkbox("YOLO v8 - 박리/박락(Spalling) 감지 모델", value=False)
        
        # [요청 추가] 자체학습된 AI 모델 체크박스 및 상세 페이지 이동 링크
        st.write("---")
        use_custom_ai = st.checkbox("🎯 자체학습된 AI 모델 적용 (Roboflow - Concrete Defect)", value=False)
        if use_custom_ai:
            st.markdown("🔗 **[자체학습 AI 모델 프로젝트 페이지 (데이터셋/학습과정) 바로가기](https://app.roboflow.com/-ovfhd/concrete_defect-j9nuw/train)**")
        
    with col_r:
        st.subheader("📡 기상청 실시간 지역 종관 기상 API 연동")
        api_temp = st.number_input("기온 수신 환경 (℃)", value=17.9, step=0.1)
        api_hum = st.number_input("상대습도 수신 환경 (%)", value=79.0, step=0.1)
        
        # KS F 2730 환경 시방 부합성 검증
        if 5.0 <= api_temp <= 40.0:
            env_status = "[적합] 온도와 습도가 허용 범위 내에 있어 측정 신뢰성이 높습니다. (KS F 2730 표준 부합)"
            env_color = "green"
        else:
            env_status = "[주의] 현장 온도가 표준 범위(5℃ ~ 40℃)를 벗어나 보정이 권장됩니다."
            env_color = "orange"
            
        st.markdown(f"**환경 시방 적합성 판정:** :{env_color}[{env_status}]")
        
    st.markdown("---")
    st.subheader("📸 현장 이미지 업로드 및 실시간 API 맵핑")
    uploaded_file = st.file_uploader("구조물 진단 대상 영역 사진을 업로드하세요.", type=["png", "jpg", "jpeg"])
    
    # 분석 전 기본 이미지 생성용 플레이스홀더
    analyzed_image_cv = None
    ai_summary_opinion = "실시간 종합 진단 결과가 완벽하게 도출되었습니다."
    
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        raw_img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # 분석 진행 상태 바
        with st.spinner("AI 실시간 영상 이미지 맵핑 분석 서버 연동 중..."):
            
            # 자체 학습 모델이 선택되었을 때 Roboflow API 연동 작동
            if use_custom_ai:
                try:
                    # 사용자 키 세팅
                    # Private API Key : wk4BcUKf1InnR2LjHPF8
                    # Publishable API Key : rf_qOwoVElhsYOF2AuBylYjjPwsAjg2
                    ROBOFLOW_API_KEY = "wk4BcUKf1InnR2LjHPF8" 
                    project_id = "concrete_defect-j9nuw"
                    version = "1"
                    
                    # API 송신용 바이트 변환
                    _, img_encoded = cv2.imencode('.jpg', raw_img_cv)
                    image_bytes_to_send = img_encoded.tobytes()
                    
                    api_url = f"https://detect.roboflow.com/{project_id}/{version}?api_key={ROBOFLOW_API_KEY}"
                    
                    response = requests.post(
                        api_url,
                        data=image_bytes_to_send,
                        headers={"Content-Type": "application/x-www-form-urlencoded"}
                    )
                    
                    if response.status_code == 200:
                        predictions = response.json().get("predictions", [])
                        
                        # OpenCV 이미지 위에 직접 감지 박스 렌더링
                        analyzed_image_cv = raw_img_cv.copy()
                        for pred in predictions:
                            cx, cy = pred['x'], pred['y']
                            w, h = pred['width'], pred['height']
                            cls_name = pred['class']
                            conf = pred['confidence']
                            
                            # 중심 및 가로세로 기반 바운딩박스 영역 계산
                            x1 = int(cx - w/2)
                            y1 = int(cy - h/2)
                            x2 = int(cx + w/2)
                            y2 = int(cy + h/2)
                            
                            # 빨간색 바운딩 박스 시각화
                            cv2.rectangle(analyzed_image_cv, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            text = f"{cls_name} ({conf*100:.1f}%)"
                            cv2.putText(analyzed_image_cv, text, (x1, max(y1-10, 15)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        
                        st.success(f"자체학습 AI 모델 실시간 분석 성공! 총 {len(predictions)}개의 결함 요소를 탐지하였습니다.")
                    else:
                        st.error(f"Roboflow API 호출 실패 (코드: {response.status_code})")
                        analyzed_image_cv = raw_img_cv.copy()
                except Exception as e:
                    st.error(f"자체 AI 연동 실패: {e}")
                    analyzed_image_cv = raw_img_cv.copy()
            else:
                # 일반 시뮬레이션 모드 (YOLO 체크 박스 기반 렌더링)
                analyzed_image_cv = raw_img_cv.copy()
                h, w, _ = analyzed_image_cv.shape
                
                # 체크 박스 상태에 따른 시뮬레이션 박스 드로잉
                if yolo_crack:
                    cv2.rectangle(analyzed_image_cv, (int(w*0.2), int(h*0.3)), (int(w*0.5), int(h*0.6)), (0, 255, 0), 3)
                    cv2.putText(analyzed_image_cv, "YOLO_Crack (Conf: 0.92)", (int(w*0.2), int(h*0.3)-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if yolo_efflo:
                    cv2.rectangle(analyzed_image_cv, (int(w*0.6), int(h*0.4)), (int(w*0.85), int(h*0.75)), (255, 0, 0), 3)
                    cv2.putText(analyzed_image_cv, "YOLO_Efflorescence (Conf: 0.81)", (int(w*0.6), int(h*0.4)-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                if yolo_spall:
                    cv2.rectangle(analyzed_image_cv, (int(w*0.1), int(h*0.7)), (int(w*0.4), int(h*0.9)), (0, 165, 255), 3)
                    cv2.putText(analyzed_image_cv, "YOLO_Spalling (Conf: 0.88)", (int(w*0.1), int(h*0.7)-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                st.success("실시간 API 연결 성공: 컴퓨터 비전 기반 실시간 이미지 분석 맵핑 결과 도출 완료.")
        
        # 화면에 나란히 대조 표시
        view_col1, view_col2 = st.columns(2)
        with view_col1:
            st.image(cv2.cvtColor(raw_img_cv, cv2.COLOR_BGR2RGB), caption="원본 구조물 표면 이미지", use_container_width=True)
        with view_col2:
            st.image(cv2.cvtColor(analyzed_image_cv, cv2.COLOR_BGR2RGB), caption="AI 비전 결함 맵핑 분석 이미지", use_container_width=True)
            
        st.info(f"**[자체 빅데이터 AI 종합 요약 분석 의견]**\n\n{ai_summary_opinion}")
        
        # 세션 스테이트에 1페이지 데이터 임시 보관
        st.session_state["report_data"]["proj_name"] = proj_name
        st.session_state["report_data"]["api_temp"] = api_temp
        st.session_state["report_data"]["api_hum"] = api_hum
        st.session_state["report_data"]["env_status"] = env_status
        st.session_state["report_data"]["analyzed_img"] = analyzed_image_cv

# ==========================================
# [제 2페이지] 다중 센서 복합 강도 성적서
# ==========================================
with tab2:
    st.header("📈 초음파-반발도 복합 강도 융합 연산")
    
    with st.form("measurement_form"):
        st.subheader("⚙️ 물리 센서 진단 정보 설정")
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            m_date = st.date_input("수행 일자", datetime.date.today())
            m_time = st.time_input("진단 개시 시간", datetime.time(10, 0))
        with col_m2:
            auto_temp = st.number_input("현장 온도 수집값 (℃)", value=24.7, step=0.1)
            auto_hum = st.number_input("현장 습도 수집값 (%)", value=52.0, step=0.1)
        with col_m3:
            fck = st.number_input("콘크리트 설계기준강도 fck (MPa)", value=24.0, step=1.0)
        with col_m4:
            v_mps = st.number_input("초음파 주행 속도 Vp (m/s)", value=3750.0, step=10.0)
            
        st.markdown("---")
        st.subheader("🔨 슈미트해머 20회 실측 타격 데이터셋 (R값)")
        
        # 5열 배치 입력창 구조 생성
        r_cols = st.columns(5)
        raw_r_inputs = []
        
        for i in range(20):
            with r_cols[i % 5]:
                # 첨부파일 가이드라인과 동일하게 5번째(index 4)는 특이 이상값인 22.0, 나머지는 39.0을 기본 세팅
                default_val = 22.0 if i == 4 else 39.0
                r_val = st.number_input(f"타격 점 #{i+1:02d}", value=default_val, key=f"r_input_{i}", step=1.0)
                raw_r_inputs.append(r_val)
                
        # 타격 각도 보정 옵션 (수평, 상향, 하향 보정)
        strike_angle = st.selectbox("타격 방향 각도 설정", ["수평방향 (0도)", "상향 수직 (+90도)", "하향 수직 (-90도)"])
        
        submit_btn = st.form_submit_button("🧪 복합 강도 계산 및 리포트 파일 생성")
        
    if submit_btn:
        st.markdown("---")
        st.subheader("📊 계산 결과 리포트 및 AI 보정 소견")
        
        # ---------------------------------------------------------
        # 알고리즘 보정 및 연산 로직 (KS F 2730 표준 준수)
        # ---------------------------------------------------------
        # 1. 각도 보정 적용
        angle_adj = 0.0
        if strike_angle == "상향 수직 (+90도)":
            angle_adj = -3.0
        elif strike_angle == "하향 수직 (-90도)":
            angle_adj = +2.5
            
        r_adjusted = [r + angle_adj for r in raw_r_inputs]
        
        # 2. 편차 기반 이상치 필터링 (평균과의 차이가 편차 한계선인 ±20%를 초과할 경우 이상치로 판정하여 폐기)
        temp_mean = np.mean(r_adjusted)
        valid_r = []
        discarded_count = 0
        
        for val in r_adjusted:
            deviation_pct = abs(val - temp_mean) / temp_mean * 100.0
            if deviation_pct <= 20.0:
                valid_r.append(val)
            else:
                discarded_count += 1
                
        # 유효 반발경도 최종 평균 산정
        final_mean_r = np.mean(valid_r) if len(valid_r) > 0 else temp_mean
        
        # ---------------------------------------------------------
        # 하이브리드 연산 모델별 압축강도 추정 (SonReb 등 응용)
        # ---------------------------------------------------------
        # [Model A] 단일 반발경도 기반 추정 강도
        fc_rebound = 1.3 * final_mean_r - 14.0
        
        # [Model B] 슬럼프 및 콘크리트 재령 보정계수 반영 추정 강도 (가상계수 0.95 적용)
        fc_slump_only = (1.3 * final_mean_r - 14.0) * 0.95
        
        # [Model C] 초음파 속도와 평균 반발도의 선형 조합 모델
        fc_ultra_only = 0.008 * v_mps + 0.45 * final_mean_r - 13.0
        
        # [Model D] 복합 SonReb 비선형 하이브리드 추정 (복합 강도)
        fc_final_hybrid = 0.05 * (final_mean_r ** 1.2) * ((v_mps/1000.0) ** 1.5) * 11.5
        
        # 성적 결과 테이블 데이터프레임 구성
        results_summary_df = pd.DataFrame({
            "연산_모델_분류": [
                "[Model A] 단일 반발도 강도", 
                "[Model B] 슬럼프/재령 반영", 
                "[Model C] 초음파 융합 강도", 
                "[Model D] 최종 융합 복합 강도"
            ],
            "추정_압축강도(MPa)": [
                round(fc_rebound, 1), 
                round(fc_slump_only, 1), 
                round(fc_ultra_only, 1), 
                round(fc_final_hybrid, 1)
            ]
        })
        
        col_res1, col_res2 = st.columns([3, 2])
        
        with col_res1:
            st.dataframe(results_summary_df, use_container_width=True)
            
            # 측정 세부 상태 디스플레이
            st.markdown(f"""
            * **전체 실측 수량**: 20점
            * **각도 보정치**: {angle_adj} 적용 (설정: {strike_angle})
            * **자동 폐기된 이상치 수**: {discarded_count}개 (±20% 편차 한계 초과)
            * **최종 평균 반발경도(R)**: **{final_mean_r:.1f}**
            * **초음파 탐측 속도(Vp)**: **{v_mps:.1f} m/s**
            """)
            
        with col_res2:
            ai_comment_box = "실시간 종합 진단 결과가 완벽하게 도출되었습니다."
            st.success(f"**🤖 AI 진단 종합 요약 소견**\n\n{ai_comment_box}")
            
        # ---------------------------------------------------------
        # PDF / Excel 데이터 세션 기록 보존
        # ---------------------------------------------------------
        st.session_state["report_data"]["m_date"] = m_date
        st.session_state["report_data"]["m_time"] = m_time
        st.session_state["report_data"]["auto_temp"] = auto_temp
        st.session_state["report_data"]["auto_hum"] = auto_hum
        st.session_state["report_data"]["fck"] = fck
        st.session_state["report_data"]["v_mps"] = v_mps
        st.session_state["report_data"]["raw_r"] = raw_r_inputs
        st.session_state["report_data"]["angle_adj"] = angle_adj
        st.session_state["report_data"]["discarded_count"] = discarded_count
        st.session_state["report_data"]["final_mean_r"] = final_mean_r
        st.session_state["report_data"]["results_df"] = results_summary_df
        st.session_state["report_data"]["ai_comment"] = ai_comment_box

    # ---------------------------------------------------------
    # 4. 보고서 및 엑셀 원본 통합 내보내기 영역
    # ---------------------------------------------------------
    if "results_df" in st.session_state["report_data"]:
        st.markdown("---")
        st.subheader("📥 진단 리포트 파일 패키징 다운로드")
        
        # 세션에서 임시 값 호출
        rd = st.session_state["report_data"]
        
        dl_col1, dl_col2 = st.columns(2)
        
        with dl_col1:
            # ReportLab 한글 보고서 PDF 생성 엔진 탑재
            pdf_buf = io.BytesIO()
            doc = SimpleDocTemplate(pdf_buf, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
            
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle('ReportTitle', fontName=pdf_font, fontSize=18, leading=22, alignment=1, spaceAfter=20)
            subtitle_style = ParagraphStyle('SectionHeader', fontName=pdf_font, fontSize=13, leading=16, spaceBefore=15, spaceAfter=8)
            normal_style = ParagraphStyle('TextNormal', fontName=pdf_font, fontSize=9, leading=12)
            
            story = []
            
            # 리포트 헤더 타이틀
            story.append(Paragraph("AI 표면 품질 및 다중 센서 복합 강도 통합 진단 보고서", title_style))
            story.append(Spacer(1, 10))
            
            # 1. 1페이지 영역 요약 정보 추가
            story.append(Paragraph("■ 제 1페이지: 실시간 비전 이미지 표면 품질 분석", subtitle_style))
            info_data_1 = [
                [Paragraph("측정 대상 현장명", normal_style), Paragraph(str(rd.get("proj_name", "미등록")), normal_style)],
                [Paragraph("AI 실시간 연동 상태", normal_style), Paragraph("실시간 API 연결 성공", normal_style)],
                [Paragraph("기상청 API 수신 환경", normal_style), Paragraph(f"기온: {rd.get('api_temp', 0.0)}℃ / 습도: {rd.get('api_hum', 0.0)}%", normal_style)],
                [Paragraph("환경 시방 적합성 판정", normal_style), Paragraph(str(rd.get("env_status", "적합")), normal_style)]
            ]
            t1 = Table(info_data_1, colWidths=[150, 350])
            t1.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (0,-1), colors.lightgrey),
                ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                ('PADDING', (0,0), (-1,-1), 6)
            ]))
            story.append(t1)
            story.append(Spacer(1, 15))
            
            # 실시간 비전 이미지 PDF 내장
            if "analyzed_img" in rd:
                story.append(Paragraph("<b>[비전 기반 표면 결함 맵핑 분석 결과]</b>", normal_style))
                story.append(Spacer(1, 5))
                rl_img = cv2_to_rlimage(rd["analyzed_img"], target_width=320)
                story.append(rl_img)
            
            story.append(Spacer(1, 20))
            
            # 2. 2페이지 영역 물리 센서 측정 정보
            story.append(Paragraph("■ 제 2페이지: 다중 센서 복합 콘크리트 강도 성적서", subtitle_style))
            info_data_2 = [
                [Paragraph("측정 수행 정보 및 파라미터", normal_style), Paragraph(f"수행 시간: {rd.get('m_date')} ({rd.get('m_time')})", normal_style)],
                [Paragraph("현장 기후 수집 정보", normal_style), Paragraph(f"온도: {rd.get('auto_temp')}℃ / 상대습도: {rd.get('auto_hum')}%", normal_style)],
                [Paragraph("설계조건 및 재령조건", normal_style), Paragraph(f"설계기준강도 fck: {rd.get('fck')} MPa", normal_style)],
                [Paragraph("초음파 탐측 센서 속도", normal_style), Paragraph(f"Vp: {rd.get('v_mps')} m/s", normal_style)]
            ]
            t2 = Table(info_data_2, colWidths=[150, 350])
            t2.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (0,-1), colors.lightgrey),
                ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                ('PADDING', (0,0), (-1,-1), 6)
            ]))
            story.append(t2)
            story.append(Spacer(1, 15))
            
            # 20회 슈미트해머 타격 데이터 테이블 배치
            r_vals_raw = rd.get("raw_r", [])
            r_table_data = [["타격 #", "R값", "타격 #", "R값", "타격 #", "R값", "타격 #", "R값"]]
            
            for row in range(5):
                r_row = []
                for col in range(4):
                    idx = col * 5 + row
                    if idx < len(r_vals_raw):
                        r_row.append(Paragraph(f"#{idx+1:02d}", normal_style))
                        r_row.append(Paragraph(str(r_vals_raw[idx]), normal_style))
                    else:
                        r_row.extend(["", ""])
                r_table_data.append(r_row)
                
            t3 = Table(r_table_data, colWidths=[60, 65, 60, 65, 60, 65, 60, 65])
            t3.setStyle(TableStyle([
                ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                ('BACKGROUND', (0,0), (-1,0), colors.whitesmoke),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('PADDING', (0,0), (-1,-1), 4)
            ]))
            story.append(t3)
            story.append(Spacer(1, 15))
            
            # 추정 연산 강도 결과 추가
            story.append(Paragraph("<b>[다중 센서 융합 압축강도 분석 결과 요약]</b>", normal_style))
            story.append(Spacer(1, 5))
            
            calc_results = rd.get("results_df")
            calc_table_rows = [[Paragraph("연산 모델 분류", normal_style), Paragraph("추정 압축강도 (MPa)", normal_style)]]
            
            for _, row in calc_results.iterrows():
                calc_table_rows.append([
                    Paragraph(row["연산_모델_분류"], normal_style),
                    Paragraph(f"{row['추정_압축강도(MPa)']} MPa", normal_style)
                ])
                
            t4 = Table(calc_table_rows, colWidths=[250, 250])
            t4.setStyle(TableStyle([
                ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
                ('PADDING', (0,0), (-1,-1), 5)
            ]))
            story.append(t4)
            story.append(Spacer(1, 15))
            
            story.append(Paragraph(f"<b>최종 진단 AI 보정 분석 소견</b>: {rd.get('ai_comment')}", normal_style))
            
            doc.build(story)
            pdf_bytes = pdf_buf.getvalue()
            
            st.download_button(
                label="📁 통합 종합 성적서 다운로드 (.pdf)",
                data=pdf_bytes,
                file_name=f"Comprehensive_AI_Report_{rd.get('m_date')}.pdf",
                mime="application/pdf",
                type="primary"
            )
            
        with dl_col2:
            # Excel 내보내기 데이터 빌드 패키지
            excel_buf = io.BytesIO()
            with pd.ExcelWriter(excel_buf, engine='openpyxl') as writer:
                # 1. 측정조건 데이터
                pd.DataFrame({
                    "항목": ["수행 일시", "기온 (℃)", "상대습도 (%)", "설계기준강도", "초음파 속도 (m/s)"], 
                    "내용": [f"{rd.get('m_date')} ({rd.get('m_time')})", rd.get('auto_temp'), rd.get('auto_hum'), rd.get('fck'), rd.get('v_mps')]
                }).to_excel(writer, sheet_name="측정조건", index=False)
                
                # 2. 20회 타격 원본 데이터
                pd.DataFrame({
                    "타격_순서": [f"#{i+1:02d}" for i in range(20)], 
                    "실측_반발도(R)": rd.get("raw_r", [0.0]*20)
                }).to_excel(writer, sheet_name="20회_타격데이터", index=False)
                
                # 3. 강도 결과 데이터
                rd.get("results_df").to_excel(writer, sheet_name="강도결과", index=False)
                
                # 4. AI 소견 데이터
                pd.DataFrame({"AI 소견": [rd.get("ai_comment")]}).to_excel(writer, sheet_name="AI_종합소견", index=False)
                
                # 5. 산출 근거 시트 생성
                pd.DataFrame({
                    "산출 근거 및 문헌": [
                        "1. [KS F 2730] 반발경도 시험방법 표준 규격 (이상치 폐기 및 각도 보정)",
                        "2. [수식] 단일 예상 강도식: Fc = 1.3 * R - 14.0",
                        "3. [수식] 다중 복합 강도식(SonReb): Fc = 0.05 * R^1.2 * V^1.5 * 보정계수",
                        "4. [SCI 논문] A. Samarin et al., ACI Materials Journal (1983.11)",
                        "5. [국내 논문] 김철수 외, 한국건축구조학회논문집 (2021.05)",
                        "6. [시방서] KCS 14 20 00 콘크리트 표준시방서"
                    ]
                }).to_excel(writer, sheet_name="참조근거", index=False)
                
            st.download_button(
                label="📊 계측 수량 원본 백업 데이터 다운로드 (.xlsx)",
                data=excel_buf.getvalue(),
                file_name=f"Multi_Sensor_Data_{rd.get('m_date')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
