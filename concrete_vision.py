# -*- coding: utf-8 -*-
"""
concrete_vision.py
------------------------------------------------------------------
직접 학습한 YOLOv8-seg 모델(concrete / crack / hand / ruler / void)로
1) 사진에서 콘크리트·자·손·결함 영역을 분할하고
2) 자(ruler) 길이로 픽셀→mm 스케일을 자동 계산하고
3) '가장 큰 빈 원' 방식으로 타격점을 1번부터 N번까지 추천하고
4) 결과를 색으로 구분한 지도로 그려주는 모듈 (Streamlit 과 독립적으로 동작)
"""
import math
import cv2
import numpy as np

# ---------------------------------------------------------------
# 클래스 이름 / 색상 (BGR)
# ---------------------------------------------------------------
DEFECT_CLASSES = ("crack", "void")          # 결함 = 빨간색
EXCLUDE_CLASSES = ("ruler", "hand")         # 자·손 = 타격 금지
COLORS = {
    "defect": (0, 0, 255),          # 빨강
    "ruler": (160, 245, 255),       # 연한 노랑
    "hand": (0, 140, 255),          # 주황
    "usable": (0, 200, 0),          # 초록 (타격 가능 영역)
    "outline": (0, 0, 0),           # 검정 (콘크리트 외곽)
    "outside": (70, 70, 70),        # 회색 사선 (대상 아님)
    "point": (255, 255, 255),
}


# ---------------------------------------------------------------
# 1) 분할
# ---------------------------------------------------------------
def load_model(path):
    """YOLOv8-seg 가중치(best.pt) 불러오기"""
    from ultralytics import YOLO
    return YOLO(path)


def segment(model, img_bgr, conf=0.25, imgsz=640):
    """
    반환: {"concrete": mask, "crack": mask, "hand": mask, "ruler": mask, "void": mask, ...}
    mask 는 입력 사진과 같은 크기의 uint8 (0 / 255)
    """
    h, w = img_bgr.shape[:2]
    res = model.predict(img_bgr, imgsz=imgsz, conf=conf, retina_masks=True, verbose=False)[0]
    names = {int(k): str(v).lower() for k, v in res.names.items()}
    masks = {n: np.zeros((h, w), np.uint8) for n in names.values()}
    scores = {n: [] for n in names.values()}
    if res.masks is None:
        return masks, scores
    cls = res.boxes.cls.cpu().numpy().astype(int)
    cf = res.boxes.conf.cpu().numpy()
    for poly, c, s in zip(res.masks.xy, cls, cf):
        if len(poly) < 3:
            continue
        n = names[c]
        cv2.fillPoly(masks[n], [np.round(poly).astype(np.int32)], 255)
        scores[n].append(float(s))
    return masks, scores


def largest_component(mask):
    """가장 큰 덩어리만 남기고 내부 구멍을 메움 (콘크리트 영역 정리용)"""
    m = (mask > 0).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return m, None
    c = max(cnts, key=cv2.contourArea)
    out = np.zeros_like(m)
    cv2.drawContours(out, [c], -1, 255, -1)
    return out, c


# ---------------------------------------------------------------
# 2) 자(ruler)로 스케일 자동 계산
# ---------------------------------------------------------------
def scale_from_ruler(ruler_mask, ruler_len_mm=300.0, min_px=80):
    """
    자 영역의 최소 외접 사각형(minAreaRect) 긴 변 = 자의 길이(픽셀)
    반환: (mm_per_px, 자 길이 px, 사각형 꼭짓점 4개) / 못 찾으면 (None, None, None)
    """
    if ruler_mask is None or cv2.countNonZero(ruler_mask) == 0:
        return None, None, None
    cnts, _ = cv2.findContours(ruler_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    long_px = max(rect[1])
    if long_px < min_px:
        return None, None, None
    return ruler_len_mm / long_px, long_px, cv2.boxPoints(rect).astype(np.int32)


# ---------------------------------------------------------------
# 3) 타격점 추천 (가장 큰 빈 원 → 주변 3cm 제외 → 반복)
# ---------------------------------------------------------------
def build_usable_mask(concrete, defects, excludes, mm_per_px, defect_buffer_mm=5.0, exclude_buffer_mm=10.0):
    """타격 가능 영역 = 콘크리트 - (결함 + 여유) - (자·손 + 여유)"""
    usable = (concrete > 0).astype(np.uint8) * 255
    def dil(m, mm):
        r = max(1, int(round(mm / mm_per_px)))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        return cv2.dilate(m, k)
    if defects is not None and cv2.countNonZero(defects):
        usable[dil(defects, defect_buffer_mm) > 0] = 0
    if excludes is not None and cv2.countNonZero(excludes):
        usable[dil(excludes, exclude_buffer_mm) > 0] = 0
    return usable


def plan_strike_points(usable, concrete, mm_per_px, n_points, spacing_mm=30.0, edge_mm=30.0, min_radius_mm=8.0):
    """
    사용자 알고리즘을 그대로 구현:
      - 타격 가능 영역 안에서 원을 1px, 2px, 3px... 키워 갈 때 가장 크게 그려지는 원의 중심
        = 거리변환(distance transform) 값이 최대인 점  → 타격점 1
      - 그 점 주위 spacing_mm(3cm) 원을 금지 영역으로 지우고 다시 계산 → 타격점 2 ... N
      - 콘크리트 외곽선에서 edge_mm 이내는 중심으로 고르지 않음
      - 가장 큰 원의 반지름이 min_radius_mm 보다 작아지면 (공간 부족) 중단
    반환: [{"no", "x", "y", "r_px", "r_mm"}, ...]
    """
    free = (usable > 0).astype(np.uint8)
    # 중심 후보 = 콘크리트 외곽에서 edge_mm 이상 안쪽
    edge_px = max(1, int(round(edge_mm / mm_per_px)))
    conc_dt = cv2.distanceTransform((concrete > 0).astype(np.uint8), cv2.DIST_L2, 5)
    center_ok = conc_dt >= edge_px
    spacing_px = max(1, int(round(spacing_mm / mm_per_px)))

    pts = []
    for k in range(int(n_points)):
        dt = cv2.distanceTransform(free, cv2.DIST_L2, 5)
        dt[~center_ok] = 0
        _, max_val, _, loc = cv2.minMaxLoc(dt)
        if max_val <= 0 or max_val * mm_per_px < min_radius_mm:
            break
        x, y = loc
        pts.append({"no": k + 1, "x": int(x), "y": int(y),
                    "r_px": float(max_val), "r_mm": float(max_val * mm_per_px)})
        cv2.circle(free, (int(x), int(y)), spacing_px, 0, -1)   # 주변 3cm 제외
    return pts


# ---------------------------------------------------------------
# 4) 결과 지도 그리기
# ---------------------------------------------------------------
def _blend(dst, mask, color, alpha):
    m = mask > 0
    if m.any():
        dst[m] = (dst[m] * (1 - alpha) + np.array(color, np.float32) * alpha).astype(np.uint8)


def render_map(img_bgr, concrete, concrete_contour, defects, ruler, hand, usable, points,
               ruler_box=None, show_radius=False):
    h, w = img_bgr.shape[:2]
    out = img_bgr.copy()
    font_scale = max(0.45, w / 1800)
    thick = max(2, int(round(w / 400)))

    # (1) 콘크리트 바깥: 어둡게 + 사선(금지 표시)
    outside = concrete == 0
    out[outside] = (out[outside] * 0.45).astype(np.uint8)
    hatch = np.zeros((h, w), np.uint8)
    step = max(14, w // 60)
    for i in range(-h, w, step):
        cv2.line(hatch, (i, 0), (i + h, h), 255, max(2, step // 6))
    hm = (hatch > 0) & outside
    out[hm] = COLORS["outside"]

    # (2) 타격 가능 영역 초록
    _blend(out, usable, COLORS["usable"], 0.30)
    # (3) 자·손·결함
    _blend(out, ruler, COLORS["ruler"], 0.65)
    _blend(out, hand, COLORS["hand"], 0.60)
    _blend(out, defects, COLORS["defect"], 0.75)
    for m, col in ((defects, COLORS["defect"]),):
        cnts, _ = cv2.findContours((m > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, col, max(1, thick // 2))

    # (4) 콘크리트 외곽 검은 테두리 (안쪽 흰 선으로 대비)
    if concrete_contour is not None:
        cv2.drawContours(out, [concrete_contour], -1, (255, 255, 255), thick + 4)
        cv2.drawContours(out, [concrete_contour], -1, COLORS["outline"], thick + 1)
    if ruler_box is not None:
        cv2.drawContours(out, [ruler_box], -1, (0, 200, 255), max(1, thick // 2))

    # (5) 타격점: 여유 반경 원 + 번호
    for p in points:
        c = (p["x"], p["y"])
        if show_radius:
            cv2.circle(out, c, int(p["r_px"]), (255, 255, 255), 1, cv2.LINE_AA)
        rad = max(7, int(w / 90))
        cv2.circle(out, c, rad, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(out, c, rad, (255, 255, 255), 2, cv2.LINE_AA)
        txt = str(p["no"])
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        cv2.putText(out, txt, (c[0] - tw // 2, c[1] + th // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def analyze(img_bgr, masks, n_points, ruler_len_mm=300.0, manual_mm_per_px=None,
            spacing_mm=30.0, edge_mm=30.0, min_radius_mm=8.0, defect_buffer_mm=5.0, show_radius=False):
    """분할 결과(masks) → 스케일, 타격 가능 영역, 타격점, 지도까지 한 번에"""
    h, w = img_bgr.shape[:2]
    z = np.zeros((h, w), np.uint8)
    get = lambda n: masks.get(n, z)

    concrete_raw = get("concrete")
    concrete_found = cv2.countNonZero(concrete_raw) > 0.02 * h * w
    if concrete_found:
        concrete, contour = largest_component(concrete_raw)
    else:  # 콘크리트를 못 찾으면 사진 전체를 대상으로 (경고 표시용 플래그)
        concrete = np.full((h, w), 255, np.uint8)
        contour = np.array([[[0, 0]], [[w - 1, 0]], [[w - 1, h - 1]], [[0, h - 1]]], np.int32)

    defects = z.copy()
    for n in DEFECT_CLASSES:
        defects = cv2.bitwise_or(defects, get(n))
    ruler, hand = get("ruler"), get("hand")
    excludes = cv2.bitwise_or(ruler, hand)

    auto_scale, ruler_px, ruler_box = scale_from_ruler(ruler, ruler_len_mm)
    if manual_mm_per_px:
        mm_per_px, scale_src = manual_mm_per_px, "수동 기준점"
    elif auto_scale:
        mm_per_px, scale_src = auto_scale, f"AI 자동 (자 {ruler_px:.0f}px = {ruler_len_mm:.0f}mm)"
    else:
        mm_per_px, scale_src = 0.5, "기본값 0.5 mm/px (자를 찾지 못함)"

    usable = build_usable_mask(concrete, defects, excludes, mm_per_px, defect_buffer_mm)
    points = plan_strike_points(usable, concrete, mm_per_px, n_points, spacing_mm, edge_mm, min_radius_mm)
    vis = render_map(img_bgr, concrete, contour, defects, ruler, hand, usable, points, ruler_box, show_radius)

    conc_px = cv2.countNonZero(concrete)
    defect_in_conc = cv2.countNonZero(cv2.bitwise_and(defects, concrete))
    return {
        "vis": vis, "points": points, "mm_per_px": mm_per_px, "scale_src": scale_src,
        "ruler_found": auto_scale is not None, "concrete_found": concrete_found,
        "concrete_area_cm2": conc_px * (mm_per_px ** 2) / 100.0,
        "usable_area_cm2": cv2.countNonZero(usable) * (mm_per_px ** 2) / 100.0,
        "defect_ratio": 100.0 * defect_in_conc / conc_px if conc_px else 0.0,
        "img_w_cm": w * mm_per_px / 10.0, "img_h_cm": h * mm_per_px / 10.0,
        "counts": {n: int(cv2.countNonZero(get(n)) > 0) for n in ("concrete", "crack", "void", "ruler", "hand")},
    }
