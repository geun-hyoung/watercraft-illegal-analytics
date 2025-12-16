# =============================================================
# 과업 2-2: 수상 레저 속 번호판 탐지 및 텍스트 추출 (OCR)
# =============================================================
# 
# 기능:
# - 학습된 번호판 탐지 모델(model/plate_detection_baseline.pt)로 번호판 영역 탐지
# - EasyOCR을 사용하여 탐지된 번호판에서 텍스트 추출
# - 번호판 형식 정규화 (AA-00-0000 또는 00-AA-0000)
# - 다양한 전처리 기법 적용 (회전, 크기 조정, 감마 보정 등)
# - 결과를 CSV 파일과 시각화 이미지로 저장
#
# 입력: data/test/plate_detection/ 폴더의 이미지 파일들
# 출력: runs/ocr/OCR_FINAL_1029/ 폴더에 결과 저장
# =============================================================

from pathlib import Path
import sys, csv, cv2, gc, traceback, time, re
import numpy as np
import ultralytics
from ultralytics import YOLO
print(f"[INFO] Ultralytics {ultralytics.__version__}")

# ---------------- 사용자 설정 -------------------
# 프로젝트 루트 기준으로 경로 설정
BASE_DIR = Path(__file__).parent.parent
WEIGHT_PATH = BASE_DIR / "model" / "plate_detection_baseline.pt"
SOURCE_DIR = BASE_DIR / "data" / "test" / "plate_detection"
OUT_ROOT    = BASE_DIR / "runs" / "ocr" / "OCR_FINAL_1029"           # 결과 루트
CONF_THRES  = 0.25
IOU_THRES   = 0.5
DEVICE      = 0                     # GPU: 0 또는 정수, CPU: 'cpu' 또는 -1
LANGS       = ['en']                # EasyOCR 언어
TEXT_MINLEN = 2                     # 너무 짧은 텍스트 제거
PAD_RATIO   = 0.06                  # 크롭 시 패딩 비율
MAX_SIDE    = 1600                  # 한 변 최대(속도/안정성), 원본 유지: None
SAVE_EVERY_N = 25                   # N장 처리 후 gc 호출

# 허용되는 알파벳 접두쌍 정의
# MB: 모터보트, PW: 수상오토바이, RB: 고무보트, YT: 세일링요트
ALLOWED_PREFIX = {"MB", "PW", "RB", "YT"}

# ==== OCR 튜닝 파라미터 ====
TARGET_H = 304                    # OCR 전처리 시 세로 표준 높이 (픽셀)
ROT_DEGS = [-12, -8, -4, 0, 4, 8, 12]  # OCR 정확도 향상을 위한 회전 각도 후보 (도)
ALLOW    = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"  # OCR 허용 문자 목록
SCALE_FACTORS = [1.0, 1.2, 1.5, 1.8]  # 번호판 크롭 영역을 확대하는 비율들 (여러 크기로 시도)

# ---------------- 경로/출력 준비 -----------------
OUT_ROOT.mkdir(parents=True, exist_ok=True)
CROP_DIR    = OUT_ROOT / "crops";   CROP_DIR.mkdir(parents=True, exist_ok=True)
OVERLAY_DIR = OUT_ROOT / "overlay"; OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH    = OUT_ROOT / "ocr_results.csv"
LOG_PATH    = OUT_ROOT / "run_log.txt"

# ---------------- 유틸/후처리 함수 ----------------------
def safe_imread(path: Path):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        img = cv2.imread(str(path))
        return img

def safe_imwrite(path: Path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        ext = path.suffix.lower() or ".jpg"
        result, buf = cv2.imencode(ext, img)
        if not result:
            raise RuntimeError(f"cv2.imencode 실패: {path}")
        buf.tofile(str(path))
    except Exception:
        ok = cv2.imwrite(str(path), img)
        if not ok:
            raise RuntimeError(f"이미지 저장 실패: {path}")

def write_csv_header(csv_path: Path):
    with open(csv_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image", "crop_path",
            "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
            "det_conf", "ocr_text", "ocr_conf",
            "norm_text", "pattern_type", "is_valid"])

def append_csv_row(csv_path: Path, row):
    with open(csv_path, "a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)

def limit_max_side(img, max_side=1600):
    if not max_side:
        return img, 1.0, 1.0
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img, 1.0, 1.0
    scale = max_side / m
    img2 = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    sy = h / img2.shape[0]
    sx = w / img2.shape[1]
    return img2, sy, sx

def pad_bbox(x1, y1, x2, y2, W, H, pad_ratio=0.06):
    w = max(0, x2 - x1); h = max(0, y2 - y1)
    px = int(w * pad_ratio); py = int(h * pad_ratio)
    X1 = max(0, x1 - px); Y1 = max(0, y1 - py)
    X2 = min(W - 1, x2 + px); Y2 = min(H - 1, y2 + py)
    if X2 <= X1 or Y2 <= Y1:
        X1 = max(0, min(X1, W - 2)); X2 = min(W - 1, max(X1 + 2, X1 + w))
        Y1 = max(0, min(Y1, H - 2)); Y2 = min(H - 1, max(Y1 + 2, Y1 + h))
    return int(X1), int(Y1), int(X2), int(Y2)

def _scale_crop(img0, box, scale):
    """원본 이미지(img0)에서 box(x1,y1,x2,y2)를 중심으로 scale배 확대 크롭 반환"""
    X1, Y1, X2, Y2 = map(int, box)
    cx = (X1 + X2) / 2.0
    cy = (Y1 + Y2) / 2.0
    w  = (X2 - X1) * scale
    h  = (Y2 - Y1) * scale
    nx1 = int(max(0, cx - w/2)); ny1 = int(max(0, cy - h/2))
    nx2 = int(min(img0.shape[1]-1, cx + w/2)); ny2 = int(min(img0.shape[0]-1, cy + h/2))
    if nx2 <= nx1 or ny2 <= ny1:
        return img0[Y1:Y2, X1:X2]  # 비정상 케이스 방어
    return img0[ny1:ny2, nx1:nx2]

def draw_box_text(img, box, text, det_conf=None, ocr_conf=None):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
    label = (text or "").strip()
    subs  = []
    if det_conf is not None and (not (np.isnan(det_conf) or np.isinf(det_conf))):
        subs.append(f"det:{float(det_conf):.2f}")
    if ocr_conf is not None and (not (np.isnan(ocr_conf) or np.isinf(ocr_conf))):
        subs.append(f"ocr:{float(ocr_conf):.2f}")
    if subs:
        label = (label + " " if label else "") + f"({', '.join(subs)})"
    if not label:
        label = "N/A"
    cv2.putText(img, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

def normalize_device(dev):
    if isinstance(dev, str):
        if dev.lower() in ("cpu",):
            return "cpu"
        try:
            return int(dev)
        except Exception:
            return "cpu"
    if isinstance(dev, int):
        return dev if dev >= 0 else "cpu"
    return "cpu"

def yolo_predict_single(detector, img, conf=0.25, iou=0.5, device="cpu"):
    kwargs = dict(conf=conf, iou=iou, device=device, verbose=False)
    try:
        return detector.predict(source=img, **kwargs)
    except TypeError:
        return detector.predict(source=img, conf=conf, device=device, verbose=False)

# ---------- 번호판 전용 정규화/검증 (1~4) ----------

# 1) 괄호류 제거 + 하이픈/공백만 유지
def _strip_junk(s: str) -> str:
    if not s:
        return ""
    s = s.upper()
    # 모든 괄호류 제거
    s = re.sub(r'[\[\]\(\)\{\}\<\>]', '', s)
    # 언더스코어 → 하이픈
    s = s.replace('_', '-')
    # 영숫자/하이픈/공백만 유지
    s = re.sub(r'[^A-Z0-9\-\s]', ' ', s)
    # 다중 공백 정리
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# 2) 자리 문맥별 혼동 교정 맵: 문자 ↔ 숫자
DIGIT_TO_LETTER = {
    '0':'O',
    '1':'I',
    '2':'Z',
    '5':'S',
    '8':'B'}
LETTER_TO_DIGIT = {
    'O':'0', 'D':'0', 'Q':'0',
    'I':'1', 'L':'1', '|':'1', '!':'1',
    'Z':'2',
    'S':'5',
    'B':'8'}

def _to_letters(two_chars: str) -> str:
    out = []
    for ch in two_chars:
        if 'A' <= ch <= 'Z':
            out.append(ch)
        elif ch in DIGIT_TO_LETTER:
            out.append(DIGIT_TO_LETTER[ch])
        else:
            out.append(ch)
    return ''.join(out)

def _to_digits(s: str) -> str:
    out = []
    for ch in s:
        if ch.isdigit():
            out.append(ch)
        elif ch in LETTER_TO_DIGIT:
            out.append(LETTER_TO_DIGIT[ch])
        else:
            out.append(ch)
    return ''.join(out)

# 3) 패턴 검사 로직(자리 고정)으로 교체: AA-00-0000 / 00-AA-0000
def _normalize_candidate(raw: str):
    if not raw:
        return "", "", False, 0
    s = _strip_junk(raw)
    if not s:
        return "", "", False, 0

    s_flat = re.sub(r'[\s\-]+', '', s)

    def try_AA00_0000(t: str):
        if len(t) != 8:
            return None
        aa = _to_letters(t[0:2])
        d2 = _to_digits(t[2:4])
        d4 = _to_digits(t[4:8])

        if not (aa.isalpha() and len(aa) == 2): return None
        if not (d2.isdigit() and len(d2) == 2): return None
        if not (d4.isdigit() and len(d4) == 4): return None

        # 허용 접두만 채택
        if aa not in ALLOWED_PREFIX:
            return None

        norm = f"{aa}-{d2}-{d4}"
        repl = sum(x != y for x, y in zip(t, aa + d2 + d4))
        return (norm, "AA-00-0000", True, repl)

    def try_00AA_0000(t: str):
        if len(t) != 8:
            return None
        d2 = _to_digits(t[0:2])
        aa = _to_letters(t[2:4])
        d4 = _to_digits(t[4:8])

        if not (d2.isdigit() and len(d2) == 2): return None
        if not (aa.isalpha() and len(aa) == 2): return None
        if not (d4.isdigit() and len(d4) == 4): return None

        if aa not in ALLOWED_PREFIX:
            return None

        norm = f"{d2}-{aa}-{d4}"
        repl = sum(x != y for x, y in zip(t, d2 + aa + d4))
        return (norm, "00-AA-0000", True, repl)

    cand = []
    r1 = try_AA00_0000(s_flat)
    if r1: cand.append(r1)
    r2 = try_00AA_0000(s_flat)
    if r2: cand.append(r2)

    if not cand:
        tokens = re.split(r'[\s\-]+', s)
        if 1 <= len(tokens) <= 3:
            t2 = ''.join(tokens)
            r1 = try_AA00_0000(t2)
            if r1: cand.append(r1)
            r2 = try_00AA_0000(t2)
            if r2: cand.append(r2)

    if not cand:
        return "", "", False, 0

    cand.sort(key=lambda x: (not x[2], x[3]))  # is_valid 우선, 교정 적을수록
    return cand[0]

    NOISE_PREFIX = set(['I', '1', '|', '!', 'L'])  # 맨 앞 노이즈 한 글자 방어용 (있으면 삭제 후보 생성)

    def _normalize_candidate(raw: str):
        if not raw:
            return "", "", False, 0
        s = _strip_junk(raw)
        if not s:
            return "", "", False, 0

        s_flat = re.sub(r'[\s\-]+', '', s)

        streams = [s_flat]

        # 4-1) 맨 앞 노이즈 한 글자 제거 스트림 추가 (예: I|1L 등)
        if len(s_flat) >= 1 and s_flat[0] in NOISE_PREFIX:
            streams.append(s_flat[1:])

        # 4-2) 앞 3글자가 모두 알파벳이면 AB/BC 접두 스트림 추가
        if len(s_flat) >= 3 and s_flat[:3].isalpha():
            letters3 = s_flat[:3]
            rest = s_flat[3:]
            # AB + rest  (3번째 글자 제거)
            streams.append(letters3[:2] + rest)
            # BC + rest  (1번째 글자 제거)
            streams.append(letters3[1:] + rest)

        # (선택) 너무 길면 8자 슬라이딩 윈도 스트림도 후보에 추가
        if len(s_flat) > 8:
            for st in (streams.copy()):  # 기존 후보 각각에 대해 8자 창 생성
                for k in range(0, len(st) - 8 + 1):
                    streams.append(st[k:k + 8])

        # 중복 제거(정렬 유지 X)
        seen = set();
        uniq_streams = []
        for t in streams:
            if t not in seen:
                uniq_streams.append(t);
                seen.add(t)

        def try_AA00_0000(t: str):
            if len(t) != 8: return None
            aa = _to_letters(t[0:2]);
            d2 = _to_digits(t[2:4]);
            d4 = _to_digits(t[4:8])
            if not (aa.isalpha() and len(aa) == 2): return None
            if not (d2.isdigit() and len(d2) == 2): return None
            if not (d4.isdigit() and len(d4) == 4): return None
            if aa not in ALLOWED_PREFIX: return None
            norm = f"{aa}-{d2}-{d4}"
            repl = sum(x != y for x, y in zip(t, aa + d2 + d4))
            return (norm, "AA-00-0000", True, repl)

        def try_00AA_0000(t: str):
            if len(t) != 8: return None
            d2 = _to_digits(t[0:2]);
            aa = _to_letters(t[2:4]);
            d4 = _to_digits(t[4:8])
            if not (d2.isdigit() and len(d2) == 2): return None
            if not (aa.isalpha() and len(aa) == 2): return None
            if not (d4.isdigit() and len(d4) == 4): return None
            if aa not in ALLOWED_PREFIX: return None
            norm = f"{d2}-{aa}-{d4}"
            repl = sum(x != y for x, y in zip(t, d2 + aa + d4))
            return (norm, "00-AA-0000", True, repl)

        cand = []
        for t in uniq_streams:
            r1 = try_AA00_0000(t)
            if r1: cand.append(r1)
            r2 = try_00AA_0000(t)
            if r2: cand.append(r2)

        if not cand:
            tokens = re.split(r'[\s\-]+', s)
            if 1 <= len(tokens) <= 3:
                t2 = ''.join(tokens)
                for t in [t2]:
                    r1 = try_AA00_0000(t);
                    r2 = try_00AA_0000(t)
                    if r1: cand.append(r1)
                    if r2: cand.append(r2)

        if not cand:
            return "", "", False, 0

        cand.sort(key=lambda x: (not x[2], x[3]))  # is_valid 우선, 교정 적을수록
        return cand[0]

def choose_best_text(ocr_result):
    if not ocr_result:
        return "", 0.0, "", "", False

    def _conf(x):
        try:
            return float(x[2]) if x[2] is not None else 0.0
        except Exception:
            return 0.0

    ocr_result = sorted(ocr_result, key=_conf, reverse=True)

    best = ("", -1.0, "", "", False)  # raw, score, norm, pattern, is_valid
    for (bbox, text, conf) in ocr_result:
        raw = "" if text is None else str(text).strip()
        norm, ptype, is_ok, repl = _normalize_candidate(raw)
        score = (_conf((None, None, conf))
                 + (0.25 if norm else 0.0)
                 + (0.35 if is_ok else 0.0)
                 - min(0.2, 0.03 * repl))
        if score > best[1]:
            best = (raw, score, norm, ptype, is_ok)

    return best[0], float(max(0.0, min(1.0, best[1]))), best[2], best[3], best[4]

# ---------------- OCR 품질 향상 유틸 ----------------------
def _resize_standard(img, target_h=176):
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return img
    scale = target_h / float(h)
    new_w = max(1, int(w * scale))
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_CUBIC)

def _gamma_correction(img, gamma=1.0):

    if img is None:
        return img
    inv = 1.0 / max(1e-6, gamma)
    table = np.array([(i / 255.0) ** inv * 255 for i in range(256)]).astype("uint8")
    if img.ndim == 2:
        return cv2.LUT(img, table)
    else:
        return cv2.LUT(img, table)

def _make_preproc_variants(crop_bgr, target_h=176):
    base = _resize_standard(crop_bgr, target_h)
    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY) if base.ndim == 3 else base.copy()

    # A) 원본
    candA = base

    # A2) 감마 0.7 (어두운 영역 밝게)
    candA_g07 = _gamma_correction(base, 0.7)

    # A3) 감마 1.3 (밝은 영역 눌러 대비 확보)
    candA_g13 = _gamma_correction(base, 1.3)

    # B) Gray+CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    candB = clahe.apply(gray)

    # C) 약 샤픈
    blur = cv2.GaussianBlur(base, (0,0), 1.0)
    candC = cv2.addWeighted(base, 1.5, blur, -0.5, 0)

    # D) Gray+Otsu 이진
    _, bw = cv2.threshold(clahe.apply(gray), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    candD = bw

    return [candA, candA_g07, candA_g13, candB, candC, candD]

def _rotate_candidates(img, degs):
    outs = []
    h, w = img.shape[:2]
    c = (w/2.0, h/2.0)
    for d in degs:
        if d == 0:
            outs.append(img)
            continue
        M = cv2.getRotationMatrix2D(c, d, 1.0)
        outs.append(cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE))
    return outs

def _tight_text_strip(crop):

    try:
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim==3 else crop.copy()
        g = cv2.bilateralFilter(g, 7, 50, 50)
        g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
        thr = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV, 31, 12)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,3))
        mor = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)
        cnts, _ = cv2.findContours(mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        H, W = g.shape[:2]
        best = None
        best_score = 0
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            ar = w / float(h + 1e-6)
            area = w*h
            score = (ar > 2.8) * (0.5 + min(0.5, area/(W*H*0.2)))  # 대략적 점수
            if score > best_score:
                best_score = score; best = (x,y,w,h)
        if best is None:
            return crop
        x,y,w,h = best
        pad = int(0.08 * max(w,h))
        X1 = max(0, x - pad); Y1 = max(0, y - pad)
        X2 = min(W-1, x+w + pad); Y2 = min(H-1, y+h + pad)
        if X2 > X1 and Y2 > Y1:
            return crop[Y1:Y2, X1:X2]
        return crop
    except:
        return crop

# ---------------- 모델 로딩 ----------------------
assert WEIGHT_PATH.exists(), f"가중치 파일이 없음: {WEIGHT_PATH}"

try:
    device_norm = normalize_device(DEVICE)
    # CUDA 호환성 확인: 실제 연산으로 테스트
    if device_norm != "cpu":
        try:
            import torch
            if torch.cuda.is_available():
                test_tensor = torch.zeros(1, device=f"cuda:{device_norm}")
                result = test_tensor + 1
                print(f"[INFO] GPU 사용 가능: {torch.cuda.get_device_name(device_norm)}")
            else:
                print("[WARN] CUDA 사용 불가, CPU로 전환합니다.")
                device_norm = "cpu"
        except (RuntimeError, Exception) as e:
            print(f"[WARN] GPU 호환성 문제 감지 (오류: {str(e)[:100]}), CPU로 전환합니다.")
            device_norm = "cpu"
    detector = YOLO(str(WEIGHT_PATH))
except Exception as e:
    print("[FATAL] YOLO 가중치 로딩 실패:", e)
    traceback.print_exc()
    sys.exit(1)

reader = None
try:
    import easyocr
    _use_gpu = (device_norm != 'cpu')
    try:
        reader = easyocr.Reader(LANGS, gpu=_use_gpu)
    except Exception:
        print("[WARN] EasyOCR GPU 초기화 실패 → CPU로 폴백")
        reader = easyocr.Reader(LANGS, gpu=False)
except Exception as e:
    print("[FATAL] EasyOCR 초기화 실패:", e)
    traceback.print_exc()
    sys.exit(1)

# ---------------- CSV 헤더/로그 -------------------
write_csv_header(CSV_PATH)
with open(LOG_PATH, "w", encoding="utf-8") as lf:
    lf.write(time.strftime("[%Y-%m-%d %H:%M:%S] 시작\n"))

# ---------------- 처리 루프 ----------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
all_imgs = [p for p in sorted(SOURCE_DIR.rglob("*")) if p.suffix.lower() in IMG_EXTS]
print(f"[INFO] 대상 이미지: {len(all_imgs)}장 (from {SOURCE_DIR})")

n_total = len(all_imgs)
n_ok = 0
n_fail = 0
n_no_det = 0
fail_list = []
no_det_list = []

for idx, img_path in enumerate(all_imgs, 1):
    try:
        img0 = safe_imread(img_path)
        if img0 is None:
            raise RuntimeError("이미지 로드 실패")

        # 큰 이미지 축소(검출 안정/속도)
        img, sy, sx = limit_max_side(img0, MAX_SIDE)

        # ---------- YOLO 검출 ----------
        res_list = yolo_predict_single(detector, img, conf=CONF_THRES, iou=IOU_THRES, device=device_norm)

        # 결과 없음 처리
        if (not res_list) or (len(res_list) == 0) or (getattr(res_list[0], "boxes", None) is None) or (len(res_list[0].boxes) == 0):
            overlay = img0.copy()
            cv2.putText(overlay, "No plate detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            out_overlay = OVERLAY_DIR / f"{img_path.stem}_overlay.jpg"
            safe_imwrite(out_overlay, overlay)
            n_no_det += 1
            no_det_list.append(str(img_path))
            n_ok += 1
            continue

        # 원본 좌표계로 복원
        boxes = res_list[0].boxes
        boxes_xyxy = boxes.xyxy
        confs      = boxes.conf
        cls_ids    = getattr(boxes, "cls", None)

        if hasattr(boxes_xyxy, "cpu"):
            boxes_xyxy = boxes_xyxy.cpu().numpy()
        if hasattr(confs, "cpu"):
            confs = confs.cpu().numpy()
        if cls_ids is not None and hasattr(cls_ids, "cpu"):
            cls_ids = cls_ids.cpu().numpy()

        overlay = img0.copy()
        H, W = img0.shape[:2]

        for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
            x1 = int(x1 * sx); x2 = int(x2 * sx)
            y1 = int(y1 * sy); y2 = int(y2 * sy)

            X1, Y1, X2, Y2 = pad_bbox(x1, y1, x2, y2, W, H, pad_ratio=PAD_RATIO)
            crop = img0[Y1:Y2, X1:X2]
            crop_path = CROP_DIR / f"{img_path.stem}_det{i:02d}.jpg"
            safe_imwrite(crop_path, crop)

            # ---------- EasyOCR (강화 버전) ----------
            norm_text = ""; pattern_type = ""; is_valid = False
            raw_text = ""; disp_conf = 0.0
            try:
                # 1) 내부 재크롭으로 텍스트 띠를 더 타이트하게
                crop_refined = crop

                # 2) 전처리 후보 × 회전 후보를 모두 시도 → 최고 득점 선택
                best = dict(raw="", conf=0.0, norm="", ptype="", valid=False)

                # (1) 크롭 확대 비율별로 재시도
                for sf in SCALE_FACTORS:
                    crop_try = _scale_crop(img0, (X1, Y1, X2, Y2), sf)

                    # (2) 전처리 후보 생성
                    for v in _make_preproc_variants(crop_try, target_h=TARGET_H):

                        # (3) 회전 후보 적용
                        for vv in _rotate_candidates(v, ROT_DEGS):
                            try:
                                # 5) EasyOCR 파라미터 조정 (아래 5번 설명과 동일)
                                try:
                                    ocr_out = reader.readtext(
                                        vv,
                                        allowlist=ALLOW,
                                        detail=1,
                                        decoder='beamsearch',
                                        contrast_ths=0.05,
                                        adjust_contrast=0.7,
                                        text_threshold=0.3,
                                        low_text=0.2,
                                        link_threshold=0.2)
                                except TypeError:

                                    ocr_out = reader.readtext(vv)

                                raw, sc, norm, ptype, valid = choose_best_text(ocr_out) if ocr_out else (
                                "", 0.0, "", "", False)
                                if sc > best["conf"]:
                                    best.update(raw=raw, conf=sc, norm=norm, ptype=ptype, valid=valid)
                            except Exception:
                                pass

                raw_text = best["raw"]
                disp_conf = best["conf"]
                norm_text = best["norm"]
                pattern_type = best["ptype"]
                is_valid = best["valid"]

                # 너무 짧은 텍스트 제거 규칙 유지
                if raw_text and len(raw_text) < TEXT_MINLEN:
                    raw_text = ""
            except Exception:
                raw_text, disp_conf = "", 0.0

            show_text = norm_text if norm_text else (raw_text if raw_text else "N/A")

            # 오버레이
            det_conf_i = float(confs[i]) if (i < len(confs) and confs[i] is not None) else None
            label = show_text
            if pattern_type:
                label += f" [{pattern_type}]"
            draw_box_text(overlay, (X1, Y1, X2, Y2), label,
                          det_conf=det_conf_i, ocr_conf=disp_conf)

            # CSV 누적
            append_csv_row(CSV_PATH, [
                str(img_path),
                str(crop_path),
                int(X1), int(Y1), int(X2), int(Y2),
                ("" if det_conf_i is None else f"{det_conf_i:.4f}"),
                raw_text,
                ("" if disp_conf is None else f"{disp_conf:.4f}"),
                norm_text,
                pattern_type,
                "1" if is_valid else "0"])

        out_overlay = OVERLAY_DIR / f"{img_path.stem}_overlay.jpg"
        safe_imwrite(out_overlay, overlay)
        n_ok += 1

    except Exception as e:
        n_fail += 1
        fail_list.append(f"{img_path} :: {repr(e)}")
        print(f"[ERROR] 처리 중 예외 발생: {img_path}")
        traceback.print_exc()
    finally:
        if idx % SAVE_EVERY_N == 0:
            gc.collect()

# ---------------- 결과/요약 ----------------------
summary = (
    f"[DONE] OCR 결과 CSV: {CSV_PATH}\n"
    f"[DONE] CROP 폴더: {CROP_DIR}\n"
    f"[DONE] 오버레이 폴더: {OVERLAY_DIR}\n"
    f"[SUMMARY] total={n_total}, ok={n_ok}, fail={n_fail}, no_detect={n_no_det}\n")

print(summary)
with open(LOG_PATH, "a", encoding="utf-8") as lf:
    lf.write(summary)
    if fail_list:
        lf.write("[FAILURES]\n" + "\n".join(fail_list) + "\n")
    if no_det_list:
        lf.write("[NO_DETECTION]\n" + "\n".join(no_det_list) + "\n")