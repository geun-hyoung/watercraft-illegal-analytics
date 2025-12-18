# =============================================================
# 과업 3: 보트 유형 분류 모델 학습
# =============================================================
# 
# 기능:
# - data/raw/classification/ 폴더의 이미지와 라벨을 사용하여 YOLO 모델 학습
# - 이미지와 라벨을 train/val로 분할
# - YOLOv8s 사전 학습 모델을 기반으로 보트 유형 분류 모델 학습
# - 학습 완료 후 model/boat_classification_baseline.pt로 저장
#
# 입력: data/raw/classification/images/, data/raw/classification/labels/
# 출력: model/boat_classification_baseline.pt (학습된 모델)
# =============================================================

from pathlib import Path
import os
import shutil
import random
import yaml
from ultralytics import YOLO
import torch

# OpenMP 라이브러리 중복 로드 경고 방지
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# =========================================================
# 설정
# =========================================================
BASE_DIR = Path(__file__).resolve().parent.parent

# 입력 경로
RAW_IMG_DIR = BASE_DIR / "data" / "raw" / "classification" / "images"
RAW_LAB_DIR = BASE_DIR / "data" / "raw" / "classification" / "labels"

# 출력 경로
DATASET_DIR = BASE_DIR / "data" / "yolo_dataset_classification"
MODEL_SAVE_DIR = BASE_DIR / "model"

# 학습 설정
SEED = 42
TRAIN_RATIO = 0.85
EPOCHS = 150
BATCH_SIZE = 32
IMGSZ = 640
DEVICE = 0  # GPU: 0 또는 정수, CPU: "cpu"
WORKERS = 0  # Windows 필수

# 클래스 정의
CLS_NAMES = ["모터보트", "수상오토바이", "고무보트", "세일링요트", "기타"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 데이터셋 필터링 설정
MIN_AREA_FRAC = 0.002
MAX_AREA_FRAC = 0.80
MAX_ASPECT = 6.0

# 경로 확인
assert RAW_IMG_DIR.exists(), f"이미지 폴더 없음: {RAW_IMG_DIR}"
assert RAW_LAB_DIR.exists(), f"라벨 폴더 없음: {RAW_LAB_DIR}"

# =========================================================
# Helper Functions
# =========================================================
def norm_stem(name: str) -> str:
    return Path(name).stem

def read_yolo(path: Path):
    """YOLO 형식 라벨 파일 읽기"""
    rows = []
    if not path.exists():
        return rows
    try:
        for ln in path.read_text(encoding="utf-8").splitlines():
            t = ln.strip().split()
            if len(t) != 5:
                continue
            rows.append([int(float(t[0])), float(t[1]), float(t[2]), float(t[3]), float(t[4])])
    except Exception:
        pass
    return rows

def write_yolo(path: Path, rows):
    """YOLO 형식 라벨 파일 쓰기"""
    with open(path, "w", encoding="utf-8") as f:
        for c, xc, yc, w, h in rows:
            f.write(f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

def keep_box(xc, yc, w, h):
    """바운딩 박스 필터링"""
    area = w * h
    if area < MIN_AREA_FRAC or area > MAX_AREA_FRAC:
        return False
    asp = max(w / h, h / w) if w > 0 and h > 0 else 999
    if asp > MAX_ASPECT:
        return False
    if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1):
        return False
    return True

# =========================================================
# 데이터셋 준비
# =========================================================
print("\n[Step 1] 데이터셋 생성 시작")
random.seed(SEED)

# 1. 파일 스캔
imgs = [p for p in sorted(RAW_IMG_DIR.rglob("*")) if p.is_file() and p.suffix.lower() in IMG_EXTS]
labs_map = {norm_stem(p.name): p for p in RAW_LAB_DIR.glob("*.txt")}

print(f" - 전체 이미지 스캔: {len(imgs)}장")
samples = []

for img_path in imgs:
    stem = norm_stem(img_path.name)
    lab_path = labs_map.get(stem)
    if not lab_path:
        continue

    # 라벨 필터링
    raw_rows = read_yolo(lab_path)
    valid_rows = []
    for r in raw_rows:
        c, xc, yc, w, h = r
        if not keep_box(xc, yc, w, h):
            continue
        valid_rows.append([c, xc, yc, w, h])

    if valid_rows:
        samples.append((stem, img_path, valid_rows))

print(f" - 유효 샘플 수: {len(samples)}장")
if not samples:
    raise ValueError("유효한 데이터가 없습니다.")

# 2. Train/Val 분할
# 클래스별로 샘플 분류
class_samples = {}
for i, sample in enumerate(samples):
    cls_id = sample[2][0][0]  # 첫 번째 박스의 클래스 ID
    if cls_id not in class_samples:
        class_samples[cls_id] = []
    class_samples[cls_id].append(i)

# 클래스별로 train/val 분할
idx_tr, idx_va = [], []
for cls_id, indices in class_samples.items():
    rng = random.Random(SEED + cls_id)
    rng.shuffle(indices)
    n_train = int(len(indices) * TRAIN_RATIO)
    idx_tr.extend(indices[:n_train])
    idx_va.extend(indices[n_train:])

# 전체 셔플
random.Random(SEED).shuffle(idx_tr)
random.Random(SEED + 1).shuffle(idx_va)

print(f" - Train: {len(idx_tr)}장, Val: {len(idx_va)}장")

# 3. 데이터셋 디렉토리 생성
if DATASET_DIR.exists():
    shutil.rmtree(DATASET_DIR)

for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
    (DATASET_DIR / sub).mkdir(parents=True, exist_ok=True)

# 4. 파일 복사
def dump(indices, split):
    for i in indices:
        stem, img_p, rows = samples[i]
        dst_img = DATASET_DIR / f"images/{split}/{img_p.name}"
        dst_lab = DATASET_DIR / f"labels/{split}/{stem}.txt"
        shutil.copy2(img_p, dst_img)
        write_yolo(dst_lab, rows)

dump(idx_tr, "train")
dump(idx_va, "val")

# 5. data.yaml 생성
yaml_path = DATASET_DIR / "data.yaml"
with open(yaml_path, "w", encoding="utf-8") as f:
    f.write(f"path: {DATASET_DIR.as_posix()}\n")
    f.write(f"train: images/train\n")
    f.write(f"val: images/val\n")
    f.write(f"nc: {len(CLS_NAMES)}\n")
    f.write(f"names:\n")
    for i, n in enumerate(CLS_NAMES):
        f.write(f"  {i}: {n}\n")

print(f"[INFO] 데이터셋 구성 완료: {DATASET_DIR}")
print(f"[INFO] data.yaml 생성: {yaml_path}")

# =========================================================
# 모델 학습
# =========================================================
print("\n[Step 2] 모델 학습 시작")
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"[INFO] GPU 사용: {torch.cuda.get_device_name(DEVICE)}")
else:
    DEVICE = "cpu"
    print("[INFO] CPU 사용")

model = YOLO("yolov8s.pt")  # YOLOv8s 사전 학습 모델

# 학습 실행 (학습 결과는 Ultralytics가 자동으로 runs/ 폴더에 저장)
results = model.train(
    data=str(yaml_path),
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH_SIZE,
    device=DEVICE,
    workers=WORKERS,

    # 형상 보존 및 환경 적응
    scale=0.5,
    degrees=5.0,
    fliplr=0.5,
    flipud=0.0,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,

    # 데이터 보강 설정
    mosaic=0.3,
    close_mosaic=15,
    copy_paste=0.18,
    translate=0.08,
    erasing=0.2,

    # 제외
    mixup=0.0,
    shear=0.0,

    # 최적화
    lr0=0.0018,
    lrf=0.01,
    momentum=0.937,
    weight_decay=5e-4,
    patience=30,
    deterministic=True,
    cls=1.3,

    project="runs",
    name="boat_classification_train",
    exist_ok=True
)

# 학습 완료 후 최종 모델 저장
RUN_DIR = Path(results.save_dir)
WEIGHTS_DIR = RUN_DIR / "weights"
BEST = WEIGHTS_DIR / "best.pt"
LAST = WEIGHTS_DIR / "last.pt"
use_weight = BEST if BEST.exists() else LAST

assert use_weight.exists(), f"체크포인트 없음: {WEIGHTS_DIR}"
print(f"[TRAIN] 완료 → RUN_DIR={RUN_DIR.name}, CKPT={'best.pt' if use_weight == BEST else 'last.pt'}")

# 최종 모델을 model/ 폴더에 저장 (다른 과업과 동일)
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
FINAL_WEIGHT = MODEL_SAVE_DIR / "boat_classification_baseline.pt"
shutil.copy2(use_weight, FINAL_WEIGHT)
print(f"[SAVE] 추론용 고정 경로로 복사: {FINAL_WEIGHT}")

print("\n✅ 보트 유형 분류 모델 학습 완료!")

