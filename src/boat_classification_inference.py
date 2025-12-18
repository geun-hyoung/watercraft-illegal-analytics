# =============================================================
# 과업 3: 보트 유형 분류 추론
# =============================================================
# 
# 기능:
# - 학습된 보트 유형 분류 모델로 테스트 이미지에서 보트 탐지 및 분류
# - 결과를 CSV 파일과 시각화 이미지로 저장
#
# 입력: data/test/ 폴더의 테스트 이미지 파일들
# 출력: result/boat_classification/ 폴더에 결과 저장
# =============================================================

from pathlib import Path
import os
import csv
import cv2
from ultralytics import YOLO
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# =========================================================
# 설정
# =========================================================
BASE_DIR = Path(__file__).resolve().parent.parent

# 모델 경로
MODEL_PATH = BASE_DIR / "model" / "boat_classification_baseline.pt"

# 입력 경로
TEST_PATH = BASE_DIR / "data" / "test"

# 출력 경로
RESULT_DIR = BASE_DIR / "result" / "boat_classification"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# 추론 설정
CONF_THRES = 0.25
IOU_THRES = 0.5
IMGSZ = 640

# 클래스 이름 (학습 시와 동일)
CLS_NAMES = ["모터보트", "수상오토바이", "고무보트", "세일링요트", "기타"]

# 이미지 확장자
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 경로 확인
assert MODEL_PATH.exists(), f"모델 파일 없음: {MODEL_PATH}"
if not TEST_PATH.exists():
    print(f"[WARN] 테스트 폴더 없음: {TEST_PATH}")
    print("[INFO] 테스트 이미지를 해당 폴더에 넣어주세요.")
    exit(1)

# =========================================================
# 장치 설정
# =========================================================
DEVICE = "cpu"
try:
    if torch.cuda.is_available():
        test_tensor = torch.zeros(1, device="cuda:0")
        result = test_tensor + 1
        gpu_name = torch.cuda.get_device_name(0)
        DEVICE = 0
        print(f"[INFO] 사용 장치: GPU ({gpu_name})")
    else:
        DEVICE = "cpu"
        print(f"[INFO] 사용 장치: CPU")
except (RuntimeError, Exception) as e:
    print(f"[WARN] GPU 사용 불가 (오류: {str(e)[:100]}), CPU로 전환합니다.")
    DEVICE = "cpu"
    print(f"[INFO] 사용 장치: CPU")

# =========================================================
# 모델 로딩
# =========================================================
print(f"\n[INFO] 모델 로딩: {MODEL_PATH}")
model = YOLO(str(MODEL_PATH))
print("[INFO] 모델 로딩 완료")

# =========================================================
# 추론 실행
# =========================================================
print(f"\n[INFO] 추론 시작: {TEST_PATH}")

# CSV 파일 준비
CSV_PATH = RESULT_DIR / "classification_results.csv"
with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "x1", "y1", "x2", "y2", "class", "class_name", "conf"])

    # 테스트 이미지 파일 찾기
    img_files = [p for p in sorted(TEST_PATH.rglob("*")) if p.is_file() and p.suffix.lower() in IMG_EXTS]
    print(f"[INFO] 처리할 이미지: {len(img_files)}장")

    # 각 이미지에 대해 추론
    for img_path in img_files:
        try:
            # 이미지 로드
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[WARN] 이미지 로드 실패: {img_path.name}")
                continue

            # 추론 실행
            results = model.predict(
                source=str(img_path),
                imgsz=IMGSZ,
                conf=CONF_THRES,
                iou=IOU_THRES,
                device=DEVICE,
                verbose=False
            )

            # PIL Image로 변환 (한글 표시를 위해)
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            
            # 한글 폰트 로드
            try:
                # Windows 기본 한글 폰트
                font_path = "C:/Windows/Fonts/malgun.ttf"
                if os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, 20)
                else:
                    font = ImageFont.load_default()
            except:
                font = ImageFont.load_default()
            
            detections = []

            # 탐지 결과 처리
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                cls_name = CLS_NAMES[cls_id] if cls_id < len(CLS_NAMES) else "기타"

                # CSV에 기록
                writer.writerow([
                    img_path.name,
                    x1, y1, x2, y2,
                    cls_id,
                    cls_name,
                    f"{conf:.4f}"
                ])

                detections.append((x1, y1, x2, y2, cls_name, conf))

                # 바운딩 박스 그리기
                color = (0, 255, 0)  # 초록색
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

                # 클래스 이름과 신뢰도 표시 (한글 지원)
                label = f"{cls_name} {conf:.2f}"
                bbox = draw.textbbox((0, 0), label, font=font)
                label_width = bbox[2] - bbox[0]
                label_height = bbox[3] - bbox[1]
                
                # 배경 사각형
                draw.rectangle(
                    [x1, y1 - label_height - 10, x1 + label_width + 4, y1],
                    fill=color
                )
                # 텍스트
                draw.text(
                    (x1 + 2, y1 - label_height - 8),
                    label,
                    fill=(255, 255, 255),
                    font=font
                )
            
            # PIL Image를 OpenCV 형식으로 변환
            result_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

            # 결과 이미지 저장
            output_img_path = RESULT_DIR / f"{img_path.stem}_result.jpg"
            cv2.imwrite(str(output_img_path), result_img)

            print(f"[INFO] 처리 완료: {img_path.name} - {len(detections)}개 탐지")

        except Exception as e:
            print(f"[ERROR] 처리 중 예외 발생: {img_path.name} - {e}")
            continue

print(f"\n✅ 추론 완료!")
print(f"[DONE] 결과 CSV: {CSV_PATH}")
print(f"[DONE] 결과 이미지: {RESULT_DIR}")
print(f"[DONE] 총 처리: {len(img_files)}장")

