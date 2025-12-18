# =============================================================
# 과업 1: 수상 레저 속 승선 인원 수 탐지
# =============================================================
# 
# 기능:
# - YOLOv11s 사전 학습 모델을 사용하여 이미지에서 사람(Person)과 보트(Boat)를 탐지
# - 사람 바운딩박스 하단부(발 위치 근사)가 보트 바운딩박스 내부에 포함되는지를 기준으로
#   보트 탑승 여부를 판단
# - IoU(Intersection over Union) 기반으로 중복 탐지된 바운딩박스를 병합
# - 각 보트별로 탑승한 인원수를 계산하여 결과 이미지에 시각화
#
# 입력: result/boat_classification/ 폴더의 보트 유형 분류 결과 이미지 파일들
# 출력: result/passenger_counting/ 폴더에 결과 이미지 저장
# =============================================================

import os
import glob
import torch
import cv2
import numpy as np
from ultralytics import YOLO

# =========================
# 1. 실행 장치 설정
# =========================
# GPU가 사용 가능하면 GPU(0)를 사용하고, 없으면 CPU 사용
# CUDA 호환성 문제가 있을 경우 CPU로 폴백
device = "cpu"  # 기본값을 CPU로 설정
try:
    if torch.cuda.is_available():
        # CUDA 호환성 확인: 실제 연산으로 테스트
        test_tensor = torch.zeros(1, device="cuda:0")
        result = test_tensor + 1
        # GPU compute capability 확인
        gpu_name = torch.cuda.get_device_name(0)
        # RTX 5070 (sm_120) 등 최신 GPU는 현재 PyTorch에서 지원하지 않을 수 있음
        # 실제 연산이 성공하면 GPU 사용
        device = 0
        print(f"[INFO] 사용 장치: GPU ({gpu_name})")
    else:
        device = "cpu"
        print(f"[INFO] 사용 장치: CPU")
except (RuntimeError, Exception) as e:
    # CUDA 오류 발생 시 CPU로 폴백
    print(f"[WARN] GPU 사용 불가 (오류: {str(e)[:100]}), CPU로 전환합니다.")
    device = "cpu"
    print(f"[INFO] 사용 장치: CPU (GPU 호환성 문제로 인한 자동 전환)")

# =========================
# 2. YOLOv11s 모델 로딩
# =========================
# 모델 경로 수정: model 폴더의 yolo11s_passenger_counting.pt 사용
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "yolo11s_passenger_counting.pt")
model = YOLO(model_path)

# =========================
# 3. IoU (Intersection over Union) 계산
# =========================
# 두 바운딩박스의 겹침 정도를 계산하는 함수
# 반환값: 0.0 ~ 1.0 사이의 값 (1.0에 가까울수록 많이 겹침)
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return inter / (area1 + area2 - inter + 1e-6)

# =========================
# 4. 인접 박스 병합 (IoU 기준)
# =========================
# 동일 객체에 대해 중복 탐지된 바운딩박스를 IoU 기준으로 병합
# iou_thresh: IoU 임계값 (이 값보다 크면 같은 객체로 간주하여 병합)
def merge_close_boxes(boxes, iou_thresh=0.6):
    merged = []
    used = [False] * len(boxes)

    for i in range(len(boxes)):
        if used[i]:
            continue

        x1, y1, x2, y2 = boxes[i]["box"]
        merged_box = [x1, y1, x2, y2]

        for j in range(i + 1, len(boxes)):
            if used[j]:
                continue

            x3, y3, x4, y4 = boxes[j]["box"]
            iou = calculate_iou(
                [x1, y1, x2, y2],
                [x3, y3, x4, y4]
            )

            if iou > iou_thresh:
                merged_box = [
                    min(merged_box[0], x3),
                    min(merged_box[1], y3),
                    max(merged_box[2], x4),
                    max(merged_box[3], y4),
                ]
                used[j] = True

        used[i] = True
        merged.append({"box": merged_box})

    return merged

# =========================
# 5. 사람의 보트 탑승 여부 판별
# =========================
# 사람 바운딩박스의 하단부(발 위치 근사)가 보트 바운딩박스 내부에 포함되는지 확인
# bottom_ratio: 사람 바운딩박스 하단에서 얼마나 위쪽을 발 위치로 볼지 (기본 15%)
def is_person_on_boat(person_box, boat_box, bottom_ratio=0.15):
    px1, py1, px2, py2 = person_box  # 사람 바운딩박스 좌표
    bx1, by1, bx2, by2 = boat_box    # 보트 바운딩박스 좌표

    # 발 위치 계산: 사람 바운딩박스 하단부의 중앙점
    foot_x = (px1 + px2) / 2  # 발의 x 좌표 (중앙)
    foot_y = py2 - (py2 - py1) * bottom_ratio  # 발의 y 좌표 (하단에서 15% 위)

    # 발 위치가 보트 바운딩박스 내부에 있는지 확인
    return (bx1 <= foot_x <= bx2) and (by1 <= foot_y <= by2)

# =========================
# 6. 사람 및 보트 탐지
# =========================
# YOLO 모델을 사용하여 이미지에서 사람과 보트를 탐지
# 반환값: (persons, boats) - 각각 바운딩박스 정보를 담은 리스트
def detect_people_and_boats(img):
    persons, boats = [], []

    # YOLO 모델로 객체 탐지 수행
    results = model.predict(img, device=device, verbose=False)[0]

    # 탐지된 모든 객체를 순회하며 사람과 보트 분류
    for i, cls_id in enumerate(results.boxes.cls.cpu().numpy().astype(int)):
        label = model.model.names[cls_id].lower()  # 클래스 이름 (소문자)
        box = results.boxes.xyxy[i].cpu().numpy()  # 바운딩박스 좌표 [x1, y1, x2, y2]

        # 사람인 경우
        if label == "person":
            persons.append({"box": box})
        # 보트 또는 배인 경우
        elif label in ("boat", "ship"):
            boats.append({"box": box})

    # 중복 탐지된 바운딩박스 병합
    # 사람은 IoU 0.2 이상이면 같은 객체로 간주 (더 엄격)
    # 보트는 IoU 0.5 이상이면 같은 객체로 간주 (덜 엄격)
    persons = merge_close_boxes(persons, iou_thresh=0.2)
    boats = merge_close_boxes(boats, iou_thresh=0.5)

    return persons, boats

# =========================
# 7. 보트별 탑승 인원 매칭
# =========================
# 각 보트에 탑승한 사람들을 매칭하여 보트별 탑승 인원수를 계산
# 반환값: boats 리스트 (각 보트에 "persons" 키로 탑승한 사람 리스트 추가됨)
def match_people_to_boats(persons, boats):
    for boat in boats:
        boat["persons"] = []  # 각 보트의 탑승 인원 리스트 초기화
        # 모든 사람에 대해 해당 보트에 탑승했는지 확인
        for person in persons:
            if is_person_on_boat(person["box"], boat["box"]):
                boat["persons"].append(person)  # 탑승한 사람으로 추가
    return boats

# =========================
# 8. 결과 시각화 및 저장
# =========================
# 탐지 결과를 이미지에 그려서 저장
# - 보트: 초록색 박스, "Boat N: M persons" 텍스트
# - 탑승 인원: 빨간색 박스
def visualize_and_save(img, boats, img_name):
    result = img.copy()  # 원본 이미지 복사

    # 각 보트에 대해 시각화
    for i, boat in enumerate(boats):
        bx1, by1, bx2, by2 = map(int, boat["box"])
        # 보트 바운딩박스 그리기 (초록색)
        cv2.rectangle(result, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
        # 보트 번호와 탑승 인원수 텍스트 표시
        cv2.putText(
            result,
            f"Boat {i+1}: {len(boat['persons'])} persons",
            (bx1, by1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        # 해당 보트에 탑승한 사람들 그리기 (빨간색)
        for person in boat["persons"]:
            px1, py1, px2, py2 = map(int, person["box"])
            cv2.rectangle(result, (px1, py1), (px2, py2), (0, 0, 255), 2)

    # 결과 저장 폴더 생성 및 이미지 저장
    # 프로젝트 루트 기준 절대 경로 사용
    base_dir = os.path.dirname(os.path.dirname(__file__))
    out_dir = os.path.join(base_dir, "result", "passenger_counting")
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, img_name), result)

# =========================
# 9. 테스트 데이터셋 실행
# =========================
# result/boat_classification/ 폴더의 보트 유형 분류 결과 이미지에 대해 승선 인원수 탐지 수행
def test_on_dataset():
    # 보트 유형 분류 결과 이미지 경로 사용
    base_dir = os.path.dirname(os.path.dirname(__file__))
    boat_classification_result_dir = os.path.join(base_dir, "result", "boat_classification")
    
    # 결과 폴더 존재 확인
    if not os.path.exists(boat_classification_result_dir):
        print(f"[INFO] 보트 유형 분류 결과 폴더가 없습니다: {boat_classification_result_dir}")
        print("[INFO] 먼저 보트 유형 분류를 실행해주세요.")
        return
    
    # 보트 유형 분류 결과 이미지 파일 찾기 (*_result.jpg)
    image_files = glob.glob(os.path.join(boat_classification_result_dir, "*_result.jpg"))
    if not image_files:
        # 결과 이미지가 없으면 원본 테스트 이미지 사용
        test_path = os.path.join(base_dir, "data", "test")
        if os.path.exists(test_path):
            image_files = glob.glob(os.path.join(test_path, "*.jpg")) + glob.glob(os.path.join(test_path, "*.png"))
            print(f"[INFO] 보트 유형 분류 결과가 없어 원본 이미지를 사용합니다: {len(image_files)}개")
        else:
            print(f"[INFO] 테스트 이미지 폴더가 없습니다: {test_path}")
            return
    else:
        print(f"[INFO] 보트 유형 분류 결과 이미지 사용: {len(image_files)}개")

    # 각 이미지에 대해 처리
    for path in sorted(image_files):
        img = cv2.imread(path)
        if img is None:
            print(f"[WARN] 이미지 로드 실패: {path}")
            continue

        # 사람과 보트 탐지
        persons, boats = detect_people_and_boats(img)
        # 보트별 탑승 인원 매칭
        boats = match_people_to_boats(persons, boats)
        # 결과 시각화 및 저장
        # 파일명에서 _result 제거하여 원본 이름으로 저장
        img_name = os.path.basename(path).replace("_result.jpg", ".jpg").replace("_result.png", ".png")
        visualize_and_save(img, boats, img_name)
        print(f"[INFO] 처리 완료: {img_name} - 보트 {len(boats)}개, 총 탑승 인원 {sum(len(b['persons']) for b in boats)}명")

# =========================
# 10. 메인 실행부
# =========================
if __name__ == "__main__":
    test_on_dataset()

