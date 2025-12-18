# Watercraft Illegal Analytics

수상 레저 불법 활동 탐지 및 분석 파이프라인

## 📋 프로젝트 개요

이 프로젝트는 수상 레저 선박의 불법 활동을 자동으로 탐지하고 분석하는 시스템입니다. 3가지 주요 과업을 수행합니다:

1. **보트 유형 분류**: 보트를 탐지하고 유형을 분류 (모터보트, 수상오토바이, 고무보트, 세일링요트, 기타)
2. **승선 인원 수 탐지**: 보트에 탑승한 인원 수를 자동으로 계산
3. **번호판 OCR**: 선박 번호판을 탐지하고 텍스트를 추출하여 정규화

## 🏗️ 프로젝트 구조

```
watercraft-illegal-analytics/
├── src/                              # 소스 코드
│   ├── boat_classification_train.py      # 보트 유형 분류 모델 학습
│   ├── boat_classification_inference.py  # 보트 유형 분류 추론
│   ├── passenger_counting.py             # 승선 인원 수 탐지
│   ├── plate_detection_train.py          # 번호판 탐지 모델 학습
│   └── plate_ocr_inference.py            # 번호판 OCR 추출
│
├── notebooks/                        # Jupyter 노트북
│   └── inference.ipynb              # 통합 추론 노트북 (3가지 과업 수행)
│
├── data/                            # 데이터 폴더
│   ├── raw/                         # 원본 데이터
│   │   ├── classification/          # 보트 유형 분류 데이터
│   │   └── plate_detection/         # 번호판 탐지 데이터
│   └── test/                        # 테스트 이미지 (공통)
│
├── model/                           # 학습된 모델 파일
│   ├── boat_classification_baseline.pt
│   ├── plate_detection_baseline.pt
│   └── yolo11s_passenger_counting.pt
│
└── result/                          # 추론 결과
    ├── boat_classification/         # 보트 유형 분류 결과
    ├── passenger_counting/          # 승선 인원 수 탐지 결과
    └── plate_ocr/                   # 번호판 OCR 결과
```

## 🚀 주요 기능

### 1. 보트 유형 분류

**학습:**
```bash
python src/boat_classification_train.py
```
- 입력: `data/raw/classification/images/`, `data/raw/classification/labels/`
- 출력: `model/boat_classification_baseline.pt`

**추론:**
```bash
python src/boat_classification_inference.py
```
- 입력: `data/test/` 폴더의 이미지 파일들
- 출력: `result/boat_classification/` (CSV + 시각화 이미지)

### 2. 승선 인원 수 탐지

```bash
python src/passenger_counting.py
```
- 입력: `result/boat_classification/*_result.jpg` (보트 유형 분류 결과 이미지)
- 출력: `result/passenger_counting/` (시각화 이미지)

### 3. 번호판 탐지 및 OCR

**학습:**
```bash
python src/plate_detection_train.py
```
- 입력: `data/raw/plate_detection/images/`, `data/raw/plate_detection/labels/`
- 출력: `model/plate_detection_baseline.pt`

**OCR 추론:**
```bash
python src/plate_ocr_inference.py
```
- 입력: `data/test/` 폴더의 이미지 파일들
- 출력: `result/plate_ocr/` (CSV + 크롭 이미지 + 오버레이 이미지)

## 📦 설치 방법

### 1. 저장소 클론
```bash
git clone https://github.com/geun-hyoung/watercraft-illegal-analytics.git
cd watercraft-illegal-analytics
```

### 2. 가상 환경 생성 및 활성화
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. 패키지 설치
```bash
pip install -r requirements.txt
```

## 📝 통합 추론 실행

모든 과업을 한 번에 실행하려면 Jupyter 노트북을 사용하세요:

```bash
jupyter notebook notebooks/inference.ipynb
```

노트북에서는 다음 순서로 실행됩니다:
1. 보트 유형 분류 → `result/boat_classification/`
2. 승선 인원 수 탐지 → `result/passenger_counting/` (보트 유형 분류 결과 사용)
3. 번호판 OCR → `result/plate_ocr/`
4. 결과 시각화 및 샘플 확인

## 🔧 주요 의존성

- **ultralytics**: YOLO 모델 학습 및 추론
- **easyocr**: OCR 텍스트 추출
- **opencv-python**: 이미지 처리
- **torch**: 딥러닝 프레임워크
- **pandas**: 데이터 처리
- **numpy**: 수치 연산
- **Pillow**: 이미지 처리 (한글 표시)

전체 패키지 목록은 `requirements.txt`를 참조하세요.

## 📊 결과 형식

### 보트 유형 분류
- `classification_results.csv`: 이미지별 보트 탐지 및 분류 결과
- `*_result.jpg`: 바운딩 박스와 클래스명이 표시된 시각화 이미지

### 승선 인원 탐지
- 각 이미지에 보트별 탑승 인원 수가 시각화된 이미지

### 번호판 OCR
- `ocr_results.csv`: 이미지별 번호판 텍스트 추출 결과
- `crops/`: 탐지된 번호판 영역 크롭 이미지
- `overlay/`: 원본 이미지에 탐지 결과를 오버레이한 시각화 이미지

## 🔒 Git 제외 항목

다음 폴더/파일은 Git에 추적되지 않습니다:
- `data/`: 데이터 파일
- `model/`: 학습된 모델 파일
- `result/`: 추론 결과
- `runs/`: 학습 실행 결과
- `venv/`: 가상 환경
- `__pycache__/`: Python 캐시 파일
