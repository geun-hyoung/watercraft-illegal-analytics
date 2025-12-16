# Watercraft Illegal Analytics

수상 레저 불법 활동 탐지 및 분석 파이프라인

## 📋 프로젝트 개요

이 프로젝트는 수상 레저 선박의 불법 활동을 자동으로 탐지하고 분석하는 시스템입니다. 주요 기능은 다음과 같습니다:

- **승선 인원 수 탐지**: 보트에 탑승한 인원 수를 자동으로 계산
- **번호판 탐지 및 OCR**: 선박 번호판을 탐지하고 텍스트를 추출하여 정규화

## 🏗️ 프로젝트 구조

```
watercraft-illegal-analytics/
├── src/                          # 소스 코드
│   ├── passenger_counting.py     # 과업 1: 승선 인원 수 탐지
│   ├── plate_detection_train.py  # 과업 2-1: 번호판 탐지 모델 학습
│   └── plate_ocr_inference.py    # 과업 2-2: 번호판 OCR 추출
│
├── notebooks/                     # Jupyter 노트북
│   ├── passenger_counting.ipynb
│   └── plate_ocr_inference.ipynb
│
├── scripts/                       # 유틸리티 스크립트
│   ├── match_image_label.py      # 이미지-라벨 매칭
│   ├── move_data.py              # 데이터 이동
│   └── setup_environment.*        # 환경 설정 스크립트
│
├── data/                          # 데이터 폴더 (Git 제외)
│   ├── raw/                       # 원본 데이터
│   ├── train/                     # 학습 데이터
│   ├── val/                       # 검증 데이터
│   └── test/                      # 테스트 데이터
│
├── model/                         # 학습된 모델 파일 (Git 제외)
│   ├── plate_detection_baseline.pt
│   └── yolo11s_passenger_counting.pt
│
├── runs/                          # 실행 결과 (Git 제외)
│   └── ocr/                       # OCR 결과
│
├── results_onboat/                # 승선 인원 탐지 결과 (Git 제외)
│
├── requirements.txt               # Python 패키지 의존성
└── README.md                      # 프로젝트 문서
```

## 🚀 주요 기능

### 1. 승선 인원 수 탐지 (`passenger_counting.py`)

- YOLOv11s 모델을 사용하여 이미지에서 사람과 보트를 탐지
- 사람의 발 위치가 보트 내부에 있는지 판단하여 탑승 여부 결정
- 각 보트별 탑승 인원 수를 계산하고 시각화

**사용법:**
```bash
python src/passenger_counting.py
```

**입력:** `data/test/plate_detection/` 폴더의 이미지 파일들  
**출력:** `results_onboat/` 폴더에 결과 이미지 저장

### 2. 번호판 탐지 모델 학습 (`plate_detection_train.py`)

- YOLOv11s 사전 학습 모델을 기반으로 번호판 탐지 모델 학습
- 데이터를 train/val로 8:2 분할
- 학습된 모델을 `model/plate_detection_baseline.pt`로 저장

**사용법:**
```bash
python src/plate_detection_train.py
```

**입력:** `data/raw/plate_detection/images/`, `data/raw/plate_detection/labels/`  
**출력:** `model/plate_detection_baseline.pt`

### 3. 번호판 OCR 추출 (`plate_ocr_inference.py`)

- 학습된 번호판 탐지 모델로 번호판 영역 탐지
- EasyOCR을 사용하여 텍스트 추출
- 번호판 형식 정규화 (AA-00-0000 또는 00-AA-0000)
- 다양한 전처리 기법 적용 (회전, 크기 조정, 감마 보정 등)

**사용법:**
```bash
python src/plate_ocr_inference.py
```

**입력:** `data/test/plate_detection/` 폴더의 이미지 파일들  
**출력:** `runs/ocr/OCR_FINAL_1029/` 폴더에 결과 저장
- `ocr_results.csv`: OCR 결과 테이블
- `crops/`: 탐지된 번호판 크롭 이미지
- `overlay/`: 시각화된 결과 이미지

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

## 🔧 주요 의존성

- **ultralytics**: YOLO 모델 학습 및 추론
- **easyocr**: OCR 텍스트 추출
- **opencv-python**: 이미지 처리
- **torch**: 딥러닝 프레임워크
- **pandas**: 데이터 처리
- **numpy**: 수치 연산

전체 패키지 목록은 `requirements.txt`를 참조하세요.

## 📝 사용 예시

### 승선 인원 수 탐지 실행
```bash
python src/passenger_counting.py
```

### 번호판 OCR 추출 실행
```bash
python src/plate_ocr_inference.py
```

## 📊 결과 형식

### 승선 인원 탐지 결과
- 각 이미지에 보트별 탑승 인원 수가 시각화된 이미지가 `results_onboat/` 폴더에 저장됩니다.

### OCR 결과
- `ocr_results.csv`: 이미지별 번호판 텍스트 추출 결과
- `crops/`: 탐지된 번호판 영역 크롭 이미지
- `overlay/`: 원본 이미지에 탐지 결과를 오버레이한 시각화 이미지

## 🔒 Git 제외 항목

다음 폴더/파일은 Git에 추적되지 않습니다:
- `data/`: 데이터 파일
- `model/`: 학습된 모델 파일
- `runs/`: 실행 결과
- `results_onboat/`: 승선 인원 탐지 결과
- `scripts/`: 유틸리티 스크립트
- `venv/`: 가상 환경
- `__pycache__/`: Python 캐시 파일

## 📄 라이선스

이 프로젝트의 라이선스 정보는 저장소를 확인하세요.

## 👥 기여자

- [geun-hyoung](https://github.com/geun-hyoung)
