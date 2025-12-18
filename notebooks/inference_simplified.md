# inference.ipynb 간소화 가이드

## 변경 사항

노트북에서 직접 구현한 코드를 제거하고, `src` 폴더의 코드를 사용하도록 변경합니다.

### 1. 승선 인원수 탐지 (Cell 4-5)

**변경 전:** 노트북에 모든 함수 구현
**변경 후:** `src/passenger_counting.py`의 함수 import

```python
# Cell 4
from passenger_counting import (
    detect_people_and_boats,
    match_people_to_boats
)

def visualize_and_save_passenger(img, boats, img_name, output_dir):
    # 결과 저장만 노트북에서 처리
    ...

# Cell 5
# 승선 인원수 탐지 실행 (기존과 동일)
```

### 2. 번호판 OCR (Cell 7-9)

**변경 전:** 노트북에 모든 OCR 함수 구현
**변경 후:** `src/plate_ocr_inference.py`를 subprocess로 실행

```python
# Cell 7
import subprocess
import shutil

# Cell 8
# plate_ocr_inference.py 실행
result = subprocess.run(
    [sys.executable, str(BASE_DIR / "src" / "plate_ocr_inference.py")],
    ...
)

# 결과를 result/plate_ocr/로 복사
```

### 3. 보트 유형 분류 (Cell 10-11)

**변경 전:** 노트북에 모든 분류 함수 구현
**변경 후:** `src/boat_classification_inference.py`를 subprocess로 실행

```python
# Cell 10
# boat_classification_inference.py 실행
result = subprocess.run(
    [sys.executable, str(BASE_DIR / "src" / "boat_classification_inference.py")],
    ...
)
```

### 4. 결과 확인 및 샘플 시각화 (Cell 13-15)

기존과 동일하게 유지하되, 결과 파일 경로만 확인

## 주요 장점

1. **코드 중복 제거**: 노트북에 중복 구현된 코드 제거
2. **유지보수 용이**: `src` 폴더의 코드만 수정하면 됨
3. **일관성**: 모든 inference가 동일한 방식으로 실행됨
4. **간결성**: 노트북이 더 간결하고 읽기 쉬워짐

