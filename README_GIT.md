# Git 최종 저장 가이드

## 방법 1: Python 스크립트 사용 (권장)

```bash
python git_commit.py
```

스크립트가 단계별로 진행 상황을 보여주고 확인을 받습니다.

## 방법 2: Windows 배치 파일 사용

```bash
git_commit.bat
```

또는 간단한 버전:

```bash
git_commit_simple.bat
```

## 방법 3: 직접 명령어 실행

```bash
# 1. 변경사항 추가
git add .

# 2. 커밋 생성
git commit -m "최종 정리: 통합 inference 노트북 및 코드 정리

- notebooks/inference.ipynb: 3가지 과업 통합 (보트 유형 분류, 승선 인원수 탐지, 번호판 OCR)
- src/plate_ocr_inference.py: 중복 함수 정의 제거 및 인덴트 오류 수정
- 결과 저장 구조 통일 (result/ 폴더)
- 불필요한 코드 및 빈 셀 제거
- 주석 및 설명 개선"

# 3. 원격 저장소에 푸시
git push
```

## 커밋 메시지 수정

커밋 메시지를 수정하려면 `git_commit.py`, `git_commit.bat`, 또는 `git_commit_simple.bat` 파일의 `COMMIT_MSG` 또는 커밋 메시지 부분을 편집하세요.

