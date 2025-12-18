@echo off
chcp 65001 >nul
REM Git 최종 저장 스크립트 (Windows - 간단 버전)

echo ==========================================
echo Git 최종 저장
echo ==========================================

echo.
echo 변경사항 추가 중...
git add .

echo.
echo 커밋 생성 중...
git commit -m "최종 정리: 통합 inference 노트북 및 코드 정리

- notebooks/inference.ipynb: 3가지 과업 통합 (보트 유형 분류, 승선 인원수 탐지, 번호판 OCR)
- src/plate_ocr_inference.py: 중복 함수 정의 제거 및 인덴트 오류 수정
- 결과 저장 구조 통일 (result/ 폴더)
- 불필요한 코드 및 빈 셀 제거
- 주석 및 설명 개선"

echo.
echo 원격 저장소에 푸시 중...
git push

echo.
echo ==========================================
echo 완료
echo ==========================================
pause

