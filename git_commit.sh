#!/bin/bash
# Git 최종 저장 스크립트

echo "=========================================="
echo "Git 최종 저장 시작"
echo "=========================================="

# Git 상태 확인
echo "[1/4] Git 상태 확인 중..."
git status

# 변경사항 추가
echo ""
echo "[2/4] 변경사항 추가 중..."
git add .

# 커밋 메시지
COMMIT_MSG="최종 정리: 통합 inference 노트북 및 코드 정리

- notebooks/inference.ipynb: 3가지 과업 통합 (보트 유형 분류, 승선 인원수 탐지, 번호판 OCR)
- src/plate_ocr_inference.py: 중복 함수 정의 제거 및 인덴트 오류 수정
- 결과 저장 구조 통일 (result/ 폴더)
- 불필요한 코드 및 빈 셀 제거
- 주석 및 설명 개선"

echo ""
echo "[3/4] 커밋 생성 중..."
echo "커밋 메시지:"
echo "$COMMIT_MSG"
echo ""
read -p "커밋을 진행하시겠습니까? (y/n): " confirm

if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
    git commit -m "$COMMIT_MSG"
    
    echo ""
    echo "[4/4] 원격 저장소에 푸시 중..."
    read -p "원격 저장소에 푸시하시겠습니까? (y/n): " push_confirm
    
    if [ "$push_confirm" = "y" ] || [ "$push_confirm" = "Y" ]; then
        git push
        echo ""
        echo "✅ 푸시 완료!"
    else
        echo ""
        echo "⚠️  푸시를 건너뛰었습니다. 나중에 'git push'를 실행하세요."
    fi
else
    echo ""
    echo "⚠️  커밋을 취소했습니다."
fi

echo ""
echo "=========================================="
echo "완료"
echo "=========================================="

