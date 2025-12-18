@echo off
chcp 65001 >nul
REM Git 최종 저장 스크립트 (Windows)

echo ==========================================
echo Git 최종 저장 시작
echo ==========================================

REM Git 상태 확인
echo [1/4] Git 상태 확인 중...
git status

REM 변경사항 추가
echo.
echo [2/4] 변경사항 추가 중...
git add .

REM 커밋 메시지
set COMMIT_MSG=최종 정리: 통합 inference 노트북 및 코드 정리

echo.
echo [3/4] 커밋 생성 중...
echo 커밋 메시지:
echo %COMMIT_MSG%
echo.
set /p confirm="커밋을 진행하시겠습니까? (y/n): "

if /i "%confirm%"=="y" (
    git commit -m "%COMMIT_MSG%"
    
    echo.
    echo [4/4] 원격 저장소에 푸시 중...
    set /p push_confirm="원격 저장소에 푸시하시겠습니까? (y/n): "
    
    if /i "!push_confirm!"=="y" (
        git push
        echo.
        echo ✅ 푸시 완료!
    ) else (
        echo.
        echo ⚠️  푸시를 건너뛰었습니다. 나중에 'git push'를 실행하세요.
    )
) else (
    echo.
    echo ⚠️  커밋을 취소했습니다.
)

echo.
echo ==========================================
echo 완료
echo ==========================================
pause

