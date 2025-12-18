#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Git 최종 저장 스크립트
"""
import subprocess
import sys
from pathlib import Path

def run_command(cmd, check=True):
    """명령어 실행"""
    try:
        result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True, encoding='utf-8')
        if result.stdout:
            print(result.stdout)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"오류 발생: {e}")
        if e.stderr:
            print(e.stderr)
        return False

def main():
    print("=" * 50)
    print("Git 최종 저장 시작")
    print("=" * 50)
    
    # Git 상태 확인
    print("\n[1/4] Git 상태 확인 중...")
    run_command("git status", check=False)
    
    # 변경사항 추가
    print("\n[2/4] 변경사항 추가 중...")
    if not run_command("git add ."):
        print("❌ git add 실패")
        return
    
    # 커밋 메시지
    commit_msg = """최종 정리: 통합 inference 노트북 및 코드 정리

- notebooks/inference.ipynb: 3가지 과업 통합 (보트 유형 분류, 승선 인원수 탐지, 번호판 OCR)
- src/plate_ocr_inference.py: 중복 함수 정의 제거 및 인덴트 오류 수정
- 결과 저장 구조 통일 (result/ 폴더)
- 불필요한 코드 및 빈 셀 제거
- 주석 및 설명 개선"""
    
    print("\n[3/4] 커밋 생성 중...")
    print("커밋 메시지:")
    print(commit_msg)
    print()
    
    confirm = input("커밋을 진행하시겠습니까? (y/n): ").strip().lower()
    if confirm != 'y':
        print("\n⚠️  커밋을 취소했습니다.")
        return
    
    # 커밋 실행
    if not run_command(f'git commit -m "{commit_msg}"'):
        print("❌ git commit 실패")
        return
    
    print("\n[4/4] 원격 저장소에 푸시 중...")
    push_confirm = input("원격 저장소에 푸시하시겠습니까? (y/n): ").strip().lower()
    
    if push_confirm == 'y':
        if not run_command("git push"):
            print("❌ git push 실패")
            return
        print("\n✅ 푸시 완료!")
    else:
        print("\n⚠️  푸시를 건너뛰었습니다. 나중에 'git push'를 실행하세요.")
    
    print("\n" + "=" * 50)
    print("완료")
    print("=" * 50)

if __name__ == "__main__":
    main()

