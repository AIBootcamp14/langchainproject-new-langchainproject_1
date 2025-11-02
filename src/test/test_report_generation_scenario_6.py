"""
시나리오 6: 단일 주식 - 보고서 + 저장 (.txt)
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.agents.financial_analyst import FinancialAnalyst
from src.agents.report_generator import ReportGenerator
from src.utils.config import Config

os.makedirs("charts", exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("logs", exist_ok=True)

def main():
    print("\n" + "="*100)
    print("  시나리오 6: 단일 주식 - 보고서 + 저장 (.txt)")
    print("="*100 + "\n")

    try:
        Config.validate_api_keys()
        print("✅ API 키 검증 완료\n")
    except ValueError as e:
        print(f"❌ API 키 오류: {str(e)}")
        return

    print("STEP 1: Financial Analyst 실행 중...")
    analyst = FinancialAnalyst()  # Config.LLM_MODEL 사용
    analysis_result = analyst.analyze("애플 주식을 분석해주세요")

    print(f"✅ 분석 완료\n")

    print("STEP 2: Report Generator 실행 중 (저장 .txt)...")
    generator = ReportGenerator()  # Config.LLM_MODEL 사용
    report_result = generator.generate_report(
        "애플 주식 분석 보고서를 txt 파일로 저장해주세요",
        analysis_result
    )

    print(f"✅ 보고서 생성 완료")
    print(f"   - 차트 수: {len(report_result['charts'])}")
    print(f"   - 저장 경로: {report_result['saved_path'] or '없음'}\n")

    print("STEP 3: 결과 검증...")

    # 차트 미생성 확인
    if len(report_result['charts']) > 0:
        print(f"⚠️  WARNING: 차트가 생성되었습니다 (예상: 0개)")
    else:
        print(f"✅ PASS: 차트 미생성")

    # .txt 파일 저장 확인
    if report_result['saved_path'] and os.path.exists(report_result['saved_path']):
        if report_result['saved_path'].endswith('.txt'):
            size = os.path.getsize(report_result['saved_path'])
            print(f"✅ PASS: .txt 파일 저장 - {report_result['saved_path']} ({size} bytes)")
        else:
            print(f"⚠️  WARNING: .txt가 아닌 형식")
    else:
        print(f"❌ FAIL: 파일이 저장되지 않았습니다")

    print(f"\n{'✅ 시나리오 6 통과' if report_result['saved_path'] and report_result['saved_path'].endswith('.txt') else '❌ 시나리오 6 실패'}\n")

if __name__ == "__main__":
    main()
