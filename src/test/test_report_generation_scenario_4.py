"""
시나리오 4: 단일 주식 - 보고서 + 차트 + 저장 (.md)
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
    print("  시나리오 4: 단일 주식 - 보고서 + 차트 + 저장 (.md)")
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

    print("STEP 2: Report Generator 실행 중 (차트 + 저장 .md)...")
    generator = ReportGenerator()  # Config.LLM_MODEL 사용
    report_result = generator.generate_report(
        "애플 주식 분석 보고서를 차트와 함께 작성하고 저장해주세요",
        analysis_result
    )

    print(f"✅ 보고서 생성 완료")
    print(f"   - 차트 수: {len(report_result['charts'])}")
    print(f"   - 저장 경로: {report_result['saved_path'] or '없음'}\n")

    print("STEP 3: 결과 검증...")

    # 차트 생성 확인
    if len(report_result['charts']) == 0:
        print(f"❌ FAIL: 차트가 생성되지 않았습니다")
    else:
        print(f"✅ PASS: 차트 생성 ({len(report_result['charts'])}개)")
        for chart in report_result['charts']:
            if os.path.exists(chart):
                print(f"   - {chart}: {os.path.getsize(chart)} bytes ✅")

    # 파일 저장 + .md 형식 확인
    if report_result['saved_path'] and os.path.exists(report_result['saved_path']):
        if report_result['saved_path'].endswith('.md'):
            print(f"✅ PASS: .md 파일 저장 - {report_result['saved_path']}")
        else:
            print(f"⚠️  WARNING: 형식 불일치")
    else:
        print(f"❌ FAIL: 파일이 저장되지 않았습니다")

    print(f"\n{'✅ 시나리오 4 통과' if len(report_result['charts']) > 0 and report_result['saved_path'] else '❌ 시나리오 4 실패'}\n")

if __name__ == "__main__":
    main()
