"""
시나리오 1: 단일 주식 - 기본 보고서만 (차트 X, 저장 X)
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.agents.financial_analyst import FinancialAnalyst
from src.agents.report_generator import ReportGenerator
from src.utils.config import Config

# 출력 디렉토리 생성
os.makedirs("charts", exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("logs", exist_ok=True)

def main():
    print("\n" + "="*100)
    print("  시나리오 1: 단일 주식 - 기본 보고서만 (차트 X, 저장 X)")
    print("="*100 + "\n")

    # API 키 확인
    try:
        Config.validate_api_keys()
        print("✅ API 키 검증 완료\n")
    except ValueError as e:
        print(f"❌ API 키 오류: {str(e)}")
        return

    # Step 1: Financial Analysis
    print("STEP 1: Financial Analyst 실행 중...")
    analyst = FinancialAnalyst()  # Config.LLM_MODEL 사용
    analysis_result = analyst.analyze("애플 주식을 분석해주세요")

    print(f"✅ 분석 완료")
    print(f"   - 분석 타입: {analysis_result.get('analysis_type', 'N/A')}")
    print(f"   - 티커: {analysis_result.get('ticker', 'N/A')}")
    print(f"   - 회사명: {analysis_result.get('company_name', 'N/A')}")
    print(f"   - 현재가: ${analysis_result.get('current_price', 0)}")
    print()

    # Step 2: Report Generation (차트 X, 저장 X)
    print("STEP 2: Report Generator 실행 중...")
    generator = ReportGenerator()  # Config.LLM_MODEL 사용
    report_result = generator.generate_report(
        "애플 주식 분석 보고서를 작성해주세요",
        analysis_result
    )

    print(f"✅ 보고서 생성 완료")
    print(f"   - 상태: {report_result['status']}")
    print(f"   - 보고서 길이: {len(report_result['report'])} chars")
    print(f"   - 생성된 차트 수: {len(report_result['charts'])}")
    print(f"   - 저장된 파일: {report_result['saved_path'] or '없음'}")
    print()

    # Step 3: Verification
    print("STEP 3: 결과 검증 중...")

    # 보고서 텍스트 확인
    report_text = report_result['report']
    if len(report_text) < 100:
        print(f"❌ FAIL: 보고서 텍스트가 너무 짧습니다 ({len(report_text)} chars)")
    else:
        print(f"✅ PASS: 보고서 텍스트 생성 ({len(report_text)} chars)")
        print(f"\n=== 전체 보고서 내용 ===")
        print("-" * 100)
        print(report_text)
        print("-" * 100)

    # 차트 미생성 확인
    if len(report_result['charts']) > 0:
        print(f"⚠️  WARNING: 차트 생성이 예상되지 않았지만 생성되었습니다: {report_result['charts']}")
    else:
        print(f"✅ PASS: 차트 미생성 (예상대로)")

    # 파일 미저장 확인
    if report_result['saved_path']:
        print(f"⚠️  WARNING: 파일 저장이 예상되지 않았지만 저장되었습니다: {report_result['saved_path']}")
    else:
        print(f"✅ PASS: 파일 미저장 (예상대로)")

    print(f"\n{'✅ 시나리오 1 통과' if report_result['status'] == 'success' else '❌ 시나리오 1 실패'}")
    print("\n" + "="*100 + "\n")

if __name__ == "__main__":
    main()
