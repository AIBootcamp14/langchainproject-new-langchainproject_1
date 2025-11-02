"""
시나리오 3: 단일 주식 - 보고서 + 저장 (.md)
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
    print("  시나리오 3: 단일 주식 - 보고서 + 저장 (.md)")
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

    print(f"✅ 분석 완료 - 티커: {analysis_result.get('ticker')}, 현재가: ${analysis_result.get('current_price', 0)}\n")

    print("STEP 2: Report Generator 실행 중 (저장 요청 - .md)...")
    generator = ReportGenerator()  # Config.LLM_MODEL 사용
    report_result = generator.generate_report(
        "애플 주식 분석 보고서를 작성하고 저장해주세요",
        analysis_result
    )

    print(f"✅ 보고서 생성 완료")
    print(f"   - 상태: {report_result['status']}")
    print(f"   - 보고서 길이: {len(report_result['report'])} chars")
    print(f"   - 차트 수: {len(report_result['charts'])}")
    print(f"   - 저장 경로: {report_result['saved_path'] or '없음'}\n")

    print("STEP 3: 결과 검증 중...")

    # 보고서 텍스트
    if len(report_result['report']) < 100:
        print(f"❌ FAIL: 보고서가 너무 짧습니다")
    else:
        print(f"✅ PASS: 보고서 생성 ({len(report_result['report'])} chars)")

    # 차트 미생성 확인
    if len(report_result['charts']) > 0:
        print(f"⚠️  WARNING: 차트가 생성되었습니다 (예상: 0개): {report_result['charts']}")
    else:
        print(f"✅ PASS: 차트 미생성 (예상대로)")

    # 파일 저장 확인
    if not report_result['saved_path']:
        print(f"❌ FAIL: 파일이 저장되지 않았습니다")
    else:
        saved_path = report_result['saved_path']
        if os.path.exists(saved_path):
            size = os.path.getsize(saved_path)
            print(f"✅ PASS: 파일 저장됨 - {saved_path} ({size} bytes)")

            if not saved_path.endswith('.md'):
                print(f"⚠️  WARNING: 형식 불일치 (예상: .md, 실제: {saved_path.split('.')[-1]})")
            else:
                print(f"✅ PASS: .md 형식 확인")

            # 파일 내용 일부 출력
            with open(saved_path, 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"\n=== 저장된 파일 내용 미리보기 ===")
                print(content[:300])
                print("...")
        else:
            print(f"❌ FAIL: 파일이 존재하지 않습니다: {saved_path}")

    print(f"\n{'✅ 시나리오 3 통과' if report_result['status'] == 'success' and report_result['saved_path'] else '❌ 시나리오 3 실패'}")
    print("\n" + "="*100 + "\n")

if __name__ == "__main__":
    main()
