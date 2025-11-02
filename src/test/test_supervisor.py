# src/test/test_supervisor.py
"""
Supervisor 개별 테스트

금융 도메인 질문을 처리할 적합한 에이전트를 선택하는 기능을 테스트합니다.
"""

from src.agents.supervisor import supervisor
from src.utils.logger import get_logger

logger = get_logger(__name__)


def test_supervisor():
    """Supervisor 단독 테스트"""

    logger.info("=" * 50)
    logger.info("Supervisor 개별 테스트 시작")
    logger.info("=" * 50)

    # 테스트 케이스 정의
    test_cases = [
        {
            "question": "나스닥이 뭐야?",
            "expected": "vector_search_agent",
            "description": "금융 용어 검색 (RAG)"
        },
        {
            "question": "듀레이션이란?",
            "expected": "vector_search_agent",
            "description": "금융 용어 정의 (RAG)"
        },
        {
            "question": "애플 주식 현재가 알려줘",
            "expected": "financial_analyst",
            "description": "주식 정보 조회"
        },
        {
            "question": "삼성전자와 LG전자 실적 비교해줘",
            "expected": "financial_analyst",
            "description": "기업 비교 분석"
        },
        {
            "question": "AAPL 종목코드 찾아줘",
            "expected": "financial_analyst",
            "description": "종목코드 검색"
        },
        {
            "question": "테슬라 재무제표 보여줘",
            "expected": "financial_analyst",
            "description": "재무제표 조회"
        }
    ]

    # 테스트 실행
    results = []
    for idx, case in enumerate(test_cases, 1):
        logger.info(f"\n[테스트 {idx}/{len(test_cases)}] {case['description']}")
        logger.info(f"질문: {case['question']}")

        state = {"question": case["question"]}
        result = supervisor(state)

        passed = result == case["expected"]

        results.append({
            "case": case,
            "result": result,
            "passed": passed
        })

        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"예상: {case['expected']}, 결과: {result} - {status}")

        if not passed:
            logger.error(f"테스트 실패: {case['description']}")

    # 결과 요약
    logger.info("\n" + "=" * 50)
    logger.info("테스트 결과 요약")
    logger.info("=" * 50)

    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed

    logger.info(f"전체: {total}개")
    logger.info(f"성공: {passed}개")
    logger.info(f"실패: {failed}개")
    logger.info(f"성공률: {(passed/total*100):.1f}%")

    if failed > 0:
        logger.warning("\n실패한 테스트:")
        for r in results:
            if not r["passed"]:
                logger.warning(f"- {r['case']['description']}: {r['case']['question']}")
                logger.warning(f"  예상: {r['case']['expected']}, 실제: {r['result']}")

    return passed == total


if __name__ == "__main__":
    success = test_supervisor()
    exit(0 if success else 1)
