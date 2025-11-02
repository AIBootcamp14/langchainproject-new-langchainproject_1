# src/test/test_request_analyst.py
"""
Request Analyst 개별 테스트

사용자 질문이 경제/금융 관련인지 판별하는 기능을 테스트합니다.
chat_history 기능과 rewrite_query 기능도 함께 테스트합니다.
"""

from src.agents.request_analyst import request_analysis, rewrite_query
from src.utils.logger import get_logger

logger = get_logger(__name__)


def test_request_analyst():
    """Request Analyst 단독 테스트"""

    logger.info("=" * 50)
    logger.info("Request Analyst 개별 테스트 시작")
    logger.info("=" * 50)

    # 테스트 케이스 정의
    test_cases = [
        {
            "question": "오늘 날씨 어때?",
            "expected": "not_finance",
            "description": "비금융 질문 (날씨)"
        },
        {
            "question": "AI가 뭐야?",
            "expected": "not_finance",
            "description": "비금융 질문 (일반 IT)"
        },
        {
            "question": "AI 시장 투자 규모가 어떻게 돼?",
            "expected": "finance",
            "description": "금융 질문 (투자/시장)"
        },
        {
            "question": "애플 주식 분석해줘",
            "expected": "finance",
            "description": "금융 질문 (주식 분석)"
        },
        {
            "question": "삼성전자와 LG전자 실적 비교해줘",
            "expected": "finance",
            "description": "금융 질문 (기업 비교)"
        }
    ]

    # 테스트 실행
    results = []
    for idx, case in enumerate(test_cases, 1):
        logger.info(f"\n[테스트 {idx}/{len(test_cases)}] {case['description']}")
        logger.info(f"질문: {case['question']}")

        state = {"question": case["question"]}
        result = request_analysis(state)

        label = result.get("label", "unknown")
        passed = label == case["expected"]

        results.append({
            "case": case,
            "result": result,
            "passed": passed
        })

        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"예상: {case['expected']}, 결과: {label} - {status}")

        if not passed:
            logger.error(f"테스트 실패: {case['description']}")
            logger.error(f"전체 응답: {result}")

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

    return passed == total


def test_chat_history_support():
    """chat_history 파라미터 지원 테스트"""

    logger.info("\n" + "=" * 50)
    logger.info("Chat History 지원 테스트 시작")
    logger.info("=" * 50)

    # 가상의 chat_history 생성
    chat_history = [
        {"role": "user", "content": "애플 주식 현재가 알려줘"},
        {"role": "assistant", "content": "Apple Inc. (AAPL)의 현재 주가는 $221.40입니다."},
        {"role": "user", "content": "테슬라는?"}
    ]

    state = {"question": "그 회사 실적은 어때?"}  # 모호한 질문

    # chat_history와 함께 호출
    result = request_analysis(state, chat_history=chat_history)

    label = result.get("label", "unknown")
    passed = label in ["finance", "not_finance"]  # 정상적으로 실행되었는지만 확인

    logger.info(f"질문: {state['question']}")
    logger.info(f"chat_history: {len(chat_history)}개 메시지")
    logger.info(f"결과: {label}")

    status = "✅ PASS" if passed else "❌ FAIL"
    logger.info(f"테스트 결과: {status}")

    return passed


def test_rewrite_query():
    """rewrite_query 함수 테스트"""

    logger.info("\n" + "=" * 50)
    logger.info("Rewrite Query 테스트 시작")
    logger.info("=" * 50)

    test_cases = [
        {
            "original_query": "주식",
            "failure_reason": "incorrect",
            "expected_needs_input": True,  # 너무 짧아서 유저 입력 필요
            "description": "짧은 쿼리 (유저 입력 필요)"
        },
        {
            "original_query": "애플 주식 현재가 알려줘",
            "failure_reason": "incorrect",
            "expected_needs_input": False,  # 충분히 명확, 재시도
            "description": "명확한 쿼리 (재시도)"
        }
    ]

    results = []
    for idx, case in enumerate(test_cases, 1):
        logger.info(f"\n[테스트 {idx}/{len(test_cases)}] {case['description']}")
        logger.info(f"원본 쿼리: {case['original_query']}")
        logger.info(f"실패 이유: {case['failure_reason']}")

        result = rewrite_query(
            original_query=case["original_query"],
            failure_reason=case["failure_reason"]
        )

        needs_input = result.get("needs_user_input", False)
        rewritten = result.get("rewritten_query", "")
        request_for_detail_msg = result.get("request_for_detail_msg")

        passed = needs_input == case["expected_needs_input"]

        results.append({"case": case, "result": result, "passed": passed})

        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"예상 needs_user_input: {case['expected_needs_input']}")
        logger.info(f"실제 needs_user_input: {needs_input}")
        logger.info(f"재작성 쿼리: {rewritten}")
        if request_for_detail_msg:
            logger.info(f"유저에게 디테일한 질문 재요청: {request_for_detail_msg}")
        logger.info(f"결과: {status}")

    # 결과 요약
    logger.info("\n" + "=" * 50)
    logger.info("Rewrite Query 테스트 결과 요약")
    logger.info("=" * 50)

    total = len(results)
    passed_count = sum(1 for r in results if r["passed"])
    failed = total - passed_count

    logger.info(f"전체: {total}개")
    logger.info(f"성공: {passed_count}개")
    logger.info(f"실패: {failed}개")

    return passed_count == total


if __name__ == "__main__":
    # 기본 request_analysis 테스트
    success1 = test_request_analyst()

    # chat_history 지원 테스트
    success2 = test_chat_history_support()

    # rewrite_query 테스트
    success3 = test_rewrite_query()

    # 전체 성공 여부
    overall_success = success1 and success2 and success3

    logger.info("\n" + "=" * 50)
    logger.info("전체 테스트 결과")
    logger.info("=" * 50)
    logger.info(f"request_analysis: {'✅ PASS' if success1 else '❌ FAIL'}")
    logger.info(f"chat_history 지원: {'✅ PASS' if success2 else '❌ FAIL'}")
    logger.info(f"rewrite_query: {'✅ PASS' if success3 else '❌ FAIL'}")
    logger.info(f"전체 결과: {'✅ 모두 통과' if overall_success else '❌ 일부 실패'}")

    exit(0 if overall_success else 1)
