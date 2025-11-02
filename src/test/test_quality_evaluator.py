# src/test/test_quality_evaluator.py
"""
Quality Evaluator 개별 테스트

에이전트가 생성한 답변의 품질을 평가하는 기능을 테스트합니다.
"""

from src.agents.quality_evaluator import QualityEvaluator
from src.utils.logger import get_logger

logger = get_logger(__name__)


def test_quality_evaluator():
    """Quality Evaluator 단독 테스트"""

    logger.info("=" * 50)
    logger.info("Quality Evaluator 개별 테스트 시작")
    logger.info("=" * 50)

    # QualityEvaluator 초기화
    evaluator = QualityEvaluator()

    # 테스트 케이스 정의
    test_cases = [
        {
            "question": "애플 주식 현재가 알려줘",
            "answer": "Apple Inc. (AAPL)의 현재 주가는 $221.40입니다. 52주 최고가 대비 97.4% 위치에 있으며, P/E 비율은 40.03입니다.",
            "expected_status": "pass",
            "expected_failure_reason": None,
            "description": "좋은 답변 (구체적, 명확)"
        },
        {
            "question": "테슬라 주식 분석해줘",
            "answer": "Tesla의 주가는 현재 상승세를 보이고 있으며, 전기차 시장에서의 입지를 강화하고 있습니다. 최근 분기 실적이 기대치를 상회했으며, 향후 성장 가능성이 높습니다.",
            "expected_status": "pass",
            "expected_failure_reason": None,
            "description": "좋은 답변 (상세한 분석)"
        },
        {
            "question": "삼성전자 주가 알려줘",
            "answer": "",
            "expected_status": "fail",
            "expected_failure_reason": "empty",
            "description": "빈 답변"
        },
        {
            "question": "현대차 주가는?",
            "answer": "주식",
            "expected_status": "fail",
            "expected_failure_reason": "empty",
            "description": "매우 짧은 답변 (10자 미만)"
        },
        {
            "question": "나스닥이 뭐야?",
            "answer": "오류가 발생했습니다. 데이터를 가져올 수 없습니다.",
            "expected_status": "fail",
            "expected_failure_reason": "error",
            "description": "에러 메시지 포함"
        },
        {
            "question": "마이크로소프트 실적 분석해줘",
            "answer": "찾을 수 없음. 다시 시도해주세요.",
            "expected_status": "fail",
            "expected_failure_reason": "error",
            "description": "'찾을 수 없음' 포함"
        }
    ]

    # 테스트 실행
    results = []
    for idx, case in enumerate(test_cases, 1):
        logger.info(f"\n[테스트 {idx}/{len(test_cases)}] {case['description']}")
        logger.info(f"질문: {case['question']}")
        logger.info(f"답변: {case['answer'][:50]}..." if len(case['answer']) > 50 else f"답변: {case['answer']}")

        eval_result = evaluator.evaluate_answer(case["question"], case["answer"])

        status = eval_result.get("status", "unknown")
        score = eval_result.get("score", 0)
        failure_reason = eval_result.get("failure_reason")

        # status와 failure_reason 모두 확인
        status_match = (status == case["expected_status"])
        reason_match = (failure_reason == case["expected_failure_reason"])
        passed = status_match and reason_match

        results.append({
            "case": case,
            "result": eval_result,
            "passed": passed
        })

        pass_fail = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"예상: status={case['expected_status']}, reason={case['expected_failure_reason']}")
        logger.info(f"결과: status={status}, reason={failure_reason}, score={score} - {pass_fail}")

        if not passed:
            logger.error(f"테스트 실패: {case['description']}")
            if not status_match:
                logger.error(f"  Status 불일치: 예상={case['expected_status']}, 실제={status}")
            if not reason_match:
                logger.error(f"  Failure reason 불일치: 예상={case['expected_failure_reason']}, 실제={failure_reason}")

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
                logger.warning(f"- {r['case']['description']}")
                logger.warning(f"  예상: status={r['case']['expected_status']}, reason={r['case']['expected_failure_reason']}")
                logger.warning(f"  실제: status={r['result']['status']}, reason={r['result']['failure_reason']}, score={r['result']['score']}")

    return passed == total


if __name__ == "__main__":
    success = test_quality_evaluator()
    exit(0 if success else 1)
