# src/test/test_retriever.py
"""
Retriever 개별 테스트

ChromaDB에서 금융 용어 문서를 검색하는 기능을 테스트합니다.
"""

from src.rag.retriever import Retriever
from src.utils.logger import get_logger

logger = get_logger(__name__)


def test_retriever():
    """Retriever 단독 테스트"""

    logger.info("=" * 50)
    logger.info("Retriever 개별 테스트 시작")
    logger.info("=" * 50)

    # Retriever 초기화
    try:
        retriever = Retriever(k=3, threshold=0.3)
        logger.info("Retriever 초기화 성공")
    except Exception as e:
        logger.error(f"Retriever 초기화 실패: {e}")
        return False

    # 테스트 케이스 정의
    test_cases = [
        {
            "query": "모바일로 주식 거래하는 앱은 뭐라고 하나요?",
            "min_results": 1,
            "description": "주식 거래 앱 검색"
        },
        {
            "query": "듀레이션이란?",
            "min_results": 1,
            "description": "금융 용어 정의 검색"
        },
        {
            "query": "나스닥이 뭐야?",
            "min_results": 1,
            "description": "시장 용어 검색"
        }
    ]

    # 테스트 실행
    results = []
    for idx, case in enumerate(test_cases, 1):
        logger.info(f"\n[테스트 {idx}/{len(test_cases)}] {case['description']}")
        logger.info(f"질문: {case['query']}")

        try:
            search_results = retriever.retrieve(case["query"])

            num_results = len(search_results)
            passed = num_results >= case["min_results"]

            results.append({
                "case": case,
                "num_results": num_results,
                "passed": passed
            })

            status = "✅ PASS" if passed else "❌ FAIL"
            logger.info(f"검색 결과: {num_results}개 문서 발견 (최소 {case['min_results']}개 필요) - {status}")

            if passed and num_results > 0:
                # 첫 번째 결과의 미리보기 출력
                doc, score = search_results[0]
                preview = doc.page_content.strip().replace("\n", " ")[:100]
                logger.info(f"상위 결과 (유사도: {score:.4f}): {preview}...")

        except Exception as e:
            logger.error(f"검색 중 오류 발생: {e}", exc_info=True)
            results.append({
                "case": case,
                "num_results": 0,
                "passed": False,
                "error": str(e)
            })

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
                logger.warning(f"- {r['case']['description']}: {r['case']['query']}")
                if "error" in r:
                    logger.warning(f"  오류: {r['error']}")

    return passed == total


if __name__ == "__main__":
    success = test_retriever()
    exit(0 if success else 1)
