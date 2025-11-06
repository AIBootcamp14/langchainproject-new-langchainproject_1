# src/rag/retriever.py
"""
Retriever Module

간단한 유사도 검색기
"""

from typing import List, Optional

from langchain_core.documents import Document

from src.rag.vector_store import VectorStore
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Retriever:
    """간단한 유사도 검색기"""

    def __init__(self, store: VectorStore | None = None, k: int = None, threshold: float = None):
        """
        Retriever 초기화

        Args:
            store: VectorStore 인스턴스 (None이면 새로 생성)
            k: 검색할 문서 개수 (None이면 Config에서 가져옴)
            threshold: 최소 유사도 점수 (None이면 Config에서 가져옴)
        """
        self.store = store or VectorStore()
        self.k = k if k is not None else Config.DEFAULT_RETRIEVAL_TOP_K
        self.threshold = threshold if threshold is not None else Config.RETRIEVAL_THRESHOLD

        logger.info(f"Retriever 초기화 - top_k: {self.k}, threshold: {self.threshold}")

    def retrieve(self, query: str, k: Optional[int] = None, threshold: float = None) -> List[Document]:
        """
        질문과 가장 가까운 문서와 점수를 찾아 반환합니다.

        Args:
            query: 검색 쿼리
            k: 검색할 문서 개수 (None이면 기본값 사용)
            threshold: 최소 유사도 점수 (None이면 기본값 사용)

        Returns:
            (Document, score) 튜플의 리스트
        """
        top_k = k if k is not None else self.k
        min_threshold = threshold if threshold is not None else self.threshold

        logger.info(f"검색 시작 - 쿼리: '{query}', top_k: {top_k}, threshold: {min_threshold}")

        candidates = self.store.retrieve_with_scores(query, k=top_k)
        filtered_results = [(doc, score) for doc, score in candidates if score >= min_threshold]

        logger.info(f"검색 완료 - {len(filtered_results)}개 문서 발견 (threshold {min_threshold} 이상)")

        return filtered_results


def test(query: str | None = None, top_k: int = None):
    """
    Retriever 테스트 함수

    Args:
        query: 검색 쿼리 (None이면 명령행 인자 사용)
        top_k: 검색할 문서 개수 (None이면 명령행 인자 사용)
    """
    if query is None:
        import argparse

        parser = argparse.ArgumentParser(description="간단한 PDF 벡터 검색 CLI")
        parser.add_argument("query", help="찾고 싶은 질문/키워드")
        parser.add_argument("--top-k", type=int, default=3, help="가져올 문서 수")
        args = parser.parse_args()
        query, top_k = args.query, args.top_k

    if top_k is None:
        top_k = 3

    logger.info(f"테스트 시작 - 쿼리: '{query}', top_k: {top_k}")

    retriever = Retriever(k=top_k)
    results = retriever.retrieve(query)

    if not results:
        print("XXX 결과가 없습니다. XXX")
        logger.warning("검색 결과 없음")
        return

    print("=" * 30)
    print(f"'{query}' 관련 문서 {len(results)}개")
    logger.info(f"검색 결과: {len(results)}개 문서")

    for idx, (doc, score) in enumerate(results, start=1):
        page = doc.metadata.get("page", "?")
        preview = doc.page_content.strip().replace("\n", " ")
        if len(preview) > 200:
            preview = preview[:200] + "..."

        print(f"\n[{idx}] page {page} | score={score:.4f}")
        print(preview)
        logger.debug(f"문서 {idx}: page={page}, score={score:.4f}, preview={preview[:50]}...")


if __name__ == "__main__":
    test("모바일로 주식 거래하는 앱은 뭐라고 하나요?")
    # test("듀레이션이라는게 뭔가요?")
    # test("ㅁㄴㅇㄹㄷㄴ")