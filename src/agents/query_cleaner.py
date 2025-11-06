# src/agents/query_cleaner.py
"""
Query Cleaner Module

사용자의 질문에 대한 문맥을 파악하고 오탈자/잘못된 정보 전달 등을 수정해 정확한 쿼리를 작성하기 위한 쿼리 재작성기입니다.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

from src.model.llm import get_llm_manager
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CleanQuery(BaseModel):
    """재작성된 쿼리 결과"""
    rewritten_query: str = Field(description="질문의 의도를 유지하면서 다른 표현으로 재작성된 사용자 질문")


def query_cleaner(state: Dict[str, Any], llm=None) -> Dict[str, str]:
    """
    사용자의 query를 대화 히스토리를 참조하여 문맥을 파악하고,
    오탈자 수정, 한글 자판 오류 수정, 짧은 응답 확장 등을 통해 명확한 쿼리로 재작성합니다.

    Structured Output(CleanQuery 모델)을 사용하여 일관된 형식으로 반환합니다.

    Args:
        state (dict): 현재 graph 상태 (question & messages 필드 포함)
        llm: LLM 모델 (선택사항, 없으면 기본 모델 사용)

    Returns:
        dict: rewritten_query 단일 필드를 포함한 딕셔너리
    """
    logger.info("=" * 10 + " Cleaning Query START! " + "=" * 10)

    question = state['question']
    messages = state.get('messages', [])

    if messages:
        logger.info(f"이전 대화 {len(messages)}개 참조 중!")
        # 최근 4개 메시지만 로깅
        for i, msg in enumerate(messages[-4:]):
            role = msg.type if hasattr(msg, 'type') else 'unknown'
            content = msg.content[:50] if hasattr(msg, 'content') else str(msg)[:50]
            logger.info(f"  #{i + 1}번째 {role}: {content}...")

    logger.info(f"정제할 질문: {question}")

    # LLM 가져오기
    if llm is None:
        llm_manager = get_llm_manager()
        llm = llm_manager.get_model(Config.LLM_MODEL, temperature=Config.LLM_TEMPERATURE)
        logger.info(f"기본 LLM 모델 사용: {Config.LLM_MODEL}")

    # 프롬프트 가져오기
    llm_manager = get_llm_manager()
    prompt = llm_manager.get_prompt('clean_query')
    chain = prompt | llm.with_structured_output(CleanQuery)
    result = chain.invoke({"input": question, "chat_history": messages})

    logger.info(f"Rewritten Query: {result.rewritten_query}")

    return {"rewritten_query": result.rewritten_query}
