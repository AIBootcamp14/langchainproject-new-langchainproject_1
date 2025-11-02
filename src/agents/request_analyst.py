# src/agents/request_analyst.py
"""
Request Analyst Module

사용자의 요청이 경제/금융 관련인지 판별하는 분류기입니다.
"""

from typing import Literal, List, Dict, Any, Optional
from pydantic import BaseModel, Field

from src.model.llm import get_llm_manager
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FinanceGate(BaseModel):
    """경제/금융 관련 여부 분류 모델"""
    label: str = Field(description="경제 금융 관련 여부 label.")


class RewriteResult(BaseModel):
    """재작성된 쿼리 결과"""
    rewritten_query: str = Field(description="질문의 의도를 유지하면서 다른 표현으로 재작성된 사용자 질문")


def request_analysis(state, llm=None, chat_history: Optional[List[Dict]] = None) -> Literal["finance", "not_finance"]:
    """
    사용자 요청을 분석하여 경제/금융 관련 여부를 판별합니다.

    Args:
        state (dict): 현재 그래프 상태 (question 필드 포함)
        llm: LLM 모델 (선택사항, 없으면 기본 모델 사용)
        chat_history: 이전 대화 기록 (선택사항, 컨텍스트 기반 분석에 사용)

    Returns:
        dict: label과 return_msg 필드를 포함한 딕셔너리
    """
    logger.info("=" * 10 + " Request Analysis THINKING START! " + "=" * 10)
    question = state['question']
    logger.info(f"분석할 질문: {question}")

    # chat_history가 있으면 컨텍스트 정보 로깅
    if chat_history:
        logger.info(f"이전 대화 {len(chat_history)}개 참조 중")
        # 최근 3개 대화만 로깅
        for i, msg in enumerate(chat_history[:3]):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')[:50]
            logger.debug(f"  [{i+1}] {role}: {content}...")

    # LLM 가져오기
    if llm is None:
        llm_manager = get_llm_manager()
        llm = llm_manager.get_model(Config.LLM_MODEL, temperature=Config.LLM_TEMPERATURE)
        logger.info(f"기본 LLM 모델 사용: {Config.LLM_MODEL}")

    # 프롬프트 가져오기
    llm_manager = get_llm_manager()
    request_analysis_prompt = llm_manager.get_prompt("request_analyst")

    # 체인 생성 및 실행
    chain = request_analysis_prompt | llm.with_structured_output(FinanceGate)
    result = chain.invoke({"question": question})

    logger.info(f"Question status: {result.label}")

    if result.label == "not_finance":
        logger.info("비금융 질문으로 분류됨")
        return {
            'return_msg': Config.NOT_FINANCE_RESPONSE,
            'label': "not_finance"
        }

    logger.info("금융 질문으로 분류됨")
    return {"label": "finance"}


def rewrite_query(
    original_query: str,
    failure_reason: str,
    chat_history: Optional[List[Dict]] = None,
    llm=None
) -> Dict[str, Any]:
    """
    quality_evaluator에서 incorrect로 판정 시, 기존 쿼리를 의미 보존한 채 재작성합니다.

    Args:
        original_query: 원본 사용자 질문
        failure_reason: 실패 이유 (empty/error/incorrect)
        chat_history: 이전 대화 기록 (선택사항)
        llm: LLM 모델 (선택사항)

    Returns:
        딕셔너리:
        - rewritten_query: 재작성된 쿼리 (str)
        - needs_user_input: 유저에게 추가 정보 필요 여부 (bool)
        - request_for_detail_msg: 유저에게 더 구체적인 질문을 요청하는 메세지 (str, needs_user_input이 True일 때)
    """
    logger.info("=" * 10 + " Query Rewrite THINKING START! " + "=" * 10)
    logger.info(f"원본 질문: {original_query}")
    logger.info(f"실패 이유: {failure_reason}")

    # chat_history 컨텍스트 확인
    if chat_history:
        logger.info(f"이전 대화 {len(chat_history)}개 참조 중")

    # LLM 가져오기
    llm_manager = get_llm_manager()
    if llm is None:
        llm = llm_manager.get_model(Config.LLM_MODEL, temperature=Config.LLM_TEMPERATURE)

    # 간단한 휴리스틱: 쿼리가 너무 짧으면 유저에게 추가 정보 요청
    if len(original_query.strip()) < 5:
        logger.info("쿼리가 너무 짧음 - 유저에게 추가 정보 요청")
        return {
            "rewritten_query": original_query,
            "needs_user_input": True,
            "request_for_detail_msg": "질문을 좀 더 구체적으로 말씀해 주시겠어요? 어떤 정보를 원하시나요?"
        }

    # 프롬프트 및 체인 구성
    rewrite_prompt = llm_manager.get_prompt("rewrite_query")
    chat_history_text = ""
    if chat_history:
        chat_history_text = "\n".join(
            f"{msg.get('role', 'unknown')}: {str(msg.get('content', ''))}"
            for msg in chat_history
        )
    prompt_inputs = {
        "original_query": original_query,
        "failure_reason": failure_reason,
        "chat_history": chat_history_text or "없음"
    }

    chain = rewrite_prompt | llm.with_structured_output(RewriteResult)
    response = chain.invoke(prompt_inputs)
    rewritten_query = response.rewritten_query.strip()

    if not rewritten_query:
        logger.warning("LLM 재작성 결과가 비어 있어 원본을 그대로 사용합니다.")
        rewritten_query = original_query
    else:
        logger.info(f"쿼리 재작성 완료: {rewritten_query}")

    return {
        "rewritten_query": rewritten_query,
        "needs_user_input": False,
        "request_for_detail_msg": None
    }


if __name__ == "__main__":
    from dotenv import load_dotenv

    # 환경변수 load
    load_dotenv()

    # 질문 예제 정의
    input1 = {"question": "오늘 날씨 어때?"}
    input2 = {"question": "AI가 뭐야 ?"}
    input3 = {"question": "AI 시장 투자 규모가 어떻게 돼 ?"}

    # request_analysis 실험
    example1 = request_analysis(input1)
    example2 = request_analysis(input2)
    example3 = request_analysis(input3)

    # request_analysis 실험 결과 출력
    print(f"Question 1: {input1['question']} \nAnswer 1: {example1.get('return_msg', 'finance')}")
    print(f"Question 2: {input2['question']} \nAnswer 2: {example2.get('return_msg', 'finance')}")
    print(f"Question 3: {input3['question']} \nAnswer 3: {example3.get('return_msg', 'finance')}")
