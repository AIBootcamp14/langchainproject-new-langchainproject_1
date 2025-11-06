# src/agents/supervisor.py
"""
Supervisor Module

금융 도메인 질문을 가장 잘 처리할 다음 단계의 분석 에이전트를 선택하는 routing 감독관입니다.
"""

from typing import Literal, Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langgraph.graph.message import add_messages

from src.model.llm import get_llm_manager
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class State(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        document: retrieved document
        messages: message history
    """
    question: str  # 사용자의 질문
    generation: str  # LLM의 생성 답변
    document: str  # 검색된 문서
    messages: Annotated[list, add_messages]  # 메시지 히스토리 저장


class AgentType(BaseModel):
    """사용자 질문에 필요한 agent를 선택합니다."""
    agent: str = Field(description="Answer 'vector_search_agent' or 'financial_analyst' or 'none' only.")


def supervisor(state: State, llm=None) -> Literal["vector_search_agent", "financial_analyst", 'none']:
    """
    금융 도메인 질문을 처리할 적합한 에이전트를 선택합니다.

    Args:
        state (State): 현재 그래프 상태 (question 필드 포함)
        llm: LLM 모델 (선택사항, 없으면 기본 모델 사용)

    Returns:
        str: 선택된 에이전트 이름
    """
    logger.info("=" * 10 + " SUPERVISOR THINKING START! " + "=" * 10)
    question = state['question']
    messages = state.get('messages', [])
    logger.info(f"라우팅할 질문: {question}")

    # LLM 가져오기
    if llm is None:
        llm_manager = get_llm_manager()
        llm = llm_manager.get_model(Config.LLM_MODEL, temperature=Config.LLM_TEMPERATURE)
        logger.info(f"기본 LLM 모델 사용: {Config.LLM_MODEL}")

    # 프롬프트 가져오기 (이미 ChatPromptTemplate으로 변환됨)
    llm_manager = get_llm_manager()
    prompt = llm_manager.get_prompt("supervisor")
    # 체인 생성 및 실행
    chain = prompt | llm.with_structured_output(AgentType)
    result = chain.invoke({"input": question, 'chat_history': messages})

    logger.info(f"Choose Agent: {result.agent}")
    return result.agent


if __name__ == "__main__":
    from dotenv import load_dotenv

    # 환경변수 load
    load_dotenv()

    # 질문 예제 정의
    input1 = {"question": "오늘 날씨 어때?"}
    input2 = {"question": "나스닥이 뭐야 ???"}
    input3 = {"question": "삼성전자와 LG전자 25년도 전반기 실적 비교를 하고 싶어"}

    # supervisor 실험
    example1 = supervisor(input1)
    example2 = supervisor(input2)
    example3 = supervisor(input3)

    print(f"Question 1: {input1['question']} \nAnswer 1: {example1}")
    print(f"Question 2: {input2['question']} \nAnswer 2: {example2}")
    print(f"Question 3: {input3['question']} \nAnswer 3: {example3}")
