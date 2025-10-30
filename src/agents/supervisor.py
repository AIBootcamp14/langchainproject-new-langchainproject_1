
from langchain_core.prompts import ChatPromptTemplate
from langchain_upstage import ChatUpstage
from langgraph.graph.message import add_messages


from pydantic import BaseModel, Field

from typing_extensions import TypedDict
from typing import Annotated, Literal

class State(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        document: retrieved document
    """

    question: str # 사용자의 질문
    generation: str # LLM의 생성 답변
    document: str # 검색된 문서
    messages: Annotated[list, add_messages] # 메시지 히스토리 저장

from pydantic import BaseModel, Field

class AgentType(BaseModel):
    """사용자 질문에 필요한 agent를 선택합니다."""
    agent: str = Field(description="Answer 'vector_search_agent' or 'financial_analyst' or 'none' only.")


def supervisor(state: State, llm)-> Literal["vector_search_agent", "financial_analyst", 'none']:

    """
    Args:
        state (dict) : The current graph state
    
    Returns:
        state (dict) : 

    """
    print('='*10,"SUPERVISOR THINKING START!",'='*10)
    question = state['question']

    # define supervisor prompt
    supervisor_prompt = ChatPromptTemplate.from_template(
        """
        당신은 금융 도메인 질문을 가장 잘 처리할 다음 단계의 "분석 에이전트"를 선택하는 routing 감독관입니다.
        
        아래 에이전트 중 질문에 가장 적합한 하나만 선택하십시오.
        - vector_search_agent: 금융용어, 주식관련 용어, 주식관련 은어 등 대한 신뢰 가능한 문서 검색에 특화(RAG 기반)
        - financial_analyst: 종목코드 찾기(TICKER), 재무제표 조회, 주식 정보 조회, 주식 비교, 특정 기간 주가 이력 조회 등 주식관련 정보 수집에 특화

        선택규칙:
        1) 오직 하나만 선택 (AND 금지)
        2) 단순 금융용어 및 주식관련 용어 등이 필요하면 vector_search_agent를 우선 선택
        3) 재무 계산, 종목 비교, 종목 코드 찾기, 기업 비교 등, 재무 분석 중심이면 financial_analyst를 우선 선택
        4) 출력은 반드시 JSON 형식만 반환 (설명, 여분 텍스트 금지)

        사용자 질문:
        {question}

        출력 형식(JSON)
        {{
            "agent": "vector_search_agent" or "financial_analyst", none
        }}
        """
    )
    chain = supervisor_prompt | llm.with_structured_output(AgentType)
    result = chain.invoke({"question": question})
    print(f"Choose Agent : {result.agent}")
    return result.agent
    




if __name__ == "__main__":
    from dotenv import load_dotenv
    from langchain_upstage import ChatUpstage

    # 환경변수 load
    load_dotenv()
    # llm 호출 및 정의
    llm = ChatUpstage(model="solar-pro")

    # 질문 예제 정의
    input1 = {"question" : "오늘 날씨 어때?"}
    input2 = {"question" : "나스닥이 뭐야 ???"}
    input3 = {"question" : "삼성전자와 LG전자 25년도 전반기 실적 비교를 하고 싶어"}

    # supervisor 실험
    example1 = supervisor(input1, llm)
    # example2 = supervisor(input2, llm)
    # example3 = supervisor(input3, llm)

    print(f"Question 1 : {input1['question']} \nAnswer 1 : {example1}")
    # print(f"Question 2 : {input2['question']} \nAnswer 2 : {example2}")
    # print(f"Question 3 : {input3['question']} \nAnswer 3 : {example3}")
    # print(result)
