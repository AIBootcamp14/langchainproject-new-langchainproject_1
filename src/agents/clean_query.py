# src/agents/clean_query.py
"""
Request Analyst Module

사용자의 질문에 대한 문맥을 파악하고 오탈자/잘못된 정보 전달 등을 수정해 정확한 쿼리를 작성하기 위한 쿼리 재작성기입니다.
"""

from typing import Literal, List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.model.llm import get_llm_manager
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)

class CleanQuery(BaseModel):
    """재작성된 쿼리 결과"""
    rewritten_query: str = Field(description="질문의 의도를 유지하면서 다른 표현으로 재작성된 사용자 질문")

def query_cleaner(state, llm = None):
    """
    사용자의 query를 문맥파악, 오탈자 수정, 의도파악 등을 통해 정확한 query로 재작성합니다.

    Arg :
        State(dict) : 현재 graph 상태(question & chat_history)
        llm : LLM 모델
    
    Returns :
        dict : rewritten_query 단일 필드를 포함한 딕셔너리
    """
    logger.info("=" * 10 + " Cleaning Query START!" + "=" * 10)
    
    question = state['question']
    messages = state.get('messages', [])

    if messages :
        logger.info(f"이전 대화 {len(messages)}개 참조 중!")

        for i, msg in enumerate(messages[-4:]):
            role = msg.type
            content = msg.content[:50]
            logger.info(f"#{i + 1}번째 {role} : {content}...")
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
    result = chain.invoke({"input": question, "messages": messages})

    logger.info(f"rewritten Query: {result.rewritten_query}")

    return {"rewritten_query" : result.rewritten_query}

if __name__ == "__main__":
    from langchain_core.messages import HumanMessage, AIMessage

    scenario1 = {'question' : "내가 방금 물어본 회사 주가가 어떻게 돼?",
                 'messages' : [HumanMessage(content = "애플회사는 뭘 하는 회사지?"),
                               AIMessage(content = "저는 금융, 경제 관련 전문 도우미입니다.")]}
    scenario2 = {'question' : '방금 물어본 회사 보고소 pdf파일로 만들어줘!',
                 'messages' : [HumanMessage(content = 'ㄴ ㅔ 이버회사와 다음회사 비교분석해줘'),
                               AIMessage(content = '네이버와 다음은 대한민국에 검색엔진 사업을 하는 회사입니다. ~~~')]
                 }
    scenario3 = {'question' : '응',
                 'messages' : [HumanMessage(content = "카카오 주가 동향 분석해줘"),
                               AIMessage(content = "**카카오**에 대한 내용은 다음과 같습니다 ~~~ __pdf파일__로 만들어드릴까요?")]}
    
    scenario4 = {'question' : '방금 뭘 물어봤지?',
                 'messages' : [HumanMessage(content = "PER이 뭐야 ?"),
                               AIMessage(content = "PER은 ~~~ 뭐입니다.")]}

    result1 = query_cleaner(state = scenario1)
    result2 = query_cleaner(scenario2)
    result3 = query_cleaner(state =scenario3)
    result4 = query_cleaner(state =scenario4)

    print(f"rewritten query : {result1['rewritten_query']}")
    print(f"rewritten query : {result2['rewritten_query']}")
    print(f"rewritten query : {result3['rewritten_query']}")
    print(f"rewritten query : {result4['rewritten_query']}")