from langchain_core.prompts import ChatPromptTemplate

from pydantic import BaseModel, Field

from typing_extensions import TypedDict
from typing import Annotated, Literal



class FinanceGate(BaseModel):
    label: str = Field(description="경제 금융 관련 여부 label.")

def request_analysis(state, llm)-> Literal["finance", "not_finance"]:

    """
    Args:
        state (dict) : The current graph state
    
    Returns:
        state (dict) : 

    """
    print('='*10,"Request Analysis THINKING START!",'='*10)
    question = state['question']

    # define supervisor prompt
    request_analysis_prompt = ChatPromptTemplate.from_template(
        """
        당신은 사용자의 질문 또는 요청이 "경제, 금융 관련"이지 판별하는 분류기 입니다.
        
        판단 기준:
        - 경제, 금융 관련(`finance`) 예시 : 주식ETF/채권/파생상품, 환율/금리/인플레이션/거시경제, 기업 실적/밸류에이션(Market Cap, PER/PBR/EV/EBITDA 등), 재무제표/회계, 개인재무(예산/저축/대출/세금), 암호자산의 시세/거래/토큰 이코노미(투자 맥락), 금융/규제/정책/공시/뉴스.
        - 비관련(`not_finance`) 예시 : 날씨/여행/요리/스포츠/게임/일상 대화, 일반 IT/프로그래밍(금융 맥락 없음), 역사/예술/문화, 비재무적 기업 소개(연혁/채용 등만).
        
        엣지 케이스 처리:
        - 기술/데이터/AI 질문이라도 "투자 의사결정/시장/재무 지표/거시경제"와 직접 연결되면 `finance`.
        - 암호화폐/블록체인 기술 자체는 `not_finance`이지만, 가격/투자/거래/시장 동향을 묻는다면 `finance`.
        - 질문이 모호하면 사용자 의도가 금융일 가능성이 있는지 보수적으로 판단하되, 근거가 부족하면 `not_finance`

        출력은 구조화된 형식으로만 반환하십시오. 추가 설명이나 여분 텍스트를 포함하지 마십시오.

        사용자 질문:
        {question}
        """
    )
    chain = request_analysis_prompt | llm.with_structured_output(FinanceGate)
    result = chain.invoke({"question": question})
    print(f"Question status : {result.label}")

    if result.label == "not_finance":
        return {"generate" : "저는 경제, 금융관련 정보를 통해 전문적으로 사용자의 요청을 도와드리는 AI입니다!\n주식, 환율, 기업 분석 등 금융 관련 질문을 해주시면 답변 도와 드릴게요 😄",
                'label' : "not_finance"}
    return {"label" : "finance"}


if __name__ == "__main__":
    from dotenv import load_dotenv
    from langchain_upstage import ChatUpstage

    # 환경변수 load
    load_dotenv()
    # llm 호출 및 정의
    llm = ChatUpstage(model="solar-pro")


    # 질문 예제 정의
    input1 = {"question" : "오늘 날씨 어때?"}
    input2 = {"question" : "AI가 뭐야 ?"}
    input3 = {"question" : "AI 시장 투자 규모가 어떻게 돼 ?"}

    # request_analysis 실험
    example1 = request_analysis(input1, llm)
    example2 = request_analysis(input2, llm)
    example3 = request_analysis(input3, llm)

    # request_analysis 실험
    print(f"Question 1 : {input1['question']} \nAnswer 1 : {example1.get('generate', 'finance')}")
    print(f"Question 2 : {input2['question']} \nAnswer 2 : {example2.get('generate', 'finance')}")
    print(f"Question 3 : {input3['question']} \nAnswer 3 : {example3.get('generate', 'finance')}")