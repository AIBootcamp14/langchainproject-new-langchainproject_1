# src/agents/financial_analyst.py
"""
Financial Analyst Agent

사용자의 질문에 따라 주식 데이터를 수집하고 분석하는 ReAct 방식의 에이전트입니다.
- 주식 검색 및 정보 조회
- 과거 가격 데이터 분석
- 웹 검색을 통한 최신 뉴스 수집
- 애널리스트 추천 정보 조회
- 다중 주식 비교 분석
"""

from typing import Dict, Any, Optional, List
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.agents.tools.financial_tools import financial_tools
from src.model.llm import get_llm_manager
from src.utils.logger import get_logger
from src.utils.config import Config

logger = get_logger(__name__)

class FinancialAnalyst:
    def __init__(self, model_name: str = None, temperature: float = 0):
        """
        Financial Analyst 에이전트를 초기화합니다.

        Args:
            model_name: 사용할 모델명 (default: Config.LLM_MODEL)
            temperature: LLM 온도 (0 = 결정적, 1 = 창의적)
        """
        if model_name is None:
            model_name = Config.LLM_MODEL
        logger.info(f"Financial Analyst 초기화 - model: {model_name}, temp: {temperature}")

        # LLM Manager에서 모델 가져오기
        llm_manager = get_llm_manager()
        self.llm = llm_manager.get_model(model_name, temperature=temperature)

        self.tools = financial_tools
        self.agent_executor = self._create_agent()

        logger.info("Financial Analyst 초기화 완료")
    
    def _create_agent(self) -> AgentExecutor:
        """ReAct 에이전트를 생성합니다."""

        # LLM Manager에서 프롬프트 가져오기
        llm_manager = get_llm_manager()
        base_prompt = llm_manager.get_prompt("financial_analyst")

        prompt = ChatPromptTemplate.from_messages([
        ("system", base_prompt.template),
        MessagesPlaceholder('messages'),
        ("human", "User Query: {input}"),
        ])


        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        def handle_parsing_error(error: Exception) -> str:
            error_msg = str(error)
            logger.warning(f"[PARSING ERROR] {error_msg[:150]}...")
            
            # 마크다운 관련 에러 감지
            if "** " in error_msg or " **" in error_msg:
                return """ERROR: You used markdown (** or __) in Action line!

WRONG:
**Action:** tool_name
Action: **tool_name**

CORRECT:
Action: tool_name
Action Input: {"key": "value"}

Remove ALL markdown from Action/Action Input lines!"""
        
            if "both a final answer and a parse-able action" in error_msg:
                return """ERROR: You wrote Action and Final Answer together!

Write ONE thing at a time:
1. Action: tool
2. Action Input: {...}
3. STOP and wait
4. (system provides Observation)
5. Continue or Final Answer"""
        
            if "not a valid tool" in error_msg:
                return """ERROR: Tool name has extra characters!

Valid tools EXACTLY:
- draw_stock_chart
- draw_valuation_radar
- save_report_to_file

Check for spaces, markdown, or typos!"""
        
            return "Format error. Use plain text for Action lines."
        
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=15,
            early_stopping_method="force",
            handle_parsing_errors=handle_parsing_error,
            return_intermediate_steps=True
        )
        
        return agent_executor
        
    def analyze(self, query: str, messages:list = None) -> Dict[str, Any]:
        """
        주어진 질문에 대해 금융 분석을 수행합니다.
        
        Args:
            query: 사용자 질문
        
        Returns:
            분석 결과를 담은 딕셔너리
        """
        if messages == None:
            messages = []
        try:
            logger.info(f"분석 시작 - query: {query}")
            
            # 에이전트 실행
            result = self.agent_executor.invoke({"input": query, "messages": messages})
            
            # 결과 추출
            output = result.get("output", {})
            
            # JSON 파싱 시도
            import json
            import re
            
            if isinstance(output, str):
                logger.debug(f"원본 출력 타입: str, 길이: {len(output)}")
                
                # 1. 코드 블록 제거 (```json ... ```)
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', output, re.DOTALL)
                if json_match:
                    output = json_match.group(1)
                    logger.debug("코드 블록에서 JSON 추출 성공")
                
                # 2. Final Answer: 이후의 JSON만 추출
                elif 'Final Answer:' in output:
                    # "Final Answer:" 이후 부분만 가져오기
                    parts = output.split('Final Answer:')
                    if len(parts) > 1:
                        json_part = parts[-1].strip()  # 마지막 Final Answer 사용
                        
                        # JSON 객체 추출 (첫 { 부터 매칭되는 } 까지)
                        brace_count = 0
                        start_idx = json_part.find('{')
                        
                        if start_idx != -1:
                            end_idx = start_idx
                            for i in range(start_idx, len(json_part)):
                                if json_part[i] == '{':
                                    brace_count += 1
                                elif json_part[i] == '}':
                                    brace_count -= 1
                                    if brace_count == 0:
                                        end_idx = i + 1
                                        break
                            
                            if end_idx > start_idx:
                                output = json_part[start_idx:end_idx]
                                logger.debug(f"Final Answer에서 JSON 추출 성공 (길이: {len(output)})")
                
                # 3. 일반 JSON 추출 (위 방법들 실패 시)
                else:
                    start = output.find('{')
                    if start != -1:
                        brace_count = 0
                        end_idx = start
                        for i in range(start, len(output)):
                            if output[i] == '{':
                                brace_count += 1
                            elif output[i] == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    end_idx = i + 1
                                    break
                        
                        if end_idx > start:
                            output = output[start:end_idx]
                            logger.debug(f"일반 JSON 추출 (길이: {len(output)})")
                
                # 4. JSON 파싱
                try:
                    output = json.loads(output)
                    logger.debug(f"JSON 파싱 성공 - analysis_type: {output.get('analysis_type', 'N/A')}")
                
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON 파싱 실패: {str(e)}")
                    logger.debug(f"파싱 시도한 문자열 (처음 300자): {output[:300]}")
                    
                    # JSON이 아니면 기본 구조로 래핑
                    output = {
                        "analysis_type": "single",
                        "ticker": "UNKNOWN",
                        "company_name": "Unknown",
                        "current_price": 0,
                        "analysis": output if len(output) < 1000 else output[:1000] + "...",
                        "metrics": {},
                        "period": "3mo"
                    }
            
            logger.info(f"분석 완료 - type: {output.get('analysis_type', 'N/A')}")
            return output
        
        except Exception as e:
            logger.error(f"분석 실패 - query: {query}, error: {str(e)}")
            import traceback
            logger.debug(f"상세 에러:\n{traceback.format_exc()}")
            
            return {
                "error": str(e),
                "analysis_type": "error",
                "ticker": "ERROR",
                "company_name": "Error",
                "current_price": 0,
                "analysis": f"분석 중 오류가 발생했습니다: {str(e)}",
                "metrics": {},
                "period": "3mo"
            }
    
    def compare_stocks(self, tickers: List[str], messages:list) -> Dict[str, Any]:
        """
        여러 주식을 비교 분석합니다.
        
        Args:
            tickers: 비교할 티커 리스트 (예: ["AAPL", "MSFT", "GOOGL"])
        
        Returns:
            비교 분석 결과 딕셔너리
        """
        try:
            logger.info(f"비교 분석 시작 - tickers: {tickers}")
            
            # 자동으로 비교 쿼리 생성
            ticker_str = ", ".join(tickers)
            query = f"{ticker_str} 주식들을 비교 분석해주세요. 각각의 장단점과 투자 추천을 포함해주세요."
            
            return self.analyze(query=query, messages = messages)
        
        except Exception as e:
            logger.error(f"비교 분석 실패 - tickers: {tickers}, error: {str(e)}")
            return {
                "error": str(e),
                "analysis_type": "comparison",
                "stocks": [],
                "comparison_analysis": f"비교 분석 중 오류가 발생했습니다: {str(e)}"
            }
    
    def invoke(self, query: str, messages : list) -> Dict[str, Any]:
        """
        analyze()의 별칭 메서드 (LangChain 스타일 호환)
        
        Args:
            query: 사용자 질문
        
        Returns:
            분석 결과 딕셔너리
        """
        return self.analyze(query=query, messages=messages)


# 편의를 위한 팩토리 함수
def create_financial_analyst(
    model_name: str = "solar-pro",
    temperature: float = 0
) -> FinancialAnalyst:
    """
    Financial Analyst 에이전트를 생성합니다.
    
    Args:
        model_name: 사용할 LLM 모델명
        temperature: LLM 온도
    
    Returns:
        FinancialAnalyst 인스턴스
    """
    return FinancialAnalyst(model_name=model_name, temperature=temperature)


if __name__ == "__main__":
    import logging
    
    # 디버그 로그 활성화
    logging.getLogger("__main__").setLevel(logging.DEBUG)
    
    # LangChain 파싱 경고는 숨김
    logging.getLogger("langchain.agents.agent").setLevel(logging.ERROR)
    
    from src.utils.config import Config
    Config.validate_api_keys()

    analyst = create_financial_analyst(model_name="solar-pro")
    
    # 단일 분석
    print("\n" + "="*80)
    print("단일 주식 분석")
    print("="*80)
    result = analyst.analyze("애플 주식 분석")
    print(f"분석 타입: {result.get('analysis_type')}")
    print(f"티커: {result.get('ticker')}")
    
    # 비교 분석
    print("\n" + "="*80)
    print("비교 분석")
    print("="*80)
    result = analyst.analyze("애플과 마이크로소프트 주식 중에 뭘 추천해?")
    print(f"분석 타입: {result.get('analysis_type')}")
    if result.get('analysis_type') == 'comparison':
        print(f"주식 수: {len(result.get('stocks', []))}")
        for stock in result.get('stocks', []):
            print(f"  - {stock['ticker']}: ${stock['current_price']}")
        print(f"추천: {result.get('recommendation', {}).get('preferred_stock')}")