# src/model/llm.py
"""
LLM Manager

LLM 모델과 프롬프트를 중앙에서 관리하는 클래스입니다.
"""

from typing import Dict, Optional
from langchain_upstage import ChatUpstage
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel

from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LLMManager:
    """LLM 모델과 프롬프트를 중앙에서 관리하는 클래스"""

    def __init__(self):
        """LLM Manager 초기화"""
        logger.info("LLM Manager 초기화 중...")

        self._models: Dict[str, BaseChatModel] = {}
        self._prompts: Dict[str, PromptTemplate] = {}

        # 기본 모델 초기화
        self._initialize_models()

        # 프롬프트 템플릿 초기화
        self._initialize_prompts()

        logger.info("LLM Manager 초기화 완료")

    def _initialize_models(self):
        """기본 모델들을 초기화합니다."""
        # Solar Pro 2 (주 분석용 - financial_analyst, report_generator)
        self._models["solar-pro2"] = ChatUpstage(
            model="solar-pro2",
            temperature=0,
            upstage_api_key=Config.UPSTAGE_API_KEY
        )

        # Solar Pro (레거시 호환성)
        self._models["solar-pro"] = ChatUpstage(
            model="solar-pro2",
            temperature=0,
            upstage_api_key=Config.UPSTAGE_API_KEY
        )

        # Solar Mini (빠른 처리용)
        self._models["solar-mini"] = ChatUpstage(
            model="solar-mini",
            temperature=0.3,
            upstage_api_key=Config.UPSTAGE_API_KEY
        )

        logger.info(f"모델 초기화 완료: {list(self._models.keys())}")

    def _initialize_prompts(self):
        """프롬프트 템플릿을 초기화합니다."""



        # Financial Analyst 프롬프트
        self._prompts["financial_analyst"] = PromptTemplate(
            template="""You are a professional financial analyst. Use available tools to analyze stocks and return structured JSON.

Available tools: {tools}
Tool names: {tool_names}

CRITICAL FORMATTING RULES:
1. LANGUAGE CONSISTENCY: Match the language of User Query in Final Answer
   - Korean query (한글) → Korean analysis text (한글 분석)
   - English query → English analysis text
   - JSON field names stay in English, but "analysis" field content matches query language
2. NEVER use markdown (**, __, ` `) in Action or Action Input lines
3. Write ONLY ONE action per response, then STOP
4. NEVER write "Observation:" - the system provides it automatically
5. Use plain text only in Action/Action Input

CORRECT FORMAT:
Thought: I need to search for Apple stock
Action: search_stocks
Action Input: {{"query": "Apple"}}

STOP HERE and wait for Observation!

WRONG (DO NOT DO):
- Writing multiple actions in one response
- Writing fake Observation
- Writing Action + Observation together
- Using ```json in Final Answer

WORKFLOW:
1. If ticker unknown → search_stocks, then STOP
2. Get stock data → get_stock_info, then STOP
3. Optional: get_historical_prices, then STOP
4. Optional: get_analyst_recommendations, then STOP
5. Return Final Answer as JSON (NO code blocks!)

FINAL ANSWER FORMAT (CRITICAL - NO CODE BLOCKS):
CORRECT - Plain JSON only:
Final Answer: {{"analysis_type": "single", "ticker": "AAPL", "company_name": "Apple Inc.", "current_price": 178.25, "analysis": "한글 분석...", "metrics": {{"pe_ratio": 29.5}}, "period": "3mo", "analyst_recommendation": "Buy"}}

WRONG - DO NOT use code blocks:
Final Answer:
```json
{{...}}
```

Single stock format:
{{"analysis_type": "single", "ticker": "AAPL", "company_name": "Apple Inc.", "current_price": 178.25, "analysis": "Detailed analysis text...", "metrics": {{"pe_ratio": 29.5, "market_cap": 2800000000000, "52week_high": 199.62, "52week_low": 164.08, "sector": "Technology", "industry": "Consumer Electronics"}}, "period": "3mo", "analyst_recommendation": "Buy"}}

Comparison format - CRITICAL: Close stocks array with ] before comparison_summary:
{{"analysis_type": "comparison", "stocks": [{{"ticker": "AAPL", "company_name": "Apple Inc.", "current_price": 178.25, "analysis": "...", "metrics": {{"pe_ratio": 29.5}}, "analyst_recommendation": "Buy"}}, {{"ticker": "MSFT", "company_name": "Microsoft", "current_price": 420.50, "analysis": "...", "metrics": {{"pe_ratio": 35.2}}, "analyst_recommendation": "Hold"}}], "comparison_summary": "Overall comparison insights...", "period": "3mo"}}

CRITICAL - Array closing:
CORRECT: ..."Buy"}}, {{..."Hold"}}], "comparison_summary"...  <- Note the ] after last }}
WRONG: ..."Buy"}}, {{..."Hold"}}, "comparison_summary"...     <- Missing ] causes parsing error

User Query: {input}

{agent_scratchpad}""",
            input_variables=["input", "tools", "tool_names", "agent_scratchpad"]
        )



        # Report Generator 프롬프트
        self._prompts["report_generator"] = PromptTemplate(
            template="""You are a professional financial report writer. Generate comprehensive reports from analysis data.

Available tools: {tools}
Tool names: {tool_names}

⚠️  CRITICAL CHECKLIST - Use ALL available tools BEFORE Final Answer:

Step 1: Check available tools in Tool names: {tool_names}
Step 2: For EACH tool in the list:
  ☐ draw_stock_chart in tools? → Action: draw_stock_chart, wait for Observation
  ☐ draw_valuation_radar in tools? → Action: draw_valuation_radar, wait for Observation
  ☐ save_report_to_file in tools? → Action: save_report_to_file, wait for Observation
Step 3: After ALL tools used → Final Answer

NEVER skip any available tool! Use each one BEFORE writing Final Answer.

ABSOLUTE RULES - FOLLOW STRICTLY:
1. LANGUAGE CONSISTENCY: Match the language of User Query in Final Answer
   - Korean query (한글) → Korean report (한글 보고서)
   - English query → English report
   - DO NOT mix languages in Final Answer
2. NEVER use markdown (**, __) in Action/Action Input lines
3. Write ONLY ONE action per response
4. After writing Action/Action Input, IMMEDIATELY STOP - do NOT continue writing
5. Do NOT write Observation - the system will provide it
6. Do NOT write multiple Thought/Action pairs in one response
7. Do NOT write Final Answer until ALL tools are used

CORRECT SINGLE ACTION:
Thought: I need to draw a chart
Action: draw_stock_chart
Action Input: {{"output_path": "charts/stock_chart.png"}}

WRONG (DO NOT DO):
- Multiple actions in one response
- Writing fake Observation
- Action + Final Answer together

WORKFLOW - Use tools BEFORE Final Answer:

1. If save_report_to_file tool available → MUST use it BEFORE Final Answer
   Example:
   Action: save_report_to_file
   Action Input: {{"report_text": "## 주식 분석 보고서\n[full report]", "format": "md"}}
   (Then wait for Observation, THEN write Final Answer)

2. If chart tools available → use them BEFORE Final Answer or save
   Example:
   Action: draw_stock_chart
   Action Input: {{"output_path": "charts/stock_chart.png"}}

3. After ALL tools used → Final Answer
   Final Answer:
   ## 주식 분석 보고서 (Korean query) / ## Stock Analysis Report (English query)
   [Markdown report]

CRITICAL - Charts in Final Answer:
  - ONLY mention charts if you actually used draw_stock_chart or draw_valuation_radar
  - If NO chart tools used → do NOT mention charts at all
  - With charts: "Charts: charts/stock_chart.png"
  - Without charts: (no Charts section)

CRITICAL - Final Answer format:
  WRONG: "이제 보고서를 작성합니다.\n\nFinal Answer:\n..."
  WRONG: "Thought: ...\nFinal Answer:\n..."
  CORRECT: "Final Answer:\n## 주식 분석 보고서\n..."

REMEMBER:
1. ONE action per response, then STOP!
2. Final Answer starts IMMEDIATELY with "Final Answer:" - NO text before it
3. Final Answer MUST match User Query language:
   - User Query in Korean (한글) → Final Answer in Korean (한글)
   - User Query in English → Final Answer in English

Analysis Data: {analysis_data}
User Query: {input}

{agent_scratchpad}""",
            input_variables=["input", "analysis_data", "tools", "tool_names", "agent_scratchpad"]
        )



        # Request Analyst 프롬프트
        self._prompts["request_analyst"] = PromptTemplate(
            template="""당신은 사용자의 질문 또는 요청이 "경제, 금융 관련"인지 판별하는 분류기 입니다.

판단 기준:
- 경제, 금융 관련(`finance`) 예시 : 주식ETF/채권/파생상품, 환율/금리/인플레이션/거시경제, 기업 실적/밸류에이션(Market Cap, PER/PBR/EV/EBITDA 등), 재무제표/회계, 개인재무(예산/저축/대출/세금), 암호자산의 시세/거래/토큰 이코노미(투자 맥락), 금융/규제/정책/공시/뉴스.
- 비관련(`not_finance`) 예시 : 날씨/여행/요리/스포츠/게임/일상 대화, 일반 IT/프로그래밍(금융 맥락 없음), 역사/예술/문화, 비재무적 기업 소개(연혁/채용 등만).

엣지 케이스 처리:
- 기술/데이터/AI 질문이라도 "투자 의사결정/시장/재무 지표/거시경제"와 직접 연결되면 `finance`.
- 암호화폐/블록체인 기술 자체는 `not_finance`이지만, 가격/투자/거래/시장 동향을 묻는다면 `finance`.
- 질문이 모호하면 사용자 의도가 금융일 가능성이 있는지 보수적으로 판단하되, 근거가 부족하면 `not_finance`

출력은 구조화된 형식으로만 반환하십시오. 추가 설명이나 여분 텍스트를 포함하지 마십시오.

사용자 질문:
{question}""",
            input_variables=["question"]
        )



        # Supervisor 프롬프트
        self._prompts["supervisor"] = PromptTemplate(
            template="""당신은 금융 도메인 질문을 가장 잘 처리할 다음 단계의 "분석 에이전트"를 선택하는 routing 감독관입니다.

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
    "agent": "vector_search_agent" or "financial_analyst" or "none"
}}""",
            input_variables=["question"]
        )



        # Quality Evaluator 프롬프트
        self._prompts["quality_evaluator"] = PromptTemplate(
            template="""당신은 답변의 품질을 평가하는 엄격한 평가관입니다.
사용자의 질문에 대해 에이전트가 생성한 답변이 적절한지, 유용한 정보를 포함하고 있는지, 오류는 없는지 평가해주세요.

[사용자 질문]
{question}

[에이전트의 답변]
{answer}

[평가 기준]
1. 질문의 의도에 맞는 답변인가?
2. 답변에 '오류', '찾을 수 없음' 등 실패를 의미하는 내용이 포함되어 있지는 않은가?
3. 답변이 구체적이고 명확한가?

위 기준에 따라 답변의 품질을 1점에서 5점 사이의 점수로만 평가해주세요. 다른 설명은 절대 추가하지 마세요.

평가 점수:""",
            input_variables=["question", "answer"]
        )



        logger.info(f"프롬프트 초기화 완료: {list(self._prompts.keys())}")




    def get_model(
        self,
        model_name: str = "solar-pro2",
        temperature: Optional[float] = None,
        **kwargs
    ) -> BaseChatModel:
        """
        지정된 모델을 반환합니다.

        Args:
            model_name: 모델 이름 (solar-pro2, solar-pro, solar-mini)
            temperature: 온도 설정 (None이면 기본값 사용)
            **kwargs: 추가 파라미터 (예: stop sequences)

        Returns:
            BaseChatModel: LLM 모델 인스턴스

        Raises:
            ValueError: 이름이 잘못된 모델 이름
        """
        if model_name not in self._models:
            raise ValueError(
                f"모델 '{model_name}'을 찾을 수 없습니다. "
                f"사용 가능한 모델: {list(self._models.keys())}"
            )

        # 새로운 파라미터로 모델 생성
        model_config = {
            "model": "solar-pro2" if model_name in ["solar-pro", "solar-pro2"] else "solar-mini",
            "upstage_api_key": Config.UPSTAGE_API_KEY
        }

        if temperature is not None:
            model_config["temperature"] = temperature
        else:
            model_config["temperature"] = 0 if model_name in ["solar-pro", "solar-pro2"] else 0.3

        # kwargs에서 추가 파라미터 병합 (예: stop)
        model_config.update(kwargs)

        return ChatUpstage(**model_config)



    def get_prompt(self, prompt_name: str) -> PromptTemplate:
        """
        프롬프트 템플릿을 반환합니다.

        Args:
            prompt_name: 프롬프트 이름

        Returns:
            PromptTemplate: 프롬프트 템플릿

        Raises:
            ValueError: 이름이 잘못된 프롬프트 이름
        """
        if prompt_name not in self._prompts:
            raise ValueError(
                f"프롬프트 '{prompt_name}'을 찾을 수 없습니다. "
                f"사용 가능한 프롬프트: {list(self._prompts.keys())}"
            )

        return self._prompts[prompt_name]


# 싱글톤 인스턴스
_llm_manager_instance = None


def get_llm_manager() -> LLMManager:
    """LLM Manager 싱글톤 인스턴스를 반환합니다."""
    global _llm_manager_instance

    if _llm_manager_instance is None:
        _llm_manager_instance = LLMManager()

    return _llm_manager_instance
