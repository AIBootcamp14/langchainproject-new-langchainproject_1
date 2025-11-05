# src/agents/financial_analyst.py
"""
Financial Analyst Agent (Structured Output 기반)

사용자의 질문에 따라 주식 데이터를 수집하고 분석합니다.
ReAct 에이전트 대신 직접 도구를 호출하는 방식으로 변경되었습니다.
"""

from typing import Dict, Any, Optional, List, Literal
from pydantic import BaseModel, Field
import json

from src.agents.tools.financial_tools import (
    search_stocks,
    get_stock_info,
    get_historical_prices,
    web_search,
    get_analyst_recommendations
)
from src.model.llm import get_llm_manager
from src.utils.logger import get_logger
from src.utils.config import Config

logger = get_logger(__name__)


class StockData(BaseModel):
    """개별 주식 데이터 모델 (comparison용)"""
    ticker: str
    company_name: str
    current_price: float
    analysis: str
    metrics: Dict[str, Any] = Field(default_factory=dict)
    analyst_recommendation: Optional[str] = None


class AnalysisResult(BaseModel):
    """Financial Analyst의 분석 결과를 위한 Structured Output 모델"""
    analysis_type: Literal["single", "comparison", "concept", "definition", "error"]

    # Single stock analysis fields
    ticker: Optional[str] = None
    company_name: Optional[str] = None
    current_price: Optional[float] = None

    # 공통 필드
    analysis: str = Field(description="분석 내용 또는 설명 (필수)")
    metrics: Optional[Dict[str, Any]] = None
    period: Optional[str] = None
    analyst_recommendation: Optional[str] = None

    # Comparison fields
    stocks: Optional[List[Dict[str, Any]]] = None
    comparison_summary: Optional[str] = None

    # Concept/Definition fields
    query: Optional[str] = None

    # Error fields
    error: Optional[str] = None


class FinancialAnalyst:
    def __init__(self, model_name: str = None, temperature: float = 0):
        """
        Financial Analyst를 초기화합니다.

        Args:
            model_name: 사용할 모델명 (default: Config.LLM_MODEL)
            temperature: LLM 온도 (0 = 결정적, 1 = 창의적)
        """
        if model_name is None:
            model_name = Config.LLM_MODEL
        logger.info(f"Financial Analyst 초기화 (Structured Output) - model: {model_name}, temp: {temperature}")

        # LLM Manager에서 모델 가져오기
        llm_manager = get_llm_manager()
        self.llm = llm_manager.get_model(model_name, temperature=temperature)

        logger.info("Financial Analyst 초기화 완료")

    def analyze(self, query: str, messages: list = None) -> Dict[str, Any]:
        """
        주어진 질문에 대해 금융 분석을 수행합니다.

        Args:
            query: 사용자 질문
            messages: 대화 히스토리 (선택사항)

        Returns:
            분석 결과를 담은 딕셔너리
        """
        if messages is None:
            messages = []

        try:
            logger.info(f"분석 시작 - query: {query}")

            # Step 1: 질문 분석 및 티커 추출
            tickers = self._extract_tickers(query)

            if not tickers:
                logger.warning("티커를 찾을 수 없음 - 개념/정의 질문으로 처리")
                return self._handle_concept_query(query)

            logger.info(f"✅ 티커 추출: {tickers}")

            # Step 2: 단일 vs 비교 분석 분기
            if len(tickers) == 1:
                # 단일 주식 분석
                ticker = tickers[0]
                stock_data = self._collect_stock_data(ticker, query)

                if not stock_data:
                    return {
                        "analysis_type": "error",
                        "ticker": ticker,
                        "company_name": "Unknown",
                        "current_price": 0,
                        "analysis": f"{ticker} 주식 정보를 가져올 수 없습니다.",
                        "error": "데이터 수집 실패"
                    }

                result = self._generate_analysis(query, stock_data, messages)
                logger.info(f"분석 완료 - type: {result.get('analysis_type', 'N/A')}")
                return result

            else:
                # 여러 주식 비교 분석
                logger.info(f"🔄 비교 분석 모드 - {len(tickers)}개 종목")
                return self._compare_multiple_stocks(tickers, query, messages)

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

    def _extract_company_names(self, query: str) -> List[str]:
        """
        질문에서 회사명을 추출합니다 (여러 개 가능).

        Args:
            query: 사용자 질문

        Returns:
            회사명 리스트 (없으면 빈 리스트)
        """
        try:
            prompt = f"""다음 질문에서 주식 종목 회사명을 추출하세요.

질문: {query}

규칙:
- 회사명만 추출 (예: "삼성전자", "애플", "테슬라")
- 여러 회사가 있으면 모두 추출
- 각 회사명은 새 줄로 구분
- 회사명이 없으면 "NONE" 반환
- 부가 설명 없이 회사명만 나열

회사명:"""

            response = self.llm.invoke(prompt)
            content = response.content.strip()

            if content == "NONE" or not content:
                return []

            # 줄바꿈으로 구분된 회사명 파싱
            companies = [line.strip() for line in content.split('\n') if line.strip()]
            # 숫자 제거 (1. 삼성전자 → 삼성전자)
            companies = [c.lstrip('0123456789.-) ').strip() for c in companies]
            companies = [c for c in companies if c and c != "NONE"]

            logger.info(f"✅ 회사명 추출: '{query}' → {companies}")
            return companies

        except Exception as e:
            logger.error(f"회사명 추출 실패: {e}")
            return []

    def _extract_tickers(self, query: str) -> List[str]:
        """
        질문에서 티커를 추출합니다 (여러 개 가능).

        Args:
            query: 사용자 질문

        Returns:
            티커 리스트 (없으면 빈 리스트)
        """
        try:
            # Step 1: 질문에서 회사명 추출
            company_names = self._extract_company_names(query)
            if not company_names:
                logger.warning("회사명을 추출할 수 없음")
                return []

            # Step 2: 각 회사명으로 티커 검색
            tickers = []
            for company_name in company_names:
                logger.info(f"티커 검색 중: {company_name}")
                result = search_stocks.invoke({"query": company_name, "max_results": 1})

                if "찾을 수 없습니다" in result or "오류" in result:
                    logger.warning(f"티커 검색 실패: {company_name}")
                    continue

                # 결과에서 첫 번째 티커 추출
                # 포맷: "• TICKER - Company Name [EXCHANGE]"
                import re
                match = re.search(r'•\s*([A-Z0-9.]+)\s*-', result)
                if match:
                    ticker = match.group(1)
                    logger.info(f"✅ 티커 추출 성공: {ticker}")
                    tickers.append(ticker)
                else:
                    logger.warning(f"티커 파싱 실패 - result: {result[:200]}")

            return tickers

        except Exception as e:
            logger.error(f"티커 추출 실패: {e}")
            return []

    def _collect_stock_data(self, ticker: str, query: str) -> Optional[Dict[str, Any]]:
        """
        티커에 대한 모든 데이터를 수집합니다.

        Args:
            ticker: 주식 티커
            query: 사용자 질문

        Returns:
            수집된 데이터 딕셔너리
        """
        try:
            collected_data = {"ticker": ticker}

            # 1. 주식 기본 정보
            logger.info(f"📊 주식 정보 조회: {ticker}")
            try:
                stock_info = get_stock_info.invoke({"ticker": ticker})
                collected_data["stock_info"] = stock_info
                logger.info(f"✅ 주식 정보 수집 완료")
            except Exception as e:
                logger.warning(f"⚠️ 주식 정보 수집 실패: {e}")
                collected_data["stock_info"] = {}

            # 2. 과거 가격 데이터
            logger.info(f"📈 과거 가격 데이터 조회: {ticker}")
            try:
                historical = get_historical_prices.invoke({"ticker": ticker, "period": "3mo", "interval": "1d"})
                collected_data["historical"] = historical
                logger.info(f"✅ 과거 데이터 수집 완료")
            except Exception as e:
                logger.warning(f"⚠️ 과거 데이터 수집 실패: {e}")
                collected_data["historical"] = ""

            # 3. 웹 검색 (뉴스/분석)
            logger.info(f"🔍 웹 검색: {query}")
            try:
                web_result = web_search.invoke({"query": f"{ticker} stock news analysis"})
                collected_data["web_search"] = web_result
                logger.info(f"✅ 웹 검색 완료")
            except Exception as e:
                logger.warning(f"⚠️ 웹 검색 실패: {e}")
                collected_data["web_search"] = ""

            # 4. 애널리스트 추천
            logger.info(f"💼 애널리스트 추천 조회: {ticker}")
            try:
                analyst_rec = get_analyst_recommendations.invoke({"ticker": ticker})
                collected_data["analyst_rec"] = analyst_rec
                logger.info(f"✅ 애널리스트 추천 수집 완료")
            except Exception as e:
                logger.warning(f"⚠️ 애널리스트 추천 수집 실패: {e}")
                collected_data["analyst_rec"] = ""

            return collected_data

        except Exception as e:
            logger.error(f"데이터 수집 실패: {e}")
            return None

    def _compare_multiple_stocks(
        self,
        tickers: List[str],
        query: str,
        messages: list
    ) -> Dict[str, Any]:
        """
        여러 주식을 비교 분석합니다.

        Args:
            tickers: 티커 리스트
            query: 사용자 질문
            messages: 대화 히스토리

        Returns:
            비교 분석 결과 딕셔너리
        """
        try:
            logger.info(f"📊 {len(tickers)}개 종목 데이터 수집 시작")

            # Step 1: 각 티커별로 데이터 수집
            stocks_data = []
            for ticker in tickers:
                logger.info(f"📈 {ticker} 데이터 수집 중...")
                stock_data = self._collect_stock_data(ticker, query)

                if stock_data:
                    stock_info = stock_data.get("stock_info", {})
                    stocks_data.append({
                        "ticker": ticker,
                        "company_name": stock_info.get("name", "Unknown"),
                        "current_price": stock_info.get("current_price", 0),
                        "metrics": stock_info.get("metrics", {}),
                        "data": stock_data  # 전체 데이터 보관
                    })
                    logger.info(f"✅ {ticker} 데이터 수집 완료")
                else:
                    logger.warning(f"⚠️ {ticker} 데이터 수집 실패")

            if not stocks_data:
                return {
                    "analysis_type": "error",
                    "stocks": [],
                    "analysis": "모든 종목의 데이터 수집에 실패했습니다.",
                    "error": "데이터 수집 실패"
                }

            # Step 2: Structured Output으로 비교 분석 생성
            logger.info("🤖 비교 분석 생성 중...")
            result = self._generate_comparison_analysis(query, stocks_data, messages)

            logger.info(f"✅ 비교 분석 완료 - {len(stocks_data)}개 종목")
            return result

        except Exception as e:
            logger.error(f"비교 분석 실패: {e}")
            return {
                "analysis_type": "error",
                "stocks": [],
                "analysis": f"비교 분석 중 오류 발생: {str(e)}",
                "error": str(e)
            }

    def _generate_comparison_analysis(
        self,
        query: str,
        stocks_data: List[Dict[str, Any]],
        messages: list
    ) -> Dict[str, Any]:
        """
        여러 종목의 비교 분석을 생성합니다.

        Args:
            query: 사용자 질문
            stocks_data: 각 종목의 수집된 데이터 리스트
            messages: 대화 히스토리

        Returns:
            비교 분석 결과 딕셔너리
        """
        # 각 종목 요약 (폴백용으로도 사용)
        stocks_summary = []
        for stock in stocks_data:
            stocks_summary.append({
                "ticker": stock["ticker"],
                "company_name": stock["company_name"],
                "current_price": stock["current_price"],
                "metrics": stock.get("metrics", {})
            })

        try:
            # Structured Output 설정
            llm_with_structure = self.llm.with_structured_output(AnalysisResult)

            # 프롬프트 구성
            analysis_prompt = f"""당신은 전문 금융 애널리스트입니다.

다음 {len(stocks_data)}개 종목을 비교 분석하세요.

사용자 질문: {query}

종목 데이터:
{json.dumps(stocks_summary, ensure_ascii=False, indent=2)}

분석 요구사항:
1. analysis_type: "comparison"
2. stocks: 각 종목의 핵심 데이터 (ticker, company_name, current_price, metrics)
3. analysis: 종목 간 비교 분석 (각 종목의 강점/약점, 투자 추천 포함, 5-7문장)
4. comparison_summary: 전체 비교 요약 (2-3문장)

CRITICAL: 간결하고 명확하게 작성하세요.
"""

            # Structured Output으로 분석 생성
            result = llm_with_structure.invoke(analysis_prompt)

            # Pydantic 모델을 딕셔너리로 변환
            return result.model_dump()

        except Exception as e:
            logger.error(f"비교 분석 생성 실패: {e}")

            # 폴백: 기본 구조로 반환
            return {
                "analysis_type": "comparison",
                "stocks": stocks_summary,
                "analysis": f"{len(stocks_data)}개 종목의 데이터를 수집했으나 비교 분석 생성에 실패했습니다.",
                "comparison_summary": "분석 생성 실패"
            }

    def _generate_analysis(
        self,
        query: str,
        stock_data: Dict[str, Any],
        messages: list
    ) -> Dict[str, Any]:
        """
        수집된 데이터를 기반으로 최종 분석을 생성합니다 (Structured Output).

        Args:
            query: 사용자 질문
            stock_data: 수집된 주식 데이터
            messages: 대화 히스토리

        Returns:
            AnalysisResult 딕셔너리
        """
        try:
            # Structured Output 설정
            llm_with_structure = self.llm.with_structured_output(AnalysisResult)

            # 데이터 요약
            ticker = stock_data.get("ticker", "UNKNOWN")
            stock_info = stock_data.get("stock_info", {})
            company_name = stock_info.get("company_name", "Unknown")
            current_price = stock_info.get("current_price", 0)
            metrics = stock_info.get("metrics", {})

            # 프롬프트 구성
            analysis_prompt = f"""당신은 전문 금융 애널리스트입니다.

수집된 데이터를 기반으로 {company_name}({ticker}) 주식에 대한 분석을 제공하세요.

사용자 질문: {query}

수집된 데이터:
- 회사명: {company_name}
- 티커: {ticker}
- 현재가: {current_price}
- 재무 지표: {json.dumps(metrics, ensure_ascii=False)[:500]}
- 과거 데이터: {str(stock_data.get('historical', ''))[:300]}
- 웹 검색 결과: {str(stock_data.get('web_search', ''))[:500]}
- 애널리스트 추천: {str(stock_data.get('analyst_rec', ''))[:300]}

분석 요구사항:
1. analysis_type: "single"
2. ticker, company_name, current_price: 위 데이터 사용
3. analysis: 종합적인 분석 의견 (3-5문장, 핵심 포인트 중심)
4. metrics: 주요 재무 지표
5. analyst_recommendation: 매수/보류/매도 중 하나

CRITICAL: 간결하고 명확하게 작성하세요.
"""

            # Structured Output으로 분석 생성
            result = llm_with_structure.invoke(analysis_prompt)

            # Pydantic 모델을 딕셔너리로 변환
            return result.model_dump()

        except Exception as e:
            logger.error(f"분석 생성 실패: {e}")

            # 폴백: 기본 구조로 반환
            stock_info = stock_data.get("stock_info", {})
            return {
                "analysis_type": "single",
                "ticker": stock_data.get("ticker", "UNKNOWN"),
                "company_name": stock_info.get("company_name", "Unknown"),
                "current_price": stock_info.get("current_price", 0),
                "analysis": f"{stock_info.get('company_name', 'Unknown')} 주식에 대한 분석 데이터를 수집했습니다.",
                "metrics": stock_info.get("metrics", {}),
                "period": "3mo",
                "analyst_recommendation": "N/A"
            }

    def _handle_concept_query(self, query: str) -> Dict[str, Any]:
        """
        개념/정의 질문을 처리합니다 (티커 없는 경우).

        Args:
            query: 사용자 질문

        Returns:
            AnalysisResult 딕셔너리
        """
        try:
            logger.info(f"개념 질문 처리: {query}")

            # LLM에게 직접 답변 요청
            concept_prompt = f"""당신은 금융 전문가입니다.

다음 질문에 대해 명확하고 간결하게 답변하세요:

질문: {query}

답변 (3-5문장):"""

            response = self.llm.invoke(concept_prompt)
            explanation = response.content.strip()

            return {
                "analysis_type": "concept",
                "query": query,
                "analysis": explanation
            }

        except Exception as e:
            logger.error(f"개념 질문 처리 실패: {e}")
            return {
                "analysis_type": "error",
                "query": query,
                "analysis": f"질문을 처리할 수 없습니다: {str(e)}",
                "error": str(e)
            }

    def compare_stocks(self, tickers: List[str], messages: list = None) -> Dict[str, Any]:
        """
        여러 주식을 비교 분석합니다.

        Args:
            tickers: 비교할 티커 리스트 (예: ["AAPL", "MSFT", "GOOGL"])
            messages: 대화 히스토리

        Returns:
            비교 분석 결과 딕셔너리
        """
        if messages is None:
            messages = []

        try:
            logger.info(f"비교 분석 시작 - tickers: {tickers}")

            # 자동으로 비교 쿼리 생성
            ticker_str = ", ".join(tickers)
            query = f"{ticker_str} 주식들을 비교 분석해주세요. 각각의 장단점과 투자 추천을 포함해주세요."

            return self.analyze(query=query, messages=messages)

        except Exception as e:
            logger.error(f"비교 분석 실패 - tickers: {tickers}, error: {str(e)}")
            return {
                "error": str(e),
                "analysis_type": "comparison",
                "stocks": [],
                "comparison_analysis": f"비교 분석 중 오류가 발생했습니다: {str(e)}"
            }

    def invoke(self, query: str, messages: list = None) -> Dict[str, Any]:
        """
        analyze()의 별칭 메서드 (LangChain 스타일 호환)

        Args:
            query: 사용자 질문
            messages: 대화 히스토리

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
    Financial Analyst를 생성합니다.

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
    print(f"분석: {result.get('analysis', '')[:200]}...")
