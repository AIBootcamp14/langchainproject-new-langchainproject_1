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
    historical: Optional[str] = None  # 과거 가격 데이터 (차트 생성용)

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
        self.llm_manager = get_llm_manager()
        self.llm = self.llm_manager.get_model(model_name, temperature=temperature)

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
        질문에서 회사명 또는 티커 심볼을 추출합니다 (여러 개 가능).

        Args:
            query: 사용자 질문

        Returns:
            회사명/티커 리스트 (없으면 빈 리스트)
        """
        try:
            # llm.py의 "extract_company_names" 프롬프트 사용
            prompt = self.llm_manager.get_prompt("extract_company_names")
            formatted_prompt = prompt.format_messages(query=query)

            response = self.llm.invoke(formatted_prompt)
            content = response.content.strip()

            if content == "NONE" or not content:
                return []

            # 줄바꿈으로 구분된 회사명 파싱
            companies = [line.strip() for line in content.split('\n') if line.strip()]
            # 숫자 제거 (1. 삼성전자 → 삼성전자)
            companies = [c.lstrip('0123456789.-) ').strip() for c in companies]
            # 헤더 라인 제거 ("회사명:", "종목:", "Company:", 등)
            companies = [c for c in companies if c and c != "NONE" and not c.endswith(':') and '회사명' not in c and '종목' not in c and 'company' not in c.lower()]

            logger.info(f"✅ 종목/티커 추출: '{query}' → {companies}")
            return companies

        except Exception as e:
            logger.error(f"종목/티커 추출 실패: {e}")
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
            # Step 1: 질문에서 회사명/티커 추출
            company_names = self._extract_company_names(query)
            if not company_names:
                logger.warning("종목명/티커를 추출할 수 없음")
                return []

            # Step 2: 각 종목명/티커로 티커 검색
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
                    # 중복 체크
                    if ticker not in tickers:
                        logger.info(f"✅ 티커 추출 성공: {ticker}")
                        tickers.append(ticker)
                    else:
                        logger.info(f"⚠️  {ticker}는 이미 추출된 티커 (중복 제거)")
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

                # 52주 최고가/최저가가 없으면 과거 데이터에서 계산
                stock_info = collected_data.get("stock_info", {})
                if (stock_info.get("52week_high", 0) == 0 or stock_info.get("52week_low", 0) == 0) and historical:
                    try:
                        # historical 데이터 파싱 (CSV 형식 또는 딕셔너리)
                        import pandas as pd
                        if isinstance(historical, str):
                            from io import StringIO
                            # 첫 줄은 메타데이터, 그 다음부터 CSV
                            lines = historical.strip().split('\n')
                            if len(lines) > 1:
                                csv_data = '\n'.join(lines[1:])
                                df = pd.read_csv(StringIO(csv_data))
                            else:
                                df = pd.DataFrame()
                        elif isinstance(historical, dict):
                            df = pd.DataFrame(historical)
                        else:
                            df = historical

                        if not df.empty and 'High' in df.columns and 'Low' in df.columns:
                            high_52w = df['High'].max()
                            low_52w = df['Low'].min()

                            # stock_info 업데이트
                            if stock_info.get("52week_high", 0) == 0:
                                stock_info["52week_high"] = high_52w
                                logger.info(f"✅ 52주 최고가 계산: {high_52w:.2f}")

                            if stock_info.get("52week_low", 0) == 0:
                                stock_info["52week_low"] = low_52w
                                logger.info(f"✅ 52주 최저가 계산: {low_52w:.2f}")

                            collected_data["stock_info"] = stock_info
                    except Exception as calc_err:
                        logger.warning(f"⚠️ 52주 데이터 계산 실패: {calc_err}")
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

                    # metrics를 stock_info 전체 데이터로 구성 (중복 제거)
                    metrics = {
                        "pe_ratio": stock_info.get("pe_ratio"),
                        "forward_pe": stock_info.get("forward_pe"),
                        "pb_ratio": stock_info.get("pb_ratio"),
                        "market_cap": stock_info.get("market_cap", 0),
                        "dividend_yield": stock_info.get("dividend_yield", 0),
                        "52week_high": stock_info.get("52week_high", 0),
                        "52week_low": stock_info.get("52week_low", 0),
                        "volume": stock_info.get("volume", 0),
                        "avg_volume": stock_info.get("avg_volume", 0),
                        "sector": stock_info.get("sector", "N/A"),
                        "industry": stock_info.get("industry", "N/A")
                    }

                    stocks_data.append({
                        "ticker": ticker,
                        "company_name": stock_info.get("name", "Unknown"),
                        "current_price": stock_info.get("current_price", 0),
                        "metrics": metrics,
                        "historical": stock_data.get("historical", ""),  # 차트 생성용
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

            # llm.py의 "analyze_comparison" 프롬프트 사용
            prompt = self.llm_manager.get_prompt("analyze_comparison")
            formatted_prompt = prompt.format_messages(
                query=query,
                stocks_summary=json.dumps(stocks_summary, ensure_ascii=False, indent=2)
            )

            # Structured Output으로 분석 생성
            result = llm_with_structure.invoke(formatted_prompt)

            # Pydantic 모델을 딕셔너리로 변환
            result_dict = result.model_dump()

            # stocks를 historical 포함된 stocks_data로 교체
            result_dict["stocks"] = stocks_data

            return result_dict

        except Exception as e:
            logger.error(f"비교 분석 생성 실패: {e}")

            # 폴백: 기본 구조로 반환
            return {
                "analysis_type": "comparison",
                "stocks": stocks_data,  # historical 포함된 stocks_data 사용
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
            company_name = stock_info.get("name", stock_info.get("company_name", "Unknown"))
            current_price = stock_info.get("current_price", 0)

            # metrics를 stock_info에서 직접 구성 (중복 제거)
            metrics = {
                "pe_ratio": stock_info.get("pe_ratio"),
                "forward_pe": stock_info.get("forward_pe"),
                "pb_ratio": stock_info.get("pb_ratio"),
                "market_cap": stock_info.get("market_cap", 0),
                "dividend_yield": stock_info.get("dividend_yield", 0),
                "52week_high": stock_info.get("52week_high", 0),
                "52week_low": stock_info.get("52week_low", 0),
                "volume": stock_info.get("volume", 0),
                "avg_volume": stock_info.get("avg_volume", 0),
                "sector": stock_info.get("sector", "N/A"),
                "industry": stock_info.get("industry", "N/A")
            }

            # historical 데이터 정보 추출
            historical_info = "없음"
            historical_data = stock_data.get('historical', '')
            if historical_data and len(historical_data.strip()) > 0:
                # 첫 줄에서 메타데이터 추출 (예: "005930.KS 과거 가격 (3mo, 1d 간격) - 총 60개 데이터 포인트")
                first_line = historical_data.strip().split('\n')[0]
                historical_info = f"수집 완료 ({first_line})"

            # llm.py의 "analyze_single_stock" 프롬프트 사용
            prompt = self.llm_manager.get_prompt("analyze_single_stock")
            formatted_prompt = prompt.format_messages(
                company_name=company_name,
                ticker=ticker,
                query=query,
                current_price=current_price,
                metrics=json.dumps(metrics, ensure_ascii=False)[:500],
                historical_info=historical_info,
                web_search=str(stock_data.get('web_search', ''))[:500],
                analyst_rec=str(stock_data.get('analyst_rec', ''))[:300]
            )

            # Structured Output으로 분석 생성
            result = llm_with_structure.invoke(formatted_prompt)

            # Pydantic 모델을 딕셔너리로 변환
            result_dict = result.model_dump()

            # historical 데이터 추가 (차트 생성용)
            result_dict["historical"] = stock_data.get("historical", "")

            # metrics를 실제 수집된 데이터로 덮어쓰기 (LLM이 잘못 생성한 경우 방지)
            result_dict["metrics"] = metrics

            return result_dict

        except Exception as e:
            logger.error(f"분석 생성 실패: {e}")

            # 폴백: 기본 구조로 반환
            stock_info = stock_data.get("stock_info", {})

            # metrics 구성 (stock_info는 평탄한 구조)
            fallback_metrics = {
                "pe_ratio": stock_info.get("pe_ratio"),
                "forward_pe": stock_info.get("forward_pe"),
                "pb_ratio": stock_info.get("pb_ratio"),
                "market_cap": stock_info.get("market_cap", 0),
                "dividend_yield": stock_info.get("dividend_yield", 0),
                "52week_high": stock_info.get("52week_high", 0),
                "52week_low": stock_info.get("52week_low", 0),
                "volume": stock_info.get("volume", 0),
                "avg_volume": stock_info.get("avg_volume", 0),
                "sector": stock_info.get("sector", "N/A"),
                "industry": stock_info.get("industry", "N/A")
            }

            return {
                "analysis_type": "single",
                "ticker": stock_data.get("ticker", "UNKNOWN"),
                "company_name": stock_info.get("company_name", "Unknown"),
                "current_price": stock_info.get("current_price", 0),
                "analysis": f"{stock_info.get('company_name', 'Unknown')} 주식에 대한 분석 데이터를 수집했습니다.",
                "metrics": fallback_metrics,
                "historical": stock_data.get("historical", ""),
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

            # llm.py의 "analyze_concept" 프롬프트 사용
            prompt = self.llm_manager.get_prompt("analyze_concept")
            formatted_prompt = prompt.format_messages(query=query)

            response = self.llm.invoke(formatted_prompt)
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
