# src/agents/tools/fianancial_tools.py
"""
Financial Tools for Financial Analyst Agent

이 모듈은 financial_analyst 에이전트가 사용하는 금융 분석 도구들을 제공합니다.
- 주식 검색 (영어 및 한국어 지원)
- 주식 기본 정보 조회 (yfinance 기반)
- 웹 검색 (Tavily API 사용 + Tavily 결과 빈 값 반환시 웹페이지 직접 로드)
- 과거 가격 데이터 조회 (yfinance 기반)
- 애널리스트 추천 정보 조회 (yfinance 기반)
"""

import json
import os
import re
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

import yfinance as yf
import pandas as pd
from deep_translator import GoogleTranslator
from tavily import TavilyClient
from langchain_core.tools import tool
from langchain_community.document_loaders import WebBaseLoader

from src.utils.logger import get_logger
from src.utils.config import Config

absolute_path = os.path.dirname(os.path.abspath(__file__))
current_path = os.path.dirname(absolute_path)
logger = get_logger(__name__)


def is_korean(text: str) -> bool:
    """텍스트에 한글이 포함되어 있는지 확인"""
    return bool(re.search('[가-힣]', text))


def translate_to_english(text: str) -> str:
    """한국어를 영어로 자동 번역"""
    try:
        translated = GoogleTranslator(source='ko', target='en').translate(text)
        return translated
    except Exception as e:
        logger.error(f"번역 실패: {e}")
        return text


# 한국 주요 기업 티커 매핑 (번역 오류 방지)
KOREAN_STOCK_TICKERS = {
    # 메이저 기업
    "삼성전자": "005930.KS",
    "삼성": "005930.KS",
    "네이버": "035420.KS",
    "카카오": "035720.KS",
    "카카오뱅크": "323410.KS",
    "카카오페이": "377300.KS",
    "엔씨소프트": "036570.KS",
    "엔씨": "036570.KS",
    "현대차": "005380.KS",
    "현대자동차": "005380.KS",
    "기아": "000270.KS",
    "sk하이닉스": "000660.KS",
    "하이닉스": "000660.KS",
    "lg전자": "066570.KS",
    "lg화학": "051910.KS",
    "lg에너지솔루션": "373220.KS",
    "포스코": "005490.KS",
    "셀트리온": "068270.KS",
    "삼성바이오로직스": "207940.KS",
    "삼성바이오": "207940.KS",
    "삼성sdi": "006400.KS",
    "sk이노베이션": "096770.KS",
    "sk": "034730.KS",
    "sk텔레콤": "017670.KS",
    "kt": "030200.KS",
    "lg유플러스": "032640.KS",
    "신한지주": "055550.KS",
    "kb금융": "105560.KS",
    "하나금융지주": "086790.KS",
    "크래프톤": "259960.KS",
}


def get_korean_ticker(company_name: str) -> Optional[str]:
    """한국 기업명에서 직접 티커를 찾습니다 (번역 오류 방지).

    Args:
        company_name: 한국어 기업명

    Returns:
        티커 심볼 (예: "035720.KS") 또는 None
    """
    # 정확한 매칭 (대소문자 무시)
    normalized = company_name.strip().lower()

    for key, ticker in KOREAN_STOCK_TICKERS.items():
        if key.lower() == normalized:
            logger.info(f"✅ 한국 기업 티커 매핑: '{company_name}' → {ticker}")
            return ticker

    # 부분 매칭 (회사명이 키에 포함되는 경우)
    for key, ticker in KOREAN_STOCK_TICKERS.items():
        if normalized in key.lower() or key.lower() in normalized:
            logger.info(f"✅ 한국 기업 티커 부분 매핑: '{company_name}' → {ticker} (키워드: {key})")
            return ticker

    return None


def load_web_page(url: str) -> str:
    """웹 페이지를 로드하고 정제된 텍스트를 반환합니다."""
    try:
        loader = WebBaseLoader(url, verify_ssl=False)
        content = loader.load()
        
        if not content:
            raise ValueError(f"페이지 내용이 비어있습니다: {url}")

        raw_content = content[0].page_content.strip()

        # 과도한 공백 정리
        while '\n\n\n' in raw_content or '\t\t\t' in raw_content:
            raw_content = raw_content.replace('\n\n\n', '\n\n')
            raw_content = raw_content.replace('\t\t\t', '\t\t')

        return raw_content
        
    except Exception as e:
        logger.error(f"웹 페이지 로드 실패 - url: {url}, error: {str(e)}")
        raise


@tool
def search_stocks(query: str, max_results: int = 10) -> str:
    """회사명, 키워드, 산업명으로 주식을 검색하여 티커 심볼을 찾습니다.

    사용자가 티커 심볼을 모르거나, 특정 분야의 주식을 찾을 때 사용합니다.
    한국어 검색어도 지원하며, 자동으로 영어로 번역하여 검색합니다.

    검색 가능한 항목:
    - 회사명 (예: "Apple", "Microsoft", "Tesla", "애플", "삼성전자")
    - 산업명 (예: "semiconductor", "electric vehicle", "반도체")
    - 키워드 (예: "AI", "cloud computing", "전기차")

    Args:
        query: 검색어 (한국어/영어 모두 가능)
              회사명, 산업명, 또는 관련 키워드를 입력
        max_results: 반환할 최대 검색 결과 수 (기본값: 10)
                    너무 많은 결과는 분석을 어렵게 하므로 적절히 조절

    Returns:
        검색된 주식 티커 목록 (포맷팅된 문자열)
        - 성공 시: 티커 심볼, 회사명, 거래소 정보를 포함한 목록
        - 실패 시: 검색 결과가 없다는 메시지 또는 오류 메시지

    Examples:
        >>> search_stocks("Apple")
        '''Apple' 검색 결과:
        ----------------------------------------------------------------------
        • AAPL - Apple Inc. [NASDAQ]
        • AAPL.MX - Apple Inc. [MEX]

        💡 상세 정보를 보려면 get_stock_info 도구를 사용하세요.
        ----------------------------------------------------------------------'''

        >>> search_stocks("애플", max_results=3)
        '''애플' 검색 결과: (영어: 'Apple')
        ----------------------------------------------------------------------
        • AAPL - Apple Inc. [NASDAQ]
        • AAPL.MX - Apple Inc. [MEX]
        • AAPL.BA - Apple Inc. [BUE]

        💡 상세 정보를 보려면 get_stock_info 도구를 사용하세요.
        ----------------------------------------------------------------------'''

        >>> search_stocks("electric vehicle")
        '''electric vehicle' 검색 결과:
        ----------------------------------------------------------------------
        • TSLA - Tesla, Inc. [NASDAQ]
        • NIO - NIO Inc. [NYSE]
        • RIVN - Rivian Automotive, Inc. [NASDAQ]
        ...
        ----------------------------------------------------------------------'''
    """
    try:
        # JSON 문자열로 전달된 경우 파싱
        if isinstance(query, str) and query.strip().startswith('{'):
            try:
                parsed = json.loads(query)
                query = parsed.get('query', query)
                max_results = parsed.get('max_results', max_results)
            except:
                pass  # JSON이 아니면 그대로 사용

        original_query = query

        # 한국 기업 티커 직접 매핑 시도 (번역 오류 방지)
        if is_korean(query):
            korean_ticker = get_korean_ticker(query)
            if korean_ticker:
                # 매핑된 티커로 직접 정보 조회
                try:
                    stock = yf.Ticker(korean_ticker)
                    info = stock.info
                    company_name = info.get('longName', info.get('shortName', query))

                    output = f"""'{original_query}' 검색 결과: (한국 주요 기업 직접 매핑)
{'-' * 70}
• {korean_ticker} - {company_name} [KRX]

💡 상세 정보를 보려면 get_stock_info 도구를 사용하세요.
{'-' * 70}"""
                    logger.info(f"한국 기업 직접 매핑 성공 - query: {query} → {korean_ticker}")
                    return output.strip()
                except Exception as e:
                    logger.warning(f"매핑된 티커 정보 조회 실패: {e}, yfinance 검색으로 폴백")

        # 한글이 포함되었지만 매핑되지 않은 경우 영어로 번역
        if is_korean(query):
            query = translate_to_english(query)
            logger.info(f"검색어 번역: '{original_query}' → '{query}'")

        logger.info(f"주식 검색 시작 - query: {query}, max_results: {max_results}")
        
        # yfinance Search API 사용
        results = yf.Search(query, max_results=max_results)
        
        if not results.quotes:
            logger.warning(f"검색 결과 없음 - query: {query}")
            
            # 한국어 검색 시 추가 안내
            if original_query != query:
                return f"""'{original_query}' (영어: '{query}')에 대한 검색 결과가 없습니다.

💡 검색 팁:
- 회사명을 조금 다르게 입력해보세요 (예: '삼성' 대신 '삼성전자')
- 영어로 직접 검색해보세요
- 티커 심볼을 알고 있다면 get_stock_info를 사용하세요"""
            
            return f"'{query}'에 대한 검색 결과가 없습니다."
        
        # 결과 포맷팅
        output = f"""'{original_query}' 검색 결과:"""
        if original_query != query:
            output += f" (영어: '{query}')"
        output += f"\n{'-' * 70}\n"
        
        for item in results.quotes:
            symbol = item['symbol']
            name = item.get('longname', item.get('shortname', '이름 없음'))
            exchange = item.get('exchange', '거래소 정보 없음')
            output += f"• {symbol} - {name} [{exchange}]\n"
        
        output += f"\n💡 상세 정보를 보려면 get_stock_info 도구를 사용하세요.\n{'-' * 70}\n"
        
        logger.info(f"주식 검색 완료 - query: {query}, 결과 수: {len(results.quotes)}")
        return output.strip()
    
    except Exception as e:
        logger.error(f"주식 검색 실패 - query: {query}, error: {str(e)}")
        return f"검색 중 오류 발생: {str(e)}"


@tool
def get_stock_info(ticker: str) -> Dict[str, Any]:
    """특정 주식의 상세 정보를 조회합니다.

    yfinance API를 사용하여 주식의 현재가, 시가총액, 밸류에이션 지표,
    52주 최고/최저가, 거래량, 기업 정보 등을 실시간으로 조회합니다.

    이 도구는 단일 주식 분석의 핵심 데이터 소스이며,
    비교 분석 시에도 각 주식별로 호출하여 데이터를 수집합니다.

    Args:
        ticker: 주식 티커 심볼
               - 미국 주식: "AAPL", "MSFT", "GOOGL", "TSLA" 등
               - 한국 주식: "005930.KS" (삼성전자), "035720.KS" (카카오) 등
               - 기타: 각 거래소의 티커 규칙을 따름

    Returns:
        주식 정보를 담은 딕셔너리 (Dict[str, Any])
        포함 필드:
        - symbol: 티커 심볼
        - name: 회사명
        - current_price: 현재가
        - previous_close: 전일 종가
        - open: 시가
        - day_high: 당일 최고가
        - day_low: 당일 최저가
        - market_cap: 시가총액
        - pe_ratio: PER (Trailing P/E Ratio)
        - forward_pe: Forward P/E Ratio
        - dividend_yield: 배당수익률
        - 52week_high: 52주 최고가
        - 52week_low: 52주 최저가
        - volume: 거래량
        - avg_volume: 평균 거래량
        - sector: 섹터
        - industry: 산업
        - country: 국가
        - website: 웹사이트 URL
        - summary: 기업 개요

        오류 발생 시: {"error": "오류 메시지"}

    Examples:
        >>> get_stock_info("AAPL")
        {
            "symbol": "AAPL",
            "name": "Apple Inc.",
            "current_price": 178.25,
            "market_cap": 2800000000000,
            "pe_ratio": 29.5,
            "52week_high": 199.62,
            "52week_low": 164.08,
            "sector": "Technology",
            "industry": "Consumer Electronics",
            ...
        }

        >>> get_stock_info("TSLA")
        {
            "symbol": "TSLA",
            "name": "Tesla, Inc.",
            "current_price": 242.84,
            "market_cap": 770000000000,
            "pe_ratio": 65.3,
            ...
        }

        >>> get_stock_info("INVALID")
        {"error": "Failed to fetch info for INVALID: ..."}
    """
    try:
        # JSON 문자열로 전달된 경우 파싱
        if isinstance(ticker, str) and ticker.strip().startswith('{'):
            try:
                parsed = json.loads(ticker)
                ticker = parsed.get('ticker', ticker)
            except:
                pass  # JSON이 아니면 그대로 사용

        logger.info(f"주식 정보 조회 시작 - ticker: {ticker}")

        stock = yf.Ticker(ticker)
        info = stock.info
        
        result = {
            "symbol": info.get('symbol', ticker),
            "name": info.get('longName', info.get('shortName', 'N/A')),
            "current_price": info.get('currentPrice', info.get('regularMarketPrice', 0)),
            "previous_close": info.get('previousClose', 0),
            "open": info.get('open', 0),
            "day_high": info.get('dayHigh', 0),
            "day_low": info.get('dayLow', 0),
            "market_cap": info.get('marketCap', 0),
            "pe_ratio": info.get('trailingPE', None),
            "forward_pe": info.get('forwardPE', None),
            "pb_ratio": info.get('priceToBook', None),
            "dividend_yield": info.get('dividendYield', 0),
            "52week_high": info.get('fiftyTwoWeekHigh', 0),
            "52week_low": info.get('fiftyTwoWeekLow', 0),
            "volume": info.get('volume', 0),
            "avg_volume": info.get('averageVolume', 0),
            "sector": info.get('sector', 'N/A'),
            "industry": info.get('industry', 'N/A'),
            "country": info.get('country', 'N/A'),
            "website": info.get('website', 'N/A'),
            "summary": info.get('longBusinessSummary', 'N/A')
        }
        
        logger.info(f"주식 정보 조회 완료 - ticker: {ticker}")
        return result
    
    except Exception as e:
        logger.error(f"주식 정보 조회 실패 - ticker: {ticker}, error: {str(e)}")
        return {"error": f"Failed to fetch info for {ticker}: {str(e)}"}


@tool
def web_search(query: str) -> str:
    """웹 검색을 수행하여 최신 뉴스, 시장 동향, 기업 정보 등을 조회합니다.

    Tavily API를 사용하여 심층 웹 검색을 수행합니다.
    yfinance에서 제공하지 않는 최신 뉴스, 시장 분석, 기업 이벤트 등을 찾을 때 유용합니다.

    주요 사용 사례:
    - 주가 급등/급락의 원인 조사
    - 최신 기업 뉴스 및 공시 사항 확인
    - 경쟁사 비교 분석을 위한 업계 동향 파악
    - 신제품 출시, M&A, 경영진 변경 등 중요 이벤트 확인
    - 섹터별 시장 트렌드 분석

    Args:
        query: 웹 검색어 (영어 권장)
              구체적이고 명확한 검색어를 사용하면 더 좋은 결과를 얻을 수 있습니다.
              예: "Apple stock price increase reason", "Tesla Q4 earnings"

    Returns:
        검색 결과 요약 (포맷팅된 문자열)
        - 검색된 결과 개수
        - 상위 5개 결과의 제목, URL, 내용 미리보기

        오류 발생 시: 오류 메시지를 포함한 딕셔너리

    Examples:
        >>> web_search("Apple stock surge January 2025")
        '''🌐 'Apple stock surge January 2025' 웹 검색 완료

        📊 검색 결과: 5개

        📝 주요 결과:

        [1] Apple Stock Surges on Strong iPhone Sales
            URL: https://example.com/article1
            내용: Apple Inc. shares jumped 5% today following better-than-expected iPhone 15 sales figures...

        [2] Why AAPL is Up Today
            URL: https://example.com/article2
            내용: Analysts cite robust services revenue and AI integration as key drivers for Apple's stock rally...
        ...'''

        >>> web_search("semiconductor industry outlook 2025")
        '''🌐 'semiconductor industry outlook 2025' 웹 검색 완료

        📊 검색 결과: 7개

        📝 주요 결과:

        [1] Chip Industry Faces Supply Chain Challenges in 2025
            URL: https://example.com/chips
            내용: The global semiconductor market is expected to grow...
        ...'''
    """
    try:
        # JSON 문자열로 전달된 경우 파싱
        if isinstance(query, str) and query.strip().startswith('{'):
            try:
                parsed = json.loads(query)
                query = parsed.get('query', query)
            except:
                pass  # JSON이 아니면 그대로 사용

        # Tavily 클라이언트 초기화
        client = TavilyClient(api_key=Config.TAVILY_API_KEY)
    
        # 검색 실행
        response = client.search(query, search_depth = "advanced", include_raw_content=True)

        results = response.get('results', [])
        
        if not results:
            logger.warning(f"검색 결과 없음 - query: {query}")
            return f"'{query}'에 대한 검색 결과가 없습니다."

        # raw_content 없는 경우 웹페이지 직접 로드
        for result in results:
            if result["raw_content"] is None:
                try:
                    result["raw_content"] = load_web_page(result["url"])
                except Exception as e:
                    logger.error(f"웹 검색 실패 - query: {query}, error: {str(e)}")
                    result["raw_content"] = result["content"]

        # 요약 생성
        output = f"🌐 '{query}' 웹 검색 완료\n\n"
        output += f"📊 검색 결과: {len(results)}개\n\n"
        output += "📝 주요 결과:\n"

        for idx, result in enumerate(results[:5], 1):
            output += f"\n[{idx}] {result.get('title', 'N/A')}\n"
            output += f"    URL: {result.get('url', 'N/A')}\n"
            output += f"    내용: {result.get('content', 'N/A')[:150]}...\n"

        logger.info(f"웹 검색 완료 - query: {query}, 결과: {len(results)}개")
        return output

    except Exception as e:
        logger.error(f"[ERROR] 웹 검색 실패 - query: {query}, error: {str(e)}")
        return {"error": f"❌ 웹 검색 중 문제가 발생했습니다: {str(e)}"}


@tool
def get_historical_prices(
    ticker: str,
    period: str = "1mo",
    interval: str = "1d"
) -> str:
    """주식의 과거 가격 데이터(OHLCV)를 조회합니다.

    yfinance를 사용하여 특정 기간의 시가, 고가, 저가, 종가, 거래량(OHLCV) 데이터를 조회합니다.
    가격 추세 분석, 기술적 분석, 차트 생성 등에 필요한 시계열 데이터를 제공합니다.

    주요 사용 사례:
    - 주가 추세 분석 (상승/하락 패턴 파악)
    - 변동성 분석 (최근 몇 개월간의 가격 변동 폭 확인)
    - 기술적 분석 기초 데이터 (이동평균, RSI 등 계산용)
    - 시간대별 비교 (장중 가격 변동 vs 일간 변동)

    Args:
        ticker: 주식 티커 심볼 (예: "AAPL", "TSLA", "MSFT")
        period: 조회 기간 (기본값: "1mo")
               선택 가능:
               - "1d": 1일
               - "5d": 5일
               - "1mo": 1개월 (기본값)
               - "3mo": 3개월
               - "6mo": 6개월
               - "1y": 1년
               - "2y": 2년
               - "5y": 5년
               - "10y": 10년
               - "ytd": 올해 초부터 현재까지
               - "max": 최대 가능 기간
        interval: 데이터 간격 (기본값: "1d")
                 선택 가능:
                 - 분 단위: "1m", "2m", "5m", "15m", "30m", "60m", "90m"
                 - 시간 단위: "1h"
                 - 일 단위: "1d" (기본값), "5d"
                 - 주/월 단위: "1wk", "1mo", "3mo"

    Returns:
        과거 가격 데이터 (포맷팅된 문자열)
        - 최근 10개 데이터 포인트를 테이블 형식으로 표시
        - 각 행: 날짜/시간, Open, High, Low, Close, Volume
        - 전체 데이터 포인트 개수 정보

        오류 발생 시: 오류 메시지

    Examples:
        >>> get_historical_prices("AAPL", period="1mo", interval="1d")
        '''
        AAPL 과거 가격 (1mo, 1d 간격):
        ================================================================================
                            Open        High         Low       Close    Volume
        Date
        2025-01-01  178.09  179.23  177.54  178.87  45234567
        2025-01-02  178.90  180.12  178.45  179.65  48765432
        ...
        2025-01-30  182.34  183.21  181.90  182.75  52134678

        총 22개 데이터 포인트
        '''

        >>> get_historical_prices("TSLA", period="5d", interval="1h")
        '''
        TSLA 과거 가격 (5d, 1h 간격):
        ================================================================================
                            Open        High         Low       Close    Volume
        Datetime
        2025-01-27 14:00  242.10  242.85  241.50  242.34  1234567
        2025-01-27 15:00  242.35  243.20  242.00  242.80  987654
        ...

        총 40개 데이터 포인트
        '''

        >>> get_historical_prices("GOOGL", period="1y", interval="1wk")
        '''
        GOOGL 과거 가격 (1y, 1wk 간격):
        ================================================================================
                            Open        High         Low       Close      Volume
        Date
        2024-01-29  143.20  145.50  142.80  144.90  112345678
        ...

        총 52개 데이터 포인트
        '''
    """
    try:
        # JSON 문자열로 전달된 경우 파싱
        if isinstance(ticker, str) and ticker.strip().startswith('{'):
            try:
                parsed = json.loads(ticker)
                ticker = parsed.get('ticker', ticker)
                period = parsed.get('period', period)
                interval = parsed.get('interval', interval)
            except:
                pass  # JSON이 아니면 그대로 사용

        logger.info(f"과거 가격 조회 시작 - ticker: {ticker}, period: {period}, interval: {interval}")

        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, interval=interval)
        
        if hist.empty:
            return f"{ticker}의 과거 가격 데이터를 찾을 수 없습니다."
        
        # 차트 생성을 위해 전체 데이터를 CSV 형식으로 반환
        # 첫 줄에 메타데이터, 그 다음 CSV 데이터
        output = f"{ticker} 과거 가격 ({period}, {interval} 간격) - 총 {len(hist)}개 데이터 포인트\n"
        output += hist.to_csv()

        logger.info(f"과거 가격 조회 완료 - ticker: {ticker}, period: {period}, rows: {len(hist)}")
        return output
    
    except Exception as e:
        logger.error(f"과거 가격 조회 실패 - ticker: {ticker}, error: {str(e)}")
        return f"과거 가격 조회 중 오류 발생: {str(e)}"


@tool
def get_analyst_recommendations(ticker: str) -> str:
    """전문 애널리스트들의 주식 추천 정보를 종합적으로 조회합니다.

    yfinance를 통해 월가 애널리스트들의 추천 등급, 목표 주가,
    최근 등급 변경 이력 등을 종합적으로 제공합니다.

    투자 의사결정에 중요한 전문가 의견을 파악하는 데 유용하며,
    특히 목표가 대비 현재가의 상승 여력을 계산하여 제공합니다.

    제공 정보:
    - 현재 추천 등급 (Strong Buy, Buy, Hold, Sell, Strong Sell)
    - 커버하는 애널리스트 수
    - 평균 목표 주가 및 목표가 범위 (최저~최고)
    - 현재가 대비 상승 여력 (%)
    - 최근 10개 추천 이력
    - 최근 10개 등급 변경 이력 (업그레이드/다운그레이드)

    Args:
        ticker: 주식 티커 심볼 (예: "AAPL", "TSLA", "MSFT")

    Returns:
        애널리스트 추천 종합 정보 (포맷팅된 문자열)
        - 현재 추천 요약 (등급, 애널리스트 수, 목표가, 상승여력)
        - 최근 추천 이력 테이블 (날짜, 기관, 등급)
        - 최근 등급 변경 테이블 (날짜, 기관, 이전 등급, 새 등급)

        오류 발생 시: 오류 메시지

    Examples:
        >>> get_analyst_recommendations("AAPL")
        '''
        AAPL 애널리스트 추천:
        ================================================================================

        현재 추천 요약:
          • 추천 등급: BUY
          • 애널리스트 수: 45명
          • 평균 목표가: $195.50
          • 목표가 범위: $175.00 ~ $220.00
          • 현재가: $178.25
          • 상승여력: +9.68%

        최근 애널리스트 추천 (최근 10개):
                    Firm              To Grade     Action
        Date
        2025-01-28  Morgan Stanley    Overweight   main
        2025-01-25  JP Morgan         Buy          up
        2025-01-22  Goldman Sachs     Buy          main
        ...

        최근 등급 변경 (최근 10개):
                    Firm              From Grade   To Grade
        Date
        2025-01-25  JP Morgan         Hold         Buy
        2025-01-15  Wells Fargo       Buy          Overweight
        ...

        총 120개 등급 변경 기록
        '''

        >>> get_analyst_recommendations("TSLA")
        '''
        TSLA 애널리스트 추천:
        ================================================================================

        현재 추천 요약:
          • 추천 등급: HOLD
          • 애널리스트 수: 38명
          • 평균 목표가: $250.00
          • 목표가 범위: $180.00 ~ $350.00
          • 현재가: $242.84
          • 상승여력: +2.95%
        ...
        '''

        >>> get_analyst_recommendations("SMALLCAP")
        '''
        [NOTE] 최근 등급 변경 이력이 없습니다.
        '''
    """
    try:
        # JSON 문자열로 전달된 경우 파싱
        if isinstance(ticker, str) and ticker.strip().startswith('{'):
            try:
                parsed = json.loads(ticker)
                ticker = parsed.get('ticker', ticker)
            except:
                pass  # JSON이 아니면 그대로 사용

        logger.info(f"애널리스트 추천 조회 시작 - ticker: {ticker}")

        stock = yf.Ticker(ticker)
        info = stock.info
        
        output = f"\n{ticker} 애널리스트 추천:\n"
        output += "=" * 80 + "\n\n"
        
        # 1. 현재 추천 등급 및 목표가
        output += "현재 추천 요약:\n"
        output += f"  • 추천 등급: {info.get('recommendationKey', 'N/A').upper()}\n"
        output += f"  • 애널리스트 수: {info.get('numberOfAnalystOpinions', 0)}명\n"
        
        if info.get('targetMeanPrice'):
            output += f"  • 평균 목표가: ${info.get('targetMeanPrice', 0):.2f}\n"
            output += f"  • 목표가 범위: ${info.get('targetLowPrice', 0):.2f} ~ ${info.get('targetHighPrice', 0):.2f}\n"
            
            current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
            if current_price > 0:
                upside = ((info.get('targetMeanPrice', 0) - current_price) / current_price) * 100
                output += f"  • 현재가: ${current_price:.2f}\n"
                output += f"  • 상승여력: {upside:+.2f}%\n"
        
        output += "\n"
        
        # 2. 최근 추천 이력 (테이블)
        recommendations = stock.recommendations
        if recommendations is not None and not recommendations.empty:
            output += "최근 애널리스트 추천 (최근 10개):\n"
            output += recommendations.tail(10).to_string()
            output += "\n\n"
        
        # 3. 최근 등급 변경 이력
        upgrades_downgrades = stock.upgrades_downgrades
        if upgrades_downgrades is not None and not upgrades_downgrades.empty:
            output += "최근 등급 변경 (최근 10개):\n"
            output += upgrades_downgrades.tail(10).to_string()
            output += f"\n\n총 {len(upgrades_downgrades)}개 등급 변경 기록"
        else:
            output += "[NOTE] 최근 등급 변경 이력이 없습니다."
        
        logger.info(f"애널리스트 추천 조회 완료 - ticker: {ticker}")
        return output
    
    except Exception as e:
        logger.error(f"애널리스트 추천 조회 실패 - ticker: {ticker}, error: {str(e)}")
        return f"애널리스트 추천 조회 중 오류 발생: {str(e)}"



# Tool 리스트 (Agent에서 사용)
financial_tools = [
    search_stocks,
    get_stock_info,
    web_search,
    get_historical_prices,
    get_analyst_recommendations,
]