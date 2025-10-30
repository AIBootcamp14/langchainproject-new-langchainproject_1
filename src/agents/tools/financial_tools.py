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
    """회사명, 키워드, 또는 산업으로 주식을 검색합니다.
    사용자가 티커를 모르거나 특정 분야의 주식을 찾을 때 사용합니다.
    한국어 입력도 자동으로 영어로 번역되어 검색됩니다.
    
    Args:
        query: 검색어 (회사명, 산업, 키워드 등 - 한국어/영어 모두 가능)
        max_results: 최대 검색 결과 수 (기본값: 10)
    
    Returns:
        검색된 주식 티커와 회사명 리스트 (포맷팅된 문자열)
    """
    try:
        original_query = query
        
        # 한글이 포함된 경우 영어로 번역
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
    """주식의 기본 정보를 조회합니다.
    현재가, 시가총액, PER, 배당수익률, 52주 최고/최저가 등을 제공합니다.
    
    Args:
        ticker: 주식 티커 심볼 (예: "AAPL", "TSLA", "005930.KS")
    
    Returns:
        주식 정보 (dict)
    """
    try:
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
    """
    주어진 query에 대해 웹 검색을 하고 결과를 반환합니다.
    주가 변동의 '이유'를 찾거나, yfinance에 없는 정보를 검색하는 등의 용도로 사용합니다.

    Args:
        query (str): 검색어

    Returns:
        검색 결과 요약 및 저장 경로 (문자열)
    """
    try:
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

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 저장할 경로 설정
        project_root = os.path.dirname(current_path)  # src
        parent_dir = os.path.dirname(project_root)     # 프로젝트 최상위 (src의 상위)
        data_dir = os.path.join(parent_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        resources_json_path = os.path.join(data_dir, f'resources_{timestamp}.json')

        # JSON 저장
        with open(resources_json_path, 'w', encoding='utf-8') as f:
            json.dump(response, f, ensure_ascii=False, indent=4)

        # 요약
        output = f"🌐 '{query}' 웹 검색 완료\n\n"
        output += f"📊 검색 결과: {len(results)}개\n"
        output += f"💾 저장 위치: {resources_json_path}\n\n"
        output += "📝 주요 결과:\n"
        
        for idx, result in enumerate(results[:5], 1):
            output += f"\n[{idx}] {result.get('title', 'N/A')}\n"
            output += f"    URL: {result.get('url', 'N/A')}\n"
            output += f"    내용: {result.get('content', 'N/A')[:150]}...\n"

        logger.info(f"웹 검색 완료 - query: {query}, 결과: {len(results)}개, 저장: {resources_json_path}")
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
    """과거 주가 데이터를 조회합니다.
    차트 생성이나 추세 분석에 필요한 OHLCV 데이터를 제공합니다.
    
    Args:
        ticker: 주식 티커 심볼
        period: 기간 ("1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")
        interval: 간격 ("1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo")
    
    Returns:
        과거 가격 데이터 (포맷팅된 문자열)
    
    Example:
        >>> get_historical_prices("AAPL", period="1mo", interval="1d")
    """
    try:
        logger.info(f"과거 가격 조회 시작 - ticker: {ticker}, period: {period}, interval: {interval}")
        
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, interval=interval)
        
        if hist.empty:
            return f"{ticker}의 과거 가격 데이터를 찾을 수 없습니다."
        
        # 최근 10개 데이터만 표시
        output = f"\n{ticker} 과거 가격 ({period}, {interval} 간격):\n"
        output += "=" * 80 + "\n"
        output += hist.tail(10).to_string()
        output += f"\n\n총 {len(hist)}개 데이터 포인트"
        
        logger.info(f"과거 가격 조회 완료 - ticker: {ticker}, period: {period}, rows: {len(hist)}")
        return output
    
    except Exception as e:
        logger.error(f"과거 가격 조회 실패 - ticker: {ticker}, error: {str(e)}")
        return f"과거 가격 조회 중 오류 발생: {str(e)}"


@tool
def get_analyst_recommendations(ticker: str) -> str:
    """애널리스트 추천 정보를 종합 조회합니다.
    추천 등급, 목표 주가, 최근 등급 변경 이력을 모두 제공합니다.
    
    Args:
        ticker: 주식 티커 심볼
    
    Returns:
        애널리스트 추천 종합 정보 (포맷팅된 문자열)
    
    Example:
        >>> get_analyst_recommendations("AAPL")
    """
    try:
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