# tests/financial_tools_test.py
"""
Financial Tools 테스트 스크립트
각 도구의 기본 동작을 순차적으로 테스트합니다.
"""

from src.agents.tools.financial_tools import (
    search_stocks,
    get_stock_info,
    web_search,
    get_historical_prices,
    get_analyst_recommendations,
)

def test_search_stocks():
    """주식 검색 테스트"""
    print("\n" + "="*80)
    print("1. 주식 검색 테스트")
    print("="*80)
    
    # 영어 검색
    print("\n[테스트 1-1] 영어 검색: 'microsoft'")
    result = search_stocks.invoke({"query": "microsoft", "max_results": 5})
    print(result)
    
    # 한국어 검색
    print("\n[테스트 1-2] 한국어 검색: '삼성전자'")
    result = search_stocks.invoke({"query": "삼성전자", "max_results": 5})
    print(result)

    print("\n[테스트 1-3] 한국어 검색: '애플'")
    result = search_stocks.invoke({"query": "애플", "max_results": 5})
    print(result)

    print("\n[테스트 1-4] 한국어 산업 검색: '반도체'")
    result = search_stocks.invoke({"query": "반도체", "max_results": 5})
    print(result)

    # 산업 검색
    print("\n[테스트 1-5] 산업 검색: 'semiconductor'")
    result = search_stocks.invoke({"query": "semiconductor", "max_results": 5})
    print(result)


def test_get_stock_info():
    """주식 정보 조회 테스트"""
    print("\n" + "="*80)
    print("2. 주식 정보 조회 테스트")
    print("="*80)
    
    tickers = ["AAPL", "TSLA", "005930.KS"]  # Apple, Tesla, 삼성전자
    
    for ticker in tickers:
        print(f"\n[테스트 2] {ticker} 정보 조회")
        result = get_stock_info.invoke({"ticker": ticker})
        
        if "error" not in result:
            print(f"  • 회사명: {result['name']}")
            print(f"  • 현재가: ${result['current_price']:.2f}")
            print(f"  • 시가총액: ${result['market_cap']:,}")
            print(f"  • PER: {result['pe_ratio']}")
            print(f"  • 섹터: {result['sector']}")
        else:
            print(f"  ❌ 오류: {result['error']}")


def test_web_search():
    """웹 검색 테스트"""
    print("\n" + "="*80)
    print("3. 웹 검색 테스트")
    print("="*80)
    
    # 주가 변동 이유 검색
    print("\n[테스트 3-1] 'Apple stock price surge reason' 검색")
    result = web_search.invoke({"query": "Apple stock price surge reason"})
    print(result)
    
    # 최신 뉴스 검색
    print("\n[테스트 3-2] 'Tesla latest news' 검색")
    result = web_search.invoke({"query": "Tesla latest news"})
    print(result)
    
    # 한국어 검색
    print("\n[테스트 3-3] '삼성전자 주가 전망' 검색")
    result = web_search.invoke({"query": "삼성전자 주가 전망"})
    print(result)


def test_get_historical_prices():
    """과거 가격 조회 테스트"""
    print("\n" + "="*80)
    print("4. 과거 가격 조회 테스트")
    print("="*80)
    
    print("\n[테스트 4-1] AAPL 1개월 일별 데이터")
    result = get_historical_prices.invoke({
        "ticker": "AAPL",
        "period": "1mo",
        "interval": "1d"
    })
    print(result)
    
    print("\n[테스트 4-2] TSLA 1주일 시간별 데이터")
    result = get_historical_prices.invoke({
        "ticker": "TSLA",
        "period": "5d",
        "interval": "1h"
    })
    print(result)


def test_get_analyst_recommendations():
    """애널리스트 추천 테스트"""
    print("\n" + "="*80)
    print("5. 애널리스트 추천 테스트")
    print("="*80)
    
    print("\n[테스트 5] AAPL 애널리스트 추천")
    result = get_analyst_recommendations.invoke({"ticker": "AAPL"})
    print(result)


def run_all_tests():
    """모든 테스트 실행"""
    print("\n" + "=="*40)
    print("Financial Tools 통합 테스트 시작")
    print("=="*40)
    
    try:
        test_search_stocks()
        test_get_stock_info()
        test_web_search()  # ✅ 추가
        test_get_historical_prices()
        test_get_analyst_recommendations()
        
        print("\n" + "=="*40)
        print("✅ 모든 테스트 완료!")
        print("=="*40)
        
    except Exception as e:
        print(f"\n❌ 테스트 실패: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 개별 테스트 실행 예시
    # test_search_stocks()
    # test_get_stock_info()
    # test_web_search()
    
    # 전체 테스트 실행
    run_all_tests()