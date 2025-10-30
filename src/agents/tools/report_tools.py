# src/agents/tools/report_tools.py
"""
Report Tools for Report Generator Agent

차트 그리기, 파일 저장 등을 담당
텍스트 보고서 생성은 ReportGenerator가 직접 처리
"""

import json
import os
import platform

from typing import Dict, Any, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from langchain_core.tools import tool

from src.utils.logger import get_logger

logger = get_logger(__name__)

# 운영체제별 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':  # macOS
    plt.rcParams['font.family'] = 'AppleGothic'
else:  # Linux
    # Linux에서 사용 가능한 한글 폰트 찾기
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    if 'NanumGothic' in available_fonts:
        plt.rcParams['font.family'] = 'NanumGothic'
    elif 'Nanum Gothic' in available_fonts:
        plt.rcParams['font.family'] = 'Nanum Gothic'
    else:
        # 한글 폰트 없으면 기본 폰트 + 마이너스만 처리
        plt.rcParams['font.family'] = 'DejaVu Sans'

# 마이너스 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False


@tool
def draw_stock_chart(output_path: str = "charts/stock_chart.png") -> str:
    """
    주식 분석 데이터를 사용하여 가격 차트를 그립니다.
    데이터는 Report Generator가 이미 준비해두었으므로, output_path만 지정하면 됩니다.

    ====================================================
    도구 호출 시 전달되는 인자와 반환값에 대한 설명 명확하게

    Args:

    Returns:

    ====================================================
    Args:
        output_path: 저장 경로 (기본값: "charts/stock_chart.png")

    Returns:
        차트 저장 결과 메시지
    """
    try:
        logger.info(f"주식 차트 생성 시작 - 원본 output_path: {repr(output_path)}")

        # Clean output_path - remove any trailing artifacts
        # LLM sometimes adds extra text: path.png"}\n\n[STOP]
        import re

        # Strategy: extract clean path with extension
        # Look for: something.extension (before any quotes, braces, or control chars)
        match = re.search(r'([a-zA-Z0-9_/.-]+\.(?:png|jpg|jpeg|pdf|svg|webp))', output_path, re.IGNORECASE)
        if match:
            output_path = match.group(1)
        else:
            # Fallback: remove everything after first problematic character
            output_path = re.split(r'["\}\]\n\r]', output_path)[0].strip()

            # Ensure it has a valid extension
            if not output_path.endswith(('.png', '.jpg', '.jpeg', '.pdf', '.svg', '.webp')):
                output_path += '.png'

        logger.info(f"정리된 output_path: {output_path}")

        # Report Generator에서 설정한 글로벌 변수에서 데이터 가져오기
        from src.agents.report_generator import _get_current_analysis_data
        financial_data_json = _get_current_analysis_data()

        if not financial_data_json or financial_data_json == "{}":
            return "❌ 분석 데이터를 찾을 수 없습니다. Report Generator가 데이터를 설정하지 않았습니다."

        # JSON 파싱
        data = json.loads(financial_data_json)
        analysis_type = data.get('analysis_type', 'single')
        
        # 디렉토리 생성
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        
        if analysis_type == 'single':
            result = _draw_single_stock_chart(data, output_path)
        elif analysis_type == 'comparison':
            result = _draw_comparison_chart(data, output_path)
        else:
            return "알 수 없는 분석 유형입니다."
        
        logger.info(f"주식 차트 생성 완료 - 저장: {output_path}")
        return result
    
    except Exception as e:
        logger.error(f"주식 차트 생성 실패: {str(e)}")
        return f"차트 생성 중 오류 발생: {str(e)}"


def _draw_single_stock_chart(data: Dict[str, Any], output_path: str) -> str:
    """
    단일 주식 메트릭 시각화 (전달받은 데이터만 사용)
    52주 고가/저가, 현재가를 막대 그래프로 표시
    """
    try:
        ticker = data.get('ticker', 'N/A')
        company_name = data.get('company_name', 'Unknown')
        current_price = data.get('current_price', 0)
        metrics = data.get('metrics', {})
        
        high_52w = metrics.get('52week_high', 0)
        low_52w = metrics.get('52week_low', 0)
        pe_ratio = metrics.get('pe_ratio', 0)
        market_cap = metrics.get('market_cap', 0)
        
        # 52주 데이터 유효성 검사
        if high_52w == 0 or low_52w == 0:
            logger.warning(f"{ticker}: 52주 고가/저가 데이터가 없습니다.")
            # 현재가만 있는 경우 추정치 사용
            if current_price > 0:
                high_52w = current_price if high_52w == 0 else high_52w
                low_52w = current_price if low_52w == 0 else low_52w
            else:
                return f"❌ {ticker}: 가격 데이터가 충분하지 않습니다."
        
        # 차트 생성
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. 가격 범위 차트
        categories = ['52W Low', 'Current', '52W High']
        prices = [low_52w, current_price, high_52w]
        colors = ['#d62728', '#2ca02c', '#1f77b4']
        
        bars = ax1.bar(categories, prices, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax1.set_title(f'{company_name} ({ticker}) - Price Range', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.grid(axis='y', alpha=0.3)
        
        # 가격 레이블
        for bar, price in zip(bars, prices):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'${price:.2f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # 현재가 위치 표시 (퍼센트)
        if high_52w > low_52w:
            position_pct = (current_price - low_52w) / (high_52w - low_52w) * 100
            ax1.text(0.5, 0.95, f'Position: {position_pct:.1f}% of 52W range',
                    transform=ax1.transAxes, ha='center', va='top',
                    fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        # 2. 주요 지표 표시
        ax2.axis('off')
        
        # Market Cap 포맷팅 (조/억 단위)
        if market_cap >= 1e12:
            market_cap_str = f"${market_cap/1e12:.2f}T"
        elif market_cap >= 1e9:
            market_cap_str = f"${market_cap/1e9:.2f}B"
        elif market_cap >= 1e6:
            market_cap_str = f"${market_cap/1e6:.2f}M"
        else:
            market_cap_str = f"${market_cap:.0f}"
        
        # P/E Ratio 포맷팅 (여기가 핵심!)
        if pe_ratio > 0:
            pe_ratio_str = f"{pe_ratio:.2f}"
        else:
            pe_ratio_str = "N/A"
        
        metrics_text = f"""[Key Metrics]

Ticker: {ticker}
Company: {company_name}

Current Price: ${current_price:.2f}
52 Week High: ${high_52w:.2f}
52 Week Low: ${low_52w:.2f}

P/E Ratio: {pe_ratio_str}
Market Cap: {market_cap_str}

Sector: {metrics.get('sector', 'N/A')}
Industry: {metrics.get('industry', 'N/A')}"""
        
        ax2.text(0.1, 0.5, metrics_text, fontsize=11, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return f"✓ 차트가 {output_path}에 저장되었습니다."
    
    except Exception as e:
        logger.error(f"단일 주식 차트 생성 실패: {str(e)}")
        import traceback
        logger.debug(f"상세 에러:\n{traceback.format_exc()}")
        return f"차트 생성 중 오류: {str(e)}"


def _draw_comparison_chart(data: Dict[str, Any], output_path: str) -> str:
    """
    비교 분석 차트 그리기 (전달받은 데이터만 사용)
    """
    try:
        stocks = data.get('stocks', [])
        if not stocks:
            return "비교할 주식 데이터가 없습니다."
        
        # 3개의 서브플롯으로 확장: 현재가, 52주 범위, 주요 지표
        fig = plt.figure(figsize=(18, 5))
        gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        
        tickers = [s['ticker'] for s in stocks]
        colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        # 1. 현재가 비교
        prices = [s.get('current_price', 0) for s in stocks]
        
        bars = ax1.bar(tickers, prices, 
                      color=[colors_list[i % len(colors_list)] for i in range(len(tickers))], 
                      alpha=0.7, edgecolor='black', linewidth=1.5)
        ax1.set_title('Current Price Comparison', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Stock', fontsize=12)
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.grid(axis='y', alpha=0.3)
        
        # 가격 레이블
        for bar, price in zip(bars, prices):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'${price:.2f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # 2. 52주 범위 비교 (정규화된 차트)
        for idx, stock in enumerate(stocks):
            ticker = stock['ticker']
            current_price = stock.get('current_price', 0)
            metrics = stock.get('metrics', {})
            high_52w = metrics.get('52week_high', 0)
            low_52w = metrics.get('52week_low', 0)
            
            # 데이터 유효성 검사
            if high_52w > 0 and low_52w > 0 and high_52w > low_52w and current_price > 0:
                position = (current_price - low_52w) / (high_52w - low_52w) * 100
                
                # 정규화된 막대 (0-100%)
                ax2.barh(ticker, position, 
                        color=colors_list[idx % len(colors_list)], 
                        alpha=0.7, height=0.6)
                
                # 위치 표시
                ax2.text(position, idx, f'{position:.1f}%', 
                        ha='left', va='center', fontsize=9, fontweight='bold')
        
        ax2.set_title('Position in 52-Week Range', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Position (%)', fontsize=12)
        ax2.set_xlim(0, 100)
        ax2.grid(axis='x', alpha=0.3)
        ax2.axvline(x=50, color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        # 3. 주요 지표 비교 (P/E Ratio & Market Cap)
        pe_ratios = [s.get('metrics', {}).get('pe_ratio', 0) for s in stocks]
        market_caps = [s.get('metrics', {}).get('market_cap', 0) / 1e12 for s in stocks]  # 조 단위
        
        x = np.arange(len(tickers))
        width = 0.35
        
        ax3_twin = ax3.twinx()
        
        bars1 = ax3.bar(x - width/2, pe_ratios, width, label='P/E Ratio', 
                       color='#1f77b4', alpha=0.7, edgecolor='black', linewidth=1.5)
        bars2 = ax3_twin.bar(x + width/2, market_caps, width, label='Market Cap ($T)', 
                            color='#ff7f0e', alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # 값 레이블
        for bar, pe in zip(bars1, pe_ratios):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{pe:.1f}',
                    ha='center', va='bottom', fontsize=8)
        
        for bar, cap in zip(bars2, market_caps):
            height = bar.get_height()
            ax3_twin.text(bar.get_x() + bar.get_width()/2., height,
                         f'{cap:.2f}T',
                         ha='center', va='bottom', fontsize=8)
        
        ax3.set_title('Key Metrics Comparison', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Stock', fontsize=12)
        ax3.set_ylabel('P/E Ratio', fontsize=12, color='#1f77b4')
        ax3_twin.set_ylabel('Market Cap ($T)', fontsize=12, color='#ff7f0e')
        ax3.set_xticks(x)
        ax3.set_xticklabels(tickers)
        ax3.tick_params(axis='y', labelcolor='#1f77b4')
        ax3_twin.tick_params(axis='y', labelcolor='#ff7f0e')
        ax3.legend(loc='upper left', fontsize=9)
        ax3_twin.legend(loc='upper right', fontsize=9)
        ax3.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return f"✓ 비교 차트가 {output_path}에 저장되었습니다."
    
    except Exception as e:
        logger.error(f"비교 차트 생성 실패: {str(e)}")
        import traceback
        logger.debug(f"상세 에러:\n{traceback.format_exc()}")
        return f"차트 생성 중 오류: {str(e)}"


@tool
def draw_valuation_radar(output_path: str = "charts/valuation_radar.png") -> str:
    """
    밸류에이션 레이더 차트를 그립니다.
    주식의 성장성, 가치, 모멘텀, 품질, 시장심리를 시각화합니다.
    데이터는 Report Generator가 이미 준비해두었으므로, output_path만 지정하면 됩니다.

    Args:
        output_path: 저장 경로 (기본값: "charts/valuation_radar.png")

    Returns:
        차트 저장 결과 메시지
    """
    try:
        logger.info("밸류에이션 레이더 차트 생성 시작")

        # Clean output_path - remove any trailing JSON artifacts or newlines
        import re
        # First, extract path from quotes if present: {"output_path": "charts/file.png"}
        if '"' in output_path:
            match = re.search(r'"([^"]+\.(?:png|jpg|jpeg|pdf|svg|webp))"', output_path, re.IGNORECASE)
            if match:
                output_path = match.group(1)
            else:
                # Try without extension requirement
                match = re.search(r'"([^"]+)"', output_path)
                if match:
                    output_path = match.group(1)

        # Remove any remaining JSON/control characters
        output_path = re.sub(r'[{}\[\]"\n\r]+.*$', '', output_path).strip()

        # Ensure it has a valid extension
        if not output_path.endswith(('.png', '.jpg', '.jpeg', '.pdf', '.svg', '.webp')):
            output_path += '.png'  # Default to PNG if no valid extension

        # Report Generator에서 설정한 글로벌 변수에서 데이터 가져오기
        from src.agents.report_generator import _get_current_analysis_data
        financial_data_json = _get_current_analysis_data()

        if not financial_data_json or financial_data_json == "{}":
            return "❌ 분석 데이터를 찾을 수 없습니다. Report Generator가 데이터를 설정하지 않았습니다."

        # JSON 파싱
        data = json.loads(financial_data_json)
        
        # 디렉토리 생성
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        
        # 5가지 주요 지표 기반 점수 계산
        categories = ['Growth', 'Value', 'Momentum', 'Quality', 'Sentiment']
        
        if data.get('analysis_type') == 'single':
            # 단일 주식 분석
            ticker = data.get('ticker', 'N/A')
            company_name = data.get('company_name', ticker)
            metrics = data.get('metrics', {})
            current_price = data.get('current_price', 0)
            
            # 점수 계산
            scores = _calculate_single_stock_scores(data)
            values = [
                scores['growth'],
                scores['value'],
                scores['momentum'],
                scores['quality'],
                scores['sentiment']
            ]
            
            # 레이더 차트 그리기
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            values_plot = values + values[:1]
            angles_plot = angles + angles[:1]
            
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            ax.plot(angles_plot, values_plot, 'o-', linewidth=2.5, 
                   color='#1f77b4', markersize=8)
            ax.fill(angles_plot, values_plot, alpha=0.25, color='#1f77b4')
            
            # 각 포인트에 값 표시
            for angle, value, category in zip(angles, values, categories):
                ax.text(angle, value + 0.1, f'{value:.2f}', 
                       ha='center', va='center', size=10, 
                       bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor='yellow', alpha=0.7))
            
            # 축 설정
            ax.set_xticks(angles)
            ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
            ax.set_ylim(0, 1)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9, color='gray')
            
            title = f"{company_name} ({ticker}) Valuation Radar"
            ax.set_title(title, size=14, pad=20, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

        elif data.get('analysis_type') == 'comparison':
            # 비교 분석: 상대적 평가 사용
            stocks = data.get('stocks', [])
            if not stocks:
                return "비교할 주식 데이터가 없습니다."
            
            # 상대적 점수 계산
            all_stocks_scores, stock_labels = _calculate_comparative_scores(stocks)
            
            if not all_stocks_scores:
                return "점수 계산에 실패했습니다."
            
            title = f"Valuation Radar Comparison: {' vs '.join(stock_labels)}"
            
            # 레이더 차트 그리기 (여러 주식)
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            angles_plot = angles + angles[:1]
            
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            
            # 각 주식별로 라인 그리기
            for idx, (scores, label) in enumerate(zip(all_stocks_scores, stock_labels)):
                values_plot = scores + scores[:1]
                color = colors_list[idx % len(colors_list)]
                
                ax.plot(angles_plot, values_plot, 'o-', linewidth=2.5, 
                    label=label, color=color, markersize=8)
                ax.fill(angles_plot, values_plot, alpha=0.15, color=color)
            
            # 축 설정
            ax.set_xticks(angles)
            ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
            ax.set_ylim(0, 1)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9, color='gray')
            ax.set_title(title, size=14, pad=20, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

        else:
            return f"알 수 없는 분석 유형입니다: {data.get('analysis_type')}"

        logger.info(f"밸류에이션 레이더 차트 생성 완료 - 저장: {output_path}")
        return f"✓ 레이더 차트가 {output_path}에 저장되었습니다."
    
    except Exception as e:
        logger.error(f"레이더 차트 생성 실패: {str(e)}")
        import traceback
        logger.debug(f"상세 에러:\n{traceback.format_exc()}")
        return f"차트 생성 중 오류 발생: {str(e)}"


def _calculate_single_stock_scores(data: Dict[str, Any]) -> Dict[str, float]:
    """
    단일 주식의 밸류에이션 점수 계산
    업계 일반적 기준 대비 평가
    """
    metrics = data.get('metrics', {})
    current_price = data.get('current_price', 0)
    
    market_cap = metrics.get('market_cap', 0)
    pe_ratio = metrics.get('pe_ratio', 20)
    high_52w = metrics.get('52week_high', 0)
    low_52w = metrics.get('52week_low', 0)
    sector = metrics.get('sector', '').lower()
    recommendation = data.get('analyst_recommendation', '').lower()
    
    scores = {}
    
    # 1. Growth Score - 시가총액 기반 (일반적 기준)
    # 시가총액이 작을수록 성장 여지 높음
    if market_cap >= 2e12:  # 2조 이상: Mega Cap
        scores['growth'] = 0.50
    elif market_cap >= 1e12:  # 1조: Large Cap
        scores['growth'] = 0.60
    elif market_cap >= 1e11:  # 1000억: Mid Cap
        scores['growth'] = 0.75
    elif market_cap >= 1e10:  # 100억: Small Cap
        scores['growth'] = 0.85
    else:  # Micro Cap
        scores['growth'] = 0.90
    
    # 2. Value Score - P/E Ratio 기반 (업계 일반 기준)
    # 일반적으로 P/E 15 이하면 저평가, 30 이상이면 고평가
    if pe_ratio > 0:
        if pe_ratio < 10:
            scores['value'] = 0.95
        elif pe_ratio < 15:
            scores['value'] = 0.85
        elif pe_ratio < 20:
            scores['value'] = 0.70
        elif pe_ratio < 25:
            scores['value'] = 0.55
        elif pe_ratio < 30:
            scores['value'] = 0.40
        elif pe_ratio < 40:
            scores['value'] = 0.25
        else:
            scores['value'] = 0.15
    else:
        scores['value'] = 0.50  # P/E가 음수이거나 없는 경우
    
    # 3. Momentum Score - 52주 범위 내 위치
    if high_52w > 0 and low_52w > 0 and high_52w > low_52w and current_price > 0:
        momentum_score = (current_price - low_52w) / (high_52w - low_52w)
        scores['momentum'] = max(0.0, min(1.0, momentum_score))
    else:
        scores['momentum'] = 0.50  # 데이터 없으면 중립
    
    # 4. Quality Score - 시가총액 + 섹터
    stable_sectors = ['healthcare', 'consumer staples', 'utilities', 'consumer defensive']
    growth_sectors = ['technology', 'communication services', 'consumer cyclical']
    
    # 시가총액 기반 품질 점수
    if market_cap >= 2e12:
        base_quality = 0.85
    elif market_cap >= 1e12:
        base_quality = 0.75
    elif market_cap >= 5e11:
        base_quality = 0.65
    elif market_cap >= 1e11:
        base_quality = 0.55
    else:
        base_quality = 0.45
    
    # 섹터 보너스
    if any(s in sector for s in stable_sectors):
        scores['quality'] = min(1.0, base_quality + 0.10)
    elif any(s in sector for s in growth_sectors):
        scores['quality'] = min(1.0, base_quality + 0.05)
    else:
        scores['quality'] = base_quality
    
    # 5. Sentiment Score - 애널리스트 추천 기반
    if 'strong buy' in recommendation:
        scores['sentiment'] = 0.95
    elif 'buy' in recommendation:
        scores['sentiment'] = 0.80
    elif 'outperform' in recommendation or 'overweight' in recommendation:
        scores['sentiment'] = 0.70
    elif 'hold' in recommendation or 'neutral' in recommendation:
        scores['sentiment'] = 0.50
    elif 'underperform' in recommendation or 'underweight' in recommendation:
        scores['sentiment'] = 0.30
    elif 'sell' in recommendation:
        scores['sentiment'] = 0.20
    elif 'strong sell' in recommendation:
        scores['sentiment'] = 0.10
    else:
        scores['sentiment'] = 0.60  # 기본값
    
    return scores


def _calculate_comparative_scores(stocks: list) -> tuple:
    """
    여러 주식의 상대적 밸류에이션 점수 계산 (Min-Max 정규화)
    Returns: (scores_list, labels_list)
    """
    if not stocks or len(stocks) == 0:
        return [], []
    
    # 모든 메트릭 수집
    all_pe_ratios = []
    all_market_caps = []
    all_momentum_raw = []
    
    for stock in stocks:
        metrics = stock.get('metrics', {})
        pe_ratio = metrics.get('pe_ratio', 20)
        market_cap = metrics.get('market_cap', 1e9)
        current_price = stock.get('current_price', 0)
        high_52w = metrics.get('52week_high', 0)
        low_52w = metrics.get('52week_low', 0)
        
        all_pe_ratios.append(pe_ratio if pe_ratio > 0 else 20)
        all_market_caps.append(market_cap)
        
        # Momentum 계산
        if high_52w > 0 and low_52w > 0 and high_52w > low_52w and current_price > 0:
            momentum = (current_price - low_52w) / (high_52w - low_52w)
        else:
            momentum = 0.5
        all_momentum_raw.append(momentum)
    
    # Min-Max 값
    min_pe = min(all_pe_ratios)
    max_pe = max(all_pe_ratios)
    min_cap = min(all_market_caps)
    max_cap = max(all_market_caps)
    
    # 각 주식 점수 계산
    all_stocks_scores = []
    stock_labels = []
    
    for idx, stock in enumerate(stocks):
        ticker = stock.get('ticker', 'N/A')
        stock_labels.append(ticker)
        
        metrics = stock.get('metrics', {})
        pe_ratio = all_pe_ratios[idx]
        market_cap = all_market_caps[idx]
        sector = metrics.get('sector', '').lower()
        
        # 1. Growth Score - 시가총액 역수 (작을수록 높음)
        if max_cap > min_cap:
            growth_score = 1 - (market_cap - min_cap) / (max_cap - min_cap)
            growth_score = 0.40 + growth_score * 0.50  # 0.4~0.9 범위
        else:
            growth_score = 0.65
        
        # 2. Value Score - P/E 역수 (낮을수록 높음)
        if max_pe > min_pe:
            value_score = 1 - (pe_ratio - min_pe) / (max_pe - min_pe)
            value_score = 0.30 + value_score * 0.65  # 0.3~0.95 범위
        else:
            value_score = 0.60
        
        # 3. Momentum Score - 이미 0-1 범위
        momentum_score = all_momentum_raw[idx]
        
        # 4. Quality Score - 시가총액 정규화
        if max_cap > min_cap:
            quality_score = (market_cap - min_cap) / (max_cap - min_cap)
            quality_score = 0.50 + quality_score * 0.45  # 0.5~0.95 범위
        else:
            quality_score = 0.70
        
        # 섹터 보너스
        stable_sectors = ['healthcare', 'consumer staples', 'utilities', 'consumer defensive']
        if any(s in sector for s in stable_sectors):
            quality_score = min(1.0, quality_score + 0.05)
        
        # 5. Sentiment Score - Value와 Quality 조합
        sentiment_score = (value_score * 0.6 + quality_score * 0.4)
        
        all_stocks_scores.append([
            growth_score,
            value_score,
            momentum_score,
            quality_score,
            sentiment_score
        ])
    
    # 디버깅용 출력
    for idx, stock in enumerate(stocks):
        ticker = stock.get('ticker')
        pe_ratio = stock.get('metrics', {}).get('pe_ratio')
        market_cap = stock.get('metrics', {}).get('market_cap')
        
        print(f"\n[DEBUG] Stock {idx}: {ticker}")
        print(f"  P/E Ratio: {pe_ratio}")
        print(f"  Market Cap: {market_cap}")
        print(f"  Growth: {all_stocks_scores[idx][0]:.3f}")
        print(f"  Value: {all_stocks_scores[idx][1]:.3f}")
        print(f"  Momentum: {all_stocks_scores[idx][2]:.3f}")
        print(f"  Quality: {all_stocks_scores[idx][3]:.3f}")
        print(f"  Sentiment: {all_stocks_scores[idx][4]:.3f}")

    return all_stocks_scores, stock_labels


@tool
def save_report_to_file(
    report_text: str, 
    format: str = "md",
    output_path: Optional[str] = None,
    chart_paths: Optional[str] = None
) -> str:
    """
    보고서를 파일로 저장합니다.
    
    Args:
        report_text: 보고서 텍스트
        format: 파일 형식 ("txt", "md", "pdf")
        output_path: 저장 경로 (없으면 자동 생성)
        chart_paths: 차트 이미지 경로들 (쉼표로 구분된 문자열, PDF에만 포함)
    
    Returns:
        저장 결과 메시지
    """
    try:
        # JSON 문자열로 전달된 경우 파싱 (LangChain tool input 처리)
        if isinstance(report_text, str) and report_text.strip().startswith('{'):
            try:
                parsed = json.loads(report_text)
                # {"report_text": "...", "format": "...", ...} 형태인 경우
                if isinstance(parsed, dict) and 'report_text' in parsed:
                    report_text = parsed['report_text']
                    format = parsed.get('format', format)
                    output_path = parsed.get('output_path', output_path)
                    chart_paths = parsed.get('chart_paths', chart_paths)
            except:
                pass  # JSON이 아니면 그대로 사용

        logger.info(f"보고서 파일 저장 시작 - format: {format}")

        # 지원하는 형식 확인
        supported_formats = ["txt", "md", "pdf"]
        if format not in supported_formats:
            return f"❌ 지원하지 않는 형식입니다: {format}. 지원 형식: {', '.join(supported_formats)}"
        
        # 출력 경로 설정
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"reports/report_{timestamp}.{format}"
        
        # 디렉토리 생성
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # 형식별 저장
        if format in ["txt", "md"]:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            logger.info(f"보고서 저장 완료 - 경로: {output_path}")
            return f"✓ 보고서가 {output_path}에 저장되었습니다."
        
        elif format == "pdf":
            # PDF 생성
            return _save_pdf_report(report_text, output_path, chart_paths)
        
    except Exception as e:
        logger.error(f"보고서 저장 실패: {str(e)}")
        return f"❌ 파일 저장 중 오류 발생: {str(e)}"


def _save_pdf_report(report_text: str, output_path: str, chart_paths: Optional[str] = None) -> str:
    """
    PDF 형식으로 보고서 저장 (reportlab 사용)
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image
        from reportlab.lib.enums import TA_LEFT, TA_CENTER
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.lib import colors
        
        # 한글 폰트 등록 (matplotlib 설정 재사용)
        korean_font = 'Helvetica'
        korean_font_bold = 'Helvetica-Bold'
        
        try:
            system = platform.system()
            font_registered = False
            
            if system == 'Windows':
                font_path = 'C:/Windows/Fonts/malgun.ttf'
                if os.path.exists(font_path):
                    pdfmetrics.registerFont(TTFont('Korean', font_path))
                    # Bold 폰트도 등록 시도
                    bold_path = 'C:/Windows/Fonts/malgunbd.ttf'
                    if os.path.exists(bold_path):
                        pdfmetrics.registerFont(TTFont('KoreanBold', bold_path))
                        korean_font_bold = 'KoreanBold'
                    else:
                        korean_font_bold = 'Korean'
                    korean_font = 'Korean'
                    font_registered = True
                    
            elif system == 'Darwin':  # macOS
                font_path = '/System/Library/Fonts/AppleGothic.ttf'
                if os.path.exists(font_path):
                    pdfmetrics.registerFont(TTFont('Korean', font_path))
                    korean_font = 'Korean'
                    korean_font_bold = 'Korean'
                    font_registered = True
                    
            else:  # Linux
                font_paths = [
                    '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
                    '/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf',
                ]
                if os.path.exists(font_paths[0]):
                    pdfmetrics.registerFont(TTFont('Korean', font_paths[0]))
                    korean_font = 'Korean'
                    font_registered = True
                    
                    if os.path.exists(font_paths[1]):
                        pdfmetrics.registerFont(TTFont('KoreanBold', font_paths[1]))
                        korean_font_bold = 'KoreanBold'
                    else:
                        korean_font_bold = 'Korean'
            
            if font_registered:
                logger.info(f"한글 폰트 등록 성공: {system}")
            else:
                logger.warning("한글 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")
                
        except Exception as e:
            logger.warning(f"폰트 등록 실패: {str(e)}")
        
        # PDF 문서 생성
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=50,
            leftMargin=50,
            topMargin=50,
            bottomMargin=50
        )
        
        # 스타일 설정
        styles = getSampleStyleSheet()
        
        # 제목 스타일 (메인 타이틀)
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=20,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=20,
            spaceBefore=10,
            alignment=TA_CENTER,
            fontName=korean_font_bold,
            leading=28
        )
        
        # 섹션 제목 스타일
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#2ca02c'),
            spaceAfter=10,
            spaceBefore=15,
            fontName=korean_font_bold,
            leading=20
        )
        
        # 본문 스타일
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['BodyText'],
            fontSize=10,
            leading=15,
            spaceAfter=8,
            alignment=TA_LEFT,
            fontName=korean_font,
            textColor=colors.black
        )
        
        # 내용 구성
        story = []
        
        # 보고서 텍스트 처리
        lines = report_text.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # 빈 줄 처리
            if not line:
                story.append(Spacer(1, 0.15 * inch))
                continue
            
            # 메인 제목 라인 감지
            if (line.startswith('# ') or 
                '===' in line or 
                'REPORT' in line.upper() or
                (i == 0 and len(line) < 100)):  # 첫 줄이 짧으면 제목으로 간주
                
                clean_line = line.replace('#', '').replace('=', '').strip()
                if clean_line:
                    story.append(Paragraph(clean_line, title_style))
                    story.append(Spacer(1, 0.2 * inch))
            
            # 섹션 제목 감지
            elif line.startswith('##') or line.startswith('###') or '---' in line:
                clean_line = line.replace('#', '').replace('-', '').strip()
                if clean_line:
                    story.append(Spacer(1, 0.1 * inch))
                    story.append(Paragraph(clean_line, heading_style))
            
            # 일반 텍스트
            else:
                # HTML 특수문자 이스케이프
                line = (line.replace('&', '&amp;')
                           .replace('<', '&lt;')
                           .replace('>', '&gt;'))
                
                # 마크다운 볼드 처리
                import re
                # **text** -> <b>text</b>
                line = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', line)
                # *text* -> <i>text</i> (단, ** 처리 후)
                line = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<i>\1</i>', line)
                
                # 리스트 아이템 처리
                if line.startswith('- ') or line.startswith('* '):
                    line = '• ' + line[2:]
                elif re.match(r'^\d+\.\s', line):
                    # 숫자 리스트는 그대로 유지
                    pass
                
                story.append(Paragraph(line, body_style))
        
        # 차트 추가
        if chart_paths:
            story.append(PageBreak())
            story.append(Paragraph("Charts & Visualizations", title_style))
            story.append(Spacer(1, 0.3 * inch))
            
            # 쉼표로 구분된 경로 처리
            paths = [p.strip() for p in chart_paths.split(',')]
            
            for idx, chart_path in enumerate(paths):
                if os.path.exists(chart_path):
                    try:
                        # 이미지 객체 생성
                        from PIL import Image as PILImage
                        
                        # 이미지 원본 크기 확인
                        img_obj = PILImage.open(chart_path)
                        img_width, img_height = img_obj.size
                        aspect = img_height / float(img_width)
                        
                        # A4 페이지 너비에 맞춰 크기 조정 (여백 고려)
                        available_width = 6.5 * inch
                        available_height = 4.5 * inch
                        
                        # 비율 유지하며 크기 조정
                        if aspect > (available_height / available_width):
                            # 높이 기준
                            display_height = available_height
                            display_width = display_height / aspect
                        else:
                            # 너비 기준
                            display_width = available_width
                            display_height = display_width * aspect
                        
                        # 이미지 추가
                        img = Image(chart_path, width=display_width, height=display_height)
                        story.append(img)
                        story.append(Spacer(1, 0.3 * inch))
                        
                        # 차트가 여러 개인 경우 페이지 분리
                        if idx < len(paths) - 1:
                            story.append(PageBreak())
                            
                    except Exception as e:
                        logger.warning(f"차트 추가 실패: {chart_path}, 오류: {str(e)}")
                        # 오류 메시지 추가
                        error_para = Paragraph(
                            f"<i>차트를 불러올 수 없습니다: {os.path.basename(chart_path)}</i>",
                            body_style
                        )
                        story.append(error_para)
                        story.append(Spacer(1, 0.2 * inch))
        
        # PDF 생성
        doc.build(story)
        
        logger.info(f"PDF 보고서 저장 완료 - 경로: {output_path}")
        return f"✓ PDF 보고서가 {output_path}에 저장되었습니다."
    
    except ImportError as e:
        logger.error(f"필요한 라이브러리가 설치되지 않았습니다: {str(e)}")
        return "❌ PDF 생성에 필요한 라이브러리를 설치해주세요: pip install reportlab Pillow"
    
    except Exception as e:
        logger.error(f"PDF 생성 실패: {str(e)}")
        import traceback
        logger.debug(f"상세 에러:\n{traceback.format_exc()}")
        return f"❌ PDF 생성 중 오류 발생: {str(e)}"


# Tool 리스트
report_tools = [
    draw_stock_chart,
    draw_valuation_radar,
    save_report_to_file
]