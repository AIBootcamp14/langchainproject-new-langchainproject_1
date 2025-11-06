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
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from langchain_core.tools import tool

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Global variable to store current analysis_data for tools
_current_analysis_data_json = None


def _set_current_analysis_data(data_json: str):
    """Set the current analysis data for tools to access"""
    global _current_analysis_data_json
    _current_analysis_data_json = data_json


def _get_current_analysis_data() -> str:
    """Get the current analysis data JSON string"""
    global _current_analysis_data_json
    return _current_analysis_data_json if _current_analysis_data_json else "{}"

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
def draw_stock_chart(
    output_path: str = "charts/stock_chart.png",
    analysis_data_json: str = None
) -> str:
    """주식 분석 데이터를 기반으로 YTD(Year-to-Date) 가격 차트를 생성하고 저장합니다.

    이 도구는 financial_analyst가 분석한 주식 데이터를 시각화합니다.
    단일 주식 분석의 경우 YTD 주가 라인 차트와 주요 지표를 표시하고,
    비교 분석의 경우 각 주식별로 YTD 라인 차트를 subplot으로 나누어 표시합니다.

    Args:
        output_path: 차트 이미지를 저장할 경로 (기본값: "charts/stock_chart.png")
                    지원 형식: .png, .jpg, .jpeg, .pdf, .svg, .webp
        analysis_data_json: 분석 데이터 JSON 문자열 (선택사항, 없으면 글로벌 변수 사용)

    Returns:
        차트 저장 결과 메시지 (성공 시 "✓ 차트가 {경로}에 저장되었습니다.", 실패 시 오류 메시지)

    Examples:
        >>> draw_stock_chart("charts/aapl_analysis.png")
        "✓ YTD 차트가 charts/aapl_analysis.png에 저장되었습니다."

        >>> draw_stock_chart("charts/comparison_chart.png")
        "✓ 비교 YTD 차트가 charts/comparison_chart.png에 저장되었습니다."
    """
    def _draw_single_stock_chart(data: Dict[str, Any], save_path: str) -> str:
        """단일 주식 YTD 라인 차트 시각화"""
        try:
            ticker = data.get('ticker', 'N/A')
            company_name = data.get('company_name', 'Unknown')
            current_price = data.get('current_price', 0)
            metrics = data.get('metrics', {})
            historical = data.get('historical', '')

            # historical 데이터 파싱
            if not historical or len(historical.strip()) == 0:
                return f"❌ {ticker}: YTD 차트를 그리기 위한 과거 가격 데이터가 없습니다."

            import pandas as pd
            from io import StringIO

            # historical 문자열 파싱 (CSV 형식)
            try:
                # 첫 줄은 메타데이터, 그 다음부터 CSV
                lines = historical.strip().split('\n')
                if len(lines) < 2:
                    return f"❌ {ticker}: 과거 가격 데이터가 부족합니다"

                # 메타데이터를 제외하고 CSV 파싱
                csv_data = '\n'.join(lines[1:])
                df = pd.read_csv(StringIO(csv_data))

                # 첫 번째 컬럼이 Date/Datetime
                if df.empty:
                    return f"❌ {ticker}: 과거 가격 데이터 파싱 실패"

                # 날짜 컬럼 파싱 (첫 번째 컬럼 또는 'Date' 컬럼)
                date_col = df.columns[0]
                df[date_col] = pd.to_datetime(df[date_col], utc=True)
                df = df.sort_values(date_col)

                # Close 컬럼 확인
                if 'Close' not in df.columns:
                    return f"❌ {ticker}: Close 컬럼이 없습니다"

                # 차트 생성
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

                # 1. YTD 라인 차트
                ax1.plot(df[date_col], df['Close'], color='#1f77b4', linewidth=2, label='Close Price')
                ax1.fill_between(df[date_col], df['Close'], alpha=0.3, color='#1f77b4')

                ax1.set_title(f'{company_name} ({ticker}) - YTD Price Chart', fontsize=14, fontweight='bold')
                ax1.set_xlabel('Date', fontsize=12)
                ax1.set_ylabel('Price ($)', fontsize=12)
                ax1.grid(True, alpha=0.3)
                ax1.legend()

                # 날짜 포맷 조정
                import matplotlib.dates as mdates
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
                plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

                # 2. 주요 지표 표시
                ax2.axis('off')

                # Market Cap 포맷팅
                market_cap = metrics.get('market_cap', 0)
                if market_cap >= 1e12:
                    market_cap_str = f"${market_cap/1e12:.2f}T"
                elif market_cap >= 1e9:
                    market_cap_str = f"${market_cap/1e9:.2f}B"
                elif market_cap >= 1e6:
                    market_cap_str = f"${market_cap/1e6:.2f}M"
                else:
                    market_cap_str = f"${market_cap:.0f}"

                # P/E Ratio 포맷팅
                pe_ratio = metrics.get('pe_ratio')
                pe_ratio_str = f"{pe_ratio:.2f}" if pe_ratio and pe_ratio > 0 else "N/A"

                # PB Ratio 포맷팅
                pb_ratio = metrics.get('pb_ratio')
                pb_ratio_str = f"{pb_ratio:.2f}" if pb_ratio and pb_ratio > 0 else "N/A"

                metrics_text = f"""Ticker: {ticker}
Company: {company_name}

Current Price: ${current_price:.2f}
52 Week High: ${metrics.get('52week_high', 0):.2f}
52 Week Low: ${metrics.get('52week_low', 0):.2f}

P/E Ratio: {pe_ratio_str}
P/B Ratio: {pb_ratio_str}
Market Cap: {market_cap_str}

Sector: {metrics.get('sector', 'N/A')}
Industry: {metrics.get('industry', 'N/A')}"""

                ax2.text(0.1, 0.5, metrics_text, fontsize=11, verticalalignment='center',
                        family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

                plt.tight_layout()
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()

                return f"✓ YTD 차트가 {save_path}에 저장되었습니다."

            except Exception as parse_err:
                logger.error(f"과거 데이터 파싱 실패: {parse_err}")
                return f"❌ {ticker}: 과거 데이터 파싱 실패 - {str(parse_err)}"

        except Exception as e:
            logger.error(f"단일 주식 차트 생성 실패: {str(e)}")
            import traceback
            logger.debug(f"상세 에러:\n{traceback.format_exc()}")
            return f"차트 생성 중 오류: {str(e)}"

    def _draw_comparison_chart(data: Dict[str, Any], save_path: str) -> str:
        """비교 분석 차트 그리기 (각 주식별 YTD 라인 차트)"""
        try:
            stocks = data.get('stocks', [])
            if not stocks:
                return "비교할 주식 데이터가 없습니다."

            import pandas as pd
            from io import StringIO
            import matplotlib.dates as mdates

            num_stocks = len(stocks)
            colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

            # 각 주식별로 YTD 차트를 그리기 위한 subplot 생성
            # 2열로 배치 (주식이 많으면 여러 행)
            ncols = 2
            nrows = (num_stocks + 1) // 2  # 올림 처리

            fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows))

            # axes를 1차원 배열로 변환 (subplot이 1개일 때 처리)
            if num_stocks == 1:
                axes = [axes]
            else:
                axes = axes.flatten() if nrows > 1 else axes

            # 각 주식별로 YTD 차트 그리기
            for idx, stock in enumerate(stocks):
                ticker = stock.get('ticker', 'N/A')
                company_name = stock.get('company_name', 'Unknown')
                current_price = stock.get('current_price', 0)
                metrics = stock.get('metrics', {})
                historical = stock.get('historical', '')

                ax = axes[idx]
                color = colors_list[idx % len(colors_list)]

                # historical 데이터 파싱
                if not historical or len(historical.strip()) == 0:
                    ax.text(0.5, 0.5, f'{ticker}: YTD 차트 데이터 없음',
                           ha='center', va='center', fontsize=12, color='red')
                    ax.set_title(f'{company_name} ({ticker})', fontsize=12, fontweight='bold')
                    ax.axis('off')
                    continue

                try:
                    # CSV 파싱 (첫 줄은 메타데이터)
                    lines = historical.strip().split('\n')
                    if len(lines) < 2:
                        ax.text(0.5, 0.5, f'{ticker}: 데이터 부족',
                               ha='center', va='center', fontsize=12, color='orange')
                        ax.set_title(f'{company_name} ({ticker})', fontsize=12, fontweight='bold')
                        ax.axis('off')
                        continue

                    csv_data = '\n'.join(lines[1:])
                    df = pd.read_csv(StringIO(csv_data))

                    # 날짜 컬럼 파싱
                    date_col = df.columns[0]
                    df[date_col] = pd.to_datetime(df[date_col], utc=True)
                    df = df.sort_values(date_col)

                    # YTD 라인 차트 그리기
                    ax.plot(df[date_col], df['Close'], color=color, linewidth=2, label='Close Price')
                    ax.fill_between(df[date_col], df['Close'], alpha=0.3, color=color)

                    # 제목 및 레이블
                    ax.set_title(f'{company_name} ({ticker}) - YTD', fontsize=12, fontweight='bold')
                    ax.set_xlabel('Date', fontsize=10)
                    ax.set_ylabel('Price ($)', fontsize=10)
                    ax.grid(True, alpha=0.3)

                    # 현재가 표시
                    ax.text(0.02, 0.98, f'Current: ${current_price:.2f}',
                           transform=ax.transAxes, fontsize=10, fontweight='bold',
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                    # 날짜 포맷 조정
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

                except Exception as parse_err:
                    logger.warning(f"{ticker} 차트 파싱 실패: {parse_err}")
                    ax.text(0.5, 0.5, f'{ticker}: 차트 파싱 실패',
                           ha='center', va='center', fontsize=12, color='orange')
                    ax.set_title(f'{company_name} ({ticker})', fontsize=12, fontweight='bold')
                    ax.axis('off')

            # 남은 빈 subplot 숨기기
            for idx in range(num_stocks, len(axes)):
                axes[idx].axis('off')

            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            return f"✓ 비교 YTD 차트가 {save_path}에 저장되었습니다."

        except Exception as e:
            logger.error(f"비교 차트 생성 실패: {str(e)}")
            return f"차트 생성 중 오류: {str(e)}"

    try:
        logger.info(f"주식 차트 생성 시작 - 원본 output_path: {repr(output_path)}")

        # Clean output_path - remove any trailing artifacts
        import re
        match = re.search(r'([a-zA-Z0-9_/.-]+\.(?:png|jpg|jpeg|pdf|svg|webp))', output_path, re.IGNORECASE)
        if match:
            output_path = match.group(1)
        else:
            output_path = re.split(r'["\}\]\n\r]', output_path)[0].strip()
            if not output_path.endswith(('.png', '.jpg', '.jpeg', '.pdf', '.svg', '.webp')):
                output_path += '.png'

        logger.info(f"정리된 output_path: {output_path}")

        # 파라미터로 전달된 데이터 우선, 없으면 글로벌 변수 사용
        if analysis_data_json:
            financial_data_json = analysis_data_json
            logger.info("📊 파라미터로 전달된 analysis_data 사용")
        else:
            financial_data_json = _get_current_analysis_data()
            logger.info("📊 글로벌 변수에서 analysis_data 가져오기")

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


@tool
def draw_valuation_radar(
    output_path: str = "charts/valuation_radar.png",
    analysis_data_json: str = None
) -> str:
    """주식의 밸류에이션 지표를 레이더 차트로 시각화합니다.

    이 도구는 주식의 5가지 핵심 지표(성장성, 가치, 모멘텀, 품질, 시장심리)를
    레이더 차트(거미줄 차트)로 시각화하여 종합적인 투자 매력도를 한눈에 파악할 수 있게 합니다.

    단일 주식 분석 시에는 업계 기준 대비 절대적 평가를 하고,
    비교 분석 시에는 여러 주식의 상대적 강점을 비교합니다.

    Args:
        output_path: 레이더 차트 이미지를 저장할 경로 (기본값: "charts/valuation_radar.png")
                    지원 형식: .png, .jpg, .jpeg, .pdf, .svg, .webp
        analysis_data_json: 분석 데이터 JSON 문자열 (선택사항, 없으면 글로벌 변수 사용)

    Returns:
        차트 저장 결과 메시지 (성공 시 "✓ 레이더 차트가 {경로}에 저장되었습니다.", 실패 시 오류 메시지)

    Examples:
        >>> draw_valuation_radar("charts/aapl_valuation.png")
        "✓ 레이더 차트가 charts/aapl_valuation.png에 저장되었습니다."

        >>> draw_valuation_radar("charts/comparison_radar.png")
        "✓ 레이더 차트가 charts/comparison_radar.png에 저장되었습니다."
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

        # 파라미터로 전달된 데이터 우선, 없으면 글로벌 변수 사용
        if analysis_data_json:
            financial_data_json = analysis_data_json
            logger.info("📊 파라미터로 전달된 analysis_data 사용 (레이더 차트)")
        else:
            financial_data_json = _get_current_analysis_data()
            logger.info("📊 글로벌 변수에서 analysis_data 가져오기 (레이더 차트)")

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
    current_price = data.get('current_price')
    if current_price is None:
        current_price = 0

    # None 체크를 명시적으로 수행
    market_cap = metrics.get('market_cap')
    if market_cap is None or market_cap <= 0:
        market_cap = 0

    pe_ratio = metrics.get('pe_ratio')
    if pe_ratio is None or pe_ratio <= 0:
        pe_ratio = 20

    high_52w = metrics.get('52week_high')
    if high_52w is None:
        high_52w = 0

    low_52w = metrics.get('52week_low')
    if low_52w is None:
        low_52w = 0

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
        # None 체크를 명시적으로 수행
        pe_ratio = metrics.get('pe_ratio')
        if pe_ratio is None or pe_ratio <= 0:
            pe_ratio = 20

        market_cap = metrics.get('market_cap')
        if market_cap is None or market_cap <= 0:
            market_cap = 1e9

        current_price = stock.get('current_price')
        if current_price is None:
            current_price = 0

        high_52w = metrics.get('52week_high')
        if high_52w is None:
            high_52w = 0

        low_52w = metrics.get('52week_low')
        if low_52w is None:
            low_52w = 0

        all_pe_ratios.append(pe_ratio)
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
    """생성된 보고서를 지정한 형식의 파일로 저장합니다.

    이 도구는 텍스트 보고서를 파일로 저장하며, TXT, Markdown, PDF 형식을 지원합니다.
    PDF 형식의 경우 차트 이미지를 함께 포함할 수 있어 완전한 보고서를 만들 수 있습니다.

    지원 형식:
    - txt: 일반 텍스트 파일 (간단한 저장용)
    - md: Markdown 파일 (구조화된 문서, 기본값)
    - pdf: PDF 파일 (차트 포함 가능, 프레젠테이션용)

    Args:
        report_text: 저장할 보고서의 텍스트 내용 (Markdown 형식 권장)
        format: 파일 형식 (기본값: "md")
               선택 가능: "txt", "md", "pdf"
        output_path: 파일을 저장할 경로 (Optional)
                    None인 경우 자동으로 "reports/report_YYYYMMDD_HHMMSS.{format}" 형식으로 생성
        chart_paths: 포함할 차트 이미지 경로들 (Optional, PDF 형식에만 사용)
                    여러 개인 경우 쉼표로 구분 (예: "charts/chart1.png,charts/chart2.png")

    Returns:
        저장 결과 메시지
        - 성공 시: "✓ 보고서가 {경로}에 저장되었습니다." 또는 "✓ PDF 보고서가 {경로}에 저장되었습니다."
        - 실패 시: "❌ " 로 시작하는 오류 메시지

    Examples:
        >>> save_report_to_file("# Stock Analysis\\n\\nApple Inc...", format="md")
        "✓ 보고서가 reports/report_20250131_143022.md에 저장되었습니다."

        >>> save_report_to_file("Report content", format="pdf", output_path="reports/aapl.pdf", chart_paths="charts/aapl.png")
        "✓ PDF 보고서가 reports/aapl.pdf에 저장되었습니다."

        >>> save_report_to_file("Simple text", format="txt", output_path="reports/summary.txt")
        "✓ 보고서가 reports/summary.txt에 저장되었습니다."
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
        
        # 한글 폰트 등록
        korean_font = 'Helvetica'
        korean_font_bold = 'Helvetica-Bold'
        font_registered = False
        
        try:
            system = platform.system()
            project_fonts_dir = Path(__file__).resolve().parents[3] / "fonts"
            
            font_pairs = []
            if project_fonts_dir.exists():
                font_pairs.extend([
                    (project_fonts_dir / "NanumGothic.ttf", project_fonts_dir / "NanumGothicBold.ttf"),
                    (project_fonts_dir / "NanumBarunGothic.ttf", project_fonts_dir / "NanumBarunGothicBold.ttf"),
                ])
            
            if system == 'Windows':
                font_pairs.append((Path('C:/Windows/Fonts/malgun.ttf'), Path('C:/Windows/Fonts/malgunbd.ttf')))
            elif system == 'Darwin':
                font_pairs.append((Path('/System/Library/Fonts/AppleGothic.ttf'), None))
            else:  # Linux
                font_pairs.append((Path('/usr/share/fonts/truetype/nanum/NanumGothic.ttf'),
                                   Path('/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf')))
            
            for regular_path, bold_path in font_pairs:
                if regular_path.exists():
                    pdfmetrics.registerFont(TTFont('Korean', str(regular_path)))
                    korean_font = 'Korean'
                    korean_font_bold = 'Korean'
                    
                    if bold_path and bold_path.exists():
                        pdfmetrics.registerFont(TTFont('KoreanBold', str(bold_path)))
                        korean_font_bold = 'KoreanBold'
                    
                    font_registered = True
                    break
        
        except Exception as e:
            logger.warning(f"폰트 등록 실패: {str(e)}")
        
        if not font_registered:
            logger.warning("한글 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")
        
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
                (i == 0 and len(line) < 100)):
                
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
        logger.info(f"사용된 폰트: {korean_font} (임베딩: True)")
        
        return f"✓ PDF 보고서가 {output_path}에 저장되었습니다. (폰트 임베딩 완료)"
            
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
