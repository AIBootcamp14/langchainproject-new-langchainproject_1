"""
Report Generator (Structured Output 기반)

financial_analyst 또는 vector_search_agent의 출력을 받아서
보고서를 생성하고, 필요시 차트를 그리고, 파일로 저장합니다.

ReAct 에이전트 대신 Structured Output으로 계획을 수립한 후 순차적으로 도구를 호출합니다.
"""

import json
import os
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

from src.agents.tools.report_tools import (
    draw_stock_chart,
    draw_valuation_radar,
    save_report_to_file,
    _set_current_analysis_data,
    _get_current_analysis_data
)
from src.model.llm import get_llm_manager
from src.utils.logger import get_logger
from src.utils.config import Config

logger = get_logger(__name__)


class ReportPlan(BaseModel):
    """보고서 생성 계획 (Structured Output)"""
    needs_stock_chart: bool = Field(description="주가 차트가 필요한가? (주가 추이, 52주 범위 등)")
    needs_valuation_chart: bool = Field(description="밸류에이션 레이더 차트가 필요한가? (재무 지표 비교)")
    needs_save: bool = Field(description="파일로 저장이 필요한가?")
    save_format: Optional[str] = Field(
        default=None,
        description="저장 형식 (pdf, md, txt 중 하나). needs_save=True일 때만 필수"
    )
    report_title: str = Field(description="보고서 제목 (회사명 또는 비교 대상 포함)")
    report_text: str = Field(description="보고서 본문 (마크다운 형식, 분석 데이터 기반)")


class ReportGenerator:
    def __init__(self, model_name: str = None, temperature: float = 0.0):
        """
        Report Generator를 초기화합니다.

        Args:
            model_name: 사용할 모델명 (default: Config.LLM_MODEL)
            temperature: LLM 온도 (0.0 = 결정적)
        """
        if model_name is None:
            model_name = Config.LLM_MODEL
        logger.info(f"Report Generator 초기화 (Structured Output) - model: {model_name}, temp: {temperature}")

        # LLM Manager에서 모델 가져오기
        llm_manager = get_llm_manager()
        self.llm = llm_manager.get_model(model_name, temperature=temperature)

        logger.info("Report Generator 초기화 완료")

    def generate_report(
        self,
        user_request: str,
        analysis_data: Dict[str, Any],
        messages: list = None,
    ) -> Dict[str, Any]:
        """
        분석 데이터를 기반으로 보고서를 생성합니다.

        Structured Output으로 계획을 수립한 후 순차적으로 도구를 호출합니다.

        Args:
            user_request: 사용자 요청 (예: "삼성 주식 분석 PDF로 저장해줘")
            analysis_data: 분석 데이터 딕셔너리
            messages: 대화 히스토리 (선택사항)

        Returns:
            Dict with keys: report, status, charts, saved_path
        """
        if messages is None:
            messages = []

        try:
            logger.info(f"보고서 생성 시작 - request: {user_request[:50]}...")

            if not analysis_data:
                logger.error("Analysis data is empty")
                return {
                    "report": "❌ 분석 데이터가 없습니다.",
                    "status": "error",
                    "charts": [],
                    "saved_path": None
                }

            # Step 1: analysis_data를 JSON으로 변환 (도구가 사용)
            analysis_json = json.dumps(analysis_data, ensure_ascii=False, indent=2)
            _set_current_analysis_data(analysis_json)

            # Step 2: LLM에게 계획 수립 요청 (Structured Output)
            logger.info("📝 보고서 계획 수립 중...")
            plan = self._create_plan(user_request, analysis_data, messages)

            logger.info(f"✅ 계획 완료 - 주가차트: {plan.needs_stock_chart}, "
                       f"밸류차트: {plan.needs_valuation_chart}, "
                       f"저장: {plan.needs_save} ({plan.save_format})")

            # Step 3: 계획에 따라 도구 순차 호출
            charts = []
            saved_path = None

            # 3-1. 주가 차트 생성
            if plan.needs_stock_chart:
                logger.info("📊 주가 차트 생성 중...")
                try:
                    chart_path = draw_stock_chart.invoke({"output_path": "charts/stock_chart.png"})
                    if "성공" in chart_path or "저장" in chart_path:
                        # "charts/xxx.png" 추출
                        import re
                        match = re.search(r'(charts/[^\s]+\.png)', chart_path)
                        if match:
                            charts.append(match.group(1))
                            logger.info(f"✅ 주가 차트 생성 완료: {match.group(1)}")
                except Exception as e:
                    logger.warning(f"⚠️ 주가 차트 생성 실패: {e}")

            # 3-2. 밸류에이션 레이더 차트 생성
            if plan.needs_valuation_chart:
                logger.info("📊 밸류에이션 레이더 차트 생성 중...")
                try:
                    chart_path = draw_valuation_radar.invoke({"output_path": "charts/valuation_radar.png"})
                    if "성공" in chart_path or "저장" in chart_path:
                        import re
                        match = re.search(r'(charts/[^\s]+\.png)', chart_path)
                        if match:
                            charts.append(match.group(1))
                            logger.info(f"✅ 밸류에이션 차트 생성 완료: {match.group(1)}")
                except Exception as e:
                    logger.warning(f"⚠️ 밸류에이션 차트 생성 실패: {e}")

            # 3-3. 파일 저장
            if plan.needs_save and plan.save_format:
                logger.info(f"💾 보고서 저장 중 ({plan.save_format})...")
                try:
                    # 파일명 생성 (제목 기반)
                    safe_title = "".join(c for c in plan.report_title if c.isalnum() or c in (' ', '_', '-'))
                    safe_title = safe_title.replace(' ', '_')[:50]
                    output_filename = f"reports/{safe_title}.{plan.save_format}"

                    # 차트 경로를 콤마 구분 문자열로 변환
                    chart_paths_str = ",".join(charts) if charts else None

                    result = save_report_to_file.invoke({
                        "report_text": plan.report_text,
                        "format": plan.save_format,
                        "output_path": output_filename,
                        "chart_paths": chart_paths_str
                    })

                    if "성공" in result or "저장" in result:
                        import re
                        match = re.search(r'(reports/[^\s]+\.(pdf|md|txt))', result)
                        if match:
                            saved_path = match.group(1)
                            logger.info(f"✅ 파일 저장 완료: {saved_path}")
                except Exception as e:
                    logger.warning(f"⚠️ 파일 저장 실패: {e}")

            logger.info(f"📄 보고서 생성 완료 - charts: {len(charts)}, saved: {saved_path is not None}")

            return {
                "report": plan.report_text,
                "status": "success",
                "charts": charts,
                "saved_path": saved_path
            }

        except Exception as e:
            logger.error(f"보고서 생성 실패: {str(e)}")
            import traceback
            logger.debug(f"상세 에러:\n{traceback.format_exc()}")

            # 폴백: 직접 보고서 생성
            try:
                fallback_report = self._generate_report_directly(analysis_data)
                return {
                    "report": fallback_report,
                    "status": "partial",
                    "charts": [],
                    "saved_path": None,
                    "error": str(e)
                }
            except:
                return {
                    "report": f"❌ 보고서 생성 중 오류가 발생했습니다: {str(e)}",
                    "status": "error",
                    "charts": [],
                    "saved_path": None,
                    "error": str(e)
                }

    def _create_plan(
        self,
        user_request: str,
        analysis_data: Dict[str, Any],
        messages: list
    ) -> ReportPlan:
        """
        LLM을 사용하여 보고서 생성 계획을 수립합니다 (Structured Output).

        Args:
            user_request: 사용자 요청
            analysis_data: 분석 데이터
            messages: 대화 히스토리

        Returns:
            ReportPlan 객체
        """
        # Structured Output 설정
        llm_with_structure = self.llm.with_structured_output(ReportPlan)

        # 프롬프트 구성
        analysis_summary = self._summarize_analysis_data(analysis_data)

        planning_prompt = f"""당신은 금융 보고서 생성 전문가입니다.

사용자 요청과 분석 데이터를 기반으로 보고서 생성 계획을 수립하세요.

사용자 요청: {user_request}

분석 데이터 요약:
{analysis_summary}

계획 수립 가이드라인:
1. **needs_stock_chart**: 주가 추이, 52주 범위, 차트 등이 언급되면 True
2. **needs_valuation_chart**: 재무 지표, 밸류에이션, 비교 분석이 언급되면 True
3. **needs_save**: 저장, 파일, PDF, MD 등이 언급되면 True
4. **save_format**:
   - "pdf" 언급 시 → "pdf"
   - "markdown", "md" 언급 시 → "md"
   - "텍스트", "txt" 언급 시 → "txt"
   - 명시 없으면 "pdf" (기본값)
5. **report_title**: 회사명 또는 비교 대상 포함 (예: "삼성전자 분석", "AAPL vs MSFT 비교")
6. **report_text**:
   - 마크다운 형식으로 작성
   - 분석 데이터의 핵심 내용 포함
   - 섹션 구조: 개요, 주가 정보, 재무 지표, 분석 의견, 추천
   - 차트가 생성될 경우 차트 참조 포함 (예: "![차트](charts/stock_chart.png)")

CRITICAL: report_text는 완전한 보고서여야 합니다. 이것이 사용자에게 전달되는 최종 결과물입니다.
"""

        # Structured Output으로 계획 생성
        plan = llm_with_structure.invoke(planning_prompt)

        return plan

    def _summarize_analysis_data(self, analysis_data: Dict[str, Any]) -> str:
        """분석 데이터를 요약하여 프롬프트에 전달할 문자열로 변환"""
        analysis_type = analysis_data.get("analysis_type", "unknown")

        if analysis_type == "single":
            return f"""타입: 단일 주식 분석
티커: {analysis_data.get('ticker', 'N/A')}
회사명: {analysis_data.get('company_name', 'N/A')}
현재가: {analysis_data.get('current_price', 'N/A')}
분석 내용: {analysis_data.get('analysis', 'N/A')[:200]}...
추천: {analysis_data.get('analyst_recommendation', 'N/A')}"""

        elif analysis_type == "comparison":
            stocks = analysis_data.get("stocks", [])
            tickers = [s.get("ticker") for s in stocks]
            return f"""타입: 비교 분석
대상 주식: {', '.join(tickers)}
주식 수: {len(stocks)}
비교 분석: {analysis_data.get('comparison_summary', 'N/A')[:200]}..."""

        elif analysis_type == "rag":
            return f"""타입: RAG 검색
질문: {analysis_data.get('query', 'N/A')}
문서 수: {len(analysis_data.get('documents', []))}"""

        else:
            # JSON 전체를 요약
            return f"""타입: {analysis_type}
데이터: {json.dumps(analysis_data, ensure_ascii=False)[:300]}..."""

    def _generate_report_directly(self, analysis_data: Dict[str, Any]) -> str:
        """
        Structured Output 실패 시 폴백: 직접 보고서 생성

        Args:
            analysis_data: 분석 데이터

        Returns:
            마크다운 형식의 보고서 텍스트
        """
        logger.info("폴백: 직접 보고서 생성")

        analysis_type = analysis_data.get("analysis_type", "single")
        llm_manager = get_llm_manager()

        try:
            if analysis_type == "single":
                ticker = analysis_data.get("ticker", "N/A")
                company = analysis_data.get("company_name", "Unknown")
                price = analysis_data.get("current_price", "N/A")
                analysis_text = analysis_data.get("analysis", "분석 내용 없음")
                recommendation = analysis_data.get("analyst_recommendation", "N/A")

                return f"""# {company}({ticker}) 분석 보고서

## 주가 정보
- 현재가: {price}

## 분석 의견
{analysis_text}

## 투자 추천
{recommendation}

---
*본 보고서는 제공된 데이터를 기반으로 생성되었습니다.*
"""

            elif analysis_type == "comparison":
                stocks = analysis_data.get("stocks", [])
                tickers = [s.get("ticker") for s in stocks]
                comparison = analysis_data.get("comparison_summary", "비교 분석 없음")

                stocks_section = "\n".join([
                    f"### {s.get('ticker')}: {s.get('company_name')}\n- 현재가: {s.get('current_price')}\n- {s.get('analysis', '')}\n"
                    for s in stocks
                ])

                return f"""# {' vs '.join(tickers)} 비교 분석

## 개별 분석
{stocks_section}

## 비교 의견
{comparison}

---
*본 보고서는 제공된 데이터를 기반으로 생성되었습니다.*
"""

            elif analysis_type == "rag":
                query = analysis_data.get("query", "")
                documents = analysis_data.get("documents", [])

                docs_text = "\n\n".join([f"**문서 {i+1}**\n{doc[:300]}..." for i, doc in enumerate(documents[:3])])

                return f"""# RAG 검색 결과

## 질문
{query}

## 검색된 정보
{docs_text}

---
*본 보고서는 검색된 문서를 기반으로 생성되었습니다.*
"""

            else:
                return f"""# 분석 보고서

{json.dumps(analysis_data, ensure_ascii=False, indent=2)}

---
*분석 타입: {analysis_type}*
"""

        except Exception as e:
            logger.error(f"폴백 보고서 생성 실패: {e}")
            return f"❌ 보고서 생성 중 오류가 발생했습니다: {str(e)}"


if __name__ == "__main__":
    import logging

    # 디버그 로그 활성화
    logging.getLogger("__main__").setLevel(logging.DEBUG)

    Config.validate_api_keys()

    # 테스트용 샘플 데이터
    SAMPLE_SINGLE_STOCK = {
        "analysis_type": "single",
        "ticker": "005930.KS",
        "company_name": "삼성전자",
        "current_price": 70000,
        "analysis": "삼성전자는 반도체 업황 회복과 AI 수요 증가로 중장기 성장성이 긍정적입니다.",
        "metrics": {
            "pe_ratio": 10.5,
            "market_cap": 400000000000000,
            "52week_high": 85000,
            "52week_low": 55000,
        },
        "analyst_recommendation": "매수"
    }

    # Report Generator 초기화
    print("\n" + "="*80)
    print("REPORT GENERATOR 테스트 (Structured Output)")
    print("="*80)

    generator = ReportGenerator()

    # 출력 디렉토리 생성
    os.makedirs("charts", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    print("\n삼성전자 분석을 PDF로 차트 포함해서 저장 테스트")
    print("-"*80)

    result = generator.generate_report(
        "삼성전자 분석을 PDF로 차트 포함해서 저장해줘",
        SAMPLE_SINGLE_STOCK
    )

    print(f"\n[결과]")
    print(f"상태: {result['status']}")
    print(f"차트: {result['charts']}")
    print(f"저장: {result['saved_path']}")
    print(f"\n보고서:\n{result['report'][:500]}...")

    print("\n✅ 테스트 완료!")
