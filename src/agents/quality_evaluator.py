# src/agents/quality_evaluator.py
"""
Quality Evaluator Module

이 모듈은 에이전트가 생성한 최종 답변의 품질을 평가하는 역할을 합니다.
'LLM-as-a-judge' 패턴을 적용하거나, 간단한 규칙 기반으로 답변이 유효한지 판단합니다.
"""

from typing import Dict, Any

from langchain_core.language_models.chat_models import BaseChatModel

from src.model.llm import get_llm_manager
from src.utils.config import Config
from src.utils.logger import get_logger

# 로거 설정
logger = get_logger(__name__)


class QualityEvaluator:
    """
    답변의 품질을 평가하는 클래스.
    """

    def __init__(self, llm: BaseChatModel = None, threshold: int = None):
        """
        LLM 모델과 품질 평가 통과 기준 점수(threshold)를 받아 초기화합니다.

        Args:
            llm (BaseChatModel): 평가에 사용할 LLM. None이면 기본 모델 사용
            threshold (int): 평가 통과 최저 점수. None이면 Config에서 가져옴
        """
        # LLM 초기화
        if llm is None:
            llm_manager = get_llm_manager()
            self.llm = llm_manager.get_model(Config.LLM_MODEL, temperature=0)
            logger.info(f"기본 LLM 모델 사용: {Config.LLM_MODEL}")
        else:
            self.llm = llm
            logger.info("사용자 제공 LLM 모델 사용")

        # Threshold 설정
        self.threshold = threshold if threshold is not None else Config.QUALITY_THRESHOLD
        logger.info(f"품질 평가 threshold 설정: {self.threshold}")

        # 프롬프트 가져오기
        llm_manager = get_llm_manager()
        evaluation_prompt = llm_manager.get_prompt("quality_evaluator")
        self.evaluation_chain = evaluation_prompt | self.llm

    def _is_critical_error(self, answer: str) -> bool:
        """
        답변에 치명적인 에러가 있는지 판단합니다.

        단순 키워드 매칭이 아닌 컨텍스트를 고려한 스마트 판단:
        1. 성공 지표가 있으면 비치명적 (부분 성공 인정)
        2. 치명적 에러 패턴 확인
        3. 일반 에러 키워드는 성공 지표 없을 때만 치명적

        Args:
            answer: 평가할 답변 텍스트

        Returns:
            True면 치명적 에러, False면 정상 또는 부분 성공
        """
        answer_lower = answer.lower()

        # 1. 성공 지표 확인 - 하나라도 있으면 부분 성공으로 인정
        success_indicators = [
            "✓", "✔", "성공", "완료", "저장되었습니다", "저장됨",
            "생성되었습니다", "생성 완료", "saved", "completed", "successfully"
        ]
        has_success = any(indicator in answer for indicator in success_indicators)

        if has_success:
            logger.debug("성공 지표 발견 - 부분 성공으로 인정")
            return False

        # 2. 치명적 에러 패턴 - 완전 실패를 의미하는 메시지
        critical_patterns = [
            "분석 데이터를 찾을 수 없습니다",
            "질문이 비어 있어",
            "적합한 에이전트를 찾을 수 없습니다",
            "보고서를 생성하지 못했습니다",
            "답변을 드릴 수 없습니다",
            "처리할 수 없습니다",
            "오류가 발생했습니다",
            "analysis_data를 찾을 수 없습니다"
        ]

        for pattern in critical_patterns:
            if pattern in answer:
                logger.warning(f"치명적 에러 패턴 감지: {pattern}")
                return True

        # 3. 일반 에러 키워드 - 성공 지표가 없고 에러만 있는 경우
        error_keywords = ["error", "failed", "could not", "unable to", "오류", "실패"]
        if any(keyword in answer_lower for keyword in error_keywords):
            logger.debug("일반 에러 키워드 발견 (성공 지표 없음)")
            return True

        return False

    def evaluate_answer(self, question: str, answer: str) -> Dict[str, Any]:
        """
        답변의 품질을 평가합니다 (module_plan.md 요구사항 준수).

        평가 순서:
        1. Empty 체크 (10자 미만)
        2. Critical Error 체크 (스마트 에러 감지)
        3. LLM-as-a-judge로 품질 평가 (최종 판단)

        Args:
            question (str): 사용자의 원본 질문.
            answer (str): 에이전트가 생성한 답변.

        Returns:
            평가 결과 딕셔너리:
            - status: "pass" or "fail"
            - score: 1-5 점수
            - failure_reason: "empty" | "error" | "incorrect" | None
        """
        logger.info("답변 품질 평가 시작...")
        logger.debug(f"질문: {question[:100]}...")
        logger.debug(f"답변: {answer[:100] if answer else '(empty)'}...")

        # 1. Empty 체크 (10자 미만)
        if not answer or len(answer.strip()) < 10:
            logger.warning("품질 평가 실패: 답변이 비어있거나 10자 미만")
            return {
                "status": "fail",
                "score": 0,
                "failure_reason": "empty"
            }

        # 2. Critical Error 체크 - 스마트 에러 감지
        if self._is_critical_error(answer):
            logger.warning(f"품질 평가 실패: 치명적 에러 감지")
            return {
                "status": "fail",
                "score": 0,
                "failure_reason": "error"
            }

        # 3. LLM-as-a-judge로 품질 평가
        try:
            response = self.evaluation_chain.invoke({
                "question": question,
                "answer": answer
            })

            # LLM의 답변에서 첫 번째 1-5 사이의 숫자만 추출
            import re
            match = re.search(r'[1-5]', response.content)
            score = int(match.group()) if match else 0
            logger.info(f"추출된 품질 점수: {score}")
            logger.debug(f"LLM 원본 응답: {response.content}")

            # 기준 점수와 비교하여 통과/실패 결정
            if score >= self.threshold:
                status = "pass"
                failure_reason = None
                logger.info(f"품질 평가 통과 (점수: {score}/{self.threshold} 이상)")
            else:
                status = "fail"
                failure_reason = "incorrect"
                logger.warning(f"품질 평가 실패 (점수: {score}/{self.threshold} 미만) - incorrect")

            return {
                "status": status,
                "score": score,
                "failure_reason": failure_reason
            }

        except Exception as e:
            logger.error(f"품질 평가 중 오류 발생: {e}", exc_info=True)
            # 오류 발생 시 안전하게 'fail' 처리
            return {
                "status": "fail",
                "score": 0,
                "failure_reason": "error"
            }
