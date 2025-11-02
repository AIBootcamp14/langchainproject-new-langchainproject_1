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

    def evaluate_answer(self, question: str, answer: str) -> Dict[str, Any]:
        """
        답변의 품질을 평가합니다 (module_plan.md 요구사항 준수).

        평가 순서:
        1. Empty 체크 (10자 미만)
        2. Error 패턴 체크 (키워드 검색)
        3. LLM-as-a-judge로 품질 평가

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

        # 2. Error 패턴 체크
        error_keywords = ["error", "failed", "could not", "unable to", "오류", "실패", "찾을 수 없"]
        answer_lower = answer.lower()
        if any(keyword in answer_lower for keyword in error_keywords):
            logger.warning(f"품질 평가 실패: 에러 키워드 감지")
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
