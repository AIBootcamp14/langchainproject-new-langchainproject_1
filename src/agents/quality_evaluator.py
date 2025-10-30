# src/agents/quality_evaluator.py

"""
Quality Evaluator Module

이 모듈은 에이전트가 생성한 최종 답변의 품질을 평가하는 역할을 합니다.
'LLM-as-a-judge' 패턴을 적용하거나, 간단한 규칙 기반으로 답변이 유효한지 판단합니다.
"""

from typing import Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel

from src.utils.logger import get_logger

# 로거 설정
logger = get_logger(__name__)

# LLM-as-a-judge를 위한 프롬프트
EVALUATION_PROMPT = ChatPromptTemplate.from_template(
    """당신은 답변의 품질을 평가하는 엄격한 평가관입니다.
사용자의 질문에 대해 에이전트가 생성한 답변이 적절한지, 유용한 정보를 포함하고 있는지, 오류는 없는지 평가해주세요.

[사용자 질문]
{question}

[에이전트의 답변]
{answer}

[평가 기준]
1. 질문의 의도에 맞는 답변인가?
2. 답변에 '오류', '찾을 수 없음' 등 실패를 의미하는 내용이 포함되어 있지는 않은가?
3. 답변이 구체적이고 명확한가?

위 기준에 따라 답변의 품질을 1점에서 5점 사이의 점수로만 평가해주세요. 다른 설명은 절대 추가하지 마세요.

평가 점수:"""
)


class QualityEvaluator:
    """
    답변의 품질을 평가하는 클래스.
    """

    def __init__(self, llm: BaseChatModel, threshold: int = 3):
        """
        LLM 모델과 품질 평가 통과 기준 점수(threshold)를 받아 초기화합니다.

        Args:
            llm (BaseChatModel): 평가에 사용할 LLM.
            threshold (int): 평가 통과 최저 점수.
        """
        self.llm = llm
        self.threshold = threshold
        self.evaluation_chain = EVALUATION_PROMPT | self.llm

    def evaluate_answer(self, question: str, answer: str) -> Dict[str, Any]:
        """
        'LLM-as-a-judge'를 이용해 답변의 품질을 평가하고 점수를 매깁니다.

        Args:
            question (str): 사용자의 원본 질문.
            answer (str): 에이전트가 생성한 답변.

        Returns:
            평가 결과 딕셔너리 ({"status": "pass" or "fail", "score": 점수})
        """
        logger.info("답변 품질 평가 시작...")

        try:
            # LLM에게 질문과 답변을 주고 평가를 요청합니다.
            response = self.evaluation_chain.invoke({
                "question": question,
                "answer": answer
            })

            # LLM의 답변에서 숫자(점수)만 추출합니다.
            score_text = "".join(filter(str.isdigit, response.content))
            score = int(score_text) if score_text else 0

            # 기준 점수와 비교하여 통과/실패를 결정합니다.
            if score >= self.threshold:
                status = "pass"
                logger.info(f"품질 평가 통과 (점수: {score}/{self.threshold} 이상)")
            else:
                status = "fail"
                logger.warning(f"품질 평가 실패 (점수: {score}/{self.threshold} 미만)")
            
            return {"status": status, "score": score}

        except Exception as e:
            logger.error(f"품질 평가 중 오류 발생: {e}")
            # 오류 발생 시 안전하게 'fail' 처리
            return {"status": "fail", "score": 0}

