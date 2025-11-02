from __future__ import annotations

from typing import Dict, List, Literal, Optional, TypedDict

from langgraph.graph import END, StateGraph

from src.agents.financial_analyst import FinancialAnalyst
from src.agents.quality_evaluator import QualityEvaluator
from src.agents.report_generator import ReportGenerator
from src.agents.request_analyst import request_analysis, rewrite_query
from src.agents.supervisor import supervisor
from src.model.llm import get_llm_manager
from src.rag.retriever import Retriever
from src.utils.config import Config

from src.utils.logger import get_logger

logger = get_logger(__name__)


class WorkflowState(TypedDict, total=False):
    """LangGraph에서 주고받는 기본 상태 구조.
    
    *** 개인적으로 필요한 상태 값들은 아래에 주석과 함께 추가 부탁드리겠습니다.*** 
    
    """
    session_id: str # 사용자 세션 id
    question: str # 사용자의 질문
    answer: str   # LLM 의 생성 답변
    route: Literal["end", "supervisor", "financial_analyst", "report_generator"]
    request_type: Literal["rag", "financial_analyst"]  # report_generator 의 2가지 task 분기
    rag_search_results: List[str]  # Rag 의 검색 결과
    analysis_data: Dict[str, object] # Rag 혹은 financial_analyst 의 최종 분석 결과
    quality_passed: bool       # quality_evaluator 에서의 품질 통과 여부  
    quality_detail: Dict[str, object]  # quality_evaluator 의 평가 결과 디테일
    retries : int  # 루프 재시도 횟수 
    


class Workflow:
    """요청 분석 → 라우팅 → 답변 생성 → 품질 평가 → (선택:재시도 루프) 까지 이어지는 워크플로우."""

    def __init__(self):
        self.llm_manager = get_llm_manager()
        self.shared_llm = self.llm_manager.get_model(Config.LLM_MODEL, temperature=Config.LLM_TEMPERATURE)

        self.retriever = Retriever()
        self.financial_analyst = FinancialAnalyst()
        self.report_generator = ReportGenerator()
        self.quality_evaluator = QualityEvaluator(llm=self.shared_llm)
        self.graph = self._build_graph()

   
    def _build_graph(self):
        graph = StateGraph(WorkflowState)

        graph.add_node("request_analyst", self.request_analyst_node)
        graph.add_node("supervisor", self.supervisor_node)
        graph.add_node("financial_analyst", self.financial_analyst_node)
        graph.add_node("report_generator", self.report_generator_node)
        graph.add_node("quality_evaluator", self.quality_evaluator_node)

        graph.set_entry_point("request_analyst")

        graph.add_conditional_edges(
            "request_analyst",
            self._route_from_request_analyst,
            {
                "end": END,
                "supervisor": "supervisor",
            },
        )

        graph.add_conditional_edges(
            "supervisor",
            self._route_from_supervisor,
            {
                "financial_analyst": "financial_analyst",
                "report_generator": "report_generator",
                "end": END,
            },
        )

        graph.add_edge("financial_analyst", "report_generator")
        graph.add_edge("report_generator", "quality_evaluator")

        graph.add_conditional_edges(
            "quality_evaluator",
            self._route_from_quality_evaluator,
            {
                "retry": "request_analyst",
                "end": END,
            },
        )

        return graph.compile()

    # ------------------------------------------------------------------ #
    # Node 
    # ------------------------------------------------------------------ #
    def request_analyst_node(self, state: WorkflowState) -> WorkflowState:
        """질문이 경제, 금융 도메인인지 확인하고 비금융이면 바로 END 로 종료됩니다."""
        question = state.get("question", "").strip()
        if not question:
            state["answer"] = "질문이 비어 있어 답변을 드릴 수 없습니다."
            state["route"] = "end"
            return state

        analysis_result = request_analysis(state, llm=self.shared_llm)
        label = analysis_result.get("label")
        if label == "finance":
            state["route"] = "supervisor"
        else:
            # 비금융 질문인 경우 안내 메시지를 그대로 전달
            state["answer"] = analysis_result.get("return_msg", "경제, 금융관련 질문이 아닙니다.")
            state["route"] = "end"
        return state

    def supervisor_node(self, state: WorkflowState) -> WorkflowState:
        """슈퍼바이저 에이전트를 호출해 다음 노드를 결정합니다."""
        agent_choice = supervisor(
            state,
            llm=self.shared_llm,
        )

        if agent_choice == "financial_analyst":
            state["route"] = "financial_analyst"
        elif agent_choice == "vector_search_agent":
            state["route"] = "report_generator"
            state["request_type"] = "rag"
        else:
            state["answer"] = "적합한 에이전트를 찾을 수 없습니다. 그래프를 종료합니다."
            state["route"] = "end"
        return state

    def financial_analyst_node(self, state: WorkflowState) -> WorkflowState:
        """재무 분석 에이전트를 실행 후, report_generator를 호출합니다."""
        question = state.get("question", "")
        logger.info(f"🔍 financial_analyst_node 시작")
        analysis_data = self.financial_analyst.analyze(question)
         # 중요: 반환값 확인
        logger.info(f"📊 analyze() 반환 타입: {type(analysis_data)}")
        logger.info(f"📊 analyze() 반환 값: {analysis_data}")
    
        state["analysis_data"] = analysis_data
        state["request_type"] = "financial_analyst"
        logger.debug(f"재무 분석 에이전트 분석 결과 data : {analysis_data}")
        # 저장 후 확인 (중요!)
        logger.info(f"✅ state에 저장 완료")
        logger.info(f"✅ state['analysis_data'] 확인: {state.get('analysis_data', 'NOT FOUND')}")
        # summary = analysis.get("summary") or analysis.get("final_answer") or str(analysis)
        # state["answer"] = summary
        return state

    def report_generator_node(self, state: WorkflowState) -> WorkflowState:
        """RAG 검색 혹은 report 작성을 수행하여 최종 답변을 생성합니다."""
        question = state.get("question", "")
        logger.info(f"📝 report_generator_node 진입")
        logger.info(f"📝 request_type: {state.get('request_type', 'NOT SET')}")
        
        if state.get("request_type","rag") == "rag":
            logger.info("📝 RAG 모드")
            results = self.retriever.retrieve(question)
            rag_search_results = []
            for doc, score in results:
                page = doc.metadata.get("page", "?")
                if isinstance(page, int):
                    page += 1  # 0-index → 1-index 변환
                source = doc.metadata.get("source", "unknown")
                rag_search_results.append(f"- (score={score:.2f}) {source} p.{page}")

            state["rag_search_results"] = rag_search_results
            
            analysis_data = {
            "analysis_type" : "rag",
            "query": question,
            "documents": [doc.page_content for doc, _ in results],
            }
            
            state["analysis_data"] = analysis_data

        else:
            # financial_analyst 에서 호출 시, 해당 분석 결과 사용
            logger.info("📝 financial_analyst 모드")
            analysis_data = state.get("analysis_data")
            if not analysis_data:
                logger.error("❌ analysis_data가 state에 없습니다!")
                state["answer"] = "분석 데이터(analysis_data)를 찾을 수 없습니다."
                return state
            
            logger.debug(f"✅ State 저장소 analysis_data 로드: {analysis_data.get('analysis_type', 'N/A')}")
      
        report = self.report_generator.generate_report(question, analysis_data)
        state["answer"] = report.get("report", "보고서를 생성하지 못했습니다.")
        logger.info(f"✅ 보고서 생성 완료 (길이: {len(state['answer'])})")
        return state

    def quality_evaluator_node(self, state: WorkflowState) -> WorkflowState:
        """생성된 답변을 평가하고 필요 시 쿼리를 재작성합니다."""
        question = state.get("question", "")
        answer = state.get("answer", "")
        result = self.quality_evaluator.evaluate_answer(question, answer)

        state["quality_detail"] = result
        state["quality_passed"] = result.get("status") == "pass"

        if not state["quality_passed"]:
            state["retries"] = state.get("retries", 0) + 1
            rewrite_result = rewrite_query(
                original_query=question,
                failure_reason=result.get("failure_reason", "incorrect"),
                llm=self.shared_llm,
            )
            logger.info(
                "quality_evaluator_node 결과 | needs_user_input=%s | rewritten_query=%s",
                rewrite_result.get("needs_user_input"),
                rewrite_result.get("rewritten_query"),
            )
            
            if rewrite_result.get("needs_user_input"):
                state["answer"] = rewrite_result.get(
                    "request_for_detail_msg", "질문을 좀 더 구체적으로 말씀해 주시겠어요? "
                )
                state["route"] = "end"
            else:
                state["question"] = rewrite_result.get("rewritten_query", question)
                state["answer"] = "질문을 다시 정제했습니다. 재시도합니다."
        else:
            state["retries"] = 0        
            
        return state

    # ------------------------------------------------------------------ #
    # Edge routing helpers
    # ------------------------------------------------------------------ #
    def _route_from_request_analyst(self, state: WorkflowState) -> Literal["end", "supervisor"]:
        return "end" if state.get("route") == "end" else "supervisor"

    def _route_from_supervisor(self, state: WorkflowState) -> Literal["financial_analyst", "report_generator", "end"]:
        return state.get("route", "financial_analyst")

    def _route_from_quality_evaluator(self, state: WorkflowState) -> Literal["retry", "end"]:
        """
        품질 평가 결과에 따라 재시도 여부를 결정합니다.
        최대 3회까지만 재시도하며, 이후에는 강제로 종료합니다.
        """
        retries = state.get("retries", 0)  # 기본값 0
        quality_passed = state.get("quality_passed", False)
        
        logger.info("quality_passed 여부: %s", quality_passed)
        # 품질 통과 시 즉시 종료
        if quality_passed:
            return "end"

        # 품질 미통과 + 재시도 3회 미만이면 retry
        if retries < 3:
            logger.info("retries 횟수: %s  3회 미만까지는 retry 시도", state["retries"])
            return "retry"

        # 품질 미통과 + 재시도 3회 초과 → 강제 종료
        logger.warning(f"⚠️ 재시도 횟수 초과 (총 {retries}회). 루프 종료.")
        state["answer"] = "3회 재시도에도 품질 기준을 충족하지 못했습니다. 답변을 종료합니다."
        return "end"

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(self, question: str) -> WorkflowState:
        """사용자 질문에 따른 그래프를 실행한 뒤 최종 상태를 반환합니다."""
        initial_state: WorkflowState = {"question": question}
        return self.graph.invoke(initial_state)


def build_workflow() -> Workflow:
    """외부에서 간편하게 워크플로우 인스턴스를 생성할 때 사용."""
    return Workflow()


__all__ = ["Workflow", "WorkflowState", "build_workflow"]


# if __name__ == "__main__":
#     # from IPython.display import Image
#     wf = build_workflow()
#     # Image(wf.graph.get_graph().draw_png())
#     mermaid_code = wf.graph.get_graph().draw_mermaid()
#     # print(wf.graph.get_graph().draw_mermaid())
#     with open("/workspace/langchain_project/img/workflow_diagram.mmd", "w", encoding="utf-8") as f:
#         f.write(mermaid_code)
#     print("워크플로우 다이어그램이 workflow_diagram.mmd 파일로 저장되었습니다.")


if __name__ == "__main__":
    workflow = build_workflow()
    sample_questions = [
        # "삼성전자와 애플의 최근 실적을 비교해줘",
        # "경제와 관련 없는 질문입니다",
        # "레버리지 ETF의 위험성을 설명해줘" 
        # "삼성전자와 애플의 최근 주가를 비교 후, 간단하게 차트를 그려줘",
        # "나스닥이 뭐야?",
        "모바일로 주식 거래하는 앱은 뭐라고 하나요?"
    ]

    for question in sample_questions:
        print("=" * 80)
        print(f"Q: {question}")
        result = workflow.run(question)
        print(f"route: {result.get('route')}")
        answer = result.get("answer")
        if isinstance(answer, str) and len(answer) > 400:
            answer = answer[:400] + "..."
        print("answer:", answer)
        if result.get("rag_search_results"):
            print("rag_search_results:")
            for line in result["rag_search_results"]:
                print(f"  {line}")
        if result.get("quality_detail"):
            print(f"quality_check: {result['quality_detail']}")
