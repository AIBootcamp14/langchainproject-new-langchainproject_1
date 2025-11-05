"""
Workflow Helper Functions

Streamlit과 CLI에서 공통으로 사용하는 workflow 관련 유틸리티 함수들입니다.
중복 코드를 제거하고 유지보수를 용이하게 합니다.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from langchain_core.messages import HumanMessage, AIMessage

from src.utils.logger import get_logger

logger = get_logger(__name__)


def convert_messages_to_langchain(
    messages: List[Dict[str, Any]]
) -> List[Union[HumanMessage, AIMessage]]:
    """
    dict 형태의 메시지를 LangChain 메시지 객체로 변환합니다.

    Args:
        messages: 변환할 메시지 리스트 (role, content 필드 포함)

    Returns:
        LangChain 메시지 객체 리스트 (HumanMessage, AIMessage)

    Example:
        >>> messages = [
        ...     {"role": "user", "content": "안녕하세요"},
        ...     {"role": "assistant", "content": "안녕하세요!"}
        ... ]
        >>> langchain_messages = convert_messages_to_langchain(messages)
    """
    langchain_messages = []

    for msg in messages:
        if msg["role"] == "user":
            langchain_messages.append(HumanMessage(content=msg["content"]))
        else:  # assistant
            langchain_messages.append(AIMessage(content=msg["content"]))

    return langchain_messages


def extract_previous_analysis_data(
    messages: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """
    메시지 히스토리에서 가장 최근 assistant 메시지의 analysis_data를 추출합니다.

    후속 질문 처리를 위해 이전 분석 데이터를 재사용할 때 사용합니다.

    Args:
        messages: 메시지 히스토리 리스트 (metadata.analysis_data 포함)

    Returns:
        가장 최근 analysis_data 또는 None

    Example:
        >>> messages = [
        ...     {"role": "user", "content": "삼성전자 분석해줘"},
        ...     {"role": "assistant", "content": "...", "metadata": {"analysis_data": {...}}}
        ... ]
        >>> analysis_data = extract_previous_analysis_data(messages)
    """
    # 역순으로 탐색하여 가장 최근 assistant 메시지 찾기
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            metadata = msg.get("metadata", {})
            if metadata.get("analysis_data"):
                logger.debug(f"✅ 이전 analysis_data 추출: type={metadata['analysis_data'].get('analysis_type')}")
                return metadata["analysis_data"]

    logger.debug("ℹ️ 이전 analysis_data 없음")
    return None


def process_chart_paths(
    result: Dict[str, Any],
    base_path: Path
) -> List[str]:
    """
    workflow 실행 결과에서 차트 경로를 추출하고 절대경로로 변환합니다.

    Args:
        result: workflow.run() 실행 결과 (current_charts 필드 포함)
        base_path: 프로젝트 루트 경로 (절대경로 변환 기준)

    Returns:
        절대경로로 변환된 차트 경로 리스트

    Example:
        >>> result = {"current_charts": ["charts/stock_chart.png"]}
        >>> base_path = Path("/home/user/project")
        >>> paths = process_chart_paths(result, base_path)
        >>> # ["/home/user/project/charts/stock_chart.png"]
    """
    image_paths = []

    if result.get("current_charts"):
        for chart_path in result["current_charts"]:
            # 상대경로를 절대경로로 변환
            abs_path = str(base_path / chart_path)
            image_paths.append(abs_path)
            logger.debug(f"📊 차트 경로 변환: {chart_path} → {abs_path}")

    return image_paths


def process_file_paths(
    result: Dict[str, Any],
    base_path: Path
) -> Dict[str, Optional[str]]:
    """
    workflow 실행 결과에서 저장된 파일 경로를 추출하고 확장자별로 분류합니다.

    Args:
        result: workflow.run() 실행 결과 (current_saved_file 필드 포함)
        base_path: 프로젝트 루트 경로 (절대경로 변환 기준)

    Returns:
        확장자별 파일 경로 딕셔너리
        {"pdf_path": str|None, "md_path": str|None, "txt_path": str|None}

    Example:
        >>> result = {"current_saved_file": "reports/analysis.pdf"}
        >>> base_path = Path("/home/user/project")
        >>> paths = process_file_paths(result, base_path)
        >>> # {"pdf_path": "/home/user/project/reports/analysis.pdf", ...}
    """
    file_paths = {
        "pdf_path": None,
        "md_path": None,
        "txt_path": None
    }

    saved_file_path = result.get("current_saved_file")

    if saved_file_path:
        # 상대경로를 절대경로로 변환
        abs_saved_path = str(base_path / saved_file_path)

        # 확장자별 분류
        ext = Path(abs_saved_path).suffix.lower()
        if ext == '.pdf':
            file_paths["pdf_path"] = abs_saved_path
            logger.debug(f"💾 PDF 경로: {abs_saved_path}")
        elif ext == '.md':
            file_paths["md_path"] = abs_saved_path
            logger.debug(f"💾 Markdown 경로: {abs_saved_path}")
        elif ext == '.txt':
            file_paths["txt_path"] = abs_saved_path
            logger.debug(f"💾 Text 경로: {abs_saved_path}")

    return file_paths


def build_response_metadata(
    result: Dict[str, Any],
    image_paths: List[str],
    file_paths: Dict[str, Optional[str]]
) -> Dict[str, Any]:
    """
    AI 응답 저장을 위한 메타데이터를 구성합니다.

    Args:
        result: workflow.run() 실행 결과 (analysis_data 포함)
        image_paths: 차트 경로 리스트 (절대경로)
        file_paths: 파일 경로 딕셔너리 (pdf_path, md_path, txt_path)

    Returns:
        DB 저장용 메타데이터 딕셔너리

    Example:
        >>> result = {"analysis_data": {...}}
        >>> image_paths = ["/path/to/chart.png"]
        >>> file_paths = {"pdf_path": "/path/to/report.pdf", ...}
        >>> metadata = build_response_metadata(result, image_paths, file_paths)
    """
    metadata = {
        "image_paths": image_paths,
        "pdf_path": file_paths.get("pdf_path"),
        "md_path": file_paths.get("md_path"),
        "txt_path": file_paths.get("txt_path"),
        "analysis_data": result.get("analysis_data")
    }

    logger.debug(f"📦 메타데이터 구성 완료 - 차트: {len(image_paths)}개, "
                f"파일: {bool(file_paths.get('pdf_path') or file_paths.get('md_path') or file_paths.get('txt_path'))}")

    return metadata


def get_project_root(current_file: str, levels_up: int = 1) -> Path:
    """
    현재 파일로부터 프로젝트 루트 경로를 계산합니다.

    Args:
        current_file: __file__ 값
        levels_up: 루트까지 올라갈 상위 디렉토리 개수
            - main.py (src/main.py): 1 (src → project_root)
            - streamlit_app.py (src/web/streamlit_app.py): 2 (src/web → src → project_root)

    Returns:
        프로젝트 루트 경로 (절대경로)

    Example:
        >>> # src/main.py에서 호출
        >>> root = get_project_root(__file__, levels_up=1)
        >>> # /home/user/ai_agent_project

        >>> # src/web/streamlit_app.py에서 호출
        >>> root = get_project_root(__file__, levels_up=2)
        >>> # /home/user/ai_agent_project
    """
    current_path = Path(current_file).parent

    for _ in range(levels_up):
        current_path = current_path.parent

    logger.debug(f"📁 프로젝트 루트: {current_path}")
    return current_path


__all__ = [
    "convert_messages_to_langchain",
    "extract_previous_analysis_data",
    "process_chart_paths",
    "process_file_paths",
    "build_response_metadata",
    "get_project_root"
]
