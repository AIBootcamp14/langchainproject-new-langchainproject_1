# src/streamlit_app.py
import streamlit as st
import uuid
import re
from pathlib import Path
from src.workflow.workflow import build_workflow
from src.database.chat_history import ChatHistoryDB
from src.utils.config import Config
from src.utils.logger import get_logger
from src.utils.workflow_helpers import (
    convert_messages_to_langchain,
    extract_previous_analysis_data,
    process_chart_paths,
    process_file_paths,
    build_response_metadata,
    get_project_root
)
from langchain_core.messages import HumanMessage, AIMessage

# 로거 초기화
logger = get_logger(__name__)

# ===== 1. 초기화 =====
@st.cache_resource
def init_resources():
    """DB와 Workflow 초기화 (캐싱)"""
    db = ChatHistoryDB()
    db.setup_database()
    workflow = build_workflow()
    return db, workflow

# ===== 2. Session ID 자동 생성 =====
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.loaded = False
    st.session_state.user_input = ""

# ===== 3. 사이드바: 대화 관리 =====
with st.sidebar:
    st.title("💬 대화 히스토리")

    # 캐시 클리어 버튼
    col1, col2, col3 = st.columns([0.01, 0.9, 0.01])
    with col2:
        if st.button("🔄 캐시 클리어 & 재시작"):
            st.cache_resource.clear()
            st.rerun()

db, workflow = init_resources()

# 사이드바 계속
with st.sidebar:

    # 새 대화 시작 버튼
    col1, col2, col3 = st.columns([0.01, 0.35, 0.2])
    with col2:
        if st.button("🆕 새 대화", use_container_width=True):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.session_state.loaded = False
            st.session_state.user_input = ""
            st.rerun()

    st.divider()

    # 대화 히스토리 목록
    st.subheader("📚 최근 대화")
    st.caption("💡 각 대화는 독립적입니다 (최대 총 20개 메시지)")

    # 요약 상태 표시
    msg_count = len(st.session_state.messages)
    if msg_count > 15:
        st.caption(f"⏳ 곧 최대 메시지 개수 도달 ({msg_count}/20)")

    # 모든 세션 목록 가져오기
    all_sessions = db.get_all_sessions(limit=20)

    if all_sessions:
        for session_info in all_sessions:
            session_id = session_info["session_id"]
            preview = session_info["preview"]
            message_count = session_info["message_count"]

            # 현재 활성 세션 표시
            is_current = (session_id == st.session_state.session_id)
            button_label = f"{'▶ ' if is_current else '  '}{preview}"

            # 세션 버튼과 삭제 버튼을 나란히 배치
            col1, col2 = st.columns([5, 2])

            with col1:
                # 세션 버튼 (클릭 시 해당 세션으로 전환)
                if st.button(
                    button_label,
                    key=f"session_{session_id}",
                    use_container_width=True,
                    type="primary" if is_current else "secondary"
                ):
                    if session_id != st.session_state.session_id:
                        # 다른 세션으로 전환
                        st.session_state.session_id = session_id
                        st.session_state.messages = []
                        st.session_state.loaded = False
                        st.session_state.user_input = ""
                        st.rerun()

            with col2:
                # 삭제 버튼
                if st.button("🗑️", key=f"delete_{session_id}", help="대화 삭제"):
                    # 세션 삭제
                    db.clear_session(session_id)

                    # 현재 활성 세션을 삭제한 경우 새 세션 생성
                    if session_id == st.session_state.session_id:
                        st.session_state.session_id = str(uuid.uuid4())
                        st.session_state.messages = []
                        st.session_state.loaded = False
                        st.session_state.user_input = ""

                    st.rerun()

            # 메시지 개수 표시
            st.caption(f"💬 {message_count}개 메시지")
    else:
        st.caption("아직 대화가 없습니다.")

# ===== 4. DB에서 이전 대화 로드 (최초 1회만) =====
if not st.session_state.loaded:
    # 현재 세션의 최근 20개 메시지만 로드
    history = db.get_history(st.session_state.session_id, limit=20)

    # 역순 정렬 (오래된 것부터)
    for msg in reversed(history):
        # 경로를 절대경로로 변환 (상대경로로 저장된 경우 대비)
        base_path = Path(__file__).parent.parent.parent

        images = msg.get("metadata", {}).get("image_paths", []) if msg.get("metadata") else []
        images_abs = []
        for img in images:
            if img and not Path(img).is_absolute():
                # 상대경로인 경우 절대경로로 변환
                images_abs.append(str(base_path / img))
            else:
                images_abs.append(img)

        # 파일 경로들도 절대경로로 변환
        def to_abs_path(path):
            if path and not Path(path).is_absolute():
                return str(base_path / path)
            return path

        pdf_path = to_abs_path(msg.get("metadata", {}).get("pdf_path") if msg.get("metadata") else None)
        md_path = to_abs_path(msg.get("metadata", {}).get("md_path") if msg.get("metadata") else None)
        txt_path = to_abs_path(msg.get("metadata", {}).get("txt_path") if msg.get("metadata") else None)

        st.session_state.messages.append({
            "role": msg["role"],
            "content": msg["content"],
            "images": images_abs,
            "pdf_path": pdf_path,
            "md_path": md_path,
            "txt_path": txt_path,
            "metadata": msg.get("metadata", {})  # 전체 metadata 포함 (analysis_data 포함)
        })

    st.session_state.loaded = True

# ===== 5. 메인: 대화 표시 =====
st.title("💰 Financial AI Agent")

for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # 이미지 표시 + 다운로드
        if msg.get("images"):
            for img_idx, img_path in enumerate(msg["images"]):
                if Path(img_path).exists():
                    st.image(img_path, width=800)

                    with open(img_path, "rb") as file:
                        st.download_button(
                            label=f"📥 차트 {img_idx+1} 다운로드",
                            data=file,
                            file_name=Path(img_path).name,
                            mime="image/png",
                            key=f"dl_hist_{idx}_{img_idx}"
                        )

        # PDF 다운로드 버튼
        if msg.get("pdf_path") and Path(msg["pdf_path"]).exists():
            with open(msg["pdf_path"], "rb") as pdf_file:
                st.download_button(
                    label="📄 PDF 보고서 다운로드",
                    data=pdf_file,
                    file_name=Path(msg["pdf_path"]).name,
                    mime="application/pdf",
                    key=f"dl_pdf_hist_{idx}"
                )

        # MD 다운로드 버튼
        if msg.get("md_path") and Path(msg["md_path"]).exists():
            with open(msg["md_path"], "r", encoding="utf-8") as md_file:
                st.download_button(
                    label="📝 Markdown 파일 다운로드",
                    data=md_file.read(),
                    file_name=Path(msg["md_path"]).name,
                    mime="text/markdown",
                    key=f"dl_md_hist_{idx}"
                )

        # TXT 다운로드 버튼
        if msg.get("txt_path") and Path(msg["txt_path"]).exists():
            with open(msg["txt_path"], "r", encoding="utf-8") as txt_file:
                st.download_button(
                    label="📄 텍스트 파일 다운로드",
                    data=txt_file.read(),
                    file_name=Path(msg["txt_path"]).name,
                    mime="text/plain",
                    key=f"dl_txt_hist_{idx}"
                )

# ===== 6. 사용자 입력 처리 =====
# 턴 수 체크
current_turn_count = db.get_turn_count(st.session_state.session_id)
is_session_limit_reached = current_turn_count >= Config.MAX_TURNS_PER_SESSION

# 세션 제한 도달 시 경고 메시지 표시
if is_session_limit_reached:
    st.warning(Config.SESSION_LIMIT_RESPONSE)

# st.chat_input 사용 (Enter로 전송) - 세션 제한 도달 시 비활성화
if prompt := st.chat_input(
    "질문을 입력하세요...",
    disabled=is_session_limit_reached,
    key="user_input_box"
):
    prompt = prompt.strip()

    # DB에 저장
    db.add_message(
        session_id=st.session_state.session_id,
        role="user",
        content=prompt
    )

    # 세션 스테이트에 추가
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "images": [],
        "pdf_path": None,
        "metadata": {}
    })

    # 유저 메시지 즉시 표시
    with st.chat_message("user"):
        st.markdown(prompt)

    # 변수 초기화 (try-except 블록 밖에서도 사용 가능하도록)
    answer = ""
    quality_passed = False
    image_paths = []
    pdf_path = None
    md_path = None
    txt_path = None
    result = {}

    try:
        with st.spinner("분석 중..."):
            # 컨텍스트 윈도우: Config에서 설정 가져오기 (0 = 무제한)
            MAX_CONTEXT_MESSAGES = Config.MAX_CONTEXT_MESSAGES
            all_messages = st.session_state.messages[:-1]  # 마지막(현재 입력) 제외

            # 컨텍스트 메시지 제한 (0이면 무제한)
            if MAX_CONTEXT_MESSAGES > 0 and len(all_messages) > MAX_CONTEXT_MESSAGES:
                previous_messages = convert_messages_to_langchain(all_messages[-MAX_CONTEXT_MESSAGES:])
                logger.info(f"📊 컨텍스트: 최근 {MAX_CONTEXT_MESSAGES}개 메시지 사용 (전체 {len(all_messages)}개 중)")
            else:
                previous_messages = convert_messages_to_langchain(all_messages)
                logger.info(f"📊 컨텍스트: 전체 {len(all_messages)}개 메시지 사용")

            # 가장 최근 assistant 메시지에서 analysis_data 추출 (헬퍼 함수 사용)
            prev_analysis_data = extract_previous_analysis_data(st.session_state.messages)

            # 멀티턴 대화 실행
            result = workflow.run(
                question=prompt,
                session_id=st.session_state.session_id,
                previous_messages=previous_messages,
                previous_analysis_data=prev_analysis_data  # 이전 분석 데이터 전달
            )

        answer = result.get("answer", "")
        quality_passed = result.get("quality_passed", False)

        # 프로젝트 루트 경로 계산 (헬퍼 함수 사용)
        # src/web/streamlit_app.py → ai_agent_project (2단계 상위)
        base_path = get_project_root(__file__, levels_up=2)

        # 차트 경로 처리 (헬퍼 함수 사용)
        image_paths = process_chart_paths(result, base_path)

        # 파일 경로 처리 (헬퍼 함수 사용)
        file_paths = process_file_paths(result, base_path)
        pdf_path = file_paths.get("pdf_path")
        md_path = file_paths.get("md_path")
        txt_path = file_paths.get("txt_path")

        # 보고서에서 파일 경로 텍스트 제거 (차트 다운로드 버튼만 표시)
        # "Charts:\n- charts/xxx.png\n- charts/yyy.png" 패턴 제거
        answer = re.sub(r'Charts?:\s*\n(?:[-•]\s*charts/[^\n]+\n?)+', '', answer, flags=re.IGNORECASE)
        # 단독 차트 경로 라인도 제거 (예: "- charts/xxx.png")
        answer = re.sub(r'^\s*[-•]\s*charts/[^\n]+\s*$', '', answer, flags=re.MULTILINE)

        # 보고서 저장 경로 텍스트 제거 (다운로드 버튼만 표시)
        # "Saved to: reports/xxx.pdf" 패턴 제거
        answer = re.sub(r'Saved\s+to:\s*reports/[^\n]+', '', answer, flags=re.IGNORECASE)
        answer = re.sub(r'저장\s*(위치|경로|됨)?:?\s*reports/[^\n]+', '', answer, flags=re.IGNORECASE)
        # 단독 보고서 경로 라인도 제거 (예: "- reports/xxx.pdf")
        answer = re.sub(r'^\s*[-•]\s*reports/[^\n]+\s*$', '', answer, flags=re.MULTILINE)

    except Exception as e:
        # 에러 발생 시 사용자에게 친절한 메시지 표시
        error_msg = f"""
### ⚠️ 분석 중 오류가 발생했습니다

죄송합니다. 요청을 처리하는 중 문제가 발생했습니다.

**가능한 해결 방법:**
- 질문을 다르게 표현해보세요
- 더 구체적인 정보를 포함해주세요 (예: 회사명, 날짜 등)
- 잠시 후 다시 시도해주세요

**기술적 오류 정보:**
```
{str(e)}
```
"""
        # 에러 로깅
        logger.error(f"Streamlit workflow 실행 오류: {e}", exc_info=True)

        # 에러 발생 시 변수 설정
        answer = error_msg
        quality_passed = False
        image_paths = []
        pdf_path = None
        md_path = None
        txt_path = None
        result = {
            "answer": error_msg,
            "quality_passed": False,
            "quality_detail": {},
            "analysis_data": {}
        }

    # 메타데이터 구성 (헬퍼 함수 사용)
    metadata = build_response_metadata(result, image_paths, file_paths)

    # DB에 저장 (analysis_data 전체 포함)
    db.add_message(
        session_id=st.session_state.session_id,
        role="assistant",
        content=answer,
        agent_name="report_generator",
        status="success" if quality_passed else "failed",
        quality_score=result.get("quality_detail", {}).get("score"),
        metadata=metadata
    )

    # 세션 스테이트에 추가 (metadata 포함)
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "images": image_paths,
        "pdf_path": pdf_path,
        "md_path": md_path,
        "txt_path": txt_path,
        "metadata": metadata
    })

    # 답변 즉시 표시 (st.rerun() 전에)
    with st.chat_message("assistant"):
        st.markdown(answer)

        # 차트 즉시 표시 + 다운로드 버튼
        if image_paths:
            for img_idx, img_path in enumerate(image_paths):
                if Path(img_path).exists():
                    st.image(img_path, width=800)

                    with open(img_path, "rb") as file:
                        st.download_button(
                            label=f"📥 차트 {img_idx+1} 다운로드",
                            data=file,
                            file_name=Path(img_path).name,
                            mime="image/png",
                            key=f"dl_new_{img_idx}"
                        )
                else:
                    st.warning(f"⚠️ 차트 파일을 찾을 수 없습니다: {img_path}")

        # PDF 다운로드 버튼
        if pdf_path and Path(pdf_path).exists():
            with open(pdf_path, "rb") as pdf_file:
                st.download_button(
                    label="📄 PDF 보고서 다운로드",
                    data=pdf_file,
                    file_name=Path(pdf_path).name,
                    mime="application/pdf",
                    key="dl_pdf_new"
                )

        # MD 다운로드 버튼
        if md_path and Path(md_path).exists():
            with open(md_path, "r", encoding="utf-8") as md_file:
                st.download_button(
                    label="📝 Markdown 파일 다운로드",
                    data=md_file.read(),
                    file_name=Path(md_path).name,
                    mime="text/markdown",
                    key="dl_md_new"
                )

        # TXT 다운로드 버튼
        if txt_path and Path(txt_path).exists():
            with open(txt_path, "r", encoding="utf-8") as txt_file:
                st.download_button(
                    label="📄 텍스트 파일 다운로드",
                    data=txt_file.read(),
                    file_name=Path(txt_path).name,
                    mime="text/plain",
                    key="dl_txt_new"
                )

    # 페이지 리렌더링
    st.rerun()
