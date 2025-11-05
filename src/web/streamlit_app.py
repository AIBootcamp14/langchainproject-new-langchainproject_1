# src/web/stream_multiturn_v.py

import streamlit as st
import uuid
from pathlib import Path
from src.workflow.workflow import build_workflow
from src.database.chat_history import ChatHistoryDB
from src.utils.markdown_cleaner import remove_markdown
from langchain_core.messages import HumanMessage, AIMessage

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

    # 캐시 클리어 버튼 (디버깅용)
    if st.button("🔄 캐시 클리어 & 재시작"):
        st.cache_resource.clear()
        st.rerun()

db, workflow = init_resources()

# 사이드바 계속
with st.sidebar:

    # 새 대화 시작 버튼
    if st.button("🆕 새 대화", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.loaded = False
        st.session_state.user_input = ""
        st.rerun()

    st.divider()

    # 대화 히스토리 목록
    st.subheader("📚 최근 대화")

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
    history = db.get_history(st.session_state.session_id, limit=20)

    # 역순 정렬 (오래된 것부터)
    for msg in reversed(history):
        st.session_state.messages.append({
            "role": msg["role"],
            "content": msg["content"],
            "images": msg.get("metadata", {}).get("image_paths", []) if msg.get("metadata") else [],
            "pdf_path": msg.get("metadata", {}).get("pdf_path") if msg.get("metadata") else None,
            "md_path": msg.get("metadata", {}).get("md_path") if msg.get("metadata") else None,
            "txt_path": msg.get("metadata", {}).get("txt_path") if msg.get("metadata") else None
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
# st.chat_input 사용 (Enter로 전송)
if prompt := st.chat_input("질문을 입력하세요..."):
    prompt = prompt.strip()

    # 사용자 메시지 표시
    with st.chat_message("user"):
        st.markdown(prompt)

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
        "pdf_path": None
    })

    # AI 응답 생성
    with st.chat_message("assistant"):
        # 변수 초기화 (try-except 블록 밖에서도 사용 가능하도록)
        answer = ""
        quality_passed = False
        image_paths = []
        pdf_path = None
        md_path = None
        txt_path = None
        result = {}  # result 변수도 초기화

        try:
            with st.spinner("분석 중..."):
                # session_state.messages를 활용하여 이전 대화 구성 (방금 추가한 현재 입력 제외)
                # 현재 입력은 이미 session_state.messages에 추가되었으므로, 마지막 것만 제외
                previous_messages = []
                for msg in st.session_state.messages[:-1]:  # 마지막(현재 입력) 제외
                    if msg["role"] == "user":
                        previous_messages.append(HumanMessage(content=msg["content"]))
                    else:
                        # 에이전트 메시지는 마크다운 제거
                        clean_content = remove_markdown(msg["content"])
                        previous_messages.append(AIMessage(content=clean_content))

                # 가장 최근 assistant 메시지에서 analysis_data 추출 (후속 질문용)
                # DB에서 직접 조회 (최신 1개만)
                prev_analysis_data = None
                history = db.get_history(st.session_state.session_id, limit=5)  # 최근 5개만
                for msg in history:  # 최신순이므로 첫 assistant 메시지가 가장 최근
                    if msg["role"] == "assistant" and msg.get("metadata", {}).get("analysis_data"):
                        prev_analysis_data = msg["metadata"]["analysis_data"]
                        break  # 가장 최근 것 사용

                # 멀티턴 대화 실행
                result = workflow.run(
                    question=prompt,
                    session_id=st.session_state.session_id,
                    previous_messages=previous_messages,
                    previous_analysis_data=prev_analysis_data  # 이전 분석 데이터 전달
                )

                answer = result.get("answer", "")
                quality_passed = result.get("quality_passed", False)

                st.markdown(answer)

                # 이미지 경로 추출
                if result.get("analysis_data", {}).get("chart_paths"):
                    image_paths = result["analysis_data"]["chart_paths"]

                # 채팅 메시지 안에서 이미지 + 다운로드 버튼 표시
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

                # 저장된 파일 경로 추출
                saved_file_path = result.get("analysis_data", {}).get("saved_file_path")

                if saved_file_path:
                    ext = Path(saved_file_path).suffix.lower()
                    if ext == '.pdf':
                        pdf_path = saved_file_path
                    elif ext == '.md':
                        md_path = saved_file_path
                    elif ext == '.txt':
                        txt_path = saved_file_path

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
            st.error(error_msg)

            # 에러 로깅
            import logging
            logger = logging.getLogger(__name__)
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

    # PDF 다운로드 버튼
    if pdf_path and Path(pdf_path).exists():
        st.write("---")
        st.subheader("📄 PDF 보고서")

        abs_path = Path(pdf_path).resolve()
        st.info(f"💾 저장 위치: `{abs_path}`")

        with open(pdf_path, "rb") as pdf_file:
            st.download_button(
                label="📥 PDF 보고서 다운로드",
                data=pdf_file,
                file_name=Path(pdf_path).name,
                mime="application/pdf",
                key="dl_pdf_new"
            )
    elif pdf_path:
        st.warning(f"⚠️ PDF 파일을 찾을 수 없습니다: {pdf_path}")

    # MD 파일 다운로드 버튼
    if md_path and Path(md_path).exists():
        st.write("---")
        st.subheader("📝 Markdown 파일")

        abs_path = Path(md_path).resolve()
        st.info(f"💾 저장 위치: `{abs_path}`")

        with open(md_path, "r", encoding="utf-8") as md_file:
            st.download_button(
                label="📥 Markdown 파일 다운로드",
                data=md_file.read(),
                file_name=Path(md_path).name,
                mime="text/markdown",
                key="dl_md_new"
            )
    elif md_path:
        st.warning(f"⚠️ MD 파일을 찾을 수 없습니다: {md_path}")

    # TXT 파일 다운로드 버튼
    if txt_path and Path(txt_path).exists():
        st.write("---")
        st.subheader("📄 텍스트 파일")

        abs_path = Path(txt_path).resolve()
        st.info(f"💾 저장 위치: `{abs_path}`")

        with open(txt_path, "r", encoding="utf-8") as txt_file:
            st.download_button(
                label="📥 텍스트 파일 다운로드",
                data=txt_file.read(),
                file_name=Path(txt_path).name,
                mime="text/plain",
                key="dl_txt_new"
            )
    elif txt_path:
        st.warning(f"⚠️ TXT 파일을 찾을 수 없습니다: {txt_path}")

    # DB에 저장 (analysis_data 전체 포함)
    db.add_message(
        session_id=st.session_state.session_id,
        role="assistant",
        content=answer,
        agent_name="report_generator",
        status="success" if quality_passed else "failed",
        quality_score=result.get("quality_detail", {}).get("score"),
        metadata={
            "image_paths": image_paths,
            "pdf_path": pdf_path,
            "md_path": md_path,
            "txt_path": txt_path,
            "analysis_data": result.get("analysis_data")  # 전체 analysis_data 저장
        }
    )

    # 세션 스테이트에 추가
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "images": image_paths,
        "pdf_path": pdf_path,
        "md_path": md_path,
        "txt_path": txt_path
    })
