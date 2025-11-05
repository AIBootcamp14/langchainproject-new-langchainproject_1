# src/web/stream_multiturn_v.py

import streamlit as st
import uuid
import re
from pathlib import Path
from src.workflow.workflow import build_workflow
from src.database.chat_history import ChatHistoryDB
from src.utils.config import Config
from src.utils.logger import get_logger
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

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

# ===== 1-1. 대화 요약 함수 =====
def summarize_conversation(messages_to_summarize: list) -> str:
    """
    중간 메시지들을 LLM으로 요약하여 컨텍스트 효율화

    Args:
        messages_to_summarize: 요약할 메시지 리스트 (dict 형태)

    Returns:
        요약된 텍스트 (200-300 토큰)
    """
    if not messages_to_summarize or len(messages_to_summarize) == 0:
        return ""

    try:
        from src.model.llm import get_llm_manager

        # LLM Manager에서 경량 모델 가져오기
        llm_manager = get_llm_manager()
        summarizer = llm_manager.get_model("solar-mini", temperature=0)

        # 요약할 대화 텍스트 구성
        conversation_text = ""
        for idx, msg in enumerate(messages_to_summarize, 1):
            role = "사용자" if msg["role"] == "user" else "AI"
            content = msg["content"]
            # 너무 긴 내용은 잘라내기 (각 메시지당 최대 500자)
            if len(content) > 500:
                content = content[:500] + "..."
            conversation_text += f"{idx}. {role}: {content}\n\n"

        # 요약 프롬프트
        summary_prompt = f"""다음은 사용자와 AI 금융 상담 에이전트의 이전 대화입니다.
이 대화의 핵심 내용을 간결하게 요약해주세요.

요약 시 포함할 내용:
- 사용자가 질문한 주요 주제 (주식, 기업명 등)
- AI가 제공한 핵심 분석 내용
- 중요한 수치나 결론

요약 형식:
"이전 대화에서 사용자는 [주제]에 대해 질문했고, AI는 [핵심 내용]을 분석했습니다."

대화:
{conversation_text}

요약 (200자 이내):"""

        # LLM 호출
        summary = summarizer.invoke(summary_prompt).content

        logger.info(f"📝 {len(messages_to_summarize)}개 메시지 요약 완료 (길이: {len(summary)}자)")
        return summary.strip()

    except Exception as e:
        logger.error(f"❌ 대화 요약 실패: {e}")
        # 요약 실패 시 간단한 폴백 메시지
        return f"[이전 대화 {len(messages_to_summarize)}개 메시지 생략됨]"

# ===== 2. Session ID 자동 생성 =====
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.loaded = False
    st.session_state.user_input = ""
    st.session_state.conversation_summary = ""
    st.session_state.last_summarized_count = 0

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
            st.session_state.conversation_summary = ""
            st.session_state.last_summarized_count = 0
            st.rerun()

    st.divider()

    # 대화 히스토리 목록
    st.subheader("📚 최근 대화")
    st.caption("💡 각 대화는 독립적입니다 (최대 20개 메시지)")

    # 요약 상태 표시
    msg_count = len(st.session_state.messages)
    if msg_count > 20:
        middle_count = msg_count - 20
        st.caption(f"📝 요약 활성: 중간 {middle_count}개 메시지 압축됨")
    elif msg_count > 15:
        st.caption(f"⏳ 곧 요약 시작 ({msg_count}/20)")

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
                        st.session_state.conversation_summary = ""
                        st.session_state.last_summarized_count = 0
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
                        st.session_state.conversation_summary = ""
                        st.session_state.last_summarized_count = 0

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
            "txt_path": txt_path
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
        "pdf_path": None
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
            # 슬라이딩 윈도우 + 요약: 컨텍스트 효율화
            # 초기 페어 유지 + 중간 메시지 요약 + 최근 메시지 유지
            MAX_CONTEXT_MESSAGES = 20  # 최대 20개 메시지
            SUMMARIZE_INTERVAL = 5  # 5개마다 재요약
            all_messages = st.session_state.messages[:-1]  # 마지막(현재 입력) 제외

            previous_messages = []  # LangChain 메시지 리스트

            if len(all_messages) > MAX_CONTEXT_MESSAGES:
                # 메시지가 많을 경우: 초기 + 요약 + 최근
                initial_pair = all_messages[:2]  # 첫 2개
                middle_messages = all_messages[2:-18]  # 중간 메시지들
                recent_messages = all_messages[-18:]  # 최근 18개

                # 중간 메시지 요약
                should_summarize = (
                    len(all_messages) - st.session_state.last_summarized_count >= SUMMARIZE_INTERVAL
                ) or (st.session_state.conversation_summary == "")

                if should_summarize and len(middle_messages) > 0:
                    logger.info(f"📝 중간 메시지 {len(middle_messages)}개 요약 시작...")
                    st.session_state.conversation_summary = summarize_conversation(middle_messages)
                    st.session_state.last_summarized_count = len(all_messages)
                    logger.info(f"✅ 요약 완료: {st.session_state.conversation_summary[:100]}...")

                # 컨텍스트 구성: 초기 2개
                for msg in initial_pair:
                    if msg["role"] == "user":
                        previous_messages.append(HumanMessage(content=msg["content"]))
                    else:
                        previous_messages.append(AIMessage(content=msg["content"]))

                # 요약 삽입 (SystemMessage)
                if st.session_state.conversation_summary:
                    previous_messages.append(
                        SystemMessage(content=f"[이전 대화 요약]\n{st.session_state.conversation_summary}")
                    )

                # 최근 18개
                for msg in recent_messages:
                    if msg["role"] == "user":
                        previous_messages.append(HumanMessage(content=msg["content"]))
                    else:
                        previous_messages.append(AIMessage(content=msg["content"]))

            else:
                # 메시지가 적을 경우: 전체 사용
                for msg in all_messages:
                    if msg["role"] == "user":
                        previous_messages.append(HumanMessage(content=msg["content"]))
                    else:
                        previous_messages.append(AIMessage(content=msg["content"]))

            # 가장 최근 assistant 메시지에서 analysis_data 추출 (후속 질문용)
            prev_analysis_data = None
            history = db.get_history(st.session_state.session_id, limit=20)
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

        # 현재 응답의 차트만 표시
        if result.get("current_charts"):
            base_path = Path(__file__).parent.parent.parent  # ai_agent_project 루트

            image_paths = []
            for chart_path in result["current_charts"]:
                # 상대경로를 절대경로로 변환
                abs_path = str(base_path / chart_path)
                image_paths.append(abs_path)
                logger.info(f"📊 현재 응답 차트 절대경로 변환: {chart_path} → {abs_path}")

        # 현재 응답의 파일만 표시
        saved_file_path = result.get("current_saved_file")

        if saved_file_path:
            base_path = Path(__file__).parent.parent.parent
            abs_saved_path = str(base_path / saved_file_path)

            ext = Path(abs_saved_path).suffix.lower()
            if ext == '.pdf':
                pdf_path = abs_saved_path
            elif ext == '.md':
                md_path = abs_saved_path
            elif ext == '.txt':
                txt_path = abs_saved_path

        # 보고서에서 "Charts:" 경로 텍스트 제거
        # "Charts:\n- charts/xxx.png\n- charts/yyy.png" 패턴 제거
        answer = re.sub(r'Charts?:\s*\n(?:[-•]\s*charts/[^\n]+\n?)+', '', answer, flags=re.IGNORECASE)
        # 단독 경로 라인도 제거 (예: "- charts/xxx.png")
        answer = re.sub(r'^\s*[-•]\s*charts/[^\n]+\s*$', '', answer, flags=re.MULTILINE)

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

    # 페이지 리렌더링 (모든 메시지를 126번째 줄 루프에서 표시)
    st.rerun()
