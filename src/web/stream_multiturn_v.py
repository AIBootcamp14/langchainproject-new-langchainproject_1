import streamlit as st
from dotenv import load_dotenv
from langchain_upstage import ChatUpstage
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


@st.cache_resource
def load_model():
    load_dotenv()
    print("model loaded ...")
    llm = ChatUpstage(model="solar-pro2", streaming = True)
    print("model load complete")
    return llm

llm = load_model()
st.title("financial AI")

# 대화 상태
if "chat_history" not in st.session_state:
    # [{"role": "user"/"ai", "message": "..."}] 형태로 유지
    st.session_state.chat_history = []

# 과거 대화 렌더링
for content in st.session_state.chat_history:
    with st.chat_message(content["role"]):
        st.markdown(content["message"])

def build_messages_with_history(user_prompt: str):
    """streamlit의 chat_history를 LangChain 메시지 객체 배열로 변환"""
    msgs = []
    # (선택) 히스토리 길이 제한
    history = st.session_state.chat_history[-12:]  # 최근 12 turn만 사용
    for item in history:
        if item["role"] == "user":
            msgs.append(HumanMessage(content=item["message"]))
        else:
            msgs.append(AIMessage(content=item["message"]))
    # 이번 사용자 입력 추가
    msgs.append(HumanMessage(content=user_prompt))
    return msgs

# 사용자 입력
if prompt := st.chat_input("메세지를 입력하세요."):
    # 1) UI + 히스토리 반영
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "message": prompt})

    # 2) 멀티턴: 과거 대화 + 이번 입력을 함께 LLM에 전달
    messages = build_messages_with_history(prompt)
    # response = llm.invoke(messages)  # ← 핵심: 메시지 배열을 전달

    # 3) 출력 + 히스토리 저장
    with st.chat_message("ai"):
        message_placeholder = st.empty()
        full_response = ""
        with st.spinner("메세지 입력 중입니다."):
            # response = st.session_state.chat_session.send_message(prompt, stream = True)
            for chunk in llm.stream(messages):
                full_response += chunk.content
                message_placeholder.markdown(full_response)
    st.session_state.chat_history.append({"role": "ai", "message": full_response})
