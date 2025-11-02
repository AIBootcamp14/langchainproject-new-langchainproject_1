import streamlit as st
from dotenv import load_dotenv
from langchain_upstage import ChatUpstage

@st.cache_resource
def load_model():
    load_dotenv()
    print("model loaded ...")
    llm = ChatUpstage(model="solar-pro2")
    print("model load complete")

    return llm
llm = load_model()
st.title("financial AI")


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# 과거의 입력이 보여지는곳
for content in st.session_state.chat_history:
    with st.chat_message(content['role']):
        st.markdown(content['message']) # 실질적인 출력이 일어나는 곳 

# 현시점의 입력이 보여지는 곳 
if prompt := st.chat_input("메세지를 입력하세요."): # 할당과 동시에 출력  
    # 사용자가 입력할 때 실행되는 구간
    with st.chat_message("user"): # 
        st.markdown(prompt)
        st.session_state.chat_history.append({'role': "user", "message" : prompt})

    with st.chat_message("ai"):
        response = llm.invoke(prompt)
        st.markdown(response.content)
        st.session_state.chat_history.append({"role":'ai',"message":response.content})