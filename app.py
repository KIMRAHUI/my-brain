import os
import shutil
import uuid
import streamlit as st
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 0. API 키 설정
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# 1. LLM 및 임베딩 설정
Settings.llm = Gemini(model="models/gemini-3.6-flash")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# 마크다운 헤더 파서
Settings.node_parser = MarkdownNodeParser()

PERSIST_DIR = "./storage"
DATA_DIR = "./notion"

# 2. 인덱스 빌드 및 로딩 캐싱
@st.cache_resource
def load_or_build_index():
    if os.path.exists(PERSIST_DIR):
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
    else:
        documents = SimpleDirectoryReader(
            DATA_DIR,
            required_exts=[".md"],
            recursive=True,
        ).load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    return index

with st.spinner("지식 베이스 엔진 로딩 중... (잠시만 기다려주세요!)"):
    index = load_or_build_index()

query_engine = index.as_query_engine(similarity_top_k=3)

# 3. Streamlit UI 설정
st.set_page_config(page_title="나만의 학습 비서", page_icon="")
st.title("나만의 학습 비서")

# [CSS 스타일 주입] '새 대화 시작하기' 버튼 노란색 및 버튼들 수평 정렬/줄바꿈 방지 최적화
st.markdown("""
    <style>
    /* 새 대화 버튼 노란색 스타일 */
    div.stButton > button[kind="primary"] {
        background-color: #FFD400 !important;
        color: #000000 !important;
        border: none !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #E6BF00 !important;
        color: #000000 !important;
    }
    /* 채팅 말풍선 내부 버튼들이 줄바꿈되지 않고 나란히 배치되도록 설정 */
    [data-testid="stHorizontalBlock"] {
        align-items: center;
    }
    </style>
""", unsafe_allow_html=True)

# 멀티 챗 세션 구조 초기화
if "chat_sessions" not in st.session_state:
    first_chat_id = str(uuid.uuid4())
    st.session_state.chat_sessions = {
        first_chat_id: {"title": "새 대화", "messages": []}
    }
    st.session_state.current_chat_id = first_chat_id

if "current_chat_id" not in st.session_state or st.session_state.current_chat_id not in st.session_state.chat_sessions:
    st.session_state.current_chat_id = list(st.session_state.chat_sessions.keys())[0]

# 사이드바 구성
with st.sidebar:
    st.markdown("### 지식 RAG 비서")
    
    if st.button("새 대화 시작하기", type="primary"):
        new_chat_id = str(uuid.uuid4())
        st.session_state.chat_sessions[new_chat_id] = {"title": "새 대화", "messages": []}
        st.session_state.current_chat_id = new_chat_id
        st.rerun()

    st.markdown("---")
    st.subheader("대화 기록 목록")
    
    chat_ids = list(st.session_state.chat_sessions.keys())
    for c_id in chat_ids:
        chat_title = st.session_state.chat_sessions[c_id]["title"]
        is_active = (c_id == st.session_state.current_chat_id)
        btn_label = f"[{chat_title}]" if is_active else f"{chat_title}"
        
        col_s1, col_s2 = st.columns([0.8, 0.2])
        with col_s1:
            if st.button(btn_label, key=f"select_{c_id}"):
                st.session_state.current_chat_id = c_id
                st.rerun()
        with col_s2:
            if len(st.session_state.chat_sessions) > 1:
                if st.button("X", key=f"del_chat_{c_id}", help="이 대화방 삭제"):
                    del st.session_state.chat_sessions[c_id]
                    st.session_state.current_chat_id = list(st.session_state.chat_sessions.keys())[0]
                    st.rerun()

    st.markdown("---")
    st.subheader("시스템 관리")
    if st.button("문서 인덱스 새로고침"):
        if os.path.exists(PERSIST_DIR):
            shutil.rmtree(PERSIST_DIR)
        st.cache_resource.clear()
        st.rerun()

current_messages = st.session_state.chat_sessions[st.session_state.current_chat_id]["messages"]

# 대화 기록 출력 (고정/삭제 버튼 줄바꿈 없이 나란히 배치)
for idx, message in enumerate(current_messages):
    with st.chat_message(message["role"]):
        # 내용 영역(0.7), 고정 버튼(0.15), 삭제 버튼(0.15) 비율로 한 줄 정렬
        col1, col2, col3 = st.columns([0.7, 0.15, 0.15])
        
        with col1:
            st.markdown(message["content"])
            
        with col2:
            is_pinned = message.get("pinned", False)
            pin_label = "해제" if is_pinned else "고정"
            if st.button(pin_label, key=f"pin_{st.session_state.current_chat_id}_{idx}", help="이 대화 고정/해제"):
                current_messages[idx]["pinned"] = not is_pinned
                st.rerun()
                
        with col3:
            if st.button("삭제", key=f"del_{st.session_state.current_chat_id}_{idx}", help="이 대화만 삭제"):
                del current_messages[idx]
                st.rerun()

# 사용자 입력
if user_query := st.chat_input("공부한 내용에 대해 무엇이든 물어보세요!"):
    if len(current_messages) == 0:
        short_title = user_query[:15] + "..." if len(user_query) > 15 else user_query
        st.session_state.chat_sessions[st.session_state.current_chat_id]["title"] = short_title

    current_messages.append({"role": "user", "content": user_query, "pinned": False})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("답변 생성 중..."):
            response = str(query_engine.query(user_query))
        st.markdown(response)
        
    current_messages.append({"role": "assistant", "content": response, "pinned": False})
    st.rerun()
