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
from llama_index.embeddings.gemini import GeminiEmbedding

# 0. API 키 설정 (Streamlit Cloud의 Secrets에 등록해서 사용)
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# 1. Gemini를 LLM/임베딩 모델로 명시 지정
Settings.llm = Gemini(model="models/gemini-3.6-flash")
Settings.embed_model = GeminiEmbedding(model_name="models/gemini-embedding-001")

# 2. 마크다운 헤더 구조 기반 파서 사용
Settings.node_parser = MarkdownNodeParser()

PERSIST_DIR = "./storage"
DATA_DIR = "./notion"  # Notion-GitHub-Sync 액션의 subdir 저장 경로에 맞춤


# 3. 인덱스 캐싱: 최초 1회만 생성, 이후 재사용
@st.cache_resource
def load_or_build_index():
    if os.path.exists(PERSIST_DIR):
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
    else:
        documents = SimpleDirectoryReader(
            DATA_DIR,
            required_exts=[".md"],  # png 등 이미지 파일은 인덱싱에서 제외
            recursive=True,
        ).load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    return index


index = load_or_build_index()
query_engine = index.as_query_engine()

# 4. Streamlit UI 설정
st.set_page_config(page_title="나만의 학습 비서", page_icon="📚")
st.title("📚 나만의 학습 비서")

# -------------------------------------------------------------
# [멀티 챗 세션 관리 구조 초기화]
# -------------------------------------------------------------
if "chat_sessions" not in st.session_state:
    # 최초 접속 시 기본 채팅방 하나 생성
    first_chat_id = str(uuid.uuid4())
    st.session_state.chat_sessions = {
        first_chat_id: {"title": "새 대화", "messages": []}
    }
    st.session_state.current_chat_id = first_chat_id

# 현재 선택된 채팅방 ID 안전 장치
if "current_chat_id" not in st.session_state or st.session_state.current_chat_id not in st.session_state.chat_sessions:
    st.session_state.current_chat_id = list(st.session_state.chat_sessions.keys())[0]

# 5. 사이드바 구성 (채팅방 목록 및 관리)
with st.sidebar:
    st.markdown("### 🤖 지식 RAG 비서")
    
    # ➕ 새 대화 시작하기 버튼
    if st.button("➕ 새 대화 시작하기", type="primary"):
        new_chat_id = str(uuid.uuid4())
        st.session_state.chat_sessions[new_chat_id] = {"title": "새 대화", "messages": []}
        st.session_state.current_chat_id = new_chat_id
        st.rerun()

    st.markdown("---")
    st.subheader("💬 대화 기록 목록")
    
    # 사이드바에 기존 대화 방 목록들을 버튼 형태로 출력
    chat_ids = list(st.session_state.chat_sessions.keys())
    for c_id in chat_ids:
        chat_title = st.session_state.chat_sessions[c_id]["title"]
        # 현재 활성화된 방과 그렇지 않은 방의 버튼 스타일 구분
        is_active = (c_id == st.session_state.current_chat_id)
        btn_label = f"📍 {chat_title}" if is_active else f"💬 {chat_title}"
        
        col_s1, col_s2 = st.columns([0.8, 0.2])
        with col_s1:
            if st.button(btn_label, key=f"select_{c_id}"):
                st.session_state.current_chat_id = c_id
                st.rerun()
        with col_s2:
            # 방 삭제 버튼 (최소 1개는 남아있도록 방어 코드 추가)
            if len(st.session_state.chat_sessions) > 1:
                if st.button("🗑️", key=f"del_chat_{c_id}", help="이 대화방 삭제"):
                    del st.session_state.chat_sessions[c_id]
                    st.session_state.current_chat_id = list(st.session_state.chat_sessions.keys())[0]
                    st.rerun()

    st.markdown("---")
    st.subheader("⚙️ 시스템 관리")
    if st.button("🔄 문서 인덱스 새로고침"):
        if os.path.exists(PERSIST_DIR):
            shutil.rmtree(PERSIST_DIR)
        st.cache_resource.clear()
        st.rerun()

# 6. 현재 선택된 채팅방의 메시지 불러오기
current_messages = st.session_state.chat_sessions[st.session_state.current_chat_id]["messages"]

# 7. 현재 채팅방의 대화 기록 화면 출력 (개별 삭제 및 고정 버튼 포함)
for idx, message in enumerate(current_messages):
    with st.chat_message(message["role"]):
        col1, col2, col3 = st.columns([0.8, 0.1, 0.1])
        
        with col1:
            st.markdown(message["content"])
            
        with col2:
            is_pinned = message.get("pinned", False)
            pin_icon = "📌" if is_pinned else "📍"
            if st.button(pin_icon, key=f"pin_{st.session_state.current_chat_id}_{idx}", help="이 대화 고정/해제"):
                current_messages[idx]["pinned"] = not is_pinned
                st.rerun()
                
        with col3:
            if st.button("🗑️", key=f"del_{st.session_state.current_chat_id}_{idx}", help="이 대화만 삭제"):
                del current_messages[idx]
                st.rerun()

# 8. 사용자의 입력 받기 (하단 고정 채팅창)
if user_query := st.chat_input("공부한 내용에 대해 무엇이든 물어보세요!"):
    
    # 첫 질문인 경우, 첫 번째 질문 내용을 해당 채팅방의 제목(Title)으로 자동 설정
    if len(current_messages) == 0:
        # 제목이 너무 길면 15글자로 자르기
        short_title = user_query[:15] + "..." if len(user_query) > 15 else user_query
        st.session_state.chat_sessions[st.session_state.current_chat_id]["title"] = short_title

    # 1. 사용자 질문 저장 및 표시
    current_messages.append({"role": "user", "content": user_query, "pinned": False})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. AI 답변 생성 및 저장, 표시
    with st.chat_message("assistant"):
        with st.spinner("답변 생성 중..."):
            response = str(query_engine.query(user_query))
        st.markdown(response)
        
    current_messages.append({"role": "assistant", "content": response, "pinned": False})
    st.rerun()
