import os
import shutil
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

# 세션 상태 초기화 (대화 기록 기억용)
if "messages" not in st.session_state:
    st.session_state.messages = []

# 사이드바 구성
with st.sidebar:
    st.markdown("### ℹ️ 안내")
    st.write("노션에 정리한 학습 노트를 기반으로 질문에 답해드려요.")
    
    # 인덱스 새로고침 버튼
    if st.button("🔄 인덱스 새로고침"):
        if os.path.exists(PERSIST_DIR):
            shutil.rmtree(PERSIST_DIR)
        st.cache_resource.clear()
        st.rerun()
        
    st.markdown("---")
    st.subheader("⚙️ 대화 관리")
    if st.button("🗑️ 전체 대화 일괄 삭제"):
        st.session_state.messages = []
        st.rerun()
        
    st.markdown("---")
    st.subheader("📌 고정된 핵심 노트")
    # 고정된(pinned=True) 메시지만 필터링해서 보여주기
    pinned_messages = [m for m in st.session_state.messages if m.get("pinned", False)]
    
    if not pinned_messages:
        st.info("고정된 대화가 없습니다. 대화 옆의 📌 버튼을 눌러보세요!")
    else:
        for p_idx, p_msg in enumerate(pinned_messages):
            with st.container(border=True):
                role_icon = "👤" if p_msg["role"] == "user" else "🤖"
                st.caption(f"{role_icon} {p_msg['role'].upper()}")
                st.write(p_msg["content"][:50] + "..." if len(p_msg["content"]) > 50 else p_msg["content"])

# 5. 기존 대화 기록 화면 출력 (개별 삭제 및 고정 버튼 포함)
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        # 레이아웃 나누기: 내용 영역(0.8), 고정 버튼(0.1), 삭제 버튼(0.1)
        col1, col2, col3 = st.columns([0.8, 0.1, 0.1])
        
        with col1:
            st.markdown(message["content"])
            
        with col2:
            # 고정 상태에 따른 아이콘 변경 (고정됨: 📌, 해제: 📍)
            is_pinned = message.get("pinned", False)
            pin_icon = "📌" if is_pinned else "📍"
            if st.button(pin_icon, key=f"pin_{idx}", help="이 대화 고정/해제"):
                st.session_state.messages[idx]["pinned"] = not is_pinned
                st.rerun()
                
        with col3:
            # 개별 삭제 버튼
            if st.button("🗑️", key=f"del_{idx}", help="이 대화만 삭제"):
                del st.session_state.messages[idx]
                st.rerun()

# 6. 사용자의 입력 받기 (하단 고정 채팅창)
if user_query := st.chat_input("공부한 내용에 대해 무엇이든 물어보세요!"):
    
    # 1. 사용자 질문 저장 및 표시
    st.session_state.messages.append({"role": "user", "content": user_query, "pinned": False})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. AI 답변 생성 및 저장, 표시
    with st.chat_message("assistant"):
        with st.spinner("답변 생성 중..."):
            response = str(query_engine.query(user_query))
        st.markdown(response)
        
    st.session_state.messages.append({"role": "assistant", "content": response, "pinned": False})
    st.rerun()
