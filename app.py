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

with st.spinner("지식 베이스 엔진 로딩 중..."):
    index = load_or_build_index()

query_engine = index.as_query_engine(similarity_top_k=3)

# 3. Streamlit UI 설정 (모바일 최적화 레이아웃 적용)
st.set_page_config(page_title="나만의 학습 비서", page_icon="", layout="centered")

# [CSS 스타일 주입] 보더선, 여백, 버튼 줄바꿈 방지 완벽 커스텀
st.markdown("""
    <style>
    div.stButton > button[kind="primary"] {
        background-color: #242424 !important;
        color: #eeeeee !important;
        border: none !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #E6BF00 !important;
        color: #000000 !important;
    }
    /* 모바일 환경에서 패딩 최적화 */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    /* 컨테이너(카드) 내부 패딩 및 보더선 잘림 방지 */
    div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlockBorderWrapper"] {
        padding-top: 0.8rem !important;
        padding-bottom: 0.6rem !important;
        padding-left: 0.8rem !important;
        padding-right: 0.8rem !important;
        border-radius: 8px !important;
    }
    /* Q. 개체 밑과 내부 요소들의 불필요한 여백 대폭 축소 (반절 이상 절감) */
    div[data-testid="stVerticalBlockBorderWrapper"] div.element-container {
        margin-bottom: -0.8rem !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] hr {
        margin-top: 0.4rem !important;
        margin-bottom: 0.4rem !important;
    }
    /* 고정/삭제 버튼 영역 간격을 극도로 좁히고 줄바꿈 절대 방지 */
    [data-testid="stHorizontalBlock"] {
        gap: 0.2rem !important;
        align-items: center !important;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="column"] {
        width: auto !important;
        flex: 1 !important;
    }
    /* 버튼 내부 텍스트 줄바꿈 방지 및 패딩 조정 */
    .stButton > button {
        white-space: nowrap !important;
        padding-left: 0.4rem !important;
        padding-right: 0.4rem !important;
        width: 100% !important;
    }
    </style>
""", unsafe_allow_html=True)

# 멀티 챗 세션 구조 초기화
if "chat_sessions" not in st.session_state:
    first_chat_id = str(uuid.uuid4())
    st.session_state.chat_sessions = {
        first_chat_id: {"title": "새 대화", "qa_pairs": []}
    }
    st.session_state.current_chat_id = first_chat_id

if "current_chat_id" not in st.session_state or st.session_state.current_chat_id not in st.session_state.chat_sessions:
    st.session_state.current_chat_id = list(st.session_state.chat_sessions.keys())[0]

# 사이드바 구성
with st.sidebar:
    st.markdown("### 지식 RAG 비서")
    
    if st.button("새 대화 시작하기", type="primary"):
        new_chat_id = str(uuid.uuid4())
        st.session_state.chat_sessions[new_chat_id] = {"title": "새 대화", "qa_pairs": []}
        st.session_state.current_chat_id = new_chat_id
        st.rerun()

    st.markdown("---")
    st.subheader("대화 기록 목록")
    
    chat_ids = list(st.session_state.chat_sessions.keys())
    for c_id in chat_ids:
        chat_title = st.session_state.chat_sessions[c_id]["title"]
        
        col_s1, col_s2 = st.columns([0.8, 0.2])
        with col_s1:
            if st.button(chat_title, key=f"select_{c_id}"):
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

# 현재 세션의 Q&A 카드 목록 가져오기
current_qa_pairs = st.session_state.chat_sessions[st.session_state.current_chat_id]["qa_pairs"]

# -------------------------------------------------------------
# [Q&A 카드 단위 렌더링 및 일괄 고정/삭제 처리]
# -------------------------------------------------------------
for idx, qa in enumerate(current_qa_pairs):
    with st.container(border=True):
        st.markdown(f"**Q. {qa['question']}**")
        st.markdown("---")
        st.markdown(qa['answer'])
        
        # 하단 컨트롤 영역 (고정 및 삭제 버튼 나란히 밀착 정렬)
        b_col1, b_col2, b_col3 = st.columns([0.6, 0.2, 0.2])
        with b_col2:
            is_pinned = qa.get("pinned", False)
            pin_label = "해제" if is_pinned else "고정"
            if st.button(pin_label, key=f"pin_{st.session_state.current_chat_id}_{idx}"):
                current_qa_pairs[idx]["pinned"] = not is_pinned
                st.rerun()
        with b_col3:
            if st.button("삭제", key=f"del_{st.session_state.current_chat_id}_{idx}"):
                del current_qa_pairs[idx]
                st.rerun()

# 사용자 입력 (하단 고정 채팅창)
if user_query := st.chat_input("공부한 내용에 대해 무엇이든 물어보세요!"):
    if len(current_qa_pairs) == 0:
        short_title = user_query[:15] + "..." if len(user_query) > 15 else user_query
        st.session_state.chat_sessions[st.session_state.current_chat_id]["title"] = short_title

    with st.spinner("답변 생성 중..."):
        response = str(query_engine.query(user_query))

    current_qa_pairs.append({
        "question": user_query,
        "answer": response,
        "pinned": False
    })
    st.rerun()
