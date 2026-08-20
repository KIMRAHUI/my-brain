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

# 0. API 키 설정
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# 1. LLM 및 임베딩 설정 (가장 안정적인 공식 모델 사용)
Settings.llm = Gemini(model="gemini-1.5-flash")
Settings.embed_model = GeminiEmbedding(model_name="models/text-embedding-004")

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
st.title("나만의 학습 비서")

# [CSS 스타일 주입] 보더선 위아래 여백 최소화 및 버튼 정렬 스타일
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
    /* 컨테이너(카드) 상하 마진 및 패딩 최소화 */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        margin-top: -0.5rem !important;
        margin-bottom: -0.5rem !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] div.element-container {
        margin-bottom: -0.6rem !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] hr {
        margin-top: 0.2rem !important;
        margin-bottom: 0.2rem !important;
    }
    /* 고정/삭제 버튼 그룹 영역 (우측 정렬 및 줄바꿈 절대 방지) */
    [data-testid="stHorizontalBlock"] {
        gap: 0.3rem !important;
        align-items: center !important;
    }
    .stButton > button {
        white-space: nowrap !important;
        padding-left: 0.4rem !important;
        padding-right: 0.4rem !important;
        min-height: 1.8rem !important;
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
    st.subheader("고정된 핵심 노트")
    
    # 모든 채팅방에서 고정(pinned=True)된 카드들을 수집하여 사이드바에 요약 표시
    all_pinned_items = []
    for s_id, s_data in st.session_state.chat_sessions.items():
        for q_item in s_data["qa_pairs"]:
            if q_item.get("pinned", False):
                all_pinned_items.append(q_item)

    if not all_pinned_items:
        st.caption("고정된 대화가 없습니다. 카드에서 고정을 눌러보세요.")
    else:
        for p_idx, p_qa in enumerate(all_pinned_items):
            with st.container(border=True):
                st.caption(f"Q. {p_qa['question']}")
                st.write(p_qa['answer'][:40] + "..." if len(p_qa['answer']) > 40 else p_qa['answer'])

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
# [Q&A 카드 단위 렌더링 및 고정/삭제 버튼 그룹화 처리]
# -------------------------------------------------------------
for idx, qa in enumerate(current_qa_pairs):
    with st.container(border=True):
        st.markdown(f"**Q. {qa['question']}**")
        st.markdown("---")
        st.markdown(qa['answer'])
        
        # 하단 컨트롤 영역: 빈 공간(왼쪽)을 두고, 고정/삭제 버튼을 우측에 나란히 배치
        b_col_space, b_col_pin, b_col_del = st.columns([0.5, 0.25, 0.25])
        with b_col_pin:
            is_pinned = qa.get("pinned", False)
            pin_label = "해제" if is_pinned else "고정"
            if st.button(pin_label, key=f"pin_{st.session_state.current_chat_id}_{idx}", use_container_width=True):
                current_qa_pairs[idx]["pinned"] = not is_pinned
                st.rerun()
        with b_col_del:
            if st.button("삭제", key=f"del_{st.session_state.current_chat_id}_{idx}", use_container_width=True):
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
