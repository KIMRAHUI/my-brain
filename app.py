import os
import shutil
import uuid

import streamlit as st
from supabase import create_client

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
    PromptTemplate,
)
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding


# -------------------------------------------------------------
# 0. API KEY / Supabase 클라이언트
# -------------------------------------------------------------

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

supabase = create_client(
    st.secrets["SUPABASE_URL"],
    st.secrets["SUPABASE_KEY"],
)


# -------------------------------------------------------------
# 1. LLM / Embedding 설정
# -------------------------------------------------------------

Settings.llm = GoogleGenAI(
    model="gemini-3.6-flash",
)

Settings.embed_model = GoogleGenAIEmbedding(
    model_name="gemini-embedding-2",
)

Settings.node_parser = MarkdownNodeParser()


PERSIST_DIR = "./storage"
DATA_DIR = "./notion"


# -------------------------------------------------------------
# 2. 인덱스 빌드 / 로딩
# -------------------------------------------------------------

@st.cache_resource
def load_or_build_index():
    if os.path.exists(PERSIST_DIR):
        storage_context = StorageContext.from_defaults(
            persist_dir=PERSIST_DIR
        )

        index = load_index_from_storage(
            storage_context
        )

    else:
        documents = SimpleDirectoryReader(
            DATA_DIR,
            required_exts=[".md"],
            recursive=True,
        ).load_data()

        index = VectorStoreIndex.from_documents(
            documents
        )

        index.storage_context.persist(
            persist_dir=PERSIST_DIR
        )

    return index


with st.spinner("지식 베이스 엔진 로딩 중..."):
    index = load_or_build_index()


qa_prompt = PromptTemplate(
    """
    아래는 사용자의 학습 노트에서 검색된 참고 자료입니다.

    ---------------------
    {context_str}
    ---------------------

    사용자 질문:
    {query_str}

    답변 규칙:
    1. 참고 자료에 관련 내용이 있으면 그것을 우선 활용하여 답변하세요.
    2. 참고 자료에 답이 없거나 부족하더라도 질문에 답할 수 있다면 일반적인 지식과 추론을 활용해서 완성도 높은 답변을 생성하세요.
    3. 참고 자료에 없는 내용을 참고 자료에 있다고 억지로 주장하지 마세요.
    4. 사용자가 예제 코드나 구현을 요청하면 실제로 실행 가능한 예제를 명확하게 작성해 주세요.
    5. 필요한 경우 참고 자료의 내용과 일반 지식을 자연스럽게 조합해서 설명해 주세요.
    """
)

query_engine = index.as_query_engine(
    similarity_top_k=3,
    text_qa_template=qa_prompt,
)


# -------------------------------------------------------------
# 3. Supabase 데이터 접근 함수
# -------------------------------------------------------------

def db_load_all_sessions():
    """모든 대화방 + 질문답변을 Supabase에서 불러와 session_state 형태로 변환."""
    sessions_res = (
        supabase.table("chat_sessions")
        .select("*")
        .order("created_at")
        .execute()
    )

    sessions = {}
    for row in sessions_res.data:
        sessions[row["id"]] = {"title": row["title"], "qa_pairs": []}

    qa_res = (
        supabase.table("qa_pairs")
        .select("*")
        .order("created_at")
        .execute()
    )

    for row in qa_res.data:
        chat_id = row["chat_session_id"]
        if chat_id in sessions:
            sessions[chat_id]["qa_pairs"].append(
                {
                    "db_id": row["id"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "pinned": row["pinned"],
                }
            )

    return sessions


def db_create_session(chat_id: str, title: str = "새 대화"):
    supabase.table("chat_sessions").insert(
        {"id": chat_id, "title": title}
    ).execute()


def db_update_session_title(chat_id: str, title: str):
    supabase.table("chat_sessions").update(
        {"title": title}
    ).eq("id", chat_id).execute()


def db_delete_session(chat_id: str):
    # qa_pairs는 FK ON DELETE CASCADE로 함께 삭제됨
    supabase.table("chat_sessions").delete().eq("id", chat_id).execute()


def db_insert_qa(chat_id: str, question: str, answer: str) -> str:
    res = (
        supabase.table("qa_pairs")
        .insert(
            {
                "chat_session_id": chat_id,
                "question": question,
                "answer": answer,
                "pinned": False,
            }
        )
        .execute()
    )
    return res.data[0]["id"]


def db_update_pin(qa_db_id: str, pinned: bool):
    supabase.table("qa_pairs").update({"pinned": pinned}).eq(
        "id", qa_db_id
    ).execute()


def db_delete_qa(qa_db_id: str):
    supabase.table("qa_pairs").delete().eq("id", qa_db_id).execute()


# -------------------------------------------------------------
# 4. Streamlit 설정
# -------------------------------------------------------------

st.set_page_config(
    page_title="나만의 학습 비서",
    page_icon="",
    layout="centered",
)


# -------------------------------------------------------------
# 5. CSS
# -------------------------------------------------------------

st.markdown(
    """
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

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    div[data-testid="stVerticalBlock"]
    div[data-testid="stVerticalBlockBorderWrapper"] {
        padding: none;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        margin-top: -0.5rem !important;
        margin-bottom: -0.5rem !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]
    div.element-container {
        margin-bottom: -0.6rem !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] hr {
        margin-top: 0.2rem !important;
        margin-bottom: 0.2rem !important;
    }

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
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------
# 6. 세션 초기화 (Supabase에서 로드, 없으면 새로 생성)
# -------------------------------------------------------------

if "chat_sessions" not in st.session_state:
    with st.spinner("이전 대화 기록 불러오는 중..."):
        loaded_sessions = db_load_all_sessions()

    if loaded_sessions:
        st.session_state.chat_sessions = loaded_sessions
        st.session_state.current_chat_id = list(loaded_sessions.keys())[0]
    else:
        first_chat_id = str(uuid.uuid4())
        db_create_session(first_chat_id, "새 대화")

        st.session_state.chat_sessions = {
            first_chat_id: {"title": "새 대화", "qa_pairs": []}
        }
        st.session_state.current_chat_id = first_chat_id


if (
    "current_chat_id" not in st.session_state
    or st.session_state.current_chat_id
    not in st.session_state.chat_sessions
):
    st.session_state.current_chat_id = (
        list(st.session_state.chat_sessions.keys())[0]
    )


# -------------------------------------------------------------
# 7. 사이드바
# -------------------------------------------------------------

with st.sidebar:
    st.markdown("### 지식 RAG 비서")

    if st.button("새 대화 시작하기", type="primary"):
        new_chat_id = str(uuid.uuid4())
        db_create_session(new_chat_id, "새 대화")

        st.session_state.chat_sessions[new_chat_id] = {
            "title": "새 대화",
            "qa_pairs": [],
        }

        st.session_state.current_chat_id = new_chat_id
        st.rerun()

    st.markdown("---")
    st.subheader("대화 기록 목록")

    chat_ids = list(st.session_state.chat_sessions.keys())

    for c_id in chat_ids:
        chat_title = st.session_state.chat_sessions[c_id]["title"]
        is_active = c_id == st.session_state.current_chat_id
        btn_label = f"[{chat_title}]" if is_active else chat_title

        col_s1, col_s2 = st.columns([0.8, 0.2])

        with col_s1:
            if st.button(btn_label, key=f"select_{c_id}"):
                st.session_state.current_chat_id = c_id
                st.rerun()

        with col_s2:
            if len(st.session_state.chat_sessions) > 1:
                if st.button(
                    "X", key=f"del_chat_{c_id}", help="이 대화방 삭제"
                ):
                    db_delete_session(c_id)  # Supabase에서도 삭제 (연쇄 삭제)
                    del st.session_state.chat_sessions[c_id]

                    st.session_state.current_chat_id = (
                        list(st.session_state.chat_sessions.keys())[0]
                    )
                    st.rerun()

    st.markdown("---")
    st.subheader("고정된 핵심 노트")

    all_pinned_items = []
    for session_data in st.session_state.chat_sessions.values():
        for q_item in session_data["qa_pairs"]:
            if q_item.get("pinned", False):
                all_pinned_items.append(q_item)

    if not all_pinned_items:
        st.caption("고정된 대화가 없습니다.")
    else:
        for p_qa in all_pinned_items:
            with st.container(border=True):
                st.caption(f"Q. {p_qa['question']}")
                st.write(
                    p_qa["answer"][:40] + "..."
                    if len(p_qa["answer"]) > 40
                    else p_qa["answer"]
                )

    st.markdown("---")
    st.subheader("시스템 관리")

    if st.button("문서 인덱스 새로고침"):
        if os.path.exists(PERSIST_DIR):
            shutil.rmtree(PERSIST_DIR)
        st.cache_resource.clear()
        st.rerun()


# -------------------------------------------------------------
# 8. 현재 채팅
# -------------------------------------------------------------

current_qa_pairs = st.session_state.chat_sessions[
    st.session_state.current_chat_id
]["qa_pairs"]


# -------------------------------------------------------------
# 9. Q&A 카드
# -------------------------------------------------------------

for idx, qa in enumerate(current_qa_pairs):
    with st.container(border=True):
        st.markdown(f"**Q. {qa['question']}**")
        st.markdown("---")
        st.markdown(qa["answer"])

        b_col_space, b_col_pin, b_col_del = st.columns([0.5, 0.25, 0.25])

        with b_col_pin:
            is_pinned = qa.get("pinned", False)
            pin_label = "해제" if is_pinned else "고정"

            if st.button(
                pin_label,
                key=f"pin_{st.session_state.current_chat_id}_{idx}",
                use_container_width=True,
            ):
                new_pinned = not is_pinned
                current_qa_pairs[idx]["pinned"] = new_pinned
                db_update_pin(qa["db_id"], new_pinned)  # Supabase 반영
                st.rerun()

        with b_col_del:
            if st.button(
                "삭제",
                key=f"del_{st.session_state.current_chat_id}_{idx}",
                use_container_width=True,
            ):
                db_delete_qa(qa["db_id"])  # Supabase에서 삭제
                del current_qa_pairs[idx]
                st.rerun()


# -------------------------------------------------------------
# 10. 사용자 질문
# -------------------------------------------------------------

if user_query := st.chat_input("공부한 내용에 대해 무엇이든 물어보세요!"):

    if len(current_qa_pairs) == 0:
        short_title = (
            user_query[:15] + "..." if len(user_query) > 15 else user_query
        )
        st.session_state.chat_sessions[
            st.session_state.current_chat_id
        ]["title"] = short_title
        db_update_session_title(
            st.session_state.current_chat_id, short_title
        )  # Supabase 반영

    with st.spinner("답변 생성 중..."):
        try:
            response = str(query_engine.query(user_query))
        except Exception as e:
            error_message = str(e)
            if "RESOURCE_EXHAUSTED" in error_message:
                response = (
                    "Gemini API 사용량 제한에 도달했습니다.\n\n"
                    "잠시 후 다시 시도하거나 "
                    "Gemini API 사용량/결제 설정을 확인해주세요."
                )
            else:
                response = (
                    f"답변 생성 중 오류가 발생했습니다.\n\n`{error_message}`"
                )

    # Supabase에 저장하고, 반환된 id를 함께 session_state에 보관
    new_qa_id = db_insert_qa(
        st.session_state.current_chat_id, user_query, response
    )

    current_qa_pairs.append(
        {
            "db_id": new_qa_id,
            "question": user_query,
            "answer": response,
            "pinned": False,
        }
    )

    st.rerun()
