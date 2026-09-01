사이드바 버튼 영역에서 긴 문장이 튀어나오는 문제를 해결하기 위해, 썸네일 텍스트를 깔끔한 해시태그 형식(`#`)으로 요약하고 글자 수를 제한하여 버튼 폭 안에 예쁘게 들어가도록 수정했습니다.

전체 코드는 다음과 같습니다.

```python
import os
import shutil
import time
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
    embed_batch_size=5,
)

Settings.node_parser = MarkdownNodeParser()


PERSIST_DIR = "./storage"
DATA_DIR = "./notion"
INDEX_BUCKET = "rag-index"


# -------------------------------------------------------------
# 2. 인덱스 Supabase Storage 백업/복원 함수
# -------------------------------------------------------------

def download_index_from_supabase() -> bool:
    """Supabase Storage에 저장된 인덱스 파일을 로컬 storage/로 내려받는다.
    성공적으로 하나 이상 받았으면 True, 저장된 게 없으면 False."""
    try:
        files = supabase.storage.from_(INDEX_BUCKET).list()
    except Exception:
        return False

    if not files:
        return False

    os.makedirs(PERSIST_DIR, exist_ok=True)
    got_any = False
    for f in files:
        name = f["name"]
        try:
            data = supabase.storage.from_(INDEX_BUCKET).download(name)
        except Exception:
            continue
        with open(os.path.join(PERSIST_DIR, name), "wb") as out:
            out.write(data)
        got_any = True

    return got_any


def upload_index_to_supabase():
    """로컬 storage/ 안의 인덱스 파일들을 전부 Supabase Storage에 업로드(덮어쓰기)."""
    if not os.path.exists(PERSIST_DIR):
        return
    for filename in os.listdir(PERSIST_DIR):
        local_path = os.path.join(PERSIST_DIR, filename)
        if not os.path.isfile(local_path):
            continue
        with open(local_path, "rb") as f:
            supabase.storage.from_(INDEX_BUCKET).upload(
                filename,
                f,
                {"content-type": "application/json", "upsert": "true"},
            )


# -------------------------------------------------------------
# 3. 인덱스 빌드 / 로딩 함수 정의
# -------------------------------------------------------------

@st.cache_resource
def load_or_build_index():
    if not os.path.exists(PERSIST_DIR):
        download_index_from_supabase()

    documents = SimpleDirectoryReader(
        DATA_DIR,
        required_exts=[".md"],
        recursive=True,
    ).load_data()

    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        try:
            storage_context = StorageContext.from_defaults(
                persist_dir=PERSIST_DIR
            )
            index = load_index_from_storage(storage_context)

            # 증분 업데이트: 이미 인덱싱된 문서는 건드리지 않고
            # 새로 추가되었거나 내용이 바뀐 문서만 골라서 임베딩
            refreshed_flags = index.refresh_ref_docs(documents)
            if any(refreshed_flags):
                index.storage_context.persist(persist_dir=PERSIST_DIR)
                upload_index_to_supabase()

            return index
        except Exception:
            # 손상되었거나 불완전하면 새로 빌드
            shutil.rmtree(PERSIST_DIR, ignore_errors=True)

    # 최초 빌드: 문서를 하나씩 순차적으로 삽입하면서 매번 로컬+Supabase에 저장.
    index = VectorStoreIndex.from_documents([])

    total = len(documents)
    progress_bar = st.progress(
        0.0, text=f"0 / {total} 문서 임베딩 중..."
    )

    for i, doc in enumerate(documents):
        for attempt in range(5):
            try:
                index.insert(doc)
                break
            except Exception as e:
                is_quota_error = "RESOURCE_EXHAUSTED" in str(e)
                if is_quota_error and attempt < 4:
                    wait_seconds = 30 * (attempt + 1)
                    progress_bar.progress(
                        i / total if total else 0.0,
                        text=(
                            f"할당량 제한으로 {wait_seconds}초 대기 후 재시도 "
                            f"({i} / {total} 완료)..."
                        ),
                    )
                    time.sleep(wait_seconds)
                    continue
                index.storage_context.persist(persist_dir=PERSIST_DIR)
                upload_index_to_supabase()
                raise

        index.storage_context.persist(persist_dir=PERSIST_DIR)
        upload_index_to_supabase()

        progress_bar.progress(
            (i + 1) / total if total else 1.0,
            text=f"{i + 1} / {total} 문서 임베딩 완료",
        )

    progress_bar.empty()
    return index


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


# -------------------------------------------------------------
# 4. Supabase 데이터 접근 함수 (대화방/질문답변)
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
# 5. Streamlit 설정
# -------------------------------------------------------------

st.set_page_config(
    page_title="나만의 학습 비서",
    page_icon="",
    layout="centered",
)


# -------------------------------------------------------------
# 6. CSS
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
# 7. 세션 초기화 (Supabase에서 로드, 없으면 새로 생성)
# -------------------------------------------------------------

if "chat_sessions" not in st.session_state:
    with st.spinner("이전 대화 기록 불러오는 중..."):
        loaded_sessions = db_load_all_sessions()

    st.session_state.chat_sessions = loaded_sessions

    if loaded_sessions:
        st.session_state.current_chat_id = list(loaded_sessions.keys())[0]
    else:
        st.session_state.current_chat_id = None


if (
    st.session_state.chat_sessions
    and (
        "current_chat_id" not in st.session_state
        or st.session_state.current_chat_id
        not in st.session_state.chat_sessions
    )
):
    st.session_state.current_chat_id = (
        list(st.session_state.chat_sessions.keys())[0]
    )


# -------------------------------------------------------------
# 8. 사이드바
# -------------------------------------------------------------

with st.sidebar:
    st.markdown("### STUDY")

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
            if st.button(
                "X", key=f"del_chat_{c_id}", help="이 대화방 삭제"
            ):
                db_delete_session(c_id)
                del st.session_state.chat_sessions[c_id]

                if st.session_state.chat_sessions:
                    st.session_state.current_chat_id = (
                        list(st.session_state.chat_sessions.keys())[0]
                    )
                else:
                    st.session_state.current_chat_id = None

                st.rerun()

    st.markdown("---")
    st.subheader("고정된 핵심 노트")

    all_pinned_items = []
    for c_id, session_data in st.session_state.chat_sessions.items():
        for q_item in session_data["qa_pairs"]:
            if q_item.get("pinned", False):
                all_pinned_items.append({"chat_id": c_id, **q_item})

    if not all_pinned_items:
        st.caption("고정된 대화가 없습니다.")
    else:
        for p_qa in all_pinned_items:
            with st.container(border=True):
                ans_text = p_qa["answer"].strip().split("\n")[0]
                clean_text = ans_text.replace("**", "").replace("*", "").replace("#", "").strip()
                if len(clean_text) > 16:
                    clean_text = clean_text[:16] + "..."
                thumbnail_text = f"#{clean_text}"

                st.caption(f"Q. {p_qa['question']}")
                if st.button(
                    f"{thumbnail_text}",
                    key=f"goto_pin_{p_qa['chat_id']}_{p_qa['db_id']}",
                    use_container_width=True,
                ):
                    st.session_state.current_chat_id = p_qa["chat_id"]
                    st.rerun()

    st.markdown("---")
    st.subheader("시스템 관리")

    if st.button("새 문서 확인 및 반영"):
        st.cache_resource.clear()
        st.rerun()

    with st.expander("⚠️ 전체 재빌드 (문제 있을 때만)"):
        st.caption("모델을 바꿨거나 인덱스가 꼬였을 때만 사용하세요. 노션 문서 전체를 다시 임베딩해 API 할당량이 많이 씁니다.")
        if st.button("전체 인덱스 새로 만들기", type="secondary"):
            if os.path.exists(PERSIST_DIR):
                shutil.rmtree(PERSIST_DIR)
            try:
                existing = supabase.storage.from_(INDEX_BUCKET).list()
                if existing:
                    supabase.storage.from_(INDEX_BUCKET).remove(
                        [f["name"] for f in existing]
                    )
            except Exception:
                pass
            st.cache_resource.clear()
            st.rerun()


# -------------------------------------------------------------
# 9. 인덱스 로딩 (엔진 로딩 스피너 위치를 아래로 이동)
# -------------------------------------------------------------

try:
    with st.spinner("지식 베이스 엔진 로딩 중..."):
        index = load_or_build_index()
except Exception as e:
    if "RESOURCE_EXHAUSTED" in str(e):
        st.error(
            "Gemini 임베딩 API 할당량을 초과했습니다.\n\n"
            "지금까지 처리된 문서는 안전하게 저장되었으니, "
            "잠시 후(할당량 리셋 이후) 페이지를 새로고침하면 "
            "이어서 진행됩니다."
        )
    else:
        st.error(f"인덱스 로딩 중 오류가 발생했습니다.\n\n`{e}`")
    st.stop()

query_engine = index.as_query_engine(
    similarity_top_k=3,
    text_qa_template=qa_prompt,
)


# -------------------------------------------------------------
# 10. 현재 채팅 (대화방이 하나도 없을 수 있음)
# -------------------------------------------------------------

if not st.session_state.chat_sessions:
    st.info("현재 대화방이 없습니다. 사이드바에서 '새 대화 시작하기'를 눌러 시작하세요.")
    st.stop()

current_qa_pairs = st.session_state.chat_sessions[
    st.session_state.current_chat_id
]["qa_pairs"]


# -------------------------------------------------------------
# 11. Q&A 카드
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
                db_update_pin(qa["db_id"], new_pinned)
                st.rerun()

        with b_col_del:
            if st.button(
                "삭제",
                key=f"del_{st.session_state.current_chat_id}_{idx}",
                use_container_width=True,
            ):
                db_delete_qa(qa["db_id"])
                del current_qa_pairs[idx]
                st.rerun()


# -------------------------------------------------------------
# 12. 사용자 질문
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
        )

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

```
