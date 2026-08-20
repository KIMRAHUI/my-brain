import os
import streamlit as st
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

# 0. API 키 설정 (Streamlit Cloud의 Secrets에 등록해서 사용)
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# 1. Gemini를 LLM/임베딩 모델로 명시 지정
Settings.llm = Gemini(model="models/gemini-1.5-flash")
Settings.embed_model = GeminiEmbedding()

PERSIST_DIR = "./storage"
DATA_DIR = "./notion"  # Notion-GitHub-Sync 액션의 subdir 저장 경로에 맞춤


# 2. 인덱스 캐싱: 최초 1회만 생성, 이후 재사용
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

# 3. UI
st.set_page_config(page_title="나만의 학습 비서", page_icon="📚")
st.title("📚 나만의 학습 비서")

with st.sidebar:
    st.markdown("### ℹ️ 안내")
    st.write("노션에 정리한 학습 노트를 기반으로 질문에 답해드려요.")
    if st.button("🔄 인덱스 새로고침"):
        # storage 폴더를 지우고 재시작하면 최신 노션 내용으로 재인덱싱됨
        import shutil

        if os.path.exists(PERSIST_DIR):
            shutil.rmtree(PERSIST_DIR)
        st.cache_resource.clear()
        st.rerun()

user_query = st.text_input("공부한 내용에 대해 무엇이든 물어보세요!")

if user_query:
    with st.spinner("답변 생성 중..."):
        response = query_engine.query(user_query)
    st.write(str(response))
