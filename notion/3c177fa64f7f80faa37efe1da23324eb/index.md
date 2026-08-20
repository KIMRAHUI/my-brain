---
notion-url: https://app.notion.com/p/3c177fa64f7f80faa37efe1da23324eb
title: 내 비서 만들기
date: '2026-08-19 23:45:00.000'
from_notion: https://app.notion.com/p/3c177fa64f7f80faa37efe1da23324eb
author: rahui kim
last_edited_time: '2026-08-20 00:37:00.000'
---
# Zero-Cost 개인 학습 RAG 비서 구축 계획서 (v2)

## 1. 프로젝트 개요

- **목표**: 노션 학습 노트를 깃허브 Private Repo에 마크다운으로 동기화하고, LlamaIndex + Gemini API로 나만의 RAG 챗봇을 구축, Streamlit Cloud로 배포하여 PC 전원과 무관하게 스마트폰으로 24시간 사용.

- **총 비용**: 0원 (무료 티어만 사용)

- **PC 필요 시점**: 최초 개발·배포 시에만 필요. 배포 후에는 클라우드에서 자체적으로 실행됨.

---

## 2. v1 대비 수정된 핵심 오류 3가지

 | # | v1의 문제 | 원인 | v2 해결 방법 | 
 | ---- | ---- | ---- | ---- | 
 | 1 | `app.py` 실행 시 OpenAI API 키 오류 발생 | LlamaIndex는 `Settings.llm`/`Settings.embed_model`을 지정 안 하면 기본값으로 OpenAI를 호출 | `Settings.llm`, `Settings.embed_model`을 Gemini로 명시 설정 | 
 | 2 | "Notion2GitHub"라는 이름의 툴은 실존하지 않음 | 실제로 존재하는 액션은 `YouXam/Notion-GitHub-Sync`이며, 자동 전체 스캔이 아니라 "이미 있는 md 파일의 notion-url을 채워주는" 방식 | 실제 액션 사용법 + 새 페이지를 자동 감지해 스텁 파일을 만들어주는 `generate_md_stubs.py` 스크립트 추가 | 
 | 3 | 매 접속마다 전체 문서를 재인덱싱 → 무료 서버 자원(RAM 1GB) 부족 위험 | 인덱스를 디스크에 저장(persist)하지 않음 | `storage_context.persist()`로 인덱스 캐싱, 앱 재시작 시 재사용 | 

---

## 3. 시스템 아키텍처


```plain text
[노션 학습 노트]
      │  (discover 잡: generate_md_stubs.py로 신규 페이지 → 빈 md 스텁 생성)
      ▼
[GitHub Private Repo: notion/{page_id}/index.md (notion-url만 있는 상태)]
      │  (sync 잡: YouXam/Notion-GitHub-Sync 액션이 실제 콘텐츠로 채움)
      ▼
[GitHub Private Repo: notion/{page_id}/index.md (콘텐츠 완성)]
      │  (Streamlit Cloud가 배포 시 repo clone)
      ▼
[Streamlit Cloud 앱: LlamaIndex 인덱싱 + persist]
      │  (질의 시 Gemini API 호출)
      ▼
[스마트폰/PC 브라우저에서 대화형 질의응답]
```

---

## 4. 단계별 실행 가이드

### [1단계] 깃허브 저장소 준비

1. GitHub에서 Private Repo 생성 (예: `my-brain`)

1. 로컬에 `git clone`

### [2단계] 노션 → 깃허브 동기화 (v2: 실제 Marketplace 액션 사용)

- 이 액션은 노션 전체를 자동 스캔해서 새 파일을 만들어주지 않습니다. 레포에 **이미 존재하는** `.md` 파일의 front-matter에 `notion-url`이 적혀 있으면, 그 노션 페이지 내용으로 파일을 덮어씁니다.

- 즉 새 학습 주제를 추가할 때마다 "빈 md 파일 + notion-url"을 먼저 만들어둬야 합니다 → 이를 자동화하는 스크립트를 5-1단계에 포함했습니다.

- 노션 페이지는 반드시 Integration에 **Connect**되어 있어야 읽힙니다.

1. [notion.so/my-integrations](https://www.notion.so/my-integrations)에서 Integration 생성, 토큰 발급

1. 동기화할 노션 페이지/DB에 해당 Integration을 **Connect**

1. 레포 `Settings → Secrets and variables → Actions`에 `NOTION_TOKEN` 등록


```yaml
name: sync-notion-to-github
on:
  schedule:
    - cron: '0 0 * * *'   # 매일 00:00 UTC 자동 동기화
  workflow_dispatch:        # 수동 실행 버튼
  push:
    branches: [main]

jobs:
  discover:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install deps
        run: pip install notion-client
      - name: Auto-generate empty md files for new Notion pages
        env:
          NOTION_TOKEN: ${{ secrets.NOTION_TOKEN }}
        run: python scripts/generate_md_stubs.py
      - name: Commit new stub files
        run: |
          git config user.name "notion-bot"
          git config user.email "bot@example.com"
          git add notion/
          git commit -m "chore: add new notion page stubs" || echo "No new stubs"
          git push

  sync:
    needs: discover
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          ref: main
      - uses: YouXam/Notion-GitHub-Sync@v2
        with:
          notion_token: ${{ secrets.NOTION_TOKEN }}
          subdir: true   # notion/{article_id}/ 형태로 저장
```

> `subdir: true`이므로 실제 문서는 `notion/{article_id}/` 폴더에 저장됩니다. 4단계 `app.py`의 `DATA_DIR`을 `"./notion"`으로 맞춰야 합니다 (아래 반영됨).

- 이 액션은 개인 개발자 1인이 관리하는 소규모 오픈소스입니다. 업데이트가 뜸하거나 이슈 대응이 느릴 수 있으니, 문제가 생기면 GitHub Issues에 등록하거나 대체 스크립트로 전환할 여지를 남겨두세요.

### [2-4단계] 빈 md 파일 자동 생성 스크립트 (`scripts/generate_md_stubs.py`)


```python
"""
scripts/generate_md_stubs.py

Notion 워크스페이스를 검색하여, 아직 레포에 없는 페이지에 대해
notion-url front-matter만 담긴 빈 md 스텁 파일을 생성한다.
이후 Notion-GitHub-Sync 액션이 이 스텁을 실제 콘텐츠로 채운다.
"""

import os
import re
from pathlib import Path
from notion_client import Client

NOTION_TOKEN = os.environ["NOTION_TOKEN"]
OUTPUT_DIR = Path("notion")  # Notion-GitHub-Sync의 subdir 저장 경로와 동일하게 맞춤

notion = Client(auth=NOTION_TOKEN)


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9가-힣]+", "-", text)
    return text.strip("-") or "untitled"


def get_page_title(page: dict) -> str:
    for prop in page.get("properties", {}).values():
        if prop.get("type") == "title":
            title_parts = prop["title"]
            if title_parts:
                return "".join(t["plain_text"] for t in title_parts)
    return "untitled"


def existing_notion_urls() -> set[str]:
    """레포에 이미 등록된 notion-url 목록을 수집 (중복 스텁 방지)."""
    urls = set()
    if not OUTPUT_DIR.exists():
        return urls
    for md_file in OUTPUT_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        match = re.search(r"notion-url:\s*(\S+)", content)
        if match:
            urls.add(match.group(1))
    return urls


def main():
    known_urls = existing_notion_urls()
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 워크스페이스 전체에서 페이지 검색 (Integration에 Connect된 페이지만 조회됨)
    results = notion.search(filter={"property": "object", "value": "page"}).get("results", [])

    created = 0
    for page in results:
        page_url = page.get("url")
        if not page_url or page_url in known_urls:
            continue  # 이미 스텁이 있으면 건너뜀

        title = get_page_title(page)
        page_id = page["id"].replace("-", "")
        stub_dir = OUTPUT_DIR / page_id
        stub_dir.mkdir(parents=True, exist_ok=True)
        stub_path = stub_dir / "index.md"

        if stub_path.exists():
            continue

        stub_path.write_text(
            f"---\nnotion-url: {page_url}\ntitle: {title}\n---\n",
            encoding="utf-8",
        )
        created += 1
        print(f"생성됨: {stub_path} ({title})")

    print(f"총 {created}개의 새 스텁 파일 생성")


if __name__ == "__main__":
    main()
```

1. Integration에 Connect된 노션 페이지 전체를 검색

1. 레포에 아직 `notion-url`로 등록 안 된 페이지를 찾아 `notion/{page_id}/index.md`에 빈 스텁 생성

1. 같은 워크플로의 다음 잡(`sync`)이 `Notion-GitHub-Sync` 액션으로 이 스텁들을 실제 노션 콘텐츠로 채움

### [3단계] 개발 환경 세팅


```bash
pip install streamlit llama-index llama-index-llms-gemini llama-index-embeddings-gemini google-generativeai
```

- Google AI Studio에서 Gemini API Key 무료 발급

### [4단계] app.py (수정된 최종 코드)


```python
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

# 0. API 키 설정 (Streamlit Cloud의 Secrets에 등록)
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# 1. Gemini를 LLM/임베딩 모델로 명시 지정 (v1 오류 수정 지점)
Settings.llm = Gemini(model="models/gemini-1.5-flash")
Settings.embed_model = GeminiEmbedding()

PERSIST_DIR = "./storage"
DATA_DIR = "./notion"   # Notion-GitHub-Sync 액션의 subdir 저장 경로에 맞춤

# 2. 인덱스 캐싱: 최초 1회만 생성, 이후 재사용 (v1 오류 수정 지점)
@st.cache_resource
def load_or_build_index():
    if os.path.exists(PERSIST_DIR):
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
    else:
        documents = SimpleDirectoryReader(DATA_DIR).load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    return index

index = load_or_build_index()
query_engine = index.as_query_engine()

# 3. UI
st.title("📚 나만의 학습 비서")
user_query = st.text_input("공부한 내용에 대해 무엇이든 물어보세요!")

if user_query:
    with st.spinner("답변 생성 중..."):
        response = query_engine.query(user_query)
    st.write(str(response))
```

### [5단계] requirements.txt


```plain text
streamlit
llama-index
llama-index-llms-gemini
llama-index-embeddings-gemini
google-generativeai
```

### [6단계] 배포

1. `app.py`, `requirements.txt`를 `my-brain` 레포에 push

1. [share.streamlit.io](https://share.streamlit.io/)에서 레포 연결 후 배포

1. **App → Settings → Secrets**에 아래 입력 (코드에 키를 직접 넣지 않기 위함):


```plain text
GOOGLE_API_KEY = "여기에_발급받은_키"
```

1. 배포된 URL을 스마트폰에서 열고 **[홈 화면에 추가]**

---

## 5. 무료 티어 제약사항 및 대응

 | 항목 | 제약 | 대응 | 
 | ---- | ---- | ---- | 
 | Streamlit Cloud | 비활성 시 슬립 모드 진입, 재접속 시 수십 초 로딩 | 매일 1회 이상 접속하면 슬립 방지 | 
 | Streamlit Cloud RAM | 약 1GB | 인덱스 persist로 재인덱싱 비용 절감, 문서량 과다 시 챕터 단위로 분할 | 
 | Gemini API 무료 티어 | 분당/일일 요청 수 제한 | 개인 학습용 사용량에서는 충분 | 
 | GitHub Actions | Private repo 무료 한도(월 2,000분) | 매시간 동기화 정도는 여유 있음 | 

---

## 6. 체크리스트

- [ ] GitHub Private Repo 생성

- [ ] Notion Integration Token 발급, 동기화할 노션 페이지에 Integration Connect

- [ ] `NOTION_TOKEN` 시크릿 등록

- [ ] `scripts/generate_md_stubs.py` 추가 및 로컬 테스트 (`pip install notion-client` 후 실행)

- [ ] `.github/workflows/sync.yml` 추가 (discover → sync 2단계 잡)

- [ ] 워크플로 수동 실행(`workflow_dispatch`)으로 스텁 생성 및 콘텐츠 채움 확인

- [ ] Gemini API Key 발급

- [ ] `app.py`, `requirements.txt` 작성 및 로컬 테스트

- [ ] Streamlit Cloud 배포 및 `GOOGLE_API_KEY` 시크릿 등록

- [ ] 모바일 홈 화면 추가 및 접속 테스트

---

[//]: # (child_page is not supported)

