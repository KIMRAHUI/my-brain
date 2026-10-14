# My Second Brain

> 노션 학습 노트를 지식 베이스로 삼아 질문하면 답해 주는 **개인용 RAG 학습 비서**
> Notion API · GitHub Actions · LlamaIndex · Gemini · Supabase · Streamlit


---

## 프로젝트 개요

| 항목 | 내용 |
| --- | --- |
| 개발 기간 | 2026.08 ~ 운영 중 |
| 개발 인원 | 1인 |
| 소개 | 노션에 흩어진 AI · 데이터 분석 학습 노트(현재 73개)를 매일 이 레포로 동기화하고, 그 노트를 근거로 질문에 답하는 챗봇입니다. 대화 기록과 검색 인덱스를 Supabase에 저장해 새로고침이나 서버 재시작 후에도 이어서 쓸 수 있고, 스마트폰 브라우저에서도 사용할 수 있습니다. |

---

## Tech Stack

| 영역 | 기술 |
| --- | --- |
| Sync | Notion API (`notion-client`), GitHub Actions |
| RAG | Python, LlamaIndex (`MarkdownNodeParser`, `VectorStoreIndex`) |
| LLM / Embedding | Google Gemini (`gemini-3.6-flash`), Gemini Embedding (`gemini-embedding-2`) |
| Database / Storage | Supabase Postgres (대화 기록), Supabase Storage (검색 인덱스 백업) |
| Frontend / 배포 | Streamlit, Streamlit Community Cloud |

---

## System Architecture

```text
[ Notion 학습 노트 워크스페이스 ]
        │  GitHub Actions (매일 00:00 UTC · 수동 실행 · main push)
        │  Notion API로 페이지 검색 → 본문 텍스트 추출
        ▼
[ 이 레포: notion/{페이지 제목}.md ]
        │
        ▼
[ Streamlit Cloud: app.py ]
   ├─ LlamaIndex 인덱싱 (Markdown 단위 분할, 상위 3개 문서 검색)
   ├─ 로컬 ./storage  ⇄  Supabase Storage(rag-index) 인덱스 백업 · 복원
   └─ Gemini API 호출 (노트 우선 + 일반 지식 보완 프롬프트)
        │
        ▼
[ Supabase Postgres: chat_sessions ─< qa_pairs ]
        │
        ▼
[ 모바일 / PC 브라우저 ]
```

---

## 주요 기능

- **노트 자동 동기화:** Notion Integration에 연결된 페이지를 GitHub Actions가 매일 가져와 `notion/` 폴더에 Markdown으로 저장하고 커밋합니다.
- **노트 기반 질의응답:** 질문과 가까운 노트 3개를 찾아 Gemini에 함께 넘깁니다. 노트에 답이 있으면 우선 활용하고, 부족하면 일반 지식으로 보완하되 노트에 없는 내용을 노트에 있다고 말하지 않도록 프롬프트에 규칙을 두었습니다.
- **대화방 관리:** 대화방 생성·삭제, 첫 질문으로 대화방 제목 자동 생성, 질문·답변 카드 단위 삭제
- **핵심 노트 고정:** 중요한 답변을 고정(Pin)하면 모든 대화방의 고정 카드가 사이드바에 모이고, 누르면 해당 대화방으로 이동합니다.
- **인덱스 관리 버튼:** "새 문서 확인 및 반영"은 바뀐 노트만 다시 임베딩하고, "전체 재빌드"는 모델을 바꿨거나 인덱스가 깨졌을 때만 쓰도록 분리했습니다.
- **모바일 UI:** Streamlit에 커스텀 CSS를 적용해 스마트폰 화면에서도 버튼과 카드가 깨지지 않게 했습니다.

---

## 핵심 구현 & 문제 해결

### 1. 새로고침하면 대화가 사라짐 → Supabase에 대화 기록 저장

- **문제:** 대화 기록을 `st.session_state`에만 두다 보니, 새로고침하거나 Streamlit Cloud가 슬립 모드에 들어가면 대화가 모두 사라졌습니다.
- **해결:**
  - 대화방(`chat_sessions`)과 질문·답변(`qa_pairs`)을 Supabase Postgres 테이블로 분리해 저장하고, 앱 시작 시 불러오도록 했습니다.
  - `qa_pairs`가 `chat_sessions`를 외래 키로 참조하고 `ON DELETE CASCADE`를 걸어, 대화방을 지우면 그 안의 질문·답변도 함께 정리되게 했습니다.

### 2. 재시작할 때마다 전체 재임베딩 + API 할당량 초과 → 인덱스 백업과 증분 업데이트

- **문제:** Streamlit Cloud는 재시작하면 로컬 파일이 사라져, 깨어날 때마다 노트 전체를 다시 임베딩해야 했습니다. 무료 임베딩 API 할당량을 넘기면(`RESOURCE_EXHAUSTED`) 처음부터 다시 시작해야 했습니다.
- **해결:**
  - 인덱스를 만들 때마다 Supabase Storage(`rag-index` 버킷)에 업로드하고, 앱이 새로 뜨면 먼저 내려받아 그대로 불러옵니다.
  - `refresh_ref_docs`로 새로 추가되거나 내용이 바뀐 노트만 다시 임베딩합니다.
  - 처음 만들 때는 문서를 하나씩 넣으면서 매번 저장하고, 할당량 오류가 나면 30초부터 대기 시간을 늘려 가며 최대 4번 재시도합니다. 그래도 실패하면 거기까지 저장된 상태로 멈추고, 할당량이 풀린 뒤 새로고침하면 이어서 진행됩니다.

### 3. 노션 동기화 방식 단순화

- **문제:** 처음에는 빈 Markdown 스텁 파일을 먼저 만들고 외부 동기화 액션이 본문을 채우는 2단계 방식(`scripts/generate_md_stubs.py` + Notion-GitHub-Sync)이었는데, 파일 이름이 페이지 ID라 노트를 알아보기 어려웠고 단계가 둘이라 관리가 번거로웠습니다.
- **해결:** 워크플로 하나(`.github/workflows/notion_to_git.yml`)에서 Notion API로 페이지를 검색하고, 페이지 제목을 파일 이름으로 바로 저장한 뒤 커밋하도록 바꿨습니다.

### 4. 슬립 모드에서 앱 깨우기

- Streamlit Cloud 무료 플랜은 한동안 접속이 없으면 앱이 잠듭니다. `main.py`는 헤드리스 Chrome(Selenium)으로 앱 주소에 접속해, 잠든 상태면 깨우기 버튼을 눌러 줍니다.
<!-- TODO: main.py를 어떻게 실행하는지(수동 / 스케줄러 등) 확인 후 한 줄 추가 -->

---

## 로컬 실행 방법

```bash
git clone https://github.com/KIMRAHUI/my-brain.git
cd my-brain
pip install -r requirements.txt
```

`.streamlit/secrets.toml`을 만들고 아래 값을 넣습니다. (이 파일은 저장소에 올리지 않습니다.)

```toml
GOOGLE_API_KEY = "..."
SUPABASE_URL   = "https://<project>.supabase.co"
SUPABASE_KEY   = "..."
```

Supabase에는 대화 기록용 테이블 두 개와 `rag-index` Storage 버킷을 만듭니다. 아래는 코드가 사용하는 컬럼 기준의 예시입니다.

```sql
create table chat_sessions (
  id          text primary key,
  title       text not null,
  created_at  timestamptz default now()
);

create table qa_pairs (
  id               uuid primary key default gen_random_uuid(),
  chat_session_id  text references chat_sessions(id) on delete cascade,
  question         text not null,
  answer           text not null,
  pinned           boolean default false,
  created_at       timestamptz default now()
);
```

```bash
streamlit run app.py
```

> 노션 동기화를 쓰려면 Notion Integration 토큰을 레포의 GitHub Secrets에 `NOTION_TOKEN`으로 등록하고, 동기화할 노션 페이지에 해당 Integration을 연결합니다.

---

## 한계와 다음 개선

- **동기화 범위:** 지금은 각 페이지의 최상위 블록 텍스트만 가져와서, 토글 안의 내용 · 이미지 · 표는 빠집니다. 하위 블록까지 재귀적으로 읽도록 개선할 계획입니다.
- **접근 제어:** 개인용으로 만들어 로그인이 없습니다. 앱 주소를 공유하기 전에 간단한 비밀번호 인증과 읽기 전용 데모 모드를 추가할 계획입니다.
- **답변 근거 표시:** 답변 아래에 참고한 노트 제목을 함께 보여 주면, 노트 기반 답인지 일반 지식인지 더 분명해질 것 같습니다.

---

## 회고

- **무료 인프라의 제약을 설계로 풀기:** 슬립 모드, 휘발되는 디스크, API 할당량처럼 무료 플랜의 제약이 오히려 저장 위치와 재시도 전략을 고민하게 만들었습니다.
- **직접 쓰는 도구라서 계속 고치게 됨:** 공부하면서 매일 쓰다 보니 대화 유실, 재임베딩 대기 같은 불편이 바로 보였고, 그걸 하나씩 고치는 과정이 그대로 이 프로젝트의 버전 기록이 되었습니다.
