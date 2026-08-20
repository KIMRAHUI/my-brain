

# 🧠 My Second Brain: Notion-Powered AI Study Assistant

방대한 학습 노트와 기술 문서를 노션에 개별적으로 정리해 두었으나, 막상 모바일이나 외부 환경에서 필요한 정보를 빠르게 탐색하거나 파편화된 지식을 종합하여 질의하기 어려운 **'검색 지옥'과 '정보 사일로(Silo) 현상'을 원천적으로 해결**하고, **Supabase DB 연동으로 대화 기록을 영구 보존**하는 제로-코스트 AI 학습 비서입니다.

---

## 1. Project Background & Pain Points

* **파편화된 지식과 검색 지옥**:
다양한 채널과 노션 페이지에 흩어져 있는 학습 노트로 인해 특정 기술이나 개념을 찾기 위해 오랜 시간을 소모하는 문제를 겪었습니다.
* **정보 사일로(Silo) 현상 방지**:
개별적으로 고립되어 관리되던 학습 자원과 기술 문서를 중앙 집중형 지식 베이스로 통합하여, 파편화된 지식 간의 연결 고리를 확보하고 언제든 종합적인 맥락에서 조회할 수 있도록 설계했습니다.
* **접근성 제약 및 세션 유실 극복 (v4 주요 개선)**:
PC 환경에 종속되지 않고 스마트폰 웹/앱 환경에서 24시간 대화할 수 있어야 하며, 브라우저 새로고침이나 클라우드 슬립 모드 진입 시 기존 메모리(`st.session_state`)가 날아가던 세션 유실 문제를 Supabase DB 연동으로 완벽히 해결했습니다.

---

## 2. Core Features

* **자동 동기화 파이프라인**:
노션 워크스페이스의 변경 사항 및 신규 페이지(`generate_md_stubs.py` + `YouXam/Notion-GitHub-Sync`)를 깃허브 레포지토리에 자동으로 마이그레이션하여 수동 관리 공수를 최소화했습니다.
* **지능형 하이브리드 RAG 질의응답**:
LlamaIndex와 Gemini API를 결합하여 내 노션을 1순위로 탐색하되, 내용이 부족할 경우 일반 지식으로 유연하게 보완하는 **하이브리드 Fallback 프롬프트**를 적용했습니다.
* **Supabase Postgres DB 영구 저장 (v4 New)**:
대화방(`chat_sessions`)과 질문-답변(`qa_pairs`)을 클라우드 DB에 영구 저장합니다. `ON DELETE CASCADE` 계층 삭제 규칙을 적용하여 방 삭제 시 하위 내용이 깔끔하게 정리되며, 중요 Q&A 카드 고정(Pin) 기능도 실시간 반영됩니다.
* **제로-코스트(Zero-Cost) 인프라**:
유료 서버 비용 없이 GitHub Actions, Streamlit Cloud, Supabase 및 구글 공식 경량 임베딩 모델을 조합하여 24시간 가동되는 시스템을 구축했습니다.
* **모바일 최적화 UI/UX**:
Streamlit 기반 인터페이스에 커스텀 CSS를 적용하여 스마트폰 브라우저(`[홈 화면에 추가]`)에서도 네이티브 앱처럼 정돈된 레이아웃을 제공합니다.

---

## 3. System Architecture

```text
[노션 학습 노트 워크스페이스]
      │  (discover 잡: generate_md_stubs.py로 신규 페이지 탐지 → 빈 md 스텁 생성)
      ▼
[GitHub Private Repo: notion/{page_id}/index.md]
      │  (sync 잡: YouXam/Notion-GitHub-Sync 액션이 실제 노션 콘텐츠로 바디 채움)
      ▼
[Streamlit Cloud 앱: LlamaIndex 인덱싱 + 디스크 캐싱(Persist)]
      │  (사용자 질의 시 하이브리드 프롬프트 규칙 적용하여 Gemini API 호출)
      ▼
[Supabase Postgres DB: chat_sessions ←(cascade)→ qa_pairs 영구 저장]
      │
      ▼
[스마트폰/PC 브라우저에서 24시간 대화형 질의응답 (새로고침해도 기록 유지)]

```

---

## 4. Tech Stack

* **Sync**: Notion API, GitHub Actions (`YouXam/Notion-GitHub-Sync`, 커스텀 스텁 생성기 `generate_md_stubs.py`)
* **Backend / RAG**: Python, LlamaIndex, Google Gemini API (`models/gemini-1.5-flash`), Google GenAI Embedding (`models/text-embedding-004`)
* **Database (v4)**: Supabase Postgres DB (`chat_sessions`, `qa_pairs`)
* **Frontend**: Streamlit Cloud (모바일 최적화 UI 및 커스텀 CSS 적용)
