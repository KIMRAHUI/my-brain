# My Second Brain: Notion 기반 AI 학습 비서

노션에 정리해 둔 학습 노트와 기술 문서를 모바일에서도 빠르게 탐색하고, 흩어진 내용을 종합해서 질의할 수 있도록 만든 개인용 AI 비서입니다. Supabase DB를 연동해 대화 기록을 영구 보존합니다.

---

## 1. 배경

- **파편화된 지식**: 여러 채널과 노션 페이지에 흩어진 학습 노트 때문에 특정 개념을 찾는 데 시간이 오래 걸렸습니다.
- **정보 사일로**: 개별적으로 관리되던 학습 자료를 하나의 지식 베이스로 통합해, 종합적인 맥락에서 조회할 수 있도록 했습니다.
- **접근성과 세션 유실 (v4)**: PC 없이 스마트폰에서도 대화할 수 있어야 했고, 새로고침이나 클라우드 슬립 모드 진입 시 `st.session_state`의 대화 기록이 사라지는 문제가 있었습니다. Supabase DB 연동으로 해결했습니다.

---

## 2. 주요 기능

- **자동 동기화 파이프라인**: 노션 워크스페이스의 변경 사항과 신규 페이지를 `generate_md_stubs.py`와 `YouXam/Notion-GitHub-Sync`로 깃허브 레포에 자동 반영합니다.
- **하이브리드 RAG 질의응답**: LlamaIndex와 Gemini API를 결합해, 노션 콘텐츠를 우선 참고하고 내용이 부족하면 일반 지식으로 보완합니다.
- **Supabase Postgres 연동 (v4)**: 대화방(`chat_sessions`)과 질문-답변(`qa_pairs`)을 DB에 저장합니다. `ON DELETE CASCADE`로 방 삭제 시 하위 데이터도 함께 정리되고, Q&A 카드 고정(Pin) 기능도 지원합니다.
- **무료 인프라**: GitHub Actions, Streamlit Cloud, Supabase, 구글 경량 임베딩 모델을 조합해 서버 비용 없이 상시 운영합니다.
- **모바일 UI**: Streamlit에 커스텀 CSS를 적용해 스마트폰 브라우저에서도 (홈 화면 추가 시) 앱과 비슷한 레이아웃으로 사용할 수 있습니다.

---

## 3. 시스템 구조

```text
[노션 학습 노트 워크스페이스]
      │  discover: generate_md_stubs.py로 신규 페이지 탐지 → 빈 md 스텁 생성
      ▼
[GitHub Private Repo: notion/{page_id}/index.md]
      │  sync: Notion-GitHub-Sync 액션이 노션 콘텐츠로 본문 채움
      ▼
[Streamlit Cloud: LlamaIndex 인덱싱 + 디스크 캐싱]
      │  질의 시 하이브리드 프롬프트로 Gemini API 호출
      ▼
[Supabase Postgres: chat_sessions ↔ qa_pairs 저장]
      │
      ▼
[모바일/PC 브라우저에서 대화형 질의응답, 새로고침해도 기록 유지]
```

---

## 4. 기술 스택

- **Sync**: Notion API, GitHub Actions (`YouXam/Notion-GitHub-Sync`, `generate_md_stubs.py`)
- **Backend / RAG**: Python, LlamaIndex, Google Gemini API (`gemini-1.5-flash`), Google GenAI Embedding (`text-embedding-004`)
- **Database (v4)**: Supabase Postgres (`chat_sessions`, `qa_pairs`)
- **Frontend**: Streamlit Cloud
