---
notion-url: https://app.notion.com/p/3c177fa64f7f80faa37efe1da23324eb
title: 내 비서 만들기
date: '2026-08-19 23:45:00.000'
from_notion: https://app.notion.com/p/Describe-3c177fa64f7f80faa37efe1da23324eb
author: rahui kim
last_edited_time: '2026-08-20 06:44:00.000'
---
<br/>

[//]: # (child_page is not supported)

# 🧠 My Second Brain: Notion-Powered AI Study Assistant (v4 Ultimate)

## 1. 프로젝트 개요 (Overview)

- **개발 배경:** 방대한 학습 내용과 기술 문서를 노션에 개별적으로 정리해 두었으나, 모바일 환경이나 브라우저 새로고침 시 기존 세션 메모리(`st.session_state`)가 초기화되어 대화 기록이 유실되는 불편함이 있었습니다. 이를 해결하고 파편화된 지식을 종합하여 질의하기 위한 **'검색 지옥'과 '정보 사일로(Silo) 현상'을 원천적으로 타파**하기 위해 구축했습니다.

- **핵심 목표:** 파편화된 지식을 하나의 중앙 집중형 지식 베이스로 통합하고, **Supabase DB 연동을 통한 대화 기록 영구 저장**을 구현하여 스마트폰 웹/앱 환경에서 브라우저 새로고침과 무관하게 24시간 대화할 수 있는 Zero-Cost(서버 비용 0원) 학습 비서 구현.

- **기술 스택:**

	- **Sync (자동화 파이프라인):** Notion API, GitHub Actions (`YouXam/Notion-GitHub-Sync`, 커스텀 스텁 생성기 `generate_md_stubs.py`)

	- **Backend / RAG:** Python, LlamaIndex, Google Gemini API (`models/gemini-1.5-flash`), Google GenAI Embedding (`models/text-embedding-004`)

	- **Database (v4 신규):** Supabase Postgres DB (`chat_sessions`, `qa_pairs` 테이블 영구 저장 및 `ON DELETE CASCADE` 계층 삭제 구조)

	- **Frontend:** Streamlit Cloud (모바일 최적화 UI, 초밀착형 카드 레이아웃 및 커스텀 CSS 적용)

## 2. 시스템 아키텍처 및 동작 원리 (Architecture)

# 


```plain text
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

- **자동 동기화 파이프라인:** 매일 정해진 시간(cron) 또는 푸시/수동 실행 시 GitHub Actions가 구동됩니다. `generate_md_stubs.py`가 노션 워크스페이스를 스캔하여 레포에 아직 없는 신규 페이지를 감지하고 빈 마크다운 스텁(`.md`)을 생성하면, `YouXam/Notion-GitHub-Sync` 액션이 실제 노션 콘텐츠를 채워 마크다운 문서로 완벽히 동기화합니다.

- **지능형 RAG 캐싱 및 메모리 최적화:** Streamlit Cloud의 제한된 무료 자원(RAM 1GB) 환경에서 무거운 로컬 임베딩 모델 대신 경량화된 구글 공식 임베딩 모델을 채택했습니다. 최초 빌드 시 생성된 벡터 인덱스를 로컬 디스크(`storage/`)에 영구 저장(`persist`)하여 속도와 자원 효율을 극대화했습니다.

- **Supabase DB 기반 영구 저장 아키텍처 (v4 Core):** 대화방(`chat_sessions`)과 질문-답변(`qa_pairs`)을 Supabase Postgres에 실시간 백업 및 동기화합니다. 대화방 삭제 시 DB의 `ON DELETE CASCADE` 규칙에 따라 하위 질문-답변이 안전하게 함께 정리되며, 개별 질문 삭제 및 고정(`pin`) 상태도 실시간으로 반영됩니다.

## 3. 고도화된 핵심 기능 (Core Features)

1. **하이브리드 RAG 프롬프트 템플릿 (Hybrid Fallback):**

	- 일반적인 RAG 구조는 노션 문서에 답변이 없으면 융통성 없이 차단하지만, 본 프로젝트는 커스텀 `PromptTemplate`을 적용하여 "1순위로 내 노션 노트를 참고하되, 내용이 부족하거나 코딩 예제 등의 질문은 제미나이의 강력한 일반 지식과 추론 능력으로 유연하게 보완"하도록 설계되었습니다.

1. **Supabase 기반 완벽한 세션 영구 보존:**

	- 스마트폰 브라우저에서 `[홈 화면에 추가]` 기능을 통해 네이티브 앱처럼 접근할 수 있으며, 클라우드 슬립이나 브라우저 새로고침으로 인해 메모리가 리셋되어도 **Supabase DB에서 대화 세션과 Q&A 기록을 즉시 불러와 끊김 없는 학습 연속성**을 제공합니다. 중요한 답변은 `[고정(Pin)]`하여 사이드바 보관함에 모아볼 수 있습니다.

1. **제로-코스트(Zero-Cost) 인프라:**

	- 유료 서버 비용 0원으로 GitHub Actions, Streamlit Cloud, Supabase, Gemini API 무료 할당량을 조합하여 유지 비용이 전혀 들지 않는 퍼펙트한 End-to-End 자동화 프로덕트를 완성했습니다.

<br/>

