---
notion-url: https://app.notion.com/p/3c177fa64f7f80faa37efe1da23324eb
title: 내 비서 만들기
date: '2026-08-19 23:45:00.000'
from_notion: https://app.notion.com/p/Describe-3c177fa64f7f80faa37efe1da23324eb
author: rahui kim
last_edited_time: '2026-08-20 04:11:00.000'
---
<br/>

[//]: # (child_page is not supported)

# 🧠 My Second Brain: Notion-Powered AI Study Assistant

### 1. 프로젝트 개요 (Overview)

- **개발 배경:** 방대한 학습 내용과 기술 문서를 노션에 개별적으로 정리해 두었으나, 막상 모바일이나 외부 환경에서 필요한 정보를 빠르게 탐색하거나 파편화된 지식을 종합하여 질의하기 어려운 '검색 지옥'과 '정보 사일로(Silo) 현상'을 원천적으로 해결하기 위해 구축했습니다.

- **핵심 목표:** 파편화된 지식을 하나의 중앙 집중형 지식 베이스로 통합하여 정보 사일로를 방지하고, PC 전원과 무관하게 클라우드 환경에서 24시간 동작하며 스마트폰 웹/앱 환경에서 대화할 수 있는 Zero-Cost(서버 비용 0원) 학습 비서 구현.

- **기술 스택:**

	- **Sync (자동화 파이프라인):** Notion API, GitHub Actions (`YouXam/Notion-GitHub-Sync`, 커스텀 스텁 생성기 `generate_md_stubs.py`)

	- **Backend / RAG:** Python, LlamaIndex, Google Gemini API (`models/gemini-1.5-flash`), Google GenAI Embedding (`models/text-embedding-004`)

	- **Frontend:** Streamlit Cloud (모바일 최적화 UI, 초밀착형 카드 레이아웃 및 커스텀 CSS 적용)

### 2. 시스템 아키텍처 및 동작 원리 (Architecture)


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
[스마트폰/PC 브라우저에서 24시간 대화형 질의응답 및 핵심 노트 고정]
```

- **자동 동기화 파이프라인:** * 매일 정해진 시간(cron) 또는 푸시/수동 실행 시 GitHub Actions가 구동됩니다.

	- `generate_md_stubs.py`가 노션 워크스페이스를 스캔하여 레포에 아직 없는 신규 페이지를 감지하고 빈 마크다운 스텁(`.md`)을 생성합니다.

	- `YouXam/Notion-GitHub-Sync` 액션이 스텁 파일에 실제 노션 페이지의 내용을 채워 넣어 마크다운 문서로 완벽히 동기화합니다.

- **지능형 RAG 캐싱 및 메모리 최적화:** * Streamlit Cloud의 제한된 무료 자원(RAM 1GB) 환경에서 무거운 허깅페이스 로컬 임베딩 모델(`torch`, `torchvision` 등)을 사용할 때 발생하는 메모리 폭발 및 패키지 충돌 문제를 원천 차단하기 위해, **경량화된 구글 공식 임베딩 모델**을 채택했습니다.

	- 최초 빌드 시 생성된 벡터 인덱스를 로컬 디스크(`storage/`)에 영구 저장(`persist`)하여, 앱 재시작 시 전체 문서를 매번 파싱하지 않고 캐시된 인덱스를 즉시 불러와 속도와 자원 효율을 극대화했습니다.

### 3. 고도화된 핵심 기능 (Core Features)

1. **하이브리드 RAG 프롬프트 템플릿 (Hybrid Fallback):**

	- 일반적인 RAG 구조는 노션 문서에 답변이 없으면 융통성 없이 "문서에 내용이 없습니다"라며 차단합니다.

	- 본 프로젝트는 커스텀 `PromptTemplate`을 적용하여 "1순위로 내 노션 노트를 참고하되, 노트에 내용이 없거나 부족한 코딩 예제 등의 질문은 제미나이의 강력한 일반 지식과 추론 능력으로 유연하게 보완"하도록 설계하여 학습 비서로서의 실용성을 극대화했습니다.

1. **모바일 최적화 UI/UX 및 세션 관리:**

	- 스마트폰 브라우저에서 [홈 화면에 추가] 기능을 통해 네이티브 앱처럼 24시간 접근할 수 있습니다.

	- 멀티 채팅 세션 구조를 지원하여 대화 주제별로 방을 분리할 수 있으며, 공부하다가 중요한 Q&A 카드는 **[고정]** 버튼을 눌러 사이드바 보관함에 상시 모아볼 수 있도록 디테일을 더했습니다.

1. **제로-코스트(Zero-Cost) 인프라:**

	- 유료 서버 비용 0원으로 GitHub Actions, Streamlit Cloud의 무료 티어 및 Gemini API 무료 할당량을 조합하여 유지 비용이 전혀 들지 않는 퍼펙트한 End-to-End 자동화 프로덕트를 완성했습니다.

<br/>

