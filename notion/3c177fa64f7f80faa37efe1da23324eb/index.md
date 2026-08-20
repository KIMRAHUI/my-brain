---
notion-url: https://app.notion.com/p/3c177fa64f7f80faa37efe1da23324eb
title: 내 비서 만들기
date: '2026-08-19 23:45:00.000'
from_notion: https://app.notion.com/p/Describe-3c177fa64f7f80faa37efe1da23324eb
author: rahui kim
last_edited_time: '2026-08-20 02:22:00.000'
---
<br/>

[//]: # (child_page is not supported)

# My Second Brain: Notion-Powered AI Study Assistant

## 1. 프로젝트 개요 (Overview)

- **개발 배경**: 방대한 학습 내용과 기술 문서를 노션에 개별적으로 정리해 두었으나, 막상 모바일이나 외부 환경에서 필요한 정보를 빠르게 탐색하거나 파편화된 지식을 종합하여 질의하기 어려운 '검색 지옥'과 '정보 사일로(Silo) 현상'을 원천적으로 해결하기 위해 구축했습니다.

- **핵심 목표**: 파편화된 지식을 하나의 중앙 집중형 지식 베이스로 통합하여 정보 사일로를 방지하고, PC 전원과 무관하게 클라우드 환경에서 24시간 동작하며 스마트폰 웹/앱 환경에서 대화할 수 있는 Zero-Cost(서버 비용 0원) 학습 비서 구현.

- **기술 스택**:

	- **Sync**: Notion API, GitHub Actions (`YouXam/Notion-GitHub-Sync`, 커스텀 스텁 생성기)

	- **Backend / RAG**: Python, LlamaIndex, Google Gemini API (`models/gemini-3.6-flash`), Hugging Face Embedding (`BAAI/bge-small-en-v1.5`)

	- **Frontend**: Streamlit Cloud (모바일 최적화 UI 및 커스텀 CSS 적용)

## 2. 시스템 아키텍처 및 동작 원리 (Architecture)


```plain text
[노션 학습 노트]
      │  (discover 잡: generate_md_stubs.py로 신규 페이지 → 빈 md 스텁 생성)
      ▼
[GitHub Private/Public Repo: notion/{page_id}/index.md]
      │  (sync 잡: YouXam/Notion-GitHub-Sync 액션이 실제 콘텐츠로 채움)
      ▼
[Streamlit Cloud 앱: LlamaIndex 인덱싱 + 디스크 캐싱(Persist)]
      │  (질의 시 Gemini API 호출)
      ▼
[스마트폰/PC 브라우저에서 대화형 질의응답]
```

- **자동 동기화 파이프라인**:

	- 매일 정해진 시간(`cron`) 또는 푸시/수동 실행 시 GitHub Actions가 동작합니다.

	- `generate_md_stubs.py`가 노션 워크스페이스를 스캔하여 레포에 아직 없는 신규 페이지를 감지하고 빈 마크다운 스텁(`.md`)을 생성합니다.

	- `YouXam/Notion-GitHub-Sync` 액션이 스텁 파일에 실제 노션 페이지의 내용을 채워 넣어 마크다운 문서로 마이그레이션합니다.

- **지능형 RAG 캐싱**:

	- Streamlit Cloud의 제한된 자원(RAM 1GB) 문제를 해결하기 위해, 최초 빌드 시 생성된 인덱스를 로컬 디스크(`storage/`)에 영구 저장(`persist`)합니다. 앱 재시작 시 전체 문서를 매번 파싱하지 않고 캐시된 인덱스를 불러와 속도를 극대화했습니다.

<br/>

