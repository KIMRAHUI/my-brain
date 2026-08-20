
# My Second Brain: Notion-Powered AI Study Assistant

방대한 학습 노트와 기술 문서를 노션에 개별적으로 정리해 두었으나, 막상 모바일이나 외부 환경에서 필요한 정보를 빠르게 탐색하거나 파편화된 지식을 종합하여 질의하기 어려운 '검색 지옥'과 '정보 사일로(Silo) 현상'을 원천적으로 해결하기 위해 구축한 제로-코스트 AI 학습 비서입니다.

---

## 1. Project Background & Pain Points

* **파편화된 지식과 검색 지옥**:
다양한 채널과 노션 페이지에 흩어져 있는 학습 노트로 인해 특정 기술이나 개념을 찾기 위해 오랜 시간을 소모하는 문제를 겪었습니다.
* **정보 사일로(Silo) 현상 방지**:
개별적으로 고립되어 관리되던 학습 자원과 기술 문서를 중앙 집중형 지식 베이스로 통합하여, 파편화된 지식 간의 연결 고리를 확보하고 언제든 종합적인 맥락에서 조회할 수 있도록 설계했습니다.
* **접근성 제약 극복**:
PC 환경에 종속되지 않고, 언제 어디서나 스마트폰 웹 환경에서 24시간 나만의 지식 베이스와 대화할 수 있는 인프라가 필요했습니다.

---

## 2. Core Features

* **자동 동기화 파이프라인**:
노션 워크스페이스의 변경 사항 및 신규 페이지를 깃허브 레포지토리에 자동으로 마이그레이션하여 수동 관리 공수를 최소화했습니다.
* **지능형 RAG 기반 질의응답**:
LlamaIndex와 Gemini API를 결합하여 문서의 단순 키워드 검색을 넘어 문맥을 이해하고 정확한 답변을 도출합니다.
* **제로-코스트(Zero-Cost) 인프라**:
유료 서버 비용 없이 GitHub Actions, Streamlit Cloud의 무료 티어 및 오픈소스 임베딩 모델을 활용해 24시간 가동되는 시스템을 구축했습니다.
* **모바일 최적화 UI/UX**:
Streamlit 기반 인터페이스에 커스텀 CSS를 적용하여 모바일 브라우저에서도 정돈된 레이아웃과 직관적인 조작감을 제공합니다.

---

## 3. System Architecture

```text
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

---

## 4. Tech Stack

* **Sync**: Notion API, GitHub Actions (`YouXam/Notion-GitHub-Sync`, 커스텀 스텁 생성기)
* **Backend / RAG**: Python, LlamaIndex, Google Gemini API (`models/gemini-3.6-flash`), Hugging Face Embedding (`BAAI/bge-small-en-v1.5`)
* **Frontend**: Streamlit Cloud (모바일 최적화 UI 및 커스텀 CSS 적용)
