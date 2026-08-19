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
