"""ko.wikipedia.org(CC BY-SA 4.0, 사이트 rightsinfo API로 직접 확인) 설명문형 항목 5개 -> DOCX.

처음엔 lcw99/wikipedia-korean-20240501(HF 미러)의 section_texts를 썼는데, 이 미러가
이미 위키표(`{|...|}`)와 소제목(`===...===`) 마크업을 잃어버린 상태라 표는 개별 문단
으로 쪼개지고, 소제목은 `=== Z 모식도 ===`처럼 마크업이 원문 그대로 새어나오는 문제가
있었다(사용자가 직접 광합성.docx에서 발견). -> 위키책/위키문헌에 이미 쓰던 `wikitext_lib`
로 라이브 위키백과 원문을 Special:Export로 직접 받아 제대로 파싱한다(표/헤딩 전부 살아있음
- 확인함, 광합성 문서에 실제 표 2개 존재).
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))
from wikitext_lib import export_pages, wikitext_events, events_to_docx
from config import DOCX_KOR_DIR

SITE = "ko.wikipedia.org"
OUT_DIR = DOCX_KOR_DIR / "Wikipedia_설명문"
WANTED = ["임진왜란", "광합성", "한글", "일반 상대성이론", "발효"]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pages = export_pages(SITE, WANTED)
    for i, title in enumerate(WANTED, 1):
        wikitext = pages.get(title)
        if not wikitext:
            print(f"[{i}] {title}: 못 찾음 - 스킵")
            continue
        events = wikitext_events(wikitext)
        out = OUT_DIR / f"{title}.docx"
        events_to_docx(events, title, out)
        n_tables = sum(1 for e in events if e[0] == "table")
        total_chars = sum(len(e[-1]) for e in events if e[0] != "table")
        print(f"[{i}/{len(WANTED)}] {title}: {total_chars}자, 표 {n_tables}개")
    print(f"완료: {len(WANTED)}개")
    print(f"저장 위치: {OUT_DIR}")


if __name__ == "__main__":
    main()
