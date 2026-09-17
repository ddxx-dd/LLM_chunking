"""en.wikibooks.org(CC BY-SA 4.0) "World History" 교재 -> DOCX.

Korean wikibooks와 달리 여기서는 한 페이지(챕터)를 docx 1개로 저장한다(책 전체를
합치면 37만자로 다른 문서보다 10배 이상 커져 코퍼스 크기 균형이 깨짐 - 베이비바
뽀개기를 제외한 것과 같은 이유). 목차/저자 등 메타 페이지는 길이 필터(2,000자 미만)로
나중에 걸러진다.

"Operation Blank Check"/"Contributors' Corner"는 길이 필터는 통과하지만(2,000자↑)
실제로는 세계사 내용이 아니라 위키책 편집자들의 편집 계획/토론 페이지라 코퍼스 주제와
안 맞음(직접 읽고 확인) - 제목으로 제외.
"""
EXCLUDE_TITLES = {"World History/Operation: Blank Check", "World History/Contributors' Corner"}
import sys
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))
from wikitext_lib import export_pages, wikitext_events, events_to_docx
from config import DOCX_ENG_DIR

SITE = "en.wikibooks.org"
BOOK = "World History"
OUT_DIR = DOCX_ENG_DIR / "Wikibooks_WorldHistory"


def list_pages():
    url = f"https://{SITE}/w/api.php?" + urllib.parse.urlencode({
        "action": "query", "list": "allpages", "apprefix": BOOK,
        "apnamespace": "0", "aplimit": "100", "format": "json",
    })
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    import json
    d = json.loads(urllib.request.urlopen(req, timeout=20).read())
    return [p["title"] for p in d["query"]["allpages"]]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    titles = [t for t in list_pages() if t not in EXCLUDE_TITLES]
    print(f"페이지 {len(titles)}개 발견")
    pages = export_pages(SITE, titles)
    n = 0
    for t in titles:
        wikitext = pages.get(t)
        if not wikitext:
            continue
        events = wikitext_events(wikitext)
        if not events:
            continue
        chap = t[len(BOOK):].lstrip("/") or "Overview"
        safe = "".join(c for c in chap if c not in '\\/:*?"<>|')[:60]
        out = OUT_DIR / f"{safe}.docx"
        events_to_docx(events, chap, out)
        n += 1
        total = sum(len(e[-1]) if e[0] != "table" else 0 for e in events)
        print(f"[{n}] {chap}: 약 {total}자")
    print(f"완료: {n}개")
    print(f"저장 위치: {OUT_DIR}")


if __name__ == "__main__":
    main()
