"""위키문헌(ko.wikisource.org, CC BY-SA 4.0)의 장편소설 전문을 docx로 저장한다.
작가 전부 사후 70년 경과로 퍼블릭도메인. 장(챕터)별로 페이지가 나뉜 작품은
MediaWiki API로 하위페이지 목록을 조회해서 순서대로 합친다(인간문제처럼 이미
단일 페이지인 작품은 build_ingan_munje_docx.py가 따로 담당).
"""
import sys
import urllib.parse
import urllib.request
import json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))
from wikitext_lib import export_pages, wikitext_events, events_to_docx
from config import DOCX_KOR_DIR

OUT_DIR = DOCX_KOR_DIR / "대용량문서"

# 실측 확인된 후보(전체 글자수): 탁류(채만식) 420,610자, 무정(이광수) 328,291자,
# 무영탑(현진건) 337,432자, 상록수(심훈) 254,068자, 태평천하(채만식) 168,744자
NOVELS = ["탁류", "무정"]


def list_subpages(site, prefix):
    url = f"https://{site}/w/api.php?" + urllib.parse.urlencode({
        "action": "query", "list": "allpages", "apprefix": prefix,
        "aplimit": "500", "format": "json",
    })
    req = urllib.request.Request(url, headers={"User-Agent": "research-bot/1.0"})
    with urllib.request.urlopen(req) as r:
        data = json.loads(r.read())
    return [p["title"] for p in data["query"]["allpages"]]


def build(name, out_path):
    subs = sorted(list_subpages("ko.wikisource.org", name + "/"))
    if not subs:
        print(f"[스킵] {name}: 하위페이지 없음")
        return
    pages = export_pages("ko.wikisource.org", subs)
    full_events = []
    for title in subs:
        wikitext = pages.get(title, "")
        if not wikitext:
            continue
        full_events.extend(wikitext_events(wikitext))
    events_to_docx(full_events, name, out_path)
    total_chars = sum(len(pages.get(t, "")) for t in subs)
    print(f"{name}: 하위페이지 {len(subs)}개, 원문 {total_chars:,}자 -> {out_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in NOVELS:
        build(name, OUT_DIR / f"{name}.docx")


if __name__ == "__main__":
    main()
