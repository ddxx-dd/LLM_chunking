"""위키문헌(ko.wikisource.org, CC BY-SA 4.0)의 강경애 "인간문제" 전문을 docx로 저장한다.
장(챕터) 구분이 아예 없는 단일 페이지 작품이라(23.7만자) 발췌 없이 그대로 쓸 수 있어
"대용량 단일 문서" 후보로 채택했다 - 원래 Wikisource_문학 구축 때는 "발췌 기준을 잡기
애매하다"는 이유로 제외했던 작품인데, 그 특성이 오히려 이 목적엔 장점이 됨.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))
from wikitext_lib import export_pages, wikitext_events, events_to_docx
from config import DOCX_KOR_DIR

OUT_DIR = DOCX_KOR_DIR / "대용량문서"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pages = export_pages("ko.wikisource.org", ["인간문제"])
    wikitext = pages["인간문제"]
    print(f"원문: {len(wikitext):,}자")
    events = wikitext_events(wikitext)
    events_to_docx(events, "인간문제", OUT_DIR / "인간문제.docx")
    print("저장 완료:", OUT_DIR / "인간문제.docx")


if __name__ == "__main__":
    main()
