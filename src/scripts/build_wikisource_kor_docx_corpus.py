"""ko.wikisource.org(CC BY-SA 4.0) 한국 근현대 문학 -> DOCX.

- 단편소설 6편, 수필·논설 3편: 페이지 1개 = docx 1개 그대로(발췌 불필요한 길이).
  "단군론"(최남선, 1930년대 원문)은 옛 한자 표기를 위한 유니코드 사용자 정의영역(PUA)
  코드포인트가 문서 전체에 11종류나 흩어져 있어(개별 복구 불가) 목록에서 제외.
- 장편소설 4편(탁류/무정/상록수/태평천하): 전체를 다 넣으면 문서당 15~40만자로 다른
  문서 대비 너무 커서, 각 소설의 초반 장(챕터) 1~2개만 발췌해 문서로 저장한다
  (자연스러운 장 구분이 있는 챕터 단위 발췌 - 임의로 문자 수로 자르지 않음).
  인간문제(강경애)는 장 구분 없는 단일 페이지(23만자)라 발췌 기준을 잡기 애매해 제외.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))
from wikitext_lib import export_pages, wikitext_events, events_to_docx
from config import DOCX_KOR_DIR

OUT_DIR = DOCX_KOR_DIR / "Wikisource_문학"

SHORT_STORIES = ["날개", "광염 소나타", "광화사", "금 따는 콩밭", "감자", "B사감과 러브레터"]
ESSAYS = ["근대소설의 승리", "국제 무역 주의의 동향", "구미 부인의 가정생활"]
NOVEL_EXCERPTS = {
    "탁류": ["탁류/제1장", "탁류/제2장"],
    "무정": ["무정/1장~20장"],
    "상록수": ["상록수/제1장", "상록수/제2장"],
    "태평천하": ["태평천하/제1장", "태평천하/제2장"],
}


def save(pages, title, out_path, display_title=None):
    wikitext = pages.get(title, "")
    if not wikitext:
        print(f"  [스킵] {title}: 원문 없음")
        return False
    events = wikitext_events(wikitext)
    events_to_docx(events, display_title or title, out_path)
    return True


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    story_dir = OUT_DIR / "단편소설"
    story_dir.mkdir(exist_ok=True)
    pages = export_pages("ko.wikisource.org", SHORT_STORIES)
    for i, t in enumerate(SHORT_STORIES, 1):
        if save(pages, t, story_dir / f"{t}.docx"):
            print(f"[단편소설 {i}/{len(SHORT_STORIES)}] {t}")

    essay_dir = OUT_DIR / "수필_논설"
    essay_dir.mkdir(exist_ok=True)
    pages = export_pages("ko.wikisource.org", ESSAYS)
    for i, t in enumerate(ESSAYS, 1):
        if save(pages, t, essay_dir / f"{t}.docx"):
            print(f"[수필/논설 {i}/{len(ESSAYS)}] {t}")

    novel_dir = OUT_DIR / "장편소설_발췌"
    novel_dir.mkdir(exist_ok=True)
    for novel, chapters in NOVEL_EXCERPTS.items():
        pages = export_pages("ko.wikisource.org", chapters)
        for t in chapters:
            chap_label = t.split("/", 1)[1] if "/" in t else "전체"
            fname = f"{novel}_{chap_label}.docx".replace("~", "-")
            if save(pages, t, novel_dir / fname, display_title=f"{novel} {chap_label}"):
                print(f"[장편소설 발췌] {t}")

    print("완료")
    print(f"저장 위치: {OUT_DIR}")


if __name__ == "__main__":
    main()
