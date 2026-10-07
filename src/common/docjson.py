"""공통 JSON(문서 1개 = JSON 1개)을 청커가 읽는 평문과 블록 위치표로 바꾼다
(docs/restructure_plan.md 4.1절).

청커는 평문 문자열 하나를 받아 text[start:end] 슬라이스로 자른다. 그래서 JSON의 블록을 이어
붙여 평문을 만들면서, 각 블록이 평문의 어느 위치에 들어갔는지(Piece)를 함께 돌려준다.
이 위치표 덕에 청크가 어느 블록(제목·표)·쪽·자막 시각에서 나왔는지 알 수 있다.

렌더 규칙
  heading(level n)        "#"*n + " " + text                      kind=heading
  paragraph/formula/code/footnote  text                           kind=para
  list_item               항목마다 별도 조각 (참고문헌 수백 개가 한 조각이 되는 것을 막고 smart가
                          항목 사이에서 자를 수 있게 한다)                 kind=para
  table                   caption이 있으면 caption 조각(para) 다음에 표 조각(table)
  cue (srt)               text                                    kind=cue
  마크다운 계열: 조각 사이 "\\n\\n" (끝에는 붙이지 않음)
  srt: 큐마다 뒤에 "\\n" (마지막 큐 뒤에도 붙음 - 기존 srt/loader.py의 SEP와 같다)
  텍스트가 비어 있는 블록은 건너뛴다. 블록 text는 저장된 그대로 쓴다(strip하지 않음).
"""
import json
from dataclasses import dataclass
from pathlib import Path

MD_SEP = "\n\n"

MD_TYPES = ("paragraph", "formula", "code", "footnote", "list_item")


@dataclass(frozen=True)
class Piece:
    """평문의 [start, end) 구간 하나. kind는 heading/para/table/cue, block_ids는 JSON 블록 id."""
    start: int
    end: int
    kind: str
    block_ids: tuple


def load_doc(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_doc(doc, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")


def _md_items(blocks):
    """마크다운 계열 블록 -> [(kind, text, block_ids)]  (평문에 들어갈 조각 목록)."""
    items = []
    for b in blocks:
        typ = b["type"]
        text = b.get("text") or ""
        if not text.strip():
            continue
        if typ == "heading":
            items.append(("heading", "#" * int(b.get("level", 1)) + " " + text, (b["id"],)))
        elif typ == "table":
            caption = (b.get("caption") or "").strip()
            if caption:
                items.append(("para", caption, (b["id"],)))
            items.append(("table", text, (b["id"],)))
        elif typ in MD_TYPES:
            items.append(("para", text, (b["id"],)))
        else:
            raise ValueError(f"블록 유형을 모르겠다: {typ!r} (id={b.get('id')})")
    return items


def render(doc):
    """doc -> (평문, [Piece]). 같은 doc이면 항상 같은 결과(결정적)."""
    blocks = doc["blocks"]
    if doc.get("fmt") == "srt":
        from srt.render import render_srt          # srt는 큐 단위 렌더(srt/render.py)

        return render_srt(doc)

    parts, pieces, pos = [], [], 0
    for k, (kind, text, ids) in enumerate(_md_items(blocks)):
        if k:
            parts.append(MD_SEP)
            pos += len(MD_SEP)
        pieces.append(Piece(pos, pos + len(text), kind, ids))
        parts.append(text)
        pos += len(text)
    return "".join(parts), pieces


def to_text(doc):
    return render(doc)[0]
