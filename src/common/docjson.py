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
SRT_SEP = "\n"

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
        parts, pieces, pos = [], [], 0
        for b in blocks:
            if b["type"] != "cue":
                raise ValueError(f"srt 문서에 cue가 아닌 블록이 있다: {b['type']!r} (id={b.get('id')})")
            text = b.get("text") or ""
            if not text:
                continue
            pieces.append(Piece(pos, pos + len(text), "cue", (b["id"],)))
            parts.append(text + SRT_SEP)
            pos += len(text) + len(SRT_SEP)
        return "".join(parts), pieces

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


def block_spans(doc):
    """조각마다 (start, end, kind) 목록. 옛 방식 legacy_smart.SmartChunker(srt 설정)가 입력으로 받는 모양이다.
    새 청커(semantic·smart)는 render()의 Piece 목록과 units.build_units를 쓴다."""
    return [(p.start, p.end, p.kind) for p in render(doc)[1]]


def to_lc_document(doc):
    """srt용 LangChain Document. 기존 mapper/timestamp_align이 읽는 metadata["units"] 형식 그대로:
    units[j] = {"start", "end", "kind": "cue", "meta": {"index", "t_start", "t_end"}}."""
    if doc.get("fmt") != "srt":
        raise ValueError("to_lc_document는 srt 문서에만 쓴다")
    from langchain_core.documents import Document

    text, pieces = render(doc)
    by_id = {b["id"]: b for b in doc["blocks"]}
    units = []
    for p in pieces:
        b = by_id[p.block_ids[0]]
        units.append({"start": p.start, "end": p.end, "kind": "cue",
                      "meta": {"index": b.get("index"), "t_start": b["start"], "t_end": b["end"]}})
    return Document(page_content=text,
                    metadata={"name": f"{doc['doc_id']}.srt", "fmt": "srt", "units": units})
