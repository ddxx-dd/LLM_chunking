"""srt(자막) 렌더 - 공통 JSON(cue 블록) -> 평문 + 조각표. 공통 렌더(common.docjson.render)는 fmt가 srt이면 여기로 위임한다.

  cue (srt)  text    kind=cue,  큐마다 뒤에 "\\n" (마지막 큐 뒤에도 붙음 - 기존 srt/loader.py의 SEP와 같다)
block_spans: 옛 smart(srt/smart.py)가 입력으로 받는 (start, end, kind) 목록. to_lc_document: 기존 번역 단계용 LangChain Document.
"""
from common.docjson import Piece, render

SRT_SEP = "\n"


def render_srt(doc):
    """srt doc -> (평문, [Piece]). 같은 doc이면 항상 같은 결과(결정적)."""
    blocks = doc["blocks"]
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


def block_spans(doc):
    """조각마다 (start, end, kind) 목록. 옛 smart(srt/smart.py의 SmartChunker)가 입력으로 받는 모양이다.
    pdf·docx 청커는 render()의 Piece 목록을 직접 쓴다."""
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
