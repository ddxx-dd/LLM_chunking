import re
from pathlib import Path

MARK_RE = re.compile(r"\[(\d+)\]\s*(.*?)(?=\s*\[\d+\]|\Z)", re.S)

def fragments(doc, chunk):
    """doc: Document(metadata["units"]=유닛 딕셔너리 리스트).
    chunk: Document(metadata["start_index"]=오프셋, TextSplitter가 채워줌)."""
    c_start = chunk.metadata["start_index"]
    c_end = c_start + len(chunk.page_content)
    out = []
    for j, u in enumerate(doc.metadata["units"]):
        s = max(u["start"], c_start)
        e = min(u["end"], c_end)
        if s < e:
            out.append((j, s, e, s == u["start"] and e == u["end"]))
    return out

def build_prompt(doc, chunk, instruction="Translate each line to English. Keep the [n] numbers exactly:\n{body}"):
    frs = fragments(doc, chunk)
    body = "\n".join("[{}] {}".format(n + 1, doc.page_content[s:e]) for n, (j, s, e, w) in enumerate(frs))
    return frs, instruction.format(body=body)

def parse_marked(raw_llm_output, frs):
    got = {int(m.group(1)): m.group(2).strip() for m in MARK_RE.finditer(raw_llm_output)}
    if not got:
        lines = [line.strip() for line in raw_llm_output.strip().split("\n") if line.strip()]
        got = {i + 1: line for i, line in enumerate(lines)}
    return [(j, got.get(n + 1, "")) for n, (j, s, e, w) in enumerate(frs)]

def merge_to_units(doc, all_translated_pieces):
    buckets = [[] for _ in doc.metadata["units"]]
    for j, text in all_translated_pieces:
        if text:
            buckets[j].append(text)
    return [" ".join(b).strip() for b in buckets]

def _fmt(sec):
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return "%02d:%02d:%02d,%03d" % (h, m, int(s), round((s - int(s)) * 1000))

def write_srt(doc, unit_texts, out_path):
    units = doc.metadata["units"]
    assert len(unit_texts) == len(units), "유닛 개수 불일치"
    blocks = []
    for i, (u, txt) in enumerate(zip(units, unit_texts), 1):
        t_start = _fmt(u["meta"]["t_start"])
        t_end = _fmt(u["meta"]["t_end"])
        blocks.append("{}\n{} --> {}\n{}\n".format(i, t_start, t_end, txt))
    Path(out_path).write_text("\n".join(blocks), encoding="utf-8")
    return out_path


def covered_units(chunk):
    """fragments()와 같은 겹침 계산이지만 부모 doc 인자 없이 chunk 하나만으로 가능 -
    split_documents()가 부모 metadata(units 포함)를 청크에 전부 복사해두기 때문.
    검색으로 낱개 청크만 들고 있는 상황(docx_compose 등)에서 쓴다."""
    c_start = chunk.metadata["start_index"]
    c_end = c_start + len(chunk.page_content)
    out = []
    for u in chunk.metadata["units"]:
        s, e = max(u["start"], c_start), min(u["end"], c_end)
        if s < e:
            out.append(u)
    return out


def chunk_bundle(docs, splitter):
    """여러 docx 문서 묶음을 문서 경계를 넘지 않고 각각 청킹한다.
    docs: Document 리스트. splitter: TextSplitter 인스턴스(split_documents 사용 -
    부모 metadata가 각 청크에 자동으로 복사되어 doc_name 등을 그대로 들고 감).
    반환: [{"doc_idx":i, "doc_name":doc.metadata["name"], "chunk":Document}, ...]"""
    out = []
    for doc_idx, doc in enumerate(docs):
        chunks = splitter.split_documents([doc])
        for c in chunks:
            out.append({"doc_idx": doc_idx, "doc_name": doc.metadata["name"], "chunk": c})
    return out
