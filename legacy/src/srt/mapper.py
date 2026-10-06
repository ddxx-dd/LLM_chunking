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
