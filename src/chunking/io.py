"""청크 파일 입출력: 데이터셋 하나를 청커 하나로 청킹해 data/chunks/<dataset>/<태그>.jsonl 로 저장한다.

- 청크 한 줄 = dict 하나(chunk_id, doc_id, dataset, chunker, params, start, end, text, n_chars, n_tokens, block_ids, partial_start/end, page_*, heading_path).
  옛 파일(chunking/v1)과 같은 모양이라 indexing/index.py, tasks/retrieval.py가 그대로 읽는다. 청크 텍스트는 항상 평문의 text[start:end].
- 태그: `<이름>__<조절값>[-추가값]` (예: `A__p90`, `fixed__T260`, `B__p90-beta0`). 정확한 값은 청크 줄의 params가 기준이다.
- 파일이 있으면 건너뛴다(이어하기, force로 다시). 임시 파일에 쓴 뒤 이름을 바꾼다. 옆에 <태그>.meta.json(설정, 청크 수, 내용 보존 검사 실패 문서).
- 내용 보존 검사(겹침 0): 청크들의 공백 아닌 글자를 이어 붙인 것 == 원문의 공백 아닌 글자. 어긋난 문서는 meta에 기록하고 계속한다.
- 글자수 청커의 T(목표 토큰)는 manifests/<dataset>.ratio.json의 글자/토큰 비율로 글자수로 바꾼다(measure_ratio로 측정).
"""
import datetime
import json
import os
import re
import time
from bisect import bisect_left, bisect_right
from collections import Counter

import numpy as np

from chunking.base import REGISTRY, DocContext, chunk_document
from common.config import CHUNKS_DIR, MANIFEST_DIR, PROCESSED_DIR
from common.docjson import load_doc, to_text

_END_RE = re.compile(r"[.!?…。\"')\]”’」』]$")


# ---------------------------------------------------------------------------
# 태그
# ---------------------------------------------------------------------------
def _fmt(v):
    return f"{v:g}" if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)


def tag_for(name, params):
    knob = REGISTRY[name].knob
    p = {k: v for k, v in params.items() if v is not None}
    if "size_chars" in p:
        head = f"{name}__c{_fmt(p.pop('size_chars'))}"
    elif knob in p:
        head = f"{name}__{knob}{_fmt(p.pop(knob))}"
    else:
        head = f"{name}__none"                                 # 예: 재분할 없는 header
    return head + "".join(f"-{k}{_fmt(v)}" for k, v in sorted(p.items()))


# ---------------------------------------------------------------------------
# 글자/토큰 비율
# ---------------------------------------------------------------------------
def ratio_path(dataset):
    return MANIFEST_DIR / f"{dataset}.ratio.json"


def load_ratio(dataset, lang):
    p = ratio_path(dataset)
    if not p.exists():
        raise FileNotFoundError(f"{p} 가 없다: 글자/토큰 비율을 먼저 측정할 것 (chunking.io.measure_ratio)")
    return json.loads(p.read_text(encoding="utf-8"))["by_lang"][lang]["chars_per_token"]


def measure_ratio(dataset, count_many, doc_ids=None):
    """chars_per_token = Σ글자 / Σ토큰 (문서별 비율의 평균이 아니라 합계 비율). 전처리된 JSON의 평문으로 잰다."""
    paths = sorted(p for p in (PROCESSED_DIR / dataset).glob("*.json") if p.name != "_report.json")
    by_lang = {}
    for p in paths:
        doc = load_doc(p)
        if doc_ids is not None and doc["doc_id"] not in doc_ids:
            continue
        text = to_text(doc)
        rec = by_lang.setdefault(doc["lang"], [0, 0, 0])      # [글자, 토큰, 문서 수]
        rec[0] += len(text)
        rec[1] += count_many([text])[0]
        rec[2] += 1
    return {"dataset": dataset, "measured_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "by_lang": {lang: {"chars_per_token": round(c / t, 4), "n_docs": n, "total_chars": c, "total_tokens": t}
                        for lang, (c, t, n) in sorted(by_lang.items())}}


def save_ratio(result):
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    ratio_path(result["dataset"]).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------------------------------------------------------------------------
# 청크 dict 만들기, 통계
# ---------------------------------------------------------------------------
def attach_meta(doc, text, pieces, spans, count_many, chunker, params):
    """spans [(start, end)] -> 청크 dict 목록. 조각표와 구간의 겹침을 이진 탐색으로 구해 block_ids·쪽·제목 경로를 채운다."""
    by_id = {b["id"]: b for b in doc["blocks"]}
    starts, ends = [p.start for p in pieces], [p.end for p in pieces]
    ntok = count_many([text[s:e] for s, e in spans]) if spans else []
    out = []
    for i, (s, e) in enumerate(spans):
        first, last = bisect_right(ends, s), bisect_left(starts, e) - 1
        hit = pieces[first:last + 1] if first <= last else []
        block_ids = []
        for p in hit:
            for bid in p.block_ids:
                if bid not in block_ids:
                    block_ids.append(bid)
        blocks = [by_id[b] for b in block_ids]
        pages = [b["page"] for b in blocks if b.get("page") is not None]
        out.append({
            "chunk_id": f"{doc['doc_id']}#{i:04d}", "doc_id": doc["doc_id"], "dataset": doc["dataset"], "chunker": chunker, "params": params,
            "start": s, "end": e, "text": text[s:e], "n_chars": e - s, "n_tokens": int(ntok[i]), "block_ids": block_ids,
            "partial_start": bool(hit) and s > hit[0].start, "partial_end": bool(hit) and e < hit[-1].end,
            "page_start": min(pages) if pages else None, "page_end": max(pages) if pages else None, "t_start": None, "t_end": None,
            "heading_path": list(blocks[0].get("heading_path") or []) if blocks else []})
    return out


def coverage_ok(text, spans):
    """청크들의 공백 아닌 글자를 이어 붙인 것이 원문의 공백 아닌 글자와 같은가 (겹침 0일 때만 의미가 있다)."""
    return re.sub(r"\s+", "", "".join(text[s:e] for s, e in spans)) == re.sub(r"\s+", "", text)


def compute_stats(chunks, docs_by_id):
    """청크 크기 분포와 구조 훼손 지표(pdf·docx). 크기: n_tokens(bge-m3)."""
    n = len(chunks)
    if n == 0:
        return {"n_chunks": 0}
    toks = np.array([c["n_tokens"] for c in chunks])
    out = {"n_chunks": n, "tokens_mean": round(float(toks.mean()), 1), "tokens_p50": float(np.percentile(toks, 50)),
           "tokens_p95": float(np.percentile(toks, 95)), "tokens_min": int(toks.min()), "tokens_max": int(toks.max()),
           "partial_start_rate": round(sum(c["partial_start"] for c in chunks) / n, 4),
           "partial_end_rate": round(sum(c["partial_end"] for c in chunks) / n, 4),
           "sentence_cut_rate": round(sum(1 for c in chunks if c["partial_end"] and not _END_RE.search(c["text"].rstrip())) / n, 4)}
    docs = {c["doc_id"] for c in chunks}
    types = {(d, b["id"]): b["type"] for d in docs for b in docs_by_id[d]["blocks"]}
    seen = Counter((c["doc_id"], b) for c in chunks for b in c["block_ids"])
    tables = [k for k, t in types.items() if t == "table"]
    out["table_split_rate"] = round(sum(1 for k in tables if seen[k] > 1) / len(tables), 4) if tables else None
    out["n_tables"] = len(tables)
    out["orphan_heading_rate"] = round(sum(1 for c in chunks if c["block_ids"] and types.get((c["doc_id"], c["block_ids"][-1])) == "heading") / n, 4)
    return out


# ---------------------------------------------------------------------------
# 청킹 실행
# ---------------------------------------------------------------------------
def chunk_documents(docs, name, params, runtime, chars_per_token=None, log=None):
    """문서 목록 -> (청크 dict 목록, 요약). 파일 입출력 없음(테스트·스크립트 공용).
    chars_per_token: 값 하나 또는 함수(dataset, lang) -> 값 (T로 글자수를 정하는 청커에만 필요)."""
    records, bad = [], []
    for doc in docs:
        cpt = chars_per_token(doc["dataset"], doc["lang"]) if callable(chars_per_token) else chars_per_token
        ctx = DocContext(doc, runtime, cpt)
        spans = chunk_document(name, ctx, **params)
        used = dict(params)
        if params.get("T") is not None:
            used["size_chars"] = ctx.size_chars(params["T"])        # 실제로 쓴 글자수를 기록
        chunks = attach_meta(doc, ctx.text, ctx.pieces, spans, runtime.count_many, name, {**used, "lang": doc["lang"]})
        if coverage_ok(ctx.text, spans) is False:
            bad.append(doc["doc_id"])
        records += chunks
        if log:
            log(f"  {doc['doc_id'][:44]:44} 청크 {len(chunks):>4}")
    return records, {"coverage_failures": bad}


def chunk_dataset(dataset, name, params, runtime, doc_ids=None, limit=None, force=False, out_dir=None, log=print):
    from preprocess.datasets import load_entries

    tag = tag_for(name, params)
    out_dir = out_dir or (CHUNKS_DIR / dataset)
    path = out_dir / f"{tag}.jsonl"
    if path.exists() and not force:
        log(f"[{dataset}/{tag}] 이미 있음, 건너뜀 (force로 다시 만들기)")
        return {"path": path, "skipped": True}
    ids = [e["doc_id"] for e in load_entries(dataset)]
    if doc_ids:
        ids = [i for i in ids if i in set(doc_ids)]
    if limit:
        ids = ids[:limit]
    log(f"[{dataset}/{tag}] 문서 {len(ids)}개, 설정 {params}")
    t0 = time.time()
    docs = (load_doc(PROCESSED_DIR / dataset / f"{i}.json") for i in ids)
    cpt = load_ratio if params.get("T") is not None else None
    records, summary = chunk_documents(docs, name, params, runtime, cpt, log=log)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    meta = {"dataset": dataset, "chunker": name, "tag": tag, "params": params, "n_docs": len(ids), "n_chunks": len(records),
            "seconds": round(time.time() - t0, 1), "created_at": datetime.datetime.now().isoformat(timespec="seconds"), **summary}
    (out_dir / f"{tag}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"[{dataset}/{tag}] 청크 {len(records)}개 → {path} | 내용 보존 검사 실패 문서 {len(summary['coverage_failures'])}개 | {meta['seconds']}초")
    return {"path": path, "skipped": False, **meta}
