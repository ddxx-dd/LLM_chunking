"""청크 파일의 통계와 내용 보존 검사 (docs/restructure_plan.md 5.5절, 1차 구현 지표).

  n_chunks, tokens_{mean,p50,p95,max}   크기 분포 (n_tokens = 데이터셋 토큰 카운터)
  over_max_rate, under_min_rate         max 초과 / min 미만 청크 비율 (min·max를 주었을 때만)
  partial_start_rate, partial_end_rate  블록 중간에서 시작/끝나는 청크 비율
  sentence_cut_rate                     partial_end이면서 끝 글자가 문장 종결 부호가 아닌 청크 비율 (pdf/docx)
  table_split_rate                      둘 이상의 청크에 걸친 표의 비율 (표 단위, pdf/docx)
  orphan_heading_rate                   마지막 블록이 제목인 청크 비율 (pdf/docx)
  cues_split_ratio                      둘 이상의 청크에 걸친 큐의 비율 (srt)
sentence_cut_rate의 한계: "et al.", "Fig. 3", "e.g.", 소수점 뒤에서 끝나면 종결 부호로 잘못 판정한다(근사값).
"""
import re
from collections import Counter

import numpy as np

_END_RE = re.compile(r"[.!?…。\"')\]”’」』]$")


def coverage_ok(text, spans):
    """청크들의 공백 아닌 글자를 이어 붙인 것이 원문의 공백 아닌 글자와 같은가 (overlap 0일 때만 의미가 있다)."""
    kept = re.sub(r"\s+", "", "".join(text[s:e] for s, e in spans))
    return kept == re.sub(r"\s+", "", text)


def compute_stats(chunks, docs_by_id, min_tokens=None, max_tokens=None):
    n = len(chunks)
    if n == 0:
        return {"n_chunks": 0}
    toks = np.array([c["n_tokens"] for c in chunks])
    out = {"n_chunks": n, "tokens_mean": round(float(toks.mean()), 1), "tokens_p50": float(np.percentile(toks, 50)),
           "tokens_p95": float(np.percentile(toks, 95)), "tokens_max": int(toks.max()),
           "partial_start_rate": round(sum(c["partial_start"] for c in chunks) / n, 4),
           "partial_end_rate": round(sum(c["partial_end"] for c in chunks) / n, 4)}
    if max_tokens is not None:
        out["over_max_rate"] = round(float((toks > max_tokens).mean()), 4)
    if min_tokens is not None:
        out["under_min_rate"] = round(float((toks < min_tokens).mean()), 4)

    fmts = {docs_by_id[c["doc_id"]]["fmt"] for c in chunks}
    if fmts == {"srt"}:
        seen, total = Counter(), 0
        for c in chunks:
            for b in c["block_ids"]:
                seen[(c["doc_id"], b)] += 1
        for doc in {c["doc_id"] for c in chunks}:
            total += sum(1 for b in docs_by_id[doc]["blocks"] if b["type"] == "cue")
        out["cues_split_ratio"] = round(sum(1 for v in seen.values() if v > 1) / max(total, 1), 4)
        return out

    out["sentence_cut_rate"] = round(sum(1 for c in chunks if c["partial_end"] and not _END_RE.search(c["text"].rstrip())) / n, 4)
    types = {(d, b["id"]): b["type"] for d, doc in docs_by_id.items() for b in doc["blocks"]}
    seen = Counter()
    for c in chunks:
        for b in c["block_ids"]:
            seen[(c["doc_id"], b)] += 1
    tables = [k for k, t in types.items() if t == "table" and k[0] in {c["doc_id"] for c in chunks}]
    out["table_split_rate"] = round(sum(1 for k in tables if seen[k] > 1) / len(tables), 4) if tables else None
    out["n_tables"] = len(tables)
    out["orphan_heading_rate"] = round(sum(1 for c in chunks if c["block_ids"] and types.get((c["doc_id"], c["block_ids"][-1])) == "heading") / n, 4)
    return out
