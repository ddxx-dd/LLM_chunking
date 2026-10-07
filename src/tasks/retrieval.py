"""검색 평가 지표 (docs/restructure_plan.md 6.2절). LLM 생성 없음 - 순수 검색 단계만 채점한다.

k는 3으로 고정해서 쓴다(기본값 K_DEFAULT, 2026-10-06 결정).
기존 지표(문서 단위): hit@3, MRR, F1
  hit@3   top-k 청크 중 정답 *문서*의 청크가 하나라도 있으면 1
  MRR     top-k에서 정답 문서 청크의 첫 순위의 역수 (없으면 0)
  F1      top-k 중 정답 문서 청크들과 정답(answer)의 SQuAD 단어 F1 최댓값 (정규화: 소문자·구두점 제거·공백 정리)
추가 지표(근거 단위):
  section_hit@3      (vectara)     정답 문서의 청크 중 '청크 단어 가운데 정답 섹션 텍스트에 들어 있는 비율' >= THRESH 인 것이 있으면 1
  evidence_recall@3  (docx_corpus) 정답 문서의 top-k 청크를 합친 글자 범위가 근거 구간 [s, e)을 덮는 비율(0~1)
  (allganize의 page_hit은 측정하지 않는다: 여러 쪽에 걸친 큰 청크가 유리하고 각주가 문서 끝에 모여 쪽 번호와 어긋나기 때문 - 결정 2026-10-06)
크기 효과 해석용: 인덱스 청크의 평균 토큰 수, 질문별 top-k 총 토큰 수의 평균.
"""
import re
from collections import Counter, defaultdict

SECTION_THRESH = 0.5
K_DEFAULT = 3
PRIMARY = {"vectara": "section_hit", "docx_corpus": "evidence_recall"}      # allganize는 근거 단위 지표가 없어 hit·MRR·F1만 본다


def span_coverage(chunks, start, end):
    """청크들의 [start, end) 합집합이 근거 구간 [start, end)을 덮는 비율(0~1). 겹치는 청크는 한 번만 센다."""
    if end <= start:
        return 0.0
    covered, cur = 0, start
    for s, e in sorted((max(c["start"], start), min(c["end"], end)) for c in chunks if c["start"] < end and c["end"] > start):
        s = max(s, cur)
        if e > s:
            covered += e - s
            cur = e
    return covered / (end - start)


def normalize(text):
    text = re.sub(r"[^\w\s]", "", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def f1_score(answer, chunk_text):
    gold, pred = normalize(answer).split(), normalize(chunk_text).split()
    if not gold or not pred:
        return 0.0
    common = sum((Counter(gold) & Counter(pred)).values())
    if common == 0:
        return 0.0
    p, r = common / len(pred), common / len(gold)
    return 2 * p * r / (p + r)


def containment(chunk_text, section_text, n=1):
    """청크의 단어 n-gram 중 정답 섹션 텍스트에 들어 있는 비율 (n=1이 계획서 정의)."""
    def grams(t):
        w = normalize(t).split()
        return [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    chunk, sec = grams(chunk_text), set(grams(section_text))
    return sum(1 for g in chunk if g in sec) / len(chunk) if chunk else 0.0


def question_row(qa, hits, k, ngram=1, thresh=SECTION_THRESH):
    """hits: [(청크 dict, 점수)] 상위 k개 -> 질문 하나의 채점 결과 dict."""
    gold = [(r, c) for r, (c, _) in enumerate(hits[:k], 1) if c["doc_id"] == qa["doc_id"]]
    row = {"qid": qa["qid"], "doc_id": qa["doc_id"], "qtype": qa["qtype"],
           "hit": bool(gold), "rank": gold[0][0] if gold else None, "mrr": 1 / gold[0][0] if gold else 0.0,
           "f1": max((f1_score(qa["answer"], c["text"]) for _, c in gold), default=0.0),
           "top_k_tokens": sum(c["n_tokens"] for c, _ in hits[:k]), "top_k": [c["chunk_id"] for c, _ in hits[:k]]}
    ev = qa.get("evidence") or {}
    if ev.get("kind") == "section_text" and ev.get("text"):
        row["section_hit"] = any(containment(c["text"], ev["text"], ngram) >= thresh for _, c in gold)
    if ev.get("kind") == "span" and ev.get("end") is not None:
        row["evidence_recall"] = span_coverage([c for _, c in gold], ev["start"], ev["end"])
    return row


def aggregate(rows):
    n = len(rows)
    if n == 0:
        return None
    out = {"n": n, "hit@k": sum(r["hit"] for r in rows) / n, "mrr": sum(r["mrr"] for r in rows) / n,
           "f1": sum(r["f1"] for r in rows) / n, "top_k_tokens_mean": sum(r["top_k_tokens"] for r in rows) / n}
    for key in ("section_hit", "evidence_recall"):
        vals = [r[key] for r in rows if key in r]
        if vals:
            out[key + "@k"] = sum(vals) / len(vals)
            out[key + "_n"] = len(vals)
    return out


def evaluate(index, qas, k=K_DEFAULT, ngram=1, thresh=SECTION_THRESH):
    """index.search(질문 목록, k) -> 질문마다 [(청크, 점수)]. 결과: overall / by_qtype / 질문별 rows / 청크 통계."""
    hits = index.search([q["question"] for q in qas], k)
    rows = [question_row(q, h, k, ngram, thresh) for q, h in zip(qas, hits)]
    by = defaultdict(list)
    for r in rows:
        by[r["qtype"]].append(r)
    toks = [c["n_tokens"] for c in index.chunks]
    return {"k": k, "overall": aggregate(rows), "by_qtype": {t: aggregate(v) for t, v in sorted(by.items())},
            "index": {"n_chunks": len(toks), "chunk_tokens_mean": sum(toks) / len(toks) if toks else 0.0,
                      "chunk_tokens_max": max(toks) if toks else 0},
            "rows": rows}
