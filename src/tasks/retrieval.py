"""검색 평가 지표 (docs/restructure_plan.md 6.2절). LLM 생성 없음 - 순수 검색 단계만 채점한다.

k는 3으로 고정해서 쓴다(기본값 K_DEFAULT, 2026-10-06 결정).
기존 지표(문서 단위): hit@3, MRR, F1
  hit@3   top-k 청크 중 정답 *문서*의 청크가 하나라도 있으면 1
  MRR     top-k에서 정답 문서 청크의 첫 순위의 역수 (없으면 0)
  F1      top-k 중 정답 문서 청크들과 정답(answer)의 SQuAD 단어 F1 최댓값 (정규화: 소문자·구두점 제거·공백 정리)
추가 지표(근거 단위):
  section_hit@3      (vectara)     정답 문서의 청크 중 '청크 단어 가운데 정답 섹션 텍스트에 들어 있는 비율' >= THRESH 인 것이 있으면 1
  evidence_recall@3  (docx_corpus) 정답 문서의 top-k 청크를 합친 글자 범위가 근거 구간 [s, e)을 덮는 비율(0~1)
  page_hit@3         (allganize)   정답 문서의 청크 중 쪽 번호 집합에 gold_page가 들어 있는 것이 있으면 1
                                   쪽 번호 집합은 청크 레코드의 `pages`(블록들의 쪽 번호). 옛 청크 파일에는 없어서 [page_start, page_end] 범위로 대신한다.
                                   (2026-10-08 결정: 쪽 번호는 블록 메타데이터에 있으므로 쓴다. 각주는 문서 끝에 모이지만 자기 원래 쪽을 유지하고,
                                    집합을 쓰면 끝쪽 청크의 쪽이 부풀려지지 않는다. 여러 쪽에 걸친 청크가 유리한 크기 편향은 남아 있다 - 나중에 다룬다)
크기 효과 해석용: 인덱스 청크의 평균 토큰 수, 질문별 top-k 총 토큰 수의 평균.
"""
import re
from collections import Counter, defaultdict

SECTION_THRESH = 0.5
K_DEFAULT = 3
PRIMARY = {"vectara": "section_hit", "docx_corpus": "evidence_recall"}      # 실험 1(옛 스크립트)용: allganize는 대표 지표가 없다 - 바꾸지 말 것
PRIMARY_V2 = {**PRIMARY, "allganize": "page_hit"}                          # 새 실행(scripts/run.py)용


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


def chunk_pages(chunk):
    """청크가 걸친 쪽 번호 집합. `pages`(새 청크 파일)가 있으면 그것, 없으면 [page_start, page_end] 범위, 쪽 정보가 없으면 빈 집합."""
    if chunk.get("pages") is not None:
        return set(chunk["pages"])
    p0, p1 = chunk.get("page_start"), chunk.get("page_end")
    return set(range(p0, p1 + 1)) if p0 is not None and p1 is not None else set()


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
    if qa.get("gold_page") is not None:
        row["page_hit"] = any(qa["gold_page"] in chunk_pages(c) for _, c in gold)
    return row


def aggregate(rows):
    n = len(rows)
    if n == 0:
        return None
    out = {"n": n, "hit@k": sum(r["hit"] for r in rows) / n, "mrr": sum(r["mrr"] for r in rows) / n,
           "f1": sum(r["f1"] for r in rows) / n, "top_k_tokens_mean": sum(r["top_k_tokens"] for r in rows) / n}
    for key in ("section_hit", "evidence_recall", "page_hit"):
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
