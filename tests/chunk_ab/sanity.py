"""고정 분할 검색 점수가 높게 나온 원인 점검(tests 전용).

  .venv311/bin/python tests/chunk_ab/sanity.py

A. 질문-정답 겹침: 질문의 내용어(4글자 이상, 불용어 제외)가 정답 절 / 정답 문서 / 다른 문서에 얼마나 있나
B. 낱말 검색 기준선: TF-IDF(임베딩 없음)로 문서 전체를 검색했을 때 문서 적중률@1·@5
C. 절 크기: 정답 절이 문서에서 차지하는 비율, 문서당 절 수, 절 하나가 문서 대부분인 경우
D. 무작위 기준선: 정답 문서의 청크 중 아무거나 5개를 골랐을 때 절 적중 기대값(판정은 fixed300.py와 같음)
출력: tests/chunk_ab/out/fixed300/sanity.md
"""
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fixed300 import EVAL, META, grams, no_math, toks  # noqa: E402

OUT = HERE / "out" / "fixed300"
STOP = set("what which when where with that this from have does their there they these those into about between under over then than also used using based how why who whom whose are was were been being the and for can could would should will shall may might must its it's our your".split())


def content(q):
    return [w for w in toks(q) if len(w) >= 4 and w not in STOP]


def main():
    ev = [json.loads(x) for x in EVAL.read_text(encoding="utf-8").splitlines() if x.strip()]
    docs = sorted(r["doc_id"] for r in ev if not r["ref_incomplete"])
    qs = json.loads((META / "queries.json").read_text(encoding="utf-8"))
    qrels = json.loads((META / "qrels.json").read_text(encoding="utf-8"))
    dset = set(docs)
    qids = [q for q in qs if qrels[q]["doc_id"] in dset]
    secs, full = {}, {}
    for d in docs:
        c = json.loads((META / "corpus" / f"{d}.json").read_text(encoding="utf-8"))
        secs[d] = [s["text"] for s in c["sections"]]
        full[d] = "\n\n".join(secs[d])
    lines = ["# 고정 분할 점수 점검", ""]

    # A 질문-정답 겹침
    df = Counter()
    vocab = {d: set(toks(full[d])) for d in docs}
    for d in docs:
        df.update(vocab[d])
    in_sec, in_doc, rare_in_doc, uniq = [], [], [], []
    for q in qids:
        d, sid = qrels[q]["doc_id"], qrels[q]["section_id"]
        ws = content(qs[q]["query"])
        if not ws:
            continue
        sw = set(toks(secs[d][sid])) if sid < len(secs[d]) else set()
        in_sec.append(sum(w in sw for w in ws) / len(ws))
        in_doc.append(sum(w in vocab[d] for w in ws) / len(ws))
        rare = [w for w in ws if df[w] <= 5]          # 984편 중 5편 이하에만 나오는 드문 낱말
        rare_in_doc.append(len([w for w in rare if w in vocab[d]]))
        uniq.append(any(df[w] == 1 and w in vocab[d] for w in ws))
    lines += ["## A. 질문과 정답의 낱말 겹침", "",
              f"- 질문 내용어 중 정답 절에 있는 비율: 평균 {np.mean(in_sec):.2f}, 중앙 {np.median(in_sec):.2f}",
              f"- 질문 내용어 중 정답 문서에 있는 비율: 평균 {np.mean(in_doc):.2f}",
              f"- 질문에 '984편 중 5편 이하에만 나오는 드문 낱말'이 정답 문서와 겹치는 질문: {np.mean([x > 0 for x in rare_in_doc]):.1%}",
              f"- 질문에 '정답 문서에만 있는 낱말'(984편 중 1편)이 있는 질문: {np.mean(uniq):.1%}",
              f"- 질문 유형: {dict(Counter(qs[q]['type'] for q in qids))}", ""]

    # B TF-IDF 문서 검색
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(sublinear_tf=True, stop_words="english", min_df=1)
    X = vec.fit_transform([full[d] for d in docs])
    Q = vec.transform([qs[q]["query"] for q in qids])
    S = (Q @ X.T).toarray()
    rank = np.argsort(-S, axis=1)[:, :5]
    gold = np.array([docs.index(qrels[q]["doc_id"]) for q in qids])
    h1 = np.mean(rank[:, 0] == gold)
    h5 = np.mean([gold[i] in rank[i] for i in range(len(qids))])
    lines += ["## B. 낱말 검색 기준선(TF-IDF, 임베딩 없이 문서 전체 검색)", "",
              f"- 문서 적중률@1 {h1:.3f}, @5 {h5:.3f}  (bge-m3 + 300자 청크: @1 0.907, @5 0.977)", ""]

    # C 절 크기
    frac, nsec, big = [], [], 0
    for q in qids:
        d, sid = qrels[q]["doc_id"], qrels[q]["section_id"]
        tot = sum(len(s.split()) for s in secs[d])
        f = len(secs[d][sid].split()) / max(1, tot) if sid < len(secs[d]) else 0
        frac.append(f)
        nsec.append(len(secs[d]))
        big += f >= 0.5
    lines += ["## C. 정답 절의 크기", "",
              f"- 정답 절이 문서에서 차지하는 낱말 비율: 평균 {np.mean(frac):.2f}, 중앙 {np.median(frac):.2f}, 90% 분위 {np.percentile(frac, 90):.2f}",
              f"- 정답 절이 문서의 절반 이상인 질문: {big}개 ({big / len(qids):.1%})",
              f"- 문서당 절 수(질문 기준): 중앙 {np.median(nsec):.0f}, 3개 이하 {sum(n <= 3 for n in nsec)}개", ""]

    # D 무작위 기준선(정답 문서 안에서 아무 청크 5개)
    for name in ("vectara", "ours"):
        by_doc = defaultdict(list)
        for x in (OUT / f"chunks_{name}.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(x)
            by_doc[r["doc_id"]].append(r["text"])
        exp_old, exp_nm, cache = [], [], {}
        for q in qids:
            d, sid = qrels[q]["doc_id"], qrels[q]["section_id"]
            if sid >= len(secs[d]):
                continue
            gg, gm = grams(toks(secs[d][sid])), grams(toks(no_math(secs[d][sid])))
            hits_old = hits_nm = 0
            for t in by_doc[d]:
                g = grams(toks(t))
                hits_old += bool(g) and len(g & gg) / len(g) >= 0.5
                g2 = grams(toks(no_math(t)))
                hits_nm += len(g2) >= 3 and len(g2 & gm) / len(g2) >= 0.5
            n = len(by_doc[d])
            exp_old.append(1 - np.prod([(n - hits_old - i) / (n - i) for i in range(min(5, n))]) if n else 0)
            exp_nm.append(1 - np.prod([(n - hits_nm - i) / (n - i) for i in range(min(5, n))]) if n else 0)
        lines += [f"## D. 무작위 기준선 — {name}: 정답 문서 청크 중 아무거나 5개", "",
                  f"- 절 적중률@5 기대값: 옛 판정 {np.mean(exp_old):.3f}, 수식 뺀 판정 {np.mean(exp_nm):.3f}", ""]
    (OUT / "sanity.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
