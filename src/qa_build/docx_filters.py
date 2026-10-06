"""docx_corpus 생성 질문의 자동 필터와 _meta 조립 (docs/restructure_plan.md 7.3절 3단계). LLM·임베딩을 쓰지 않는다.

입력  candidates.json(근거 구간) + 생성 결과 {cid: {question, answer, type, source, support} | {skip: 이유}}
출력  queries.json / answers.json / qrels.json (vectara _meta와 같은 모양) + rejected.json(건너뜀·탈락 이유)

필터는 어휘 통계뿐이라 특정 검색기에 편향되지 않는다(검색기에 맞춘 질문이 되는 순환을 피한다).
  support     근거 인용이 구간에 그대로 있다(공백 차이는 무시)
  grounding   정답의 내용어 중 구간에 있는 비율 >= GROUNDING_MIN (숫자 포함)
  deictic     질문에 "the document / this section / the above" 같은 지시어가 없다
  specific    질문이 정답 문서를 특정한다: 질문을 어휘(TF-IDF)로 코퍼스 문서 전체와 비교했을 때 정답 문서가 SPECIFIC_MAX_RANK위 안.
              (처음에는 "희귀 단어(DF<=3)가 있어야 한다"로 했으나, 파일럿에서 그 규칙에 걸린 질문 15개가 전부 정답 문서가 1위였다 -
              대리 지표가 의도(문서 특정성)와 어긋나서 순위 기준으로 바꿨다. 순위는 거르는 데만 쓰고 질문은 바꾸지 않는다.)
  unique      근거 인용이 **코퍼스 전체에서** 한 번만 나온다(같은 규정이 다른 절에 반복되거나, 규격서처럼 판(edition)만 다른 문서가 따로 있어
              같은 문장이 다른 문서에도 있는 경우 - 청커가 다른 사본을 찾아도 "틀린 구간"으로 채점되어 청커 비교가 왜곡된다)
  copy        질문이 구간 문장을 그대로 베끼지 않는다(질문의 연속 COPY_NGRAM 단어가 구간에 그대로 있으면 탈락).
              법령·기관 이름처럼 대문자 단어가 많은 구간(NAME_RATIO 이상)은 이름이라 베끼는 것이 정상이므로 세지 않는다
  duplicate   같은 문서의 다른 질문과 내용어 겹침(Jaccard)이 DUP_JACCARD 이상
"""
import json
import math
import re
from collections import Counter

from common.config import PROCESSED_DIR
from common.docjson import load_doc, to_text

GROUNDING_MIN = 0.6
SPECIFIC_MAX_RANK = 10                   # 정답 문서가 어휘 순위 이 값 이하여야 한다(문서 100개 중)
COPY_NGRAM = 8
DUP_JACCARD = 0.7
NAME_RATIO = 0.4
VALID_TYPES = ("extractive", "abstractive", "table")
VALID_SOURCES = ("text", "text-table", "table")

_WORD = re.compile(r"[A-Za-z0-9$%][A-Za-z0-9$%.\-']*")
_STOP = set("""a an the of to in on at by for from with and or but if as is are was were be been being it its this that these those
which who whom whose what when where why how not no do does did can could may might must shall should will would than then there
their they them he she his her you your we our us i me my also any all each other such more most some per into over under after
before between about above below up down out off only same so very too s""".split())
_DEICTIC = re.compile(r"\b(the (passage|document|text|excerpt|section above|above|following|table above|context)|this (passage|document|text|"
                      r"section|excerpt|table|paragraph|report|article)|the above|as (described|mentioned|stated) (above|earlier))\b", re.I)


def words(text):
    return [w.lower().strip(".-'") for w in _WORD.findall(text)]


def _capitalized(text):
    """words(text)와 같은 길이의 목록: 그 단어가 대문자로 시작하면 1(문장 첫 단어는 0)."""
    out = []
    for m in _WORD.finditer(text):
        w = m.group().strip(".-'")
        if w:
            out.append(int(m.group()[0].isupper() and bool(text[:m.start()].strip(" (\"'"))))
    return out


def content_words(text):
    return [w for w in words(text) if w and w not in _STOP and (len(w) > 2 or w.isdigit())]


def _norm(text):
    return re.sub(r"\s+", " ", text).strip()


class CorpusStats:
    """문서별 단어 빈도 tf[doc_id] = Counter와, 단어가 나오는 문서 수 df. 질문 -> 문서 어휘 순위를 계산한다."""

    def __init__(self, tf, texts=None):
        self.tf = tf
        self.texts = texts or {}                      # doc_id -> 공백을 한 칸으로 줄인 소문자 본문(근거 인용의 중복 검사용)
        self.df = Counter(w for c in tf.values() for w in c)
        self.n = len(tf)

    def count_in_corpus(self, quote):
        """근거 인용(공백·대소문자 무시)이 코퍼스 전체 본문에 모두 몇 번 나오는지. 본문을 모르면 1."""
        if not self.texts:
            return 1
        q = _norm(quote).lower()
        return sum(t.count(q) for t in self.texts.values())

    def rank(self, question, doc_id):
        """질문과 문서 전체를 TF-IDF 점수로 견줬을 때 doc_id의 순위(1 = 가장 잘 맞음). 동점은 문서 id 순."""
        qw = set(content_words(question))
        score = {d: sum((1 + math.log(c[w])) * math.log(self.n / self.df[w]) for w in qw if c[w] and self.df[w]) for d, c in self.tf.items()}
        order = sorted(score, key=lambda d: (-score[d], d))
        return order.index(doc_id) + 1


def corpus_stats(dataset="docx_corpus"):
    """data/processed/<dataset>/*.json 전체 -> CorpusStats."""
    tf, texts = {}, {}
    for path in sorted((PROCESSED_DIR / dataset).glob("*.json")):
        if path.name.startswith("_"):
            continue
        doc = load_doc(path)
        text = to_text(doc)
        tf[doc["doc_id"]] = Counter(content_words(text))
        texts[doc["doc_id"]] = _norm(text).lower()
    return CorpusStats(tf, texts)


def check_record(cand, gen, stats):
    """생성 레코드 하나 -> 탈락 이유 목록(비어 있으면 통과)와 지표 dict."""
    reasons, info = [], {}
    q, a, sup = gen.get("question", ""), gen.get("answer", ""), gen.get("support", "")
    if gen.get("type") not in VALID_TYPES or gen.get("source") not in VALID_SOURCES:
        reasons.append("schema")
    if not q.strip() or not a.strip():
        return ["empty"], info
    span = cand["text"]
    if _norm(sup) not in _norm(span):
        reasons.append("support")
    if stats.count_in_corpus(sup) > 1:
        reasons.append("unique")
    aw = content_words(a)
    sw = set(content_words(span))
    info["grounding"] = round(sum(w in sw for w in aw) / len(aw), 3) if aw else 0.0
    if info["grounding"] < GROUNDING_MIN:
        reasons.append("grounding")
    if _DEICTIC.search(q):
        reasons.append("deictic")
    info["doc_rank"] = stats.rank(q, cand["doc_id"])
    if info["doc_rank"] > SPECIFIC_MAX_RANK:
        reasons.append("specific")
    qw, spw = words(q), words(span)
    grams = {tuple(spw[i:i + COPY_NGRAM]) for i in range(len(spw) - COPY_NGRAM + 1)}
    caps = _capitalized(q)
    info["copied"] = any(tuple(qw[i:i + COPY_NGRAM]) in grams and sum(caps[i:i + COPY_NGRAM]) / COPY_NGRAM < NAME_RATIO
                          for i in range(len(qw) - COPY_NGRAM + 1))
    if info["copied"]:
        reasons.append("copy")
    if gen.get("source") in ("table", "text-table") and not cand["has_table"]:
        reasons.append("source_without_table")
    return reasons, info


def find_duplicates(kept):
    """kept: [(cid, doc_id, question)] -> 같은 문서에서 내용어 Jaccard >= DUP_JACCARD인 뒤쪽 질문의 cid 집합."""
    dup, seen = set(), {}
    for cid, doc_id, q in kept:
        ws = set(content_words(q))
        for other, ows in seen.get(doc_id, []):
            if ws and ows and len(ws & ows) / len(ws | ows) >= DUP_JACCARD:
                dup.add(cid)
                break
        else:
            seen.setdefault(doc_id, []).append((cid, ws))
    return dup


def assemble(cands, generated, stats, out_dir):
    """통과한 질문으로 queries/answers/qrels.json과 rejected.json을 out_dir에 쓴다. -> 집계 dict."""
    from pathlib import Path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rejected, passed, infos = {}, [], {}
    for cid in sorted(generated):
        gen = generated[cid]
        if cid not in cands:
            rejected[cid] = ["unknown_candidate"]
        elif "skip" in gen:
            rejected[cid] = ["skip: " + gen["skip"]]
        else:
            reasons, infos[cid] = check_record(cands[cid], gen, stats)
            if reasons:
                rejected[cid] = reasons
            else:
                passed.append(cid)
    dup = find_duplicates([(c, cands[c]["doc_id"], generated[c]["question"]) for c in passed])
    for c in sorted(dup):
        rejected[c] = ["duplicate"]
    passed = [c for c in passed if c not in dup]
    queries, answers, qrels = {}, {}, {}
    for k, cid in enumerate(passed):
        c, g = cands[cid], generated[cid]
        qid = f"dq-{k:04d}"
        queries[qid] = {"query": g["question"], "type": g["type"], "source": g["source"], "topic": c["topic"], "doc_type": c["type"]}
        answers[qid] = g["answer"]
        qrels[qid] = {"doc_id": c["doc_id"], "cid": cid, "start": c["start"], "end": c["end"], "block_ids": c["block_ids"],
                      "n_tokens": c["n_tokens"], "support": g["support"], "text": c["text"]}
    for name, obj in (("queries", queries), ("answers", answers), ("qrels", qrels), ("rejected", rejected)):
        (out_dir / f"{name}.json").write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"generated": len(generated), "passed": len(queries), "rejected": len(rejected),
            "reasons": _count([r.split(":")[0] for v in rejected.values() for r in v]), "infos": infos}


def _count(items):
    out = {}
    for x in items:
        out[x] = out.get(x, 0) + 1
    return dict(sorted(out.items(), key=lambda t: -t[1]))
