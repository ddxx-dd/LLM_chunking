"""raw/_meta 의 질문·정답 원본 -> data/qa/<dataset>.jsonl (docs/restructure_plan.md 3.3·7.1·7.2·8절).

vectara   queries/answers/qrels.json + corpus/<doc>.json 의 정답 섹션 텍스트(evidence.text). 원본 전체(문서 1000편, 질문 3045개) 중 source에 image가 든 질문은 excluded
docx_corpus raw/docx_corpus/_meta 의 queries/answers/qrels.json (qa_build/docx_candidates.py -> Claude 생성 -> docx_filters.py 로 만든 것).
          근거는 청커와 무관하게 고른 구간 [start, end) 이고(evidence.kind="span"), 전처리를 바꾸면 위치가 틀어지므로 읽을 때마다
          그 구간 텍스트가 현재 to_text()와 맞는지 검사한다(어긋나면 문제로 보고).
allganize rag_evaluation_result.csv 의 target_file_name -> documents.csv 의 orig_file_name -> doc_id,
          gold_page = target_page_no. context_type=image 질문은 excluded=True, split=None (이미지는 처리하지 않는다).

split(tune|eval)은 문서 단위로 20% / 80%, 층화(vectara: 표 질문이 있는 문서 여부, allganize: domain), 시드 42.
같은 문서의 질문은 같은 split에 들어간다(누수 방지). 제외된 질문은 split=None.
"""
import csv
import json
import random
from collections import Counter, defaultdict

from common.config import MANIFEST_DIR, QA_DIR, RAW_DIR

SEED = 42
TUNE_RATIO = 0.2


def assign_splits(doc_strata, seed=SEED, tune_ratio=TUNE_RATIO):
    """doc_strata: {doc_id: 층} -> {doc_id: 'tune'|'eval'}. 층마다 정렬 후 시드로 섞어 앞쪽 20%(반올림)를 tune으로."""
    groups = defaultdict(list)
    for doc, stratum in doc_strata.items():
        groups[stratum].append(doc)
    out = {}
    for stratum in sorted(groups, key=str):
        docs = sorted(groups[stratum])
        random.Random(f"{seed}:{stratum}").shuffle(docs)
        n_tune = round(len(docs) * tune_ratio)
        for i, d in enumerate(docs):
            out[d] = "tune" if i < n_tune else "eval"
    return out


def build_vectara(meta_dir=None):
    meta_dir = meta_dir or RAW_DIR / "vectara_pdf" / "_meta"
    queries = json.loads((meta_dir / "queries.json").read_text(encoding="utf-8"))
    answers = json.loads((meta_dir / "answers.json").read_text(encoding="utf-8"))
    qrels = json.loads((meta_dir / "qrels.json").read_text(encoding="utf-8"))
    manifest_ids = [m["paper_id"] for m in json.loads((MANIFEST_DIR / "vectara.json").read_text(encoding="utf-8"))]
    known = set(manifest_ids)

    sections = {}
    for doc in manifest_ids:
        f = meta_dir / "corpus" / f"{doc}.json"
        sections[doc] = {str(s["section_id"]): s["text"] for s in json.loads(f.read_text(encoding="utf-8"))["sections"]} if f.exists() else {}

    recs, problems = [], Counter()
    for qid, q in queries.items():
        rel = qrels.get(qid) or {}
        doc = rel.get("doc_id")
        if doc not in known:
            problems["문서가 매니페스트에 없음"] += 1
            continue
        sec_text = sections[doc].get(str(rel.get("section_id")))
        if sec_text is None:
            problems["정답 섹션 텍스트 없음"] += 1
        recs.append({"qid": qid, "dataset": "vectara", "question": q["query"], "answer": answers.get(qid, ""), "doc_id": doc,
                     "qtype": q["type"], "source": q["source"],
                     "evidence": {"kind": "section_text", "section_id": rel.get("section_id"), "text": sec_text},
                     "gold_page": None, "excluded": "image" in q["source"]})        # 이미지 근거 질문은 이미지를 처리하지 않으므로 제외(allganize와 같은 정책)
    usable = [r for r in recs if not r["excluded"]]
    problems["이미지 질문 제외"] = len(recs) - len(usable)
    strata = {d: "table" if any(r["doc_id"] == d and r["source"] != "text" for r in usable) else "text" for d in {r["doc_id"] for r in usable}}
    split = assign_splits(strata)                                                  # 문서 단위 분할: 쓸 수 있는 질문이 있는 문서만(나머지 문서는 코퍼스의 방해 문서)
    for r in recs:
        r["split"] = None if r["excluded"] else split[r["doc_id"]]
    return recs, {k: v for k, v in problems.items() if v}


def build_allganize(meta_dir=None):
    meta_dir = meta_dir or RAW_DIR / "allganize_pdf" / "_meta"
    manifest = json.loads((MANIFEST_DIR / "allganize.json").read_text(encoding="utf-8"))
    by_orig = {m["orig_file_name"]: m for m in manifest}
    assert len(by_orig) == len(manifest), "allganize orig_file_name 중복"
    rows = list(csv.DictReader(open(meta_dir / "rag_evaluation_result.csv", encoding="utf-8")))

    recs, problems = [], Counter()
    for i, r in enumerate(rows):
        m = by_orig.get(r["target_file_name"])
        if m is None:
            problems["문서가 매니페스트에 없음"] += 1
            continue
        excluded = r["context_type"] == "image"
        recs.append({"qid": f"allganize-{i:04d}", "dataset": "allganize", "question": r["question"], "answer": r["target_answer"],
                     "doc_id": m["doc_id"], "qtype": r["context_type"], "source": r["context_type"],
                     "evidence": {"kind": "none"}, "gold_page": int(r["target_page_no"]) if r["target_page_no"].strip() else None,
                     "excluded": excluded, "domain": m["domain"]})
    # 층화·분할은 제외되지 않은 질문이 있는 문서만 대상으로 한다
    docs_domain = {r["doc_id"]: r["domain"] for r in recs if not r["excluded"]}
    split = assign_splits(docs_domain)
    for r in recs:
        r["split"] = None if r["excluded"] else split[r["doc_id"]]
    return recs, dict(problems)


def build_docx_corpus(meta_dir=None):
    from common.config import PROCESSED_DIR
    from common.docjson import load_doc, to_text

    meta_dir = meta_dir or RAW_DIR / "docx_corpus" / "_meta"
    queries = json.loads((meta_dir / "queries.json").read_text(encoding="utf-8"))
    answers = json.loads((meta_dir / "answers.json").read_text(encoding="utf-8"))
    qrels = json.loads((meta_dir / "qrels.json").read_text(encoding="utf-8"))
    known = {m["file"].rsplit(".", 1)[0]: m for m in json.loads((MANIFEST_DIR / "docx_corpus.json").read_text(encoding="utf-8"))}

    texts, recs, problems = {}, [], Counter()
    for qid, q in queries.items():
        rel = qrels.get(qid) or {}
        doc = rel.get("doc_id")
        if doc not in known:
            problems["문서가 매니페스트에 없음"] += 1
            continue
        if doc not in texts:
            texts[doc] = to_text(load_doc(PROCESSED_DIR / "docx_corpus" / f"{doc}.json"))
        if texts[doc][rel["start"]:rel["end"]] != rel["text"]:
            problems["근거 구간이 현재 전처리와 맞지 않음(stale)"] += 1
            continue
        recs.append({"qid": qid, "dataset": "docx_corpus", "question": q["query"], "answer": answers.get(qid, ""), "doc_id": doc,
                     "qtype": q["type"], "source": q["source"], "topic": q["topic"], "doc_type": q["doc_type"],
                     "evidence": {"kind": "span", "start": rel["start"], "end": rel["end"], "block_ids": rel["block_ids"],
                                  "text": rel["text"], "support": rel.get("support")},
                     "gold_page": None, "excluded": False})
    strata = {d: known[d]["topic"] for d in {r["doc_id"] for r in recs}}      # 문서 단위 분할, 주제로 층화
    split = assign_splits(strata)
    for r in recs:
        r["split"] = split[r["doc_id"]]
    return recs, dict(problems)


BUILDERS = {"vectara": build_vectara, "allganize": build_allganize, "docx_corpus": build_docx_corpus}


def build(dataset):
    """-> (레코드, 문제 집계). 파일은 쓰지 않는다."""
    if dataset not in BUILDERS:
        raise KeyError(f"QA를 만들 수 없는 데이터셋: {dataset!r} (srt는 QA 없음)")
    return BUILDERS[dataset]()


def write_qa(dataset, recs):
    QA_DIR.mkdir(parents=True, exist_ok=True)
    path = QA_DIR / f"{dataset}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def load_qa(dataset, split=None, doc_ids=None, include_excluded=False):
    """data/qa/<dataset>.jsonl -> 레코드 목록. split='tune'|'eval', doc_ids=허용 문서 집합."""
    path = QA_DIR / f"{dataset}.jsonl"
    recs = [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]   # splitlines()는 U+0085 등에서도 나눈다
    if not include_excluded:
        recs = [r for r in recs if not r["excluded"]]
    if split:
        recs = [r for r in recs if r["split"] == split]
    if doc_ids is not None:
        keep = set(doc_ids)
        recs = [r for r in recs if r["doc_id"] in keep]
    return recs
