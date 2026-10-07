"""평가 실행 (docs/restructure_plan.md 6.3·10절). 지금은 --task retrieval 만 있다.

  source env.sh
  # 데이터셋 하나: 그 데이터셋의 청크만 든 인덱스에서 평가
  .venv311/bin/python scripts/run.py --task retrieval --dataset vectara --tag A__p90,fixed__T260 [--split eval] [--k 3] [--force]
  # 여러 데이터셋을 한 인덱스에 합쳐서 평가(질문은 데이터셋별로 채점). 같은 태그가 각 데이터셋에 있어야 한다.
  .venv311/bin/python scripts/run.py --task retrieval --datasets vectara,docx_corpus,allganize --tag A__p90

- --tag: data/chunks/<dataset>/ 의 청크 파일 이름(확장자 제외), 쉼표로 여러 개, 또는 all(데이터셋 하나일 때).
- 임베딩 정밀도(--precision, 기본 fp16): 인덱스(청크·질문)를 bge-m3 fp16으로 임베딩한다(fp32 대비 top-3 99% 일치, 약 4배 빠름).
  fp16 인덱스는 data/index/<dataset>/<태그>@fp16 에 따로 캐시하고, 결과 JSON에 precision을 적는다. 옛 fp32 결과(실험 1)와 섞이지 않는다.
- 질문은 청크 파일에 들어 있는 문서의 것만 쓴다(표본 청킹이면 그 문서들의 질문만).
- 결과: 하나 = results/<dataset>/retrieval/<태그>[@split].json, 합침 = results/combined/<dataset>/<태그>[@split].json.
  이미 있으면 건너뛴다(--force로 덮어쓰기). 정밀도가 다른 결과가 있으면 덮어쓰지 않고 알린다.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common.config import CHUNKS_DIR, DATASETS, RESULTS_DIR  # noqa: E402

MAX_SEQ = 8192   # bge-m3 최대 입력 토큰


def make_encoder(precision):
    if precision == "fp16":
        from chunking.embed import load_model, make_encode

        return make_encode(load_model(), batch_size=16)       # 긴 청크의 메모리 폭주를 막으려고 배치를 작게
    from common.models import get_embeddings
    from indexing.index import make_encoder as old_make

    return old_make(get_embeddings()._client)


def index_for(dataset, tag, encode, doc_ids, precision, force):
    """청크 파일 -> 인덱스(캐시: fp16은 <태그>@fp16, fp32는 옛 이름 그대로)."""
    from indexing.index import build_index, load_chunks, load_index, save_index

    chunks = load_chunks(dataset, tag, doc_ids)
    idx_tag = f"{tag}@fp16" if precision == "fp16" else tag
    idx = None if force else load_index(chunks, encode, dataset, idx_tag)
    if idx is None:
        idx = build_index(chunks, encode, MAX_SEQ)
        save_index(idx, dataset, idx_tag)
    return idx


def skip_existing(out, force, precision):
    if not out.exists() or force:
        return False
    had = json.loads(out.read_text(encoding="utf-8")).get("precision", "fp32")
    note = "" if had == precision else f" (정밀도가 다르다: 기존 {had}, 요청 {precision} - 덮어쓰려면 --force)"
    print(f"[{out.parent.parent.name}/{out.stem}] 결과가 이미 있음, 건너뜀{note}")
    return True


def line(dataset, tag, res, k):
    from tasks.retrieval import PRIMARY

    o, prim = res["overall"], PRIMARY.get(dataset)
    return (f"[{dataset}/{tag}] 문서 {res['n_docs']} 질문 {res['n_questions']} 청크 {res['index']['n_chunks']} "
            f"(평균 {res['index']['chunk_tokens_mean']:.0f}토큰, 최대 {res['index']['chunk_tokens_max']}) | hit@{k} {o['hit@k']:.3f} MRR {o['mrr']:.3f} F1 {o['f1']:.3f}"
            + (f" {prim}@{k} {o[prim + '@k']:.3f}" if prim and prim + "@k" in o else "") + f" | top-{k} 평균 {o['top_k_tokens_mean']:.0f}토큰")


def run_single(dataset, tags, split, k, ngram, force, precision):
    from indexing.index import load_chunks
    from qa_build.build_qa import load_qa
    from tasks.retrieval import evaluate

    encode = make_encoder(precision)
    out_dir = RESULTS_DIR / dataset / "retrieval"
    out_dir.mkdir(parents=True, exist_ok=True)
    for tag in tags:
        out = out_dir / f"{tag}{'@' + split if split else ''}.json"
        if skip_existing(out, force, precision):
            continue
        docs = sorted({c["doc_id"] for c in load_chunks(dataset, tag)})
        qas = load_qa(dataset, split=split, doc_ids=docs)
        res = evaluate(index_for(dataset, tag, encode, docs, precision, force), qas, k=k, ngram=ngram)
        res.update(dataset=dataset, tag=tag, split=split, n_docs=len(docs), n_questions=len(qas), ngram=ngram, precision=precision, mode="single")
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(line(dataset, tag, res, k))


def run_combined(datasets, tags, split, k, ngram, force, precision):
    from indexing.index import combine_indexes, load_chunks
    from qa_build.build_qa import load_qa
    from tasks.retrieval import evaluate

    encode = make_encoder(precision)
    for tag in tags:
        outs = {ds: RESULTS_DIR / "combined" / ds / f"{tag}{'@' + split if split else ''}.json" for ds in datasets}
        if all(skip_existing(o, force, precision) for o in outs.values()):
            continue
        docs = {ds: sorted({c["doc_id"] for c in load_chunks(ds, tag)}) for ds in datasets}
        comb = combine_indexes([index_for(ds, tag, encode, docs[ds], precision, force) for ds in datasets], encode)
        print(f"[combined/{tag}] 합친 인덱스: 청크 {len(comb.chunks)}개, 문서 {sum(len(v) for v in docs.values())}개")
        for ds in datasets:
            qas = load_qa(ds, split=split, doc_ids=docs[ds])
            res = evaluate(comb, qas, k=k, ngram=ngram)
            res.update(dataset=ds, tag=tag, split=split, n_docs=len(docs[ds]), n_questions=len(qas), ngram=ngram, precision=precision,
                       mode="combined", combined_datasets=list(datasets))
            outs[ds].parent.mkdir(parents=True, exist_ok=True)
            outs[ds].write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
            print(line(ds, tag, res, k))


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", required=True, choices=["retrieval"])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dataset", choices=sorted(DATASETS))
    g.add_argument("--datasets", help="쉼표로 구분한 데이터셋들을 한 인덱스로 합쳐 평가")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", choices=["tune", "eval"])
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--ngram", type=int, default=1, help="section_hit 의 포함 비율 단위(계획서 정의 = 1)")
    ap.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--force", action="store_true")
    return ap


def main(argv=None):
    ns = build_parser().parse_args(argv)
    if ns.datasets:
        datasets = [d.strip() for d in ns.datasets.split(",")]
        bad = [d for d in datasets if d not in DATASETS]
        if bad or len(datasets) < 2:
            sys.exit(f"--datasets에는 서로 다른 데이터셋 2개 이상을 줄 것 (모르는 이름: {bad})")
        if ns.tag == "all":
            sys.exit("합침 모드에서는 --tag all을 쓸 수 없다(태그를 직접 줄 것)")
        run_combined(datasets, ns.tag.split(","), ns.split, ns.k, ns.ngram, ns.force, ns.precision)
        return
    tags = sorted(p.stem for p in (CHUNKS_DIR / ns.dataset).glob("*.jsonl")) if ns.tag == "all" else ns.tag.split(",")
    run_single(ns.dataset, tags, ns.split, ns.k, ns.ngram, ns.force, ns.precision)


if __name__ == "__main__":
    main()
