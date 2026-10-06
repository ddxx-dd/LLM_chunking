"""원본 -> 공통 JSON (docs/restructure_plan.md 4.5절).

  source env.sh
  .venv311/bin/python scripts/preprocess.py --dataset vectara [--docs id1,id2] [--limit N] [--force]

- 결과: data/processed/<dataset>/<doc_id>.json, 문서별 점검 요약 data/processed/<dataset>/_report.json
- 이미 있는 JSON은 건너뛴다(이어하기). --force는 다시 만들고 하위 산출물을 3.7절 규칙대로 정리한다:
  data/chunks/<dataset>/, data/index/<dataset>/ 삭제, results/<dataset>/ -> results/_stale/ 로 이동,
  docx_corpus는 QA 근거 스팬 재계산 경고.
- PDF는 기본으로 PyMuPDF 정제(clean=True)본을 변환한다(docling-parse가 놓치는 줄을 살림, pdf_prep.CLEAN_OPTIONS).
  --no-pdf-clean 은 원본 PDF를 그대로 변환한다(A/B 비교용). docx·srt에는 영향이 없다.
- 이미 있던 JSON을 다시 만들 때는 이전 결과와의 글자 수 차이(delta_chars)를 보고서에 남긴다.
"""
import argparse
import datetime
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common.config import CHUNKS_DIR, DATASETS, INDEX_DIR, PROCESSED_DIR, QA_DIR, RESULTS_DIR  # noqa: E402
from common.docjson import save_doc  # noqa: E402
from preprocess.datasets import load_entries  # noqa: E402
from preprocess.validate import validate_doc  # noqa: E402


def invalidate_downstream(dataset):
    """전처리를 다시 하면 낡아지는 하위 산출물을 정리한다(3.7절)."""
    for d in (CHUNKS_DIR / dataset, INDEX_DIR / dataset):
        if d.exists():
            shutil.rmtree(d)
            print(f"[force] 삭제: {d}", flush=True)
    r = RESULTS_DIR / dataset
    if r.exists():
        dst = RESULTS_DIR / "_stale" / f"{dataset}_{datetime.date.today():%Y%m%d}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(r), str(dst))
        print(f"[force] results/{dataset} -> {dst}", flush=True)
    if dataset == "docx_corpus" and (QA_DIR / "docx_corpus.jsonl").exists():
        print("[force] 경고: data/qa/docx_corpus.jsonl 의 근거 스팬(to_text 오프셋)이 낡았다. "
              "build_qa.py --dataset docx_corpus 로 근거를 다시 계산할 것.", flush=True)


def prev_stats(path):
    """이미 있는 변환 결과의 (글자 수, 블록 수). 없거나 읽을 수 없으면 None."""
    if not path.exists():
        return None
    try:
        blocks = json.loads(path.read_text(encoding="utf-8"))["blocks"]
        return {"total_chars": sum(len(b.get("text") or "") for b in blocks), "n_blocks": len(blocks)}
    except Exception:
        return None


def converter_for(dataset, pdf_clean=True):
    fmt = DATASETS[dataset]["fmt"]
    if fmt == "pdf":
        from preprocess.pdf_prep import convert_pdf, get_converter
        conv = get_converter()
        return lambda e: convert_pdf(e, conv, clean=pdf_clean)
    if fmt == "docx":
        from preprocess.docx_prep import convert_docx, get_converter
        conv = get_converter()
        return lambda e: convert_docx(e, conv)
    from preprocess.srt_prep import convert_srt
    return convert_srt


def write_report(path, dataset, docs):
    totals = {"docs": len(docs),
              "ok": sum(1 for d in docs.values() if d["status"] == "ok"),
              "warn": sum(1 for d in docs.values() if d["status"] == "warn"),
              "error": sum(1 for d in docs.values() if d["status"] == "error"),
              "fail": sum(1 for d in docs.values() if d["status"] == "fail")}
    path.write_text(json.dumps({"dataset": dataset, "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
                                "totals": totals, "docs": docs}, ensure_ascii=False, indent=1), encoding="utf-8")
    return totals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--docs", help="쉼표로 구분한 doc_id만 처리")
    ap.add_argument("--limit", type=int, help="매니페스트 앞에서 N개만")
    ap.add_argument("--force", action="store_true", help="JSON을 다시 만들고 하위 산출물 정리")
    ap.add_argument("--no-pdf-clean", action="store_true",
                    help="PDF를 정제(PyMuPDF clean) 없이 원본 그대로 변환 (A/B 비교용, pdf 데이터셋에만 해당)")
    args = ap.parse_args()

    entries = load_entries(args.dataset)
    if args.docs:
        want = {d.strip() for d in args.docs.split(",") if d.strip()}
        missing = want - {e["doc_id"] for e in entries}
        if missing:
            sys.exit(f"매니페스트에 없는 doc_id: {sorted(missing)}")
        entries = [e for e in entries if e["doc_id"] in want]
    if args.limit:
        entries = entries[:args.limit]

    out_dir = PROCESSED_DIR / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "_report.json"
    docs = json.loads(report_path.read_text(encoding="utf-8"))["docs"] if report_path.exists() else {}

    if args.force:
        invalidate_downstream(args.dataset)

    todo = [e for e in entries if args.force or not (out_dir / f"{e['doc_id']}.json").exists()]
    print(f"[{args.dataset}] 대상 {len(entries)}개 중 변환 {len(todo)}개 (건너뜀 {len(entries) - len(todo)})", flush=True)
    if args.no_pdf_clean and DATASETS[args.dataset]["fmt"] != "pdf":
        print(f"[{args.dataset}] --no-pdf-clean 은 PDF 데이터셋에만 의미가 있어 무시한다", flush=True)
    convert = converter_for(args.dataset, pdf_clean=not args.no_pdf_clean) if todo else None

    t_all = time.time()
    try:
        for i, e in enumerate(todo, 1):
            t = time.time()
            prev = prev_stats(out_dir / f"{e['doc_id']}.json")
            try:
                doc = convert(e)
                v = validate_doc(doc)
                save_doc(doc, out_dir / f"{e['doc_id']}.json")
                status = "error" if v["errors"] else ("warn" if v["warnings"] else "ok")
                docs[e["doc_id"]] = {"status": status, "seconds": round(time.time() - t, 1), **v}
            except Exception as ex:          # 변환 실패는 기록하고 계속 진행
                docs[e["doc_id"]] = {"status": "fail", "seconds": round(time.time() - t, 1),
                                     "errors": [f"{type(ex).__name__}: {str(ex)[:300]}"], "warnings": [],
                                     "trace": traceback.format_exc()[-1200:]}
            d = docs[e["doc_id"]]
            if prev and "stats" in d:                  # 다시 만든 경우: 이전 결과 대비 글자 수 차이
                d["prev"] = prev
                d["delta_chars"] = d["stats"]["total_chars"] - prev["total_chars"]
            delta = f" Δ글자={d['delta_chars']:+d}" if d.get("delta_chars") else ""
            print(f"[{i}/{len(todo)}] {e['doc_id'][:48]:48} {d['status']:5} {d['seconds']:>6}s "
                  f"blocks={d.get('stats', {}).get('n_blocks', '-')}{delta} {d['errors'][:1] or d['warnings'][:1] or ''}", flush=True)
            if i % 10 == 0:
                write_report(report_path, args.dataset, docs)
    finally:
        totals = write_report(report_path, args.dataset, docs)
    print(f"[{args.dataset}] 완료 {time.time() - t_all:.0f}s | {totals} | 보고서: {report_path}", flush=True)
    redone = [d for d in (docs[e["doc_id"]] for e in todo if e["doc_id"] in docs) if d.get("prev")]
    if redone:
        inc = [d["delta_chars"] for d in redone if d["delta_chars"] > 0]
        dec = [d["delta_chars"] for d in redone if d["delta_chars"] < 0]
        print(f"[{args.dataset}] 이전 결과 대비: 다시 만든 {len(redone)}편 중 글자 증가 {len(inc)}편(+{sum(inc)}자), "
              f"감소 {len(dec)}편({sum(dec)}자), 같음 {len(redone) - len(inc) - len(dec)}편", flush=True)


if __name__ == "__main__":
    main()
