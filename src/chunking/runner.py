"""데이터셋 하나를 청커 하나로 청킹해 data/chunks/<dataset>/<청커>__<설정>.jsonl 로 저장한다
(docs/restructure_plan.md 3.2·5절).

- 문서 순서는 매니페스트 순서, 입력은 data/processed/<dataset>/<doc_id>.json.
- 파일이 이미 있으면 건너뛴다(이어하기). force=True면 다시 만든다. 쓸 때는 임시 파일에 쓴 뒤 이름을 바꾼다.
- 청크 파일 옆에 <태그>.meta.json을 둔다: 설정, 청크 수, 내용 보존 검사 실패 문서, 청커가 센 통계(info_totals: 단위·컷·합치기·min/max 위반 개수).
- 내용 보존 검사(overlap 0일 때): 청크들의 공백 아닌 글자를 이어 붙인 것 == 원문의 공백 아닌 글자.
  어긋난 문서는 meta에 기록하고 계속 진행한다(조용히 넘어가지 않게 보고한다).
"""
import datetime
import json
import os
import time

from chunking.splitters import chunk
from chunking.units import DEFAULT_RULES, DEFAULT_SIZE_CONTROL, rules_for, size_control_for
from chunking.stats import coverage_ok
from common.config import CHUNKS_DIR, PROCESSED_DIR, dataset_cfg
from common.docjson import load_doc, to_text

CHUNKERS = ("fixed", "semantic", "smart")


def default_cfg(dataset, chunker, overrides=None):
    cfg = dict(dataset_cfg(dataset)["chunkers"][chunker])
    cfg.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return cfg


def tag_for(chunker, cfg):
    """파일 이름에 쓰는 사람이 읽는 설정 요약. 정확한 값은 청크 줄의 params가 기준이다."""
    if chunker == "fixed":
        base = f"t{cfg['target_tokens']}" if cfg.get("target_tokens") and not cfg.get("chunk_size_chars") else f"c{cfg.get('chunk_size_chars')}"
        ov = cfg.get("overlap_ratio", 0.0)
        return f"fixed__{base}" + (f"-o{round(ov * 100)}" if ov else "") + ("" if cfg.get("strip_whitespace", True) else "-nostrip")
    if chunker == "smart" and cfg.get("algorithm") == "legacy":
        return f"smart__min{cfg['min_tokens']}-max{cfg['max_tokens']}-w{cfg.get('window', 2)}"
    if chunker in ("semantic", "smart"):
        tag = f"{chunker}__p{cfg['percentile']:g}-min{cfg['min_tokens']}-max{cfg['max_tokens']}"
        if cfg.get("buffer_size", 0) != 0:                     # 기본(0)이 아니면 이름에 남긴다(1 = LangChain 기본)
            tag += f"-b{cfg['buffer_size']}"
        rules = rules_for(chunker, cfg)
        if rules != DEFAULT_RULES[chunker]:                    # 기본 규칙 세트가 아니면 이름에 남긴다(어블레이션)
            tag += f"-r{rules.code()}"
        if size_control_for(chunker, cfg) != DEFAULT_SIZE_CONTROL[chunker]:     # 기본 크기 제어가 아니면 이름에 남긴다(어블레이션)
            tag += f"-sc{size_control_for(chunker, cfg)}"
        if "min_weight" in cfg and size_control_for(chunker, cfg) == "dp":
            tag += f"-mw{cfg['min_weight']:g}"
        return tag
    raise KeyError(chunker)


def chunk_documents(docs, chunker, cfg, models, log=None):
    """문서 목록 -> (청크 dict 목록, 요약). 파일 입출력 없음(테스트·스크립트 공용)."""
    records, bad, totals = [], [], {}
    for doc in docs:
        chunks, info = chunk(doc, chunker, cfg, models)
        for k, v in info.items():
            if isinstance(v, int):
                totals[k] = totals.get(k, 0) + v            # 청커가 세어 준 통계(예: min/max 위반 개수)
        overlap = (chunks[0]["params"].get("chunk_overlap_chars", 0) if chunks else 0)
        if not overlap and coverage_ok(to_text(doc), [(c["start"], c["end"]) for c in chunks]) is False:
            bad.append(doc["doc_id"])
        records += chunks
        if log:
            log(f"  {doc['doc_id'][:44]:44} 청크 {len(chunks):>4}")
    summary = {"coverage_failures": bad}
    if totals:
        summary["info_totals"] = totals
    return records, summary


def chunk_dataset(dataset, chunker, models, overrides=None, doc_ids=None, limit=None, force=False, out_dir=None, log=print):
    from preprocess.datasets import load_entries

    cfg = default_cfg(dataset, chunker, overrides)
    tag = tag_for(chunker, cfg)
    out_dir = out_dir or (CHUNKS_DIR / dataset)
    path = out_dir / f"{tag}.jsonl"
    if path.exists() and not force:
        log(f"[{dataset}/{tag}] 이미 있음, 건너뜀 (--force로 다시 만들기)")
        return {"path": path, "skipped": True}
    ids = [e["doc_id"] for e in load_entries(dataset)]
    if doc_ids:
        ids = [i for i in ids if i in set(doc_ids)]
    if limit:
        ids = ids[:limit]
    log(f"[{dataset}/{tag}] 문서 {len(ids)}개, 설정 {cfg}")
    t0 = time.time()
    docs = (load_doc(PROCESSED_DIR / dataset / f"{i}.json") for i in ids)
    records, summary = chunk_documents(docs, chunker, cfg, models, log=log)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    meta = {"dataset": dataset, "chunker": chunker, "tag": tag, "cfg": cfg, "n_docs": len(ids), "n_chunks": len(records),
            "seconds": round(time.time() - t0, 1), "created_at": datetime.datetime.now().isoformat(timespec="seconds"), **summary}
    (out_dir / f"{tag}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"[{dataset}/{tag}] 청크 {len(records)}개 → {path} | "
        f"내용 보존 검사 실패 문서 {len(summary['coverage_failures'])}개 | {meta['seconds']}초")
    return {"path": path, "skipped": False, **meta}
