"""전처리할 문서 목록: data/manifests/<데이터셋>.json → [{doc_id, dataset, fmt, lang, title, source_file}].

데이터셋마다 manifest 필드가 다르다: vectara는 paper_id(원본 data/raw/vectara_pdf/<id>.pdf)와 title,
allganize는 doc_id와 path, docx_corpus는 file(doc_id = 파일 이름 줄기)과 path. 제목이 없는 데이터셋은 1차가 첫 제목 블록으로 채운다.
EXCLUDE: 글자 층이 깨져 OCR 없이는 쓸 수 없는 문서(전처리하지 않는다).
"""
import json
from pathlib import Path

from common.config import DATASETS, MANIFEST_DIR

EXCLUDE = {("allganize", "_2019_제1회_증시콘서트_자료집_최종_"): "글자 층 깨짐(글자의 81%가 엉뚱한 문자)",
           ("allganize", "20080813_E_2"): "한글 글꼴 글자 층 이상(OCR 없이는 확인·복구 불가)"}


def documents(dataset):
    """데이터셋의 문서 목록(manifest 순서, 제외 문서 빼고)."""
    if dataset not in DATASETS:
        raise KeyError(f"모르는 데이터셋: {dataset!r} (선택: {sorted(DATASETS)})")
    info = DATASETS[dataset]
    out = []
    for e in json.loads((MANIFEST_DIR / f"{dataset}.json").read_text(encoding="utf-8")):
        if dataset == "vectara":
            doc_id, path, title = e["paper_id"], f"data/raw/{info['raw']}/{e['paper_id']}.pdf", " ".join((e.get("title") or "").split()) or None
        elif dataset == "docx_corpus":
            doc_id, path, title = Path(e["file"]).stem, e["path"], None
        else:
            doc_id, path, title = e["doc_id"], e["path"], None
        if (dataset, doc_id) in EXCLUDE:
            continue
        out.append({"doc_id": doc_id, "dataset": dataset, "fmt": info["fmt"], "lang": info["lang"], "title": title, "source_file": path})
    return out


def document(dataset, doc_id):
    d = next((x for x in documents(dataset) if x["doc_id"] == doc_id), None)
    if d is None:
        raise KeyError(f"{dataset}:{doc_id} 가 manifest에 없거나 제외 문서")
    return d
