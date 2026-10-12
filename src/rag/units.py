"""트랙 A 입력 준비: 전처리 JSON 읽기 → 깨끗한 텍스트 → 단위(문단·문장).

data/meta_text(Docling 전처리 결과)는 수정하지 않고 읽기만 한다. 변환 규칙은 AGENTS.md 3절 '입력: 깨끗한 텍스트'.
지금은 문서 읽기(A2-1)만 있다. 깨끗한 텍스트(A2-2)와 문장 단위(A2-3)는 다음 단계에서 추가한다.
"""
import json

from common.config import DATASETS, META_DIR, REPORT_FILE


def _check(dataset):
    if dataset not in DATASETS:
        raise KeyError(f"모르는 데이터셋: {dataset!r} (선택: {sorted(DATASETS)})")


def doc_ids(dataset):
    """전처리된 문서의 doc_id 목록(정렬). 실행 보고서(_report.json)만 빼고, '_'로 시작하는 doc_id는 포함한다."""
    _check(dataset)
    return sorted(p.stem for p in (META_DIR / dataset).glob("*.json") if p.name != REPORT_FILE)


def load_doc(dataset, doc_id):
    """전처리 JSON 하나(dict: doc_id, title, lang, blocks …)."""
    _check(dataset)
    return json.loads((META_DIR / dataset / f"{doc_id}.json").read_text(encoding="utf-8"))
