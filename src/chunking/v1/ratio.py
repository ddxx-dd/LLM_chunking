"""fixed 청커의 글자 수 환산용 "글자/토큰 비율" 측정 (docs/restructure_plan.md 5.4.1절).

fixed는 글자 수로 자르는 청커라서, 다른 청커와 크기를 맞추려면 "목표 토큰 수에 해당하는 글자 수"가 필요하다.
이 비율은 언어·토크나이저·문서 성격(수식·표)에 따라 달라서 추정하지 않고 전처리된 JSON의 평문으로 잰다.

  chunks_per_token 이 아니라  chars_per_token = Σ 글자 수 / Σ 토큰 수   (문서별 비율의 평균이 아니라 합계 비율)
  chunk_size_chars = round(target_tokens × chars_per_token)

결과는 data/manifests/<dataset>.ratio.json 에 저장한다(매니페스트가 목록이라 같은 파일에 넣지 못함).
전처리를 다시 하면(3.7절) 비율도 다시 잰다.
"""
import datetime
import json
import statistics

from common.config import MANIFEST_DIR, PROCESSED_DIR, dataset_cfg
from common.docjson import load_doc, to_text
from common.models import get_tokenizer


def ratio_path(dataset):
    return MANIFEST_DIR / f"{dataset}.ratio.json"


def _processed_docs(dataset):
    d = PROCESSED_DIR / dataset
    paths = sorted(p for p in d.glob("*.json") if p.name != "_report.json")    # doc_id가 "_"로 시작하는 문서도 있다
    if not paths:
        raise FileNotFoundError(f"전처리 결과가 없다: {d} (먼저 scripts/preprocess.py)")
    return [load_doc(p) for p in paths]


def measure_ratio(dataset, sample_max=None):
    """-> 측정 결과 dict. 문서가 sample_max보다 많으면 시드 고정 무작위 표본(기본: 전부)."""
    cfg = dataset_cfg(dataset)
    kind = cfg["counter"]
    tok = get_tokenizer(kind)
    tok.model_max_length = 10 ** 9        # 긴 평문을 토큰 수만 세므로 길이 경고를 끈다
    docs = _processed_docs(dataset)
    if sample_max and len(docs) > sample_max:
        import random
        docs = random.Random(42).sample(docs, sample_max)

    by_lang = {}
    for doc in docs:
        text = to_text(doc)
        n_tok = len(tok.encode(text, add_special_tokens=False))
        by_lang.setdefault(doc["lang"], []).append((len(text), n_tok))

    result = {"dataset": dataset, "counter": kind, "tokenizer": getattr(tok, "name_or_path", kind),
              "measured_at": datetime.datetime.now().isoformat(timespec="seconds"), "by_lang": {}}
    for lang, pairs in sorted(by_lang.items()):
        chars = sum(c for c, _ in pairs)
        toks = sum(t for _, t in pairs)
        per_doc = sorted(c / t for c, t in pairs if t)
        result["by_lang"][lang] = {
            "chars_per_token": round(chars / toks, 4), "n_docs": len(pairs),
            "total_chars": chars, "total_tokens": toks,
            "per_doc_min": round(per_doc[0], 3), "per_doc_p50": round(statistics.median(per_doc), 3),
            "per_doc_max": round(per_doc[-1], 3)}
    return result


def save_ratio(result):
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    ratio_path(result["dataset"]).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")


def load_ratio(dataset, lang):
    """측정값(chars_per_token)을 읽는다. 없으면 측정부터 하라고 알린다."""
    p = ratio_path(dataset)
    if not p.exists():
        raise FileNotFoundError(f"{p} 가 없다: scripts/measure_ratio.py --dataset {dataset} 를 먼저 실행할 것")
    return json.loads(p.read_text(encoding="utf-8"))["by_lang"][lang]["chars_per_token"]


def fixed_chunk_size(dataset, lang, target_tokens, overlap_ratio=0.0):
    """-> (chunk_size_chars, chunk_overlap_chars)."""
    cpt = load_ratio(dataset, lang)
    size = round(target_tokens * cpt)
    return size, round(overlap_ratio * size)
