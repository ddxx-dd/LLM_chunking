"""LongBench 원본 읽기: data/raw/longbench/<이름>.jsonl → 문서 목록과 질문 목록(파일로 따로 만들지 않는다).

원문을 그대로 쓴다(2026-10-17 사용자 결정: "공평성을 위해 최대한 원문을 그대로"). 청커 입력은 원본 context 그대로
(HotpotQA의 'Passage N:' 줄과 마침표 뒤 빠진 공백 'inmates.It'도 그대로). 지표는 원본의 input(질문)·answers로 잰다.
  hotpotqa     질문마다 자기 문맥(방해 패시지 포함) → 문서 200편, doc_id = 원본 _id
  narrativeqa  같은 문맥이 질문마다 반복된다(200줄, 문서 20편) → 문서 단위로 한 번만. 영화 대본 6편(HTML 태그가 섞인 문서)은 쓰지 않는다
               → 책 14편·137문제. doc_id = 처음 나온 순서의 'nqa00'~'nqa19'(대본 포함 20편 기준이라 번호가 바뀌지 않음)
GovReport는 2026-10-17 사용자 결정으로 제외·삭제.
"""
import json
import re

from common.config import LONGBENCH, RAW_DIR

TAG = re.compile(r"</?[a-zA-Z]+[^>]*>")


def is_script(context):
    """영화 대본(HTML 태그가 섞인 문서). 책은 태그가 거의 없다(최대 1개)."""
    return len(TAG.findall(context)) >= 5


def _records(name):
    if name not in LONGBENCH:
        raise KeyError(f"모르는 LongBench 데이터셋: {name!r} (선택: {sorted(LONGBENCH)})")
    return [json.loads(x) for x in (RAW_DIR / "longbench" / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]


def _doc_id(name, recs):
    """원본 줄 → doc_id."""
    if name == "hotpotqa":
        return {r["_id"]: r["_id"] for r in recs}
    ids = {}
    for r in recs:
        ids.setdefault(r["context"], f"nqa{len(ids):02d}")
    return {r["_id"]: ids[r["context"]] for r in recs}


def documents(name):
    """문서 목록 [{doc_id, dataset, lang, text}] (text = 원본 context 그대로)."""
    recs = _records(name)
    ids, out, seen = _doc_id(name, recs), [], set()
    for r in recs:
        d = ids[r["_id"]]
        if d in seen or is_script(r["context"]):
            continue
        seen.add(d)
        out.append({"doc_id": d, "dataset": name, "lang": LONGBENCH[name]["lang"], "text": r["context"]})
    return out


def questions(name):
    """질문 목록 [{_id, doc_id, question, answers, length}] (대본 문서의 질문은 뺀다)."""
    recs = _records(name)
    ids = _doc_id(name, recs)
    return [{"_id": r["_id"], "doc_id": ids[r["_id"]], "question": r["input"], "answers": r["answers"], "length": r["length"]}
            for r in recs if not is_script(r["context"])]
