"""LongBench 단위: data/raw/longbench/<이름>.jsonl → processed_para·processed_sen + data/qa/<이름>.jsonl (pdf·docx 경로와 따로).

  source env.sh
  PYTHONPATH=src .venv311/bin/python -m preprocess.longbench --dataset all [--out data]

원문을 거의 그대로 쓴다(2026-10-15 사용자 결정). 데이터셋별 처리
  hotpotqa    'Passage N:' 줄을 지우고 단위마다 패시지 번호(passage, 0부터)를 남긴다. 한 줄 = 한 단위(위키 원문 구조).
              마침표 뒤 공백이 빠진 곳('inmates.It')에 공백 하나를 넣는다(짧은 대문자 조각 약어 'D.Ed.Min.' 'H.Con.Res.'는 제외).
  narrativeqa 영화 대본(HTML 태그가 섞인 문서 6편)은 쓰지 않는다. 책은 원문 그대로, 빈 줄로 나뉜 블록 = 한 단위.
  gov_report  원문 그대로, 문서 전체 = 한 단위(원문에 문단 구분이 없다).
문장 나누기: nltk punkt + pdf·docx와 같은 학술 약어 + LongBench용 약어(법안·법률 번호 H.R., P.L., S., H.Con.Res. …),
  pdf·docx와 같은 영어 보정(fix_en) + 물음표·느낌표 뒤 소문자 시작은 합침('Coffee, Tea or Me? and Monte Walsh').
  검증: tests/sent_split/lb_check.py(표본 160문단, 후보 694곳 판정: F1 HotpotQA 99.0%, NarrativeQA 99.5%, GovReport 99.5%).
출력
  <out>/processed_para/<이름>/<doc_id>.json  문서 필드 + units[{text, kind, heading, pages, formulas, (passage), n_tokens}]
  <out>/processed_sen/<이름>/<doc_id>.json   문서 필드 + units[{text, kind, para, heading, pages, (passage), n_tokens}]
  <out>/qa/<이름>.jsonl                       {_id, doc_id, question, answers, length}
"""
import argparse
import json
import re
from pathlib import Path

from common.config import DATA_DIR, LONGBENCH, RAW_DIR, ROOT
from preprocess.sentences import ABBREV, LETTER, fix_en

PASSAGE = re.compile(r"Passage (\d+):")
MISSING = re.compile(r"(?<=[a-z0-9)\"'”’])\.(?=[A-Z][a-z])")          # 'inmates.It', '1993.Following', 'kg).The'
DOTTED = re.compile(r"(?:^|[\s(\[])(?:[A-Z][A-Za-z]{0,3}\.)+[A-Z][A-Za-z]{0,3}$")   # 'D.Ed.Min', 'Prof.Dr', 'H.Con.Res'
TAG = re.compile(r"</?[a-zA-Z]+[^>]*>")
LB_ABBREV = {"h.r", "p.l", "s", "h.con.res", "s.con.res", "h.res", "s.res", "h.j.res", "s.j.res"}
QE_END = re.compile(r"[?!][\"'”’)\]]*$")
_PUNKT = None


def fix_space(t):
    """마침표 뒤 빠진 공백을 넣는다. (고친 글, 고친 수)."""
    out, last, n = [], 0, 0
    for m in MISSING.finditer(t):
        if DOTTED.search(t[max(0, m.start() - 30):m.start()]):
            continue
        out.append(t[last:m.end()] + " ")
        last, n = m.end(), n + 1
    return "".join(out) + t[last:], n


def is_script(context):
    """영화 대본(HTML 태그가 섞인 문서). 책은 태그가 거의 없다(최대 1개)."""
    return len(TAG.findall(context)) >= 5


def split(text):
    """글 → 문장 목록(pdf·docx의 sentences.split과 같은 흐름, 토크나이저·보정만 LongBench용)."""
    global _PUNKT
    if _PUNKT is None:
        import nltk
        from nltk.tokenize.punkt import PunktTokenizer
        path = str(ROOT / ".home" / "nltk_data")
        if path not in nltk.data.path:
            nltk.data.path.insert(0, path)
        _PUNKT = PunktTokenizer("english")                   # pdf·docx 쪽과 다른 객체(약어 목록이 섞이지 않게)
        _PUNKT._params.abbrev_types.update(ABBREV | LB_ABBREV)
    sents = fix_en([s for s in (" ".join(x.split()) for x in _PUNKT.tokenize(text)) if s])
    merged = []
    for s in sents:
        if merged and re.match(r"[a-z]", s) and QE_END.search(merged[-1]):
            merged[-1] = merged[-1] + " " + s
        else:
            merged.append(s)
    out, carry = [], ""
    for s in merged:
        if not LETTER.search(s):
            carry = (carry + " " + s).strip()
            continue
        out.append((carry + " " + s).strip() if carry else s)
        carry = ""
    if carry:
        if out:
            out[-1] = out[-1] + " " + carry
        else:
            out = [carry]
    return out


def unit(text, **extra):
    return {"text": text, "kind": "para", "heading": "", "pages": [], "formulas": [], **extra}


def documents(name):
    """(문서 목록, QA 목록, 기록). 문서 = {doc_id, title, units}."""
    recs = [json.loads(x) for x in (RAW_DIR / "longbench" / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]
    docs, qa, log = [], [], {"records": len(recs)}
    if name == "hotpotqa":
        n_fix = 0
        for r in recs:
            units, p = [], -1
            for line in r["context"].split("\n"):
                if PASSAGE.fullmatch(line.strip()):
                    p += 1
                    continue
                if line.strip():
                    t, n = fix_space(line.strip())
                    n_fix += n
                    units.append(unit(t, passage=p))
            docs.append({"doc_id": r["_id"], "title": None, "units": units})
            qa.append({"_id": r["_id"], "doc_id": r["_id"], "question": r["input"], "answers": r["answers"], "length": r["length"]})
        log["space_fixed"] = n_fix
    elif name == "narrativeqa":
        ids = {}
        for r in recs:
            ids.setdefault(r["context"], f"nqa{len(ids):02d}")      # 처음 나온 순서로 번호(대본 포함 20편 기준이라 번호가 바뀌지 않음)
        skipped = 0
        for c, d in ids.items():
            if is_script(c):
                continue
            units = [unit(b.strip()) for b in re.split(r"\n\s*\n", c) if b.strip()]
            docs.append({"doc_id": d, "title": None, "units": units})
        for r in recs:
            if is_script(r["context"]):
                skipped += 1
                continue
            qa.append({"_id": r["_id"], "doc_id": ids[r["context"]], "question": r["input"], "answers": r["answers"], "length": r["length"]})
        log.update(documents_all=len(ids), scripts_excluded=sum(is_script(c) for c in ids), questions_excluded=skipped)
    elif name == "gov_report":
        for r in recs:
            docs.append({"doc_id": r["_id"], "title": None, "units": [unit(r["context"].strip())]})
            qa.append({"_id": r["_id"], "doc_id": r["_id"], "question": r["input"], "answers": r["answers"], "length": r["length"]})
    else:
        raise KeyError(f"모르는 LongBench 데이터셋: {name!r} (선택: {sorted(LONGBENCH)})")
    return docs, qa, log


def to_sentences(units):
    out = []
    for k, u in enumerate(units):
        flat = " ".join(u["text"].split())
        for s in split(flat) or [flat]:
            out.append({"text": s, "kind": u["kind"], "para": k, "heading": "", "pages": [],
                        **({"passage": u["passage"]} if "passage" in u else {})})
    return out


def main():
    from common.models import make_token_counter

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default="all", help="all 또는 " + ", ".join(LONGBENCH))
    ap.add_argument("--out", default=str(DATA_DIR), help="출력 루트(기본 data/)")
    ns = ap.parse_args()
    out = Path(ns.out)
    count_many = make_token_counter()
    for name in (list(LONGBENCH) if ns.dataset == "all" else [ns.dataset]):
        docs, qa, log = documents(name)
        for d in docs:
            base = {"doc_id": d["doc_id"], "dataset": name, "lang": LONGBENCH[name]["lang"], "title": d["title"], "fmt": "longbench", "n_pages": None}
            sents = to_sentences(d["units"])
            for us in (d["units"], sents):
                for u, n in zip(us, count_many([u["text"] for u in us])):
                    u["n_tokens"] = n
            para = {**base, "source": f"raw/longbench/{name}.jsonl", "n_units": len(d["units"]),
                    "n_tokens": sum(u["n_tokens"] for u in d["units"]), "units": d["units"]}
            sen = {**base, "source": f"processed_para/{name}", "n_paras": len(d["units"]), "n_units": len(sents),
                   "n_tokens": sum(u["n_tokens"] for u in sents), "units": sents}
            for sub, rec in (("processed_para", para), ("processed_sen", sen)):
                dst = out / sub / name / f"{d['doc_id']}.json"
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        qdst = out / "qa" / f"{name}.jsonl"
        qdst.parent.mkdir(parents=True, exist_ok=True)
        qdst.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in qa), encoding="utf-8")
        print(f"{name}: 문서 {len(docs)}, 질문 {len(qa)}, 문단 {sum(len(d['units']) for d in docs)} | {log}", flush=True)


if __name__ == "__main__":
    main()
