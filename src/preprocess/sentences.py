"""문장 단위: processed_para(문단 단위) → processed_sen(문장 단위).

  source env.sh
  PYTHONPATH=src .venv311/bin/python -m preprocess.sentences --dataset all [--out data]

문장 나누기 도구(2026-10-15 비교, tests/sent_split): 영어 nltk punkt + 학술 약어(F1 97.2%, 논문 96%),
  한국어 Kiwi + 개조식 구조 표시 앞 자르기(F1 87%; Kiwi만 79%, SaT 85%, spaCy 51%).
규칙
  - 표(table)·코드(code) 단위와 혼자 남은 제목(heading) 단위는 나누지 않고 그대로 한 단위.
  - 수식(processed_para의 formulas 위치) 안에서는 자르지 않는다: 경계가 수식 안에 떨어지면 앞 문장과 합친다(수식 끝에서 끝나는 문장은 그대로).
  - 문단·목록·캡션·각주 단위는 문장으로 나눈다. 단위 안 줄바꿈(2차의 조각 붙이기로 생긴 것)은 공백으로 본다.
  - 문단에 붙은 제목은 그 문단의 첫 문장 앞에 줄바꿈으로 붙인다.
  - 글자가 하나도 없는 조각('(1)', '2')은 다음 문장 앞에 붙인다(마지막이면 앞 문장 뒤).
  - 영어 보정: 정리·절 라벨만 있는 문장('Proof.', 'Lemma 2.1.', '4.2.')은 다음 문장 앞에 붙이고,
    짧은 약어('i.i.d.', 'w.r.t.', 'etc.', 'P.C.')로 끝난 문장 뒤 소문자로 시작하는 문장은 합친다.
출력: <out>/processed_sen/<데이터셋>/<doc_id>.json — 문서 필드 + units[{text, kind, para(몇 번째 문단), heading, pages, n_tokens}]
"""
import argparse
import json
import re
from pathlib import Path

from common.config import DATA_DIR, DATASETS, ROOT
from preprocess.sources import documents

KEEP = ("table", "code", "heading")
LETTER = re.compile(r"[A-Za-z0-9가-힣ㄱ-ㆎ一-鿿]")
# 한국어 개조식 구조 표시 앞에서 자른다: □ ■ ▪ ◦ ○ ● ㅇ ※ ▣ ◆ ◇, 주석 '* '(뒤에 조사가 오면 문장 안 각주 표시라 자르지 않음), ① ~ ⑳, '가. ', '1) '
KO_MARK = re.compile(r"\s(?=(?:[□■▪◦○●ㅇ※▣◆◇]\s?|\*\s(?![의을를이가은는에와과로도만]\s)|[①-⑳]|[가-하]\.\s|\d{1,2}\)\s)\S)")
_KIWI = None


# nltk 영어 punkt에 더하는 학술 약어(마침표 뒤에서 자르지 않음): 'Smith et al. (2020)', 'Fig. 3', 'Eq. (2)', 'Sec. 4', 'pp. 12'.
# 판정 자료(tests/sent_split/out/v2_labels.json)로 F1 95.7% → 97.2%, 재현율은 그대로(99.1%).
ABBREV = {"al", "et al", "fig", "figs", "eq", "eqs", "sec", "secs", "ref", "refs", "vol", "vols", "pp", "approx", "resp", "vs", "cf",
          "no", "nos", "ch", "app", "thm", "lem", "def", "prop", "cor", "tab", "e.g", "i.e", "ca", "ed", "eds"}
_PUNKT = None


def split_en(text):
    global _PUNKT
    if _PUNKT is None:
        import nltk
        from nltk.tokenize.punkt import PunktTokenizer
        path = str(ROOT / ".home" / "nltk_data")
        if path not in nltk.data.path:
            nltk.data.path.insert(0, path)
        _PUNKT = PunktTokenizer("english")
        _PUNKT._params.abbrev_types.update(ABBREV)
    return _PUNKT.tokenize(text)


def split_ko(text):
    global _KIWI
    if _KIWI is None:
        from kiwipiepy import Kiwi
        _KIWI = Kiwi()
    return [p for s in _KIWI.split_into_sents(text) for p in KO_MARK.split(s.text) if p and p.strip()]


# 영어 보정 1: 문장 전체가 정리·절 라벨('Proof.', 'Lemma 2.1.', 'Theorem 5 (Main).', '4.2.')이면 다음 문장 앞에 붙인다.
EN_LABEL = re.compile(r"^(?:\(?[A-Z]?\d{1,3}(?:\.\d{1,3})*\.?\)?|(?:Proof|Lemma|Theorem|Proposition|Corollary|Definition|Remark|Example|Assumption|Claim"
                      r"|Conjecture|Fact|Note|Step|Case|Problem|Exercise|Algorithm|Observation|Hypothesis|Notation|Condition|Property)"
                      r"(?: [A-Z]?\d{1,3}(?:\.\d{1,3})*[a-z]?)?(?: \([^)]{1,40}\))? ?[.:])$")
# 영어 보정 2: 앞 문장이 짧은 약어('etc.', 's.t.', 'a.k.a.', 'r.h.s.', 'e.g.', 한 글자+점)로 끝나고 다음 문장이 소문자로 시작하면 합친다.
EN_ABBR_END = re.compile(r"(?:\b(?:etc|i\.e|e\.g|a\.k\.a|s\.t|r\.h\.s|l\.h\.s|w\.r\.t|resp|viz|cf|vs|approx|al|ca|incl|esp|eqn?|figs?|secs?|refs?|no)\."
                         r"|(?:^|[\s(\[])(?:[A-Za-z]\.){1,3})[)\],]?$", re.I)


def fix_en(sents):
    out, carry = [], ""
    for s in sents:
        if carry:
            s, carry = carry + " " + s, ""
        if out and re.match(r"[a-z]", s) and EN_ABBR_END.search(out[-1]):
            out[-1] = out[-1] + " " + s
        elif EN_LABEL.match(s):
            carry = s
        else:
            out.append(s)
    if carry:
        out.append(carry)
    return out


def split(text, lang):
    """글 → 문장 목록(공백 정리, 글자 없는 조각은 다음 문장에 붙임)."""
    sents = [" ".join(s.split()) for s in (split_ko(text) if lang == "ko" else split_en(text))]
    if lang != "ko":
        sents = fix_en([s for s in sents if s])
    out, carry = [], ""
    for s in sents:
        if not s:
            continue
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


def keep_formulas(sents, body, spans):
    """경계가 수식 [시작, 끝) 안(시작·끝 제외)에 떨어진 문장을 앞 문장과 합친다. sents·body는 공백 정리된 글."""
    if not spans:
        return sents
    out, pos = [], 0
    for s in sents:
        k = body.find(s, pos)
        if out and k > 0 and any(a < k < b for a, b in spans):
            out[-1] = out[-1] + " " + s
        else:
            out.append(s)
        pos = k + len(s) if k >= 0 else pos
    return out


def norm_spans(text, spans, body):
    """원래 단위 글의 수식 위치 → 공백 정리된 body 안 위치."""
    out, pos = [], 0
    for a, b in spans:
        f = " ".join(text[a:b].split())
        k = body.find(f, pos) if f else -1
        if k >= 0:
            out.append((k, k + len(f)))
            pos = k + len(f)
    return out


def to_sentences(rec):
    """processed_para 문서 → 문장 단위 목록(n_tokens 없이)."""
    units = []
    for k, u in enumerate(rec["units"]):
        if u["kind"] in KEEP:
            units.append({"text": u["text"], "kind": u["kind"], "para": k, "heading": u["heading"], "pages": u["pages"]})
            continue
        head = u["heading"] if u["heading"] and u["text"].startswith(u["heading"] + "\n") else ""   # 제목이 앞에 붙은 문단만
        body = u["text"][len(head) + 1:] if head else u["text"]
        flat = " ".join(body.split())
        sents = split(flat, rec.get("lang") or "en") or [flat]
        sents = keep_formulas(sents, flat, norm_spans(u["text"], u.get("formulas", []), flat))
        for j, s in enumerate(sents):
            units.append({"text": (head + "\n" + s) if head and j == 0 else s, "kind": u["kind"], "para": k,
                          "heading": head if j == 0 else "", "pages": u["pages"]})
    return units


def main():
    from common.models import make_token_counter

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default="all", help="all 또는 " + ", ".join(DATASETS))
    ap.add_argument("--doc", action="append", help="이 doc_id만(여러 번 줄 수 있음)")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default=str(DATA_DIR), help="입출력 루트(기본 data/)")
    ns = ap.parse_args()
    out = Path(ns.out)
    count_many = make_token_counter()
    names = list(DATASETS) if ns.dataset == "all" else [ns.dataset]
    docs = [d for ds in names for d in documents(ds) if not ns.doc or d["doc_id"] in ns.doc]
    docs = [d for k, d in enumerate(docs) if k % ns.workers == ns.worker]
    for d in docs:
        src = out / "processed_para" / d["dataset"] / f"{d['doc_id']}.json"
        if not src.exists():                         # 2차에서 제외된 문서(깨진 글)
            continue
        rec = json.loads(src.read_text(encoding="utf-8"))
        units = to_sentences(rec)
        for u, n in zip(units, count_many([u["text"] for u in units])):
            u["n_tokens"] = n
        sen = {k: rec[k] for k in ("doc_id", "dataset", "lang", "title", "fmt", "n_pages")}
        sen.update(source=f"processed_para/{d['dataset']}", n_paras=len(rec["units"]), n_units=len(units),
                   n_tokens=sum(u["n_tokens"] for u in units), units=units)
        dst = out / "processed_sen" / d["dataset"] / f"{d['doc_id']}.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(sen, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{d['dataset']}:{d['doc_id'][:40]}: 문단 {len(rec['units'])} → 문장 {len(units)}", flush=True)


if __name__ == "__main__":
    main()
