"""문장 단위 규칙 조합 회귀 검증(tests 전용): preprocess.sentences(현재 V1)에 후보 규칙(아래 조각 붙이기)을 켜고 끄며 전체 데이터에 같은 지표를 잰다.

  source env.sh && PYTHONPATH=src .venv311/bin/python tests/sent_split/regress.py [--variants V0,V1,...] [--sample 15]

조합(VARIANTS: 규칙 손잡이)
  fix_en  영어 라벨 붙이기·약어 뒤 소문자 합치기
  glue    3낱말 이하 조각 붙이기(종결로 끝나면 앞, 아니면 뒤)
  num     숫자 뒤 점('2024. 4. 12.', '13.')은 종결 아님
  runin   영어에서 문단을 넘는 종결 조각은 소문자 시작만 앞에(문단 머리 제목은 다음 문단 쪽)
  frag    조각 정의 A: 3낱말 이하 / B: A 중 온전한 문장 제외 / C: B + 한국어는 공백 뺀 8글자 이하
  num_ko  num을 한국어에만
  V0 도구 + 수식 보호, V1 +fix_en, V2 +glue(A), V3 +num, V4 +runin, V4B·V4C 조각 정의만 바꿈, V5 = V4C에서 num→num_ko, V6 = V5에서 runin 끔
지표
  보존: 문서 글자(공백 무시)가 processed_para와 같은지, 표·코드·제목 단위가 그대로인지, 수식이 한 문장 안에 있는지
  판정 자료: out/v2_labels.json 경계의 정밀도·재현율·F1(표본 문단 하나를 단위로 to_sentences 전체 적용)
  분포: 데이터셋별 문장 수, 3낱말 이하 조각, 150토큰(근사: 낱말 120) 초과, 문단을 넘어 붙은 문장
  차이: 기준 조합과 달라진 문장을 데이터셋별로 무작위 추출 → out/regress_diff_<V>.jsonl (사람 판정용)
출력: out/regress_summary.md
"""
import argparse
import glob
import json
import random
import re
import sys
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))
from compare import boundaries  # noqa: E402
from preprocess import sentences as S  # noqa: E402

OUT = HERE / "out"
ORIG_FIX_EN = S.fix_en
VARIANTS = {"V0": {}, "V1": {"fix_en"}, "V2": {"fix_en", "glue"}, "V3": {"fix_en", "glue", "num"},
            "V4": {"fix_en", "glue", "num", "runin"}, "V4B": {"fix_en", "glue", "num", "runin", "B"}, "V4C": {"fix_en", "glue", "num", "runin", "C"},
            "V5": {"fix_en", "glue", "num_ko", "runin", "C"}, "V6": {"fix_en", "glue", "num_ko", "C"}}
ON = set()

# ── 조각 붙이기 후보(src에는 넣지 않음, 2026-10-15 사용자 결정: src는 V1, 청커 지표를 본 뒤 필요하면 V5를 옮긴다) ──
FRAG_WORDS = 3
TERMINAL = re.compile(r"(?:[.?!。]|[다음함임됨])[\s\"'”’)\]□■]*$")
TERMINAL_NUM = re.compile(r"(?:(?<![\d\s.])\.|[?!。]|[다음함임됨])[\s\"'”’)\]□■]*$")   # 숫자 뒤 점('2024. 4. 12.', '13.')은 종결 아님


def body(u):
    return u["text"][len(u["heading"]) + 1:] if u["heading"] else u["text"]


def complete(t, lang):
    """짧아도 온전한 문장: 영어 대문자 시작·2낱말 이상·글자 뒤 종결, 한국어 2어절 이상·'~다.' '~요.' '~까?'."""
    if lang == "ko":
        return len(t.split()) >= 2 and bool(re.search(r"(?:다|요)\.[\"'”’)]*$|까\?$", t))
    return len(t.split()) >= 2 and bool(re.match(r"[A-Z\"'“‘(]", t)) and bool(re.search(r"[A-Za-z][.?!][\"'”’)]*$", t))


def is_fragment(u, lang):
    t = body(u)
    if "C" in ON:
        short = len("".join(t.split())) <= 8 if lang == "ko" else len(t.split()) <= FRAG_WORDS
        return short and not complete(t, lang)
    return len(t.split()) <= FRAG_WORDS and not ("B" in ON and complete(t, lang))


def terminal(t, lang):
    num = "num" in ON or ("num_ko" in ON and lang == "ko")
    return bool((TERMINAL_NUM if num else TERMINAL).search(t))


def _group(u):
    return None if u["kind"] in S.KEEP else ("footnote" if u["kind"] == "footnote" else "body")


def _join(a, b, main):
    """a 뒤에 b를 잇는다(문단을 넘으면 줄바꿈). main(조각이 아닌 쪽)의 para를 따른다."""
    pages = a["pages"] + [p for p in b["pages"] if p not in a["pages"]]
    return {"text": a["text"] + (" " if a["para"] == b["para"] else "\n") + b["text"], "kind": main["kind"], "para": main["para"],
            "heading": a["heading"], "pages": pages}


def glue_fragments(units, lang):
    """조각을 이웃 문장에 붙인다: 종결로 끝나면 앞, 아니면(또는 제목이 달렸으면) 뒤. 표·코드·제목 제외, 각주는 각주끼리."""
    out, carry = [], None
    for u in units:
        g = _group(u)
        if carry is not None:
            if g == _group(carry):
                u = _join(carry, u, u)
            elif out and _group(out[-1]) == _group(carry):
                out[-1] = _join(out[-1], carry, out[-1])
            else:
                out.append(carry)
            carry = None
        if g and is_fragment(u, lang):
            back = out and _group(out[-1]) == g and not u["heading"] and terminal(u["text"], lang)
            if back and "runin" in ON and lang != "ko" and out[-1]["para"] != u["para"] and not re.match(r"[a-z]", u["text"]):
                back = False                                 # 영어 문단 머리 제목('Shift values.')은 다음 문단 쪽
            if back:
                out[-1] = _join(out[-1], u, out[-1])
            else:
                carry = u
            continue
        out.append(u)
    if carry is not None:
        if out and _group(out[-1]) == _group(carry):
            out[-1] = _join(out[-1], carry, out[-1])
        else:
            out.append(carry)
    return out


def use(v):
    ON.clear()
    ON.update(VARIANTS[v])
    S.fix_en = ORIG_FIX_EN if "fix_en" in ON else (lambda xs: xs)


def sentences(rec):
    units = S.to_sentences(rec)
    return glue_fragments(units, rec.get("lang") or "en") if "glue" in ON else units


def run_doc(args):
    v, f = args
    use(v)
    rec = json.loads(Path(f).read_text(encoding="utf-8"))
    units = sentences(rec)
    J = lambda xs: "".join("".join(x.split()) for x in xs)
    keep_p = [u["text"] for u in rec["units"] if u["kind"] in S.KEEP]
    keep_s = [u["text"] for u in units if u["kind"] in S.KEEP]
    sj = [J([u["text"]]) for u in units]
    f_split = 0
    for u in rec["units"]:
        for a, b in u.get("formulas", []):
            fm = J([u["text"][a:b]])
            if fm and not any(fm in x for x in sj):
                f_split += 1
    bod = [u for u in units if u["kind"] not in S.KEEP]
    return {"doc": Path(f).stem, "ds": rec["dataset"], "sents": [u["text"] for u in units],
            "text_ok": J(u["text"] for u in units) == J(u["text"] for u in rec["units"]), "keep_ok": keep_p == keep_s,
            "formula_split": f_split, "n": len(units),
            "frag": sum(len(body(u).split()) <= 3 for u in bod), "long": sum(len(body(u).split()) > 120 for u in bod),
            "cross": sum("\n" in body(u) for u in bod)}


def label_score(v):
    use(v)
    Sm = [json.loads(x) for x in (OUT / "v2_samples.jsonl").read_text(encoding="utf-8").splitlines()]
    C = [json.loads(x) for x in (OUT / "v2_candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    lab = {int(k): x for k, x in json.loads((OUT / "v2_labels.json").read_text(encoding="utf-8")).items()}
    B = {}
    for i, x in enumerate(Sm):
        rec = {"lang": x["lang"], "units": [{"text": x["text"], "kind": "para", "heading": "", "pages": [1]}]}
        B[i] = boundaries(x["text"], [u["text"] for u in sentences(rec)])
    res = {}
    for lang in ("en", "ko"):
        cs = [c for c in C if c["lang"] == lang and lab.get(c["id"]) is not None]
        true = [c for c in cs if lab[c["id"]] == 1]
        cut = [c for c in cs if c["pos"] in B[c["i"]]]
        tp = sum(lab[c["id"]] == 1 for c in cut)
        p, r = tp / max(1, len(cut)), tp / max(1, len(true))
        extra = sum(1 for i, b in B.items() if Sm[i]["lang"] == lang for q in b if q not in {c["pos"] for c in C if c["i"] == i})
        res[lang] = (p, r, 2 * p * r / max(1e-9, p + r), extra)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=",".join(VARIANTS), help="첫 조합이 차이 비교의 기준")
    ap.add_argument("--sample", type=int, default=15, help="기준 조합과 달라진 문장 표본 수(데이터셋마다)")
    ns = ap.parse_args()
    files = sorted(f for f in glob.glob(str(ROOT / "data" / "processed_para" / "*" / "*.json")) if not f.endswith(".dropped.json"))
    rows, base = [], None
    for v in ns.variants.split(","):
        with Pool(16) as pool:
            docs = pool.map(run_doc, [(v, f) for f in files], chunksize=8)
        ls = label_score(v)
        row = {"v": v, "label": ls, "text_bad": sum(not d["text_ok"] for d in docs), "keep_bad": sum(not d["keep_ok"] for d in docs),
               "formula_split": sum(d["formula_split"] for d in docs), "ds": {}}
        for ds in sorted({d["ds"] for d in docs}):
            dd = [d for d in docs if d["ds"] == ds]
            row["ds"][ds] = {k: sum(d[k] for d in dd) for k in ("n", "frag", "long", "cross")}
        if base is not None:
            rnd = random.Random(7)
            old = {d["doc"]: set(d["sents"]) for d in base}
            changed = [(d["ds"], d["doc"], s) for d in docs for s in d["sents"] if s not in old[d["doc"]]]
            row["changed"] = len(changed)
            with open(OUT / f"regress_diff_{v}.jsonl", "w", encoding="utf-8") as fh:
                for ds in sorted({c[0] for c in changed}):
                    cs = [c for c in changed if c[0] == ds]
                    for c in rnd.sample(cs, min(ns.sample, len(cs))):
                        fh.write(json.dumps({"ds": c[0], "doc": c[1], "text": c[2]}, ensure_ascii=False) + "\n")
        rows.append(row)
        base = docs if base is None else base
        print(v, "완료", flush=True)

    L = ["# 문장 단위 규칙 조합 비교", "",
         "| 조합 | 영어 P/R/F1 | 한국어 P/R/F1 | 글자 달라진 문서 | 표·코드·제목 바뀐 문서 | 두 문장에 걸친 수식 | 기준 조합과 달라진 문장 |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        e, k = r["label"]["en"], r["label"]["ko"]
        L.append(f"| {r['v']} | {e[0]:.1%}/{e[1]:.1%}/{e[2]:.1%} | {k[0]:.1%}/{k[1]:.1%}/{k[2]:.1%} | {r['text_bad']} | {r['keep_bad']} | {r['formula_split']} | {r.get('changed', '-')} |")
    L += ["", "| 조합 | 데이터셋 | 문장 수 | 3낱말 이하 조각 | 120낱말 초과 | 문단 넘어 붙은 문장 |", "|---|---|---|---|---|---|"]
    for r in rows:
        for ds, x in r["ds"].items():
            L.append(f"| {r['v']} | {ds} | {x['n']} | {x['frag']} ({x['frag'] / max(1, x['n']):.1%}) | {x['long']} | {x['cross']} |")
    (OUT / "regress_summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
