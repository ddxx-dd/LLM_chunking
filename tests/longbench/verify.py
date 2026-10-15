"""LongBench 단위 검증(tests 전용): src/preprocess/longbench.py 결과를 원본 jsonl과 비교한다.

  source env.sh && PYTHONPATH=src .venv311/bin/python tests/longbench/verify.py [--sample 20]

검사
  원문 보존: 공백을 무시하면 문단 단위를 이은 글 = 원본 context(HotpotQA는 'Passage N:' 줄을 뺀 것, 넣은 공백은 무시)
  문장 = 문단: 공백을 무시하면 문장 단위를 이은 글 = 문단 단위를 이은 글
  패시지: HotpotQA 단위의 패시지 번호 수 = 원본 'Passage N:' 머리글 수, 빈 패시지 없음
  정답 문자열: 정답(소문자·공백 정리)이 문맥에 들어 있는 질문 수, 원본과 처리 후 비교
  분포: 문단·문장 수, 토큰 분포, 3낱말 이하 문장, 150토큰 초과 문장
  표본: 데이터셋마다 무작위 문장(앞 문장 끝과 함께) → out/sample_sentences.jsonl (사람 판정용)
"""
import argparse
import json
import random
import re
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common.config import LONGBENCH  # noqa: E402
from preprocess.longbench import PASSAGE, is_script  # noqa: E402

OUT = HERE / "out"
NS = lambda t: "".join(t.split())                         # 공백 무시
NORM = lambda t: " ".join(t.lower().split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=20)
    ns = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(7)
    L, picks = ["# LongBench 단위 검증", ""], []
    for name in LONGBENCH:
        recs = [json.loads(x) for x in (ROOT / "data" / "raw" / "longbench" / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]
        qa = {q["_id"]: q for q in map(json.loads, (ROOT / "data" / "qa" / f"{name}.jsonl").read_text(encoding="utf-8").splitlines())}
        bad_text = bad_sen = bad_pass = 0
        ans_raw = ans_new = n_q = 0
        paras, sents = [], []
        seen = set()
        for r in recs:
            if name == "narrativeqa" and is_script(r["context"]):
                continue
            q = qa[r["_id"]]
            P = json.loads((ROOT / "data" / "processed_para" / name / f"{q['doc_id']}.json").read_text(encoding="utf-8"))
            S = json.loads((ROOT / "data" / "processed_sen" / name / f"{q['doc_id']}.json").read_text(encoding="utf-8"))
            ptext = " ".join(u["text"] for u in P["units"])
            n_q += 1
            ans_raw += any(NORM(a) in NORM(r["context"]) for a in r["answers"])
            ans_new += any(NORM(a) in NORM(ptext) or NORM(a) in NORM(" ".join(u["text"] for u in S["units"])) for a in r["answers"])
            if q["doc_id"] in seen:
                continue
            seen.add(q["doc_id"])
            raw = r["context"]
            if name == "hotpotqa":
                heads = [ln for ln in raw.split("\n") if PASSAGE.fullmatch(ln.strip())]
                raw = "\n".join(ln for ln in raw.split("\n") if not PASSAGE.fullmatch(ln.strip()))
                ps = [u["passage"] for u in P["units"]]
                bad_pass += sorted(set(ps)) != list(range(len(heads)))
            bad_text += NS(raw) != NS(ptext)
            bad_sen += NS(ptext) != NS(" ".join(u["text"] for u in S["units"]))
            paras += [u["n_tokens"] for u in P["units"]]
            for k, u in enumerate(S["units"]):
                sents.append((u["n_tokens"], len(u["text"].split())))
                picks.append((name, q["doc_id"], S["units"][k - 1]["text"][-60:] if k else "", u["text"]))
        L += [f"## {name}", "",
              f"- 문서 {len(seen)}, 질문 {n_q} | 원문과 글자가 다른 문서 {bad_text} | 문장≠문단 문서 {bad_sen}" + (f" | 패시지 번호 불일치 문서 {bad_pass}" if name == "hotpotqa" else ""),
              f"- 정답 문자열이 문맥에 있는 질문: 원본 {ans_raw} → 처리 후 {ans_new}" if LONGBENCH[name]["task"] == "qa" else "- (요약 과제: 정답 문자열 검사 없음)",
              f"- 문단 {len(paras)}개, 토큰 중앙값 {st.median(paras)}, 최대 {max(paras)}",
              f"- 문장 {len(sents)}개, 토큰 중앙값 {st.median(t for t, _ in sents)}, 90% {sorted(t for t, _ in sents)[int(.9 * len(sents))]}, 최대 {max(t for t, _ in sents)}, "
              f"3낱말 이하 {sum(w <= 3 for _, w in sents)} ({sum(w <= 3 for _, w in sents) / len(sents):.1%}), 150토큰 초과 {sum(t > 150 for t, _ in sents)}", ""]
    with open(OUT / "sample_sentences.jsonl", "w", encoding="utf-8") as fh:
        for name in LONGBENCH:
            ps = [p for p in picks if p[0] == name]
            for p in rnd.sample(ps, min(ns.sample, len(ps))):
                fh.write(json.dumps({"ds": p[0], "doc": p[1], "prev_end": p[2], "text": p[3]}, ensure_ascii=False) + "\n")
    (OUT / "summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
