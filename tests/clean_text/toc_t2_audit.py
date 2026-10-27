"""docx 목차 줄 지우기(stage2 T2) 전수 점검(tests 전용, 2026-10-26).

  source env.sh && PYTHONPATH=src .venv311/bin/python tests/clean_text/toc_t2_audit.py

사용자 원칙: "목차가 덜 지워지더라도 본문은 절대 지우면 안돼".
T2 = 번호로 끝나는 짧은 줄(12낱말 이하)이 3개 이상 이어지는 구간(stage2.toc_run_ids). 묶음(이어진 구간)마다 목차 증거를 잰다.
  E1 뒤에 다시 나옴: 줄 끝 번호를 뗀 제목(3낱말 이상이면 그대로, 짧으면 4글자 이상)이 묶음 뒤 블록 글에 다시 나오는 비율
  E2 번호 증가: 끝 번호가 앞 줄보다 작아지지 않는 비율
  E3 목차 제목: 묶음 바로 앞 5블록 안에 'Contents'·'목차'·'Table of Contents' 제목
  판정: 확실 = E1 ≥ 0.5, 또는 E3 이고 E2 ≥ 0.8 / 그 밖은 '의심' → 전부 출력해서 사람이 읽는다
출력: tests/clean_text/out/toc_t2_audit.md (집계 + 의심 묶음 전부 + 확실 묶음 무작위 표본)
"""
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from preprocess import stage2  # noqa: E402

NUM_END = re.compile(r"\s*\.{0,}\s*(\d{1,4})\s*$")
HEAD = re.compile(r"^\s*((table of )?contents|목\s*차|차\s*례|index)\s*:?\s*$", re.I)


def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9가-힣 ]+", " ", t.lower())).strip()


def runs_of(ids):
    out, cur = [], []
    for i in sorted(ids):
        if cur and i != cur[-1] + 1:
            out.append(cur); cur = []
        cur.append(i)
    if cur:
        out.append(cur)
    return out


def main():
    rnd = random.Random(0)
    sure, sus = [], []
    for p in sorted((ROOT / "data" / "meta_text_v2" / "docx_corpus").glob("*.json")):
        x = json.loads(p.read_text(encoding="utf-8"))
        blocks = x["blocks"] if isinstance(x, dict) else x
        ids = stage2.toc_run_ids(blocks)
        for run in runs_of(ids):
            lines = [blocks[i]["text"].strip() for i in run]
            after = norm(" ".join(b.get("text", "") for b in blocks[run[-1] + 1:]))
            titles = [norm(NUM_END.sub("", l)) for l in lines]
            ok = [t for t in titles if (len(t.split()) >= 3 or len(t) >= 4)]
            e1 = sum(t in after for t in ok) / max(1, len(ok))
            nums = [int(m.group(1)) for l in lines if (m := NUM_END.search(l))]
            e2 = sum(b >= a for a, b in zip(nums, nums[1:])) / max(1, len(nums) - 1)
            e3 = any(HEAD.match(blocks[j].get("text", "").strip()) for j in range(max(0, run[0] - 5), run[0]))
            rec = (p.stem, run[0], len(run), e1, e2, e3, lines)
            (sure if (e1 >= 0.5 or (e3 and e2 >= 0.8)) else sus).append(rec)
    n_sure, n_sus = sum(r[2] for r in sure), sum(r[2] for r in sus)
    L = ["# docx T2(번호로 끝나는 목차 줄 지우기) 전수 점검", "",
         f"묶음 {len(sure) + len(sus)}개 / 줄 {n_sure + n_sus:,}개. 목차 확실(E1 ≥ 0.5 또는 목차 제목 + 번호 증가 ≥ 0.8): 묶음 {len(sure)}, 줄 {n_sure:,} / "
         f"의심: 묶음 {len(sus)}, 줄 {n_sus:,}", "",
         f"## 의심 묶음 전부({len(sus)}개) - 사람이 읽고 판정", ""]
    for d, s, n, e1, e2, e3, lines in sus:
        L.append(f"### {d} (블록 {s}~, {n}줄) E1 {e1:.2f} E2 {e2:.2f} 목차제목 {e3}")
        L += [f"    {l[:160]}" for l in lines[:30]] + (["    …"] if n > 30 else []) + [""]
    L += [f"## 확실 묶음 무작위 15개(표본)", ""]
    for d, s, n, e1, e2, e3, lines in rnd.sample(sure, min(15, len(sure))):
        L.append(f"- {d[:45]} ({n}줄) E1 {e1:.2f} E2 {e2:.2f} 목차제목 {e3} | " + " / ".join(l[:60] for l in lines[:4]))
    (HERE / "out" / "toc_t2_audit.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:3]))


if __name__ == "__main__":
    main()
