"""현재 LumberChunker 코사인 판('둘로 나눠 항상 자르기', θ = 700) vs LangChain recursive 초기값 그대로: RAG 검색 지표(tests 전용).

  source env.sh && PYTHONPATH=src .venv311/bin/python tests/chunking/lumber_vs_recursive.py

lumber: tests/chunking/lumber_split.py(문단을 θ까지 모으고, 자리마다 (1 − 앞뒤 코사인) × 크기 균형^0.5가 가장 큰 곳에서 항상 자름)
recursive: RecursiveCharacterTextSplitter() 초기값 그대로(chunk_size 4000글자, chunk_overlap 200, 구분자 \\n\\n → \\n → 공백 → 글자)
데이터·지표: lumber_d.py와 같음(vectara QA 100편 + 방해 100편 seed 0, NarrativeQA 14권; 문서 Hit@3·MRR@10, 근거 Hit@3·Precision@3·MRR@10,
  근거 재현율 top-3·예산 1,500). 조정할 값이 없어 질문 전체로 잰다. NarrativeQA는 대본(nqa17)을 뺀 결과도 함께 적는다.
출력: tests/chunking/out/lumber_vs_recursive.md
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))
import compare as C  # noqa: E402
from lumber_d import evaluate  # noqa: E402
from lumber_split import THETA, lumber_split  # noqa: E402
from v4_v5_lumber import qa_sets, safe_encoder  # noqa: E402
from chunking.sentences import paragraphs  # noqa: E402
from common.models import make_token_counter  # noqa: E402
from preprocess import longbench  # noqa: E402

NAMES = ["lumber(θ 700, α 0.5)", "recursive(초기값)"]


def main():
    import faiss
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    rec = RecursiveCharacterTextSplitter()
    count, encode = make_token_counter(), safe_encoder()
    rng = np.random.default_rng(0)
    rnd = random.Random(0)
    qa = qa_sets()
    vq = sorted(d for d in qa["vectara"] if (ROOT / "data" / "clean_text" / "vectara" / f"{d}.json").exists())
    allv = sorted(f.stem for f in (ROOT / "data" / "clean_text" / "vectara").glob("*.json") if not f.name.endswith(".dropped.json"))
    qdocs = rnd.sample(vq, 100)
    distract = rnd.sample([d for d in allv if d not in qa["vectara"]], 100)
    vdocs = {d: json.loads((ROOT / "data" / "clean_text" / "vectara" / f"{d}.json").read_text(encoding="utf-8"))["text"] for d in qdocs + distract}
    books = longbench.documents("narrativeqa")
    data = {"vectara": ({d: (t, t.split("\n\n")) for d, t in vdocs.items()}, [(d, q, ev) for d in qdocs for q, ev in qa["vectara"][d]]),
            "narrativeqa": ({b["doc_id"]: (b["text"], paragraphs(b["text"])) for b in books},
                            [(q["doc_id"], q["question"], q["answers"]) for q in longbench.questions("narrativeqa")])}
    L = ["# 현재 LumberChunker 코사인 판(θ 700, α 0.5) vs recursive 초기값(4000글자, 겹침 200)", "",
         "지표: FAISS 데이터셋 전체 색인(방해 문서 포함), top-3 / MRR@10 / 예산 1,500토큰. 괄호 = recursive 대비 짝지은 차이(* = 95% 구간이 0을 포함하지 않음).", ""]
    for ds, (docs, qrecs) in data.items():
        chunks = {n: {} for n in NAMES}
        for d, (text, units) in docs.items():
            T = count(units)
            rg, _ = lumber_split(T, encode(units), THETA, alpha=0.5)
            chunks[NAMES[0]][d] = ["\n\n".join(units[i:j]) for i, j in rg]
            chunks[NAMES[1]][d] = [c for c in rec.split_text(text) if c.strip()]
        print(ds, "청킹 완료", flush=True)
        dtext = {d: t for d, (t, _) in docs.items()}
        res = {n: evaluate(ds, chunks[n], qrecs, count, encode, faiss, dtext) for n in NAMES}
        print(ds, "평가 완료", flush=True)
        L += [f"## {ds} (색인 문서 {len(docs)}편, 질문 {len(res[NAMES[0]]['문서 Hit@3'])}개)", "",
              "| 청커 | 청크 | 평균 토큰 | 중앙값 | 최대 | 50토큰 미만 |", "|---|---|---|---|---|---|"]
        for n in NAMES:
            t = np.array([x for cs in chunks[n].values() for x in count(cs)])
            L.append(f"| {n} | {len(t):,} | {t.mean():.0f} | {np.median(t):.0f} | {t.max()} | {np.mean(t < 50):.1%} |")
        subsets = [("전체", None)] + ([("대본 nqa17 제외", "nqa17")] if ds == "narrativeqa" else [])
        for title, drop in subsets:
            keep = [i for i, d in enumerate(res[NAMES[0]]["_doc"]) if d != drop]
            L += ["", f"### 검색 ({title}, 질문 {len(keep)}개)", "", "| 지표 | " + " | ".join(NAMES) + " |", "|---|---|---|"]
            for mt in [k for k in res[NAMES[0]] if not k.startswith("_")]:
                a = np.array([res[NAMES[0]][mt][i] for i in keep], float); b = np.array([res[NAMES[1]][mt][i] for i in keep], float)
                if mt == "top-3 토큰":
                    L.append(f"| {mt}(평균) | {a.mean():.0f} | {b.mean():.0f} |")
                    continue
                ma, ha = C.ci(a, rng); mb, hb = C.ci(b, rng); dm, dh = C.ci(a - b, rng)
                L.append(f"| {mt} | {'**' if ma >= mb else ''}{ma:.3f}{'**' if ma >= mb else ''} ±{ha:.3f} ({dm:+.3f}{'*' if abs(dm) > dh else ''}) | "
                         f"{'**' if mb > ma else ''}{mb:.3f}{'**' if mb > ma else ''} ±{hb:.3f} |")
        L.append("")
    (HERE / "out" / "lumber_vs_recursive.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
