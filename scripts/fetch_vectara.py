"""Vectara Open RAG Benchmark(arXiv PDF) 전체를 data/raw/vectara_pdf 로 받는다 (docs/restructure_plan.md 7절).

  .venv311/bin/python scripts/fetch_vectara.py meta          # 질문·정답·qrels·pdf_urls·corpus(섹션 텍스트) 1000편 (HF 데이터셋, 빠름)
  .venv311/bin/python scripts/fetch_vectara.py pdf [--delay 3]   # PDF 1000편 (arXiv, 예의상 요청 사이 --delay 초 대기, 이어받기 가능)
  .venv311/bin/python scripts/fetch_vectara.py manifest      # data/manifests/vectara.json 을 1000편으로 다시 만든다(기존 100편 필드 유지)

출처: https://huggingface.co/datasets/vectara/open_ragbench (CC BY-NC 4.0) 의 pdf/arxiv/ 폴더. 시스템 파이썬이 아니라 venv의 requests 없이 urllib만 쓴다.
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common.config import MANIFEST_DIR, RAW_DIR  # noqa: E402

HF = "https://huggingface.co/datasets/vectara/open_ragbench/resolve/main/pdf/arxiv"
HF_API = "https://huggingface.co/api/datasets/vectara/open_ragbench/tree/main/pdf/arxiv"
DEST = RAW_DIR / "vectara_pdf"
META = DEST / "_meta"
UA = {"User-Agent": "Mozilla/5.0 (research; DIP-challenge-lab chunking benchmark)"}


def get(url, timeout=120):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch_meta():
    META.mkdir(parents=True, exist_ok=True)
    for name in ("queries.json", "answers.json", "qrels.json", "pdf_urls.json"):
        data = get(f"{HF}/{name}")
        json.loads(data)                                                # 깨진 파일 방지
        (META / name).write_bytes(data)
        print(f"[meta] {name} {len(data) // 1024}KB")
    (META / "corpus").mkdir(exist_ok=True)
    ids = list(json.loads((META / "pdf_urls.json").read_text(encoding="utf-8")))
    done = 0
    for i, pid in enumerate(ids, 1):
        out = META / "corpus" / f"{pid}.json"
        if out.exists() and out.stat().st_size > 0:
            done += 1
            continue
        for attempt in range(4):
            try:
                data = get(f"{HF}/corpus/{pid}.json")
                json.loads(data)
                out.write_bytes(data)
                break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                if attempt == 3:
                    print(f"[meta] 실패 {pid}: {e}")
                time.sleep(2 * (attempt + 1))
        if i % 100 == 0:
            print(f"[meta] corpus {i}/{len(ids)}")
    print(f"[meta] corpus 파일 {len(list((META / 'corpus').glob('*.json')))}/{len(ids)}개 (이미 있던 것 {done})")


def fetch_pdf(delay):
    urls = json.loads((META / "pdf_urls.json").read_text(encoding="utf-8"))
    DEST.mkdir(parents=True, exist_ok=True)
    todo = [p for p in urls if not ((DEST / f"{p}.pdf").exists() and (DEST / f"{p}.pdf").stat().st_size > 10_000)]
    print(f"[pdf] 전체 {len(urls)}편, 받을 것 {len(todo)}편 (대기 {delay}초)", flush=True)
    fails = []
    for i, pid in enumerate(todo, 1):
        out = DEST / f"{pid}.pdf"
        ok = False
        for attempt in range(4):
            try:
                data = get(urls[pid], timeout=180)
                if not data.startswith(b"%PDF") or len(data) < 10_000:
                    raise ValueError(f"PDF가 아님({len(data)}B)")
                out.write_bytes(data)
                ok = True
                break
            except (urllib.error.URLError, TimeoutError, ValueError) as e:
                err = e
                time.sleep(10 * (attempt + 1))
        if not ok:
            fails.append(pid)
            print(f"[pdf] 실패 {pid}: {err}", flush=True)
        if i % 25 == 0 or i == len(todo):
            print(f"[pdf] {i}/{len(todo)} (실패 {len(fails)})", flush=True)
        time.sleep(delay)
    (DEST / "_failed.json").write_text(json.dumps(fails), encoding="utf-8")
    print(f"[pdf] 끝: 실패 {len(fails)}편")


def build_manifest():
    urls = json.loads((META / "pdf_urls.json").read_text(encoding="utf-8"))
    qrels = json.loads((META / "qrels.json").read_text(encoding="utf-8"))
    queries = json.loads((META / "queries.json").read_text(encoding="utf-8"))
    path = MANIFEST_DIR / "vectara.json"
    old = {e["paper_id"]: e for e in json.loads(path.read_text(encoding="utf-8"))}
    nq, srcs = {}, {}
    for qid, r in qrels.items():
        nq[r["doc_id"]] = nq.get(r["doc_id"], 0) + 1
        srcs.setdefault(r["doc_id"], set()).add(queries[qid]["source"])
    excl_path = MANIFEST_DIR / "vectara.excluded.json"                       # 변환 실패 등으로 뺀 문서(사유 기록)
    excluded = json.loads(excl_path.read_text(encoding="utf-8")) if excl_path.exists() else {}
    out = []
    for pid in urls:
        if pid in excluded:
            continue
        c = json.loads((META / "corpus" / f"{pid}.json").read_text(encoding="utf-8"))
        e = dict(old.get(pid, {}))
        e.update(paper_id=pid, url=urls[pid], title=c.get("title"), published=c.get("published"), categories=c.get("categories"),
                 n_queries=nq.get(pid, 0), query_sources=sorted(srcs.get(pid, [])), has_queries=pid in nq)
        e.setdefault("origin", "full_rest")                              # 기존 100편(v1_50, v2_added_50)은 그대로 두고 새로 받는 문서만 표시
        out.append(e)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[manifest] 제외 문서 {len(excluded)}편 {sorted(excluded)}")
    print(f"[manifest] {len(out)}편 (질문 있는 정답 문서 {sum(e['has_queries'] for e in out)}편, 어려운 오답 {sum(not e['has_queries'] for e in out)}편) → {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["meta", "pdf", "manifest"])
    ap.add_argument("--delay", type=float, default=3.0)
    a = ap.parse_args()
    {"meta": fetch_meta, "pdf": lambda: fetch_pdf(a.delay), "manifest": build_manifest}[a.what]()
