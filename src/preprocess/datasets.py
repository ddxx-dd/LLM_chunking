"""전처리 대상 문서 목록 (data/manifests/<dataset>.json 기준). allganize·srt 매니페스트는 원본 메타에서
처음 한 번 만든다(분할 정보 split은 7단계에서 추가)."""
import csv
import json
import re
from pathlib import Path

from common.config import MANIFEST_DIR, RAW_DIR, ROOT, dataset_cfg

# legacy/src/config.py 의 SUBTITLE_DATASETS (영어 파일, 한국어 파일)
SRT_MOVIES = {
    "Noah": ("Noah_Eng.srt", "노아.srt"),
    "Deadpool": ("Deadpool_Eng.srt", "데드풀.srt"),
    "InsidiousChapter2": ("Insidious_Chapter2_Eng.srt", "인시디어스2.srt"),
    "DoctorStrange": ("Doctor_Strange_Eng.srt", "닥터스트레인지.srt"),
    "CaptainAmericaCivilWar": ("Captain_America_Civil_War_Eng.srt", "캡틴아메리카시빌워.srt"),
}


def _rel(p):
    return str(Path(p).resolve().relative_to(ROOT))


def _write_manifest(name, data):
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    (MANIFEST_DIR / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def ensure_allganize_manifest():
    path = MANIFEST_DIR / "allganize.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(open(RAW_DIR / "allganize_pdf" / "_meta" / "documents.csv", encoding="utf-8")))
    docs = []
    for r in rows:
        stem = Path(r["file_name"]).stem
        docs.append({"doc_id": stem, "domain": r["domain"], "file_name": r["file_name"],
                     "orig_file_name": r["orig_file_name"], "pages": int(r["pages"]), "url": r["url"],
                     "path": f"data/raw/allganize_pdf/{r['domain']}/{r['file_name']}"})
    assert len({d["doc_id"] for d in docs}) == len(docs), "allganize doc_id 중복"
    _write_manifest("allganize", docs)
    return docs


def ensure_srt_manifest():
    path = MANIFEST_DIR / "srt.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    movies = []
    for key, (en, ko) in SRT_MOVIES.items():
        assert (RAW_DIR / "srt_eng" / en).exists(), en
        assert (RAW_DIR / "srt_kor" / ko).exists(), ko
        movies.append({"movie": key, "en_file": en, "ko_file": ko, "en_doc": f"{key}.en", "ko_doc": f"{key}.ko"})
    _write_manifest("srt", movies)
    return movies


def _vectara_title(paper_id):
    p = RAW_DIR / "vectara_pdf" / "_meta" / "corpus" / f"{paper_id}.json"
    if p.exists():
        t = json.loads(p.read_text(encoding="utf-8")).get("title")
        if t:
            return re.sub(r"\s+", " ", t).strip()
    return None


def load_entries(dataset):
    """-> [{doc_id, dataset, lang, path(절대), source_file(상대), title(선택)}] (매니페스트 순서)."""
    cfg = dataset_cfg(dataset)
    out = []
    if dataset == "vectara":
        for m in json.loads((MANIFEST_DIR / "vectara.json").read_text(encoding="utf-8")):
            p = RAW_DIR / "vectara_pdf" / f"{m['paper_id']}.pdf"
            out.append({"doc_id": m["paper_id"], "dataset": dataset, "lang": cfg["lang"], "path": p,
                        "source_file": _rel(p), "title": _vectara_title(m["paper_id"]) or m.get("title")})
    elif dataset == "allganize":
        for m in ensure_allganize_manifest():
            p = ROOT / m["path"]
            out.append({"doc_id": m["doc_id"], "dataset": dataset, "lang": cfg["lang"], "path": p,
                        "source_file": m["path"], "domain": m["domain"]})
    elif dataset == "docx_corpus":
        for m in json.loads((MANIFEST_DIR / "docx_corpus.json").read_text(encoding="utf-8")):
            p = ROOT / m["path"]
            out.append({"doc_id": Path(m["file"]).stem, "dataset": dataset, "lang": cfg["lang"], "path": p,
                        "source_file": m["path"], "topic": m["topic"]})
    elif dataset == "srt":
        for m in ensure_srt_manifest():
            for lang, fname, folder, doc in (("en", m["en_file"], "srt_eng", m["en_doc"]),
                                             ("ko", m["ko_file"], "srt_kor", m["ko_doc"])):
                p = RAW_DIR / folder / fname
                out.append({"doc_id": doc, "dataset": dataset, "lang": lang, "path": p,
                            "source_file": _rel(p), "title": m["movie"]})
    else:
        raise KeyError(dataset)
    ids = [e["doc_id"] for e in out]
    assert len(set(ids)) == len(ids), f"{dataset}: doc_id 중복"
    return out
