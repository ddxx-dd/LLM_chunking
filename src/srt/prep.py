"""srt -> 공통 JSON (docs/restructure_plan.md 4.4절). 기존 legacy/src/srt/loader.py의 인코딩 순차 시도,
큐 정규식, 태그 제거, 정규화(NFC·공백)를 그대로 옮긴다. 태그 제거 후 빈 큐는 버리고 개수를 log에 기록한다.
"""
import re
import unicodedata
from pathlib import Path

ENCODINGS = ("utf-8-sig", "utf-8", "cp949", "euc-kr", "utf-16")

CUE_RE = re.compile(
    r"(\d+)\s*\n(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})[^\n]*\n(.*?)(?=\n\s*\n|\Z)",
    re.S,
)
TAG_RE = re.compile(r"<[^>]{1,40}>|\{\\[^}]{0,40}\}")


def read_text(filepath):
    """(텍스트, 사용한 인코딩)."""
    raw = Path(filepath).read_bytes()
    for enc in ENCODINGS:
        try:
            return raw.decode(enc), enc
        except (UnicodeDecodeError, LookupError):
            continue
    raise ValueError("디코딩 실패: " + str(filepath))


def normalize(s):
    s = unicodedata.normalize("NFC", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub("[\u200b\u200c\u200d\u200e\u200f\ufeff\xa0]", " ", s)   # 제로폭 문자·BOM·NBSP
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def to_sec(ts):
    ts = ts.replace(".", ",")
    h, m, rest = ts.split(":")
    s, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_srt_text(raw):
    """SRT 원문 -> (cue 블록 목록, log)."""
    matches = CUE_RE.findall(raw)
    log = {"cues_found": len(matches), "dropped_empty": 0}
    blocks = []
    for num, t0, t1, body in matches:
        line = normalize(" ".join(TAG_RE.sub("", body).split()))
        if not line:
            log["dropped_empty"] += 1
            continue
        blocks.append({"id": f"c{len(blocks) + 1}", "type": "cue", "text": line,
                       "index": int(num), "start": to_sec(t0), "end": to_sec(t1)})
    return blocks, log


def convert_srt(entry):
    """entry: {doc_id, dataset, lang, path, source_file, title}."""
    raw, enc = read_text(entry["path"])
    blocks, log = parse_srt_text(raw)
    log["encoding"] = enc
    return {
        "doc_id": entry["doc_id"], "dataset": entry["dataset"], "fmt": "srt", "lang": entry["lang"],
        "title": entry.get("title") or entry["doc_id"],
        "source_file": entry["source_file"],
        "parser": "regex-srt", "parser_options": {"encodings": list(ENCODINGS)},
        "n_pages": None,
        "log": log,
        "blocks": blocks,
    }


# ---- 영화 쌍 목록과 전처리 대상 문서 목록 (원래 preprocess/datasets.py에 있던 srt 부분)
import json

from common.config import MANIFEST_DIR, RAW_DIR, ROOT

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


def ensure_srt_manifest():
    path = MANIFEST_DIR / "srt.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    movies = []
    for key, (en, ko) in SRT_MOVIES.items():
        assert (RAW_DIR / "srt_eng" / en).exists(), en
        assert (RAW_DIR / "srt_kor" / ko).exists(), ko
        movies.append({"movie": key, "en_file": en, "ko_file": ko, "en_doc": f"{key}.en", "ko_doc": f"{key}.ko"})
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    (MANIFEST_DIR / "srt.json").write_text(json.dumps(movies, ensure_ascii=False, indent=1), encoding="utf-8")
    return movies


def srt_entries(dataset="srt"):
    """-> [{doc_id, dataset, lang, path, source_file, title}] (영화 쌍마다 en, ko 두 문서)."""
    out = []
    for m in ensure_srt_manifest():
        for lang, fname, folder, doc in (("en", m["en_file"], "srt_eng", m["en_doc"]),
                                         ("ko", m["ko_file"], "srt_kor", m["ko_doc"])):
            p = RAW_DIR / folder / fname
            out.append({"doc_id": doc, "dataset": dataset, "lang": lang, "path": p,
                        "source_file": _rel(p), "title": m["movie"]})
    return out
