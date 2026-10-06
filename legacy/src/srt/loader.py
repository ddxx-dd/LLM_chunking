import re
import unicodedata
from pathlib import Path

from langchain_core.documents import Document

SEP = "\n"
ENCODINGS = ("utf-8-sig", "utf-8", "cp949", "euc-kr", "utf-16")

def read_text(filepath):
    raw = Path(filepath).read_bytes()
    for enc in ENCODINGS:
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    raise ValueError("디코딩 실패: " + str(filepath))

def normalize(s):
    s = unicodedata.normalize("NFC", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub("[​‌‍‎‏﻿\xa0]", " ", s)
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()

class _Builder:
    """LangChain Document(page_content, metadata)로 조립 - Unit은 metadata["units"]의
    딕셔너리 리스트로 들어간다({"start":,"end":,"kind":,"meta":{...}})."""
    def __init__(self):
        self.parts = []
        self.units = []
        self.pos = 0

    def add(self, text, kind, meta=None):
        if not text:
            return
        start = self.pos
        end = self.pos + len(text)
        self.units.append({"start": start, "end": end, "kind": kind, "meta": meta or {}})
        self.parts.append(text + SEP)
        self.pos += len(text) + len(SEP)

    def done(self, name, fmt, log):
        return Document(page_content="".join(self.parts),
                         metadata={"name": name, "fmt": fmt, "log": log, "units": self.units})

CUE_RE = re.compile(
    r"(\d+)\s*\n(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})[^\n]*\n(.*?)(?=\n\s*\n|\Z)",
    re.S
)
TAG_RE = re.compile(r"<[^>]{1,40}>|\{\\[^}]{0,40}\}")

def _to_sec(ts):
    ts = ts.replace(".", ",")
    h, m, rest = ts.split(":")
    s, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

def load_srt(filepath):
    raw = read_text(filepath)
    b = _Builder()
    log = {"cues_found": 0, "dropped_empty": 0}
    blocks = CUE_RE.findall(raw)
    log["cues_found"] = len(blocks)

    for num, t0, t1, body in blocks:
        line = normalize(" ".join(TAG_RE.sub("", body).split()))
        if not line:
            log["dropped_empty"] += 1
            continue
        meta = {"index": int(num), "t_start": _to_sec(t0), "t_end": _to_sec(t1)}
        b.add(line, "cue", meta)

    return b.done(Path(filepath).name, "srt", log)
