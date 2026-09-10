import json
import re
import sys
import unicodedata
from pathlib import Path
from docx import Document as DocxDocument
from docx.table import Table
from docx.text.paragraph import Paragraph

sys.path.append(str(Path(__file__).resolve().parent.parent))
from core import Doc, Unit

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
    s = re.sub(r"[\u200b\u200e\u200f\ufeff\xa0]", " ", s)
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()

class _Builder:
    def __init__(self):
        self.parts = []
        self.units = []
        self.pos = 0

    def add(self, text, kind, meta=None):
        if not text:
            return
        start = self.pos
        end = self.pos + len(text)
        self.units.append(Unit(start, end, kind, meta or {}))
        self.parts.append(text + SEP)
        self.pos += len(text) + len(SEP)

    def done(self, name, fmt, log):
        return Doc(name, "".join(self.parts), self.units, fmt, log)

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
        # 화자 표기("JOHN: ")도 원본이 실제로 보여준 정보라 지우지 않고 그대로 둔다 -
        # SDH(청각장애인용) 자막처럼 화자 표기 자체가 대사의 일부인 경우가 있고,
        # 번역 대상에서 제외할 근거가 없다.
        line = normalize(" ".join(TAG_RE.sub("", body).split()))
        if not line:
            log["dropped_empty"] += 1
            continue
        meta = {"index": int(num), "t_start": _to_sec(t0), "t_end": _to_sec(t1)}
        b.add(line, "cue", meta)

    return b.done(Path(filepath).name, "srt", log)

def _iter_body(doc):
    for child in doc.element.body.iterchildren():
        tag = child.tag.split("}")[-1]
        if tag == "p":
            yield Paragraph(child, doc)
        elif tag == "tbl":
            yield Table(child, doc)

def load_docx(filepath):
    doc = DocxDocument(filepath)
    b = _Builder()
    log = {"paras": 0, "tables": 0, "rows": 0}
    tbl_no = 0

    for item in _iter_body(doc):
        if isinstance(item, Paragraph):
            t = normalize(item.text)
            # 길이 기준으로 버리지 않는다 - 1~2글자짜리 소제목/라벨도 docx의
            # 일부이므로 그대로 보존한다. 완전히 빈 문단만 노이즈로 취급.
            if not t:
                continue
            style = item.style.name if item.style else ""
            is_head = style.startswith(("Heading", "제목", "Title"))
            b.add(t, "para", {"style": style, "heading": is_head})
            log["paras"] += 1

        elif isinstance(item, Table):
            tbl_no += 1
            log["tables"] += 1
            if not item.rows:
                continue
            header = [c.text.strip() for c in item.rows[0].cells]
            for ri, row in enumerate(item.rows[1:]):
                values = [c.text.strip() for c in row.cells]
                pairs = {k: v for k, v in zip(header, values) if k and v}
                if pairs:
                    b.add(json.dumps(pairs, ensure_ascii=False), "row", {"table": tbl_no, "row": ri, "header": header})
                    log["rows"] += 1

    return b.done(Path(filepath).name, "docx", log)

def load_file(filepath):
    ext = Path(filepath).suffix.lower()
    if ext == ".srt":
        return load_srt(str(filepath))
    if ext == ".docx":
        return load_docx(str(filepath))
    raise ValueError("지원하지 않는 형식: " + ext)

def load_docx_bundle(paths):
    """여러 docx 파일(같은 큰 주제의 소주제 묶음이든, 서로 다른 주제 묶음이든)을
    각각 독립된 Doc으로 로드한다. 문서 경계를 넘어 청킹/검색하지 않도록
    묶지 않고 리스트로 반환 -> 상위 파이프라인이 문서별로 청킹하게 한다."""
    paths = sorted(Path(p) for p in paths)
    return [load_docx(str(p)) for p in paths]