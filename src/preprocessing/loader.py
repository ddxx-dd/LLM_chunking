import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
import fitz  # PyMuPDF
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
    s = re.sub(r"[\u200b\u200c\u200d\u200e\u200f\ufeff\xa0]", " ", s)
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
            # 왼쪽 위 구석 칸이 비어있고 그 아래로 행 이름이 오는 표(예: 재무 항목표)에서
            # 헤더가 빈 문자열이면 `if k and v`가 그 칸의 값을 통째로 버렸음(행 이름 자체가
            # 사라져서 서로 다른 행이 구분조차 안 됨) - 빈 헤더는 위치 이름으로 대체하고,
            # 값이 있는지만으로 판단한다.
            header = [h or f"col{i+1}" for i, h in enumerate(header)]
            for ri, row in enumerate(item.rows[1:]):
                values = [c.text.strip() for c in row.cells]
                pairs = {k: v for k, v in zip(header, values) if v}
                if pairs:
                    b.add(json.dumps(pairs, ensure_ascii=False), "row", {"table": tbl_no, "row": ri, "header": header})
                    log["rows"] += 1

    return b.done(Path(filepath).name, "docx", log)

FOOTNOTE_SIZE_RATIO = 0.9  # 본문 폰트 크기의 90% 미만이면 각주/참고문헌급으로 간주

def _repeated_lines(pdf, threshold=0.3):
    """줄(line) 단위로 반복 빈도를 계산해 러닝헤더/푸터를 찾는다 - 실제 필터링도
    줄 단위라 단위를 맞춰야 한다(블록 단위로 계산하면 반복 헤더가 블록/줄 불일치로
    새는 버그가 있었음 - 실측 발견). 페이지 번호처럼 숫자만 다른 것도 잡으려고
    숫자를 #으로 치환한 뒤 정규화된 문자열이 전체 페이지의 threshold 이상에서
    나타나면 반복 부속물로 본다."""
    counts = Counter()
    n_pages = 0
    for page in pdf:
        n_pages += 1
        keys = set()
        for block in page.get_text("dict")["blocks"]:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                text = "".join(s["text"] for s in line["spans"]).strip()
                if text:
                    keys.add(re.sub(r"\d+", "#", text))
        for k in keys:
            counts[k] += 1
    return {k for k, c in counts.items() if c / n_pages >= threshold}

def _dominant_body_size(pdf):
    """문서 전체에서 글자 수 기준으로 가장 흔한 폰트 크기 = 본문 크기.
    span 개수가 아니라 글자 수로 가중해야 짧지만 큰 제목이 압도적으로 많은
    본문 글자 수를 못 이긴다."""
    char_counts = Counter()
    for page in pdf:
        for block in page.get_text("dict")["blocks"]:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    n = len(span["text"].strip())
                    if n:
                        char_counts[round(span["size"], 1)] += n
    return char_counts.most_common(1)[0][0]

def _bbox_overlaps(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1

def _merge_page_spanning_tables(pages_raw_tables):
    """표가 페이지 경계에서 잘리면 find_tables()가 별도 Table로 인식하는데, 이어지는
    쪽엔 헤더 행이 없어서 그 페이지의 첫 데이터 행이 헤더로 잘못 인식되는 문제가
    있었다(실측 발견). "바로 다음 페이지 + 그 페이지의 첫 표 + 컬럼 수 동일"이면
    같은 표의 연속으로 보고 행을 그대로 이어붙인다(헤더 재추출 없이 전부 데이터로
    취급). 위치(bbox) 기반 판정("표 하단이 페이지 하단 여백 근처")은 실측해보니 안
    맞는 경우가 있어서 버리고 이 조건만 쓴다."""
    surviving = []
    active = None
    for pi, tables in enumerate(pages_raw_tables):
        for idx, (bbox, rows) in enumerate(tables):
            if not rows:
                continue
            is_first_table_on_page = (idx == 0)
            if (active is not None
                    and is_first_table_on_page
                    and pi == active["last_page"] + 1
                    and active["rows"]
                    and len(rows[0]) == len(active["rows"][0])):
                active["rows"].extend(rows)
                active["last_page"] = pi
            else:
                if active is not None:
                    surviving.append(active)
                active = {"first_page": pi, "first_bbox": bbox, "rows": list(rows), "last_page": pi}
    if active is not None:
        surviving.append(active)
    return surviving

def _clean_pdf_line(s):
    """PDF 한 줄 텍스트 정리 - normalize()와 달리 좌우 공백을 보존한다(줄 끝
    공백 유무가 문단 병합 시 어절 경계 판별의 신호라 지우면 안 됨). NFC 정규화 +
    투명 서식문자(ZWSP/ZWNJ/ZWJ 등) 치환 + 내부 연속 공백/탭 축약만 적용한다."""
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"[​‌‍‎‏﻿\xa0]", " ", s)
    s = re.sub(r"[ \t]+", " ", s)
    return s

def _needs_join_space(buf):
    """버퍼 끝이 문장부호인데 공백이 없으면 문장경계 판별(split_sentences의
    "(?<=[.!?])\\s+")이 그 지점을 못 잡는다 - 공백 하나를 보정해줘야 한다. 원본
    줄 끝에 이미 공백이 있으면(단어/어절 경계) 그대로 신뢰하고 아무것도 안
    붙인다(대부분의 경우 - 한국어는 줄바꿈 시 하이픈 없이 음절 중간을 그냥
    끊는 게 정상 조판 관행이라, 공백 유무를 원본 그대로 따르는 게 맞다).
    각주번호("1)")가 문장부호 바로 뒤에 공백 없이 붙는 게 한국 학술지의
    표준 표기라(실측 발견), 끝의 각주번호 패턴은 떼고 나서 검사해야 한다."""
    if not buf or buf.endswith((" ", "\n")):
        return False
    check = re.sub(r"\d+\)$", "", buf.rstrip())
    return bool(check) and check[-1] in ".!?"

def load_pdf(filepath):
    pdf = fitz.open(str(filepath))
    b = _Builder()
    log = {"paras": 0, "footnotes": 0, "tables": 0, "rows": 0}

    dominant_size = _dominant_body_size(pdf)
    footnote_threshold = dominant_size * FOOTNOTE_SIZE_RATIO
    repeated = _repeated_lines(pdf)

    # 표는 미리 문서 전체를 훑어서 페이지 경계 병합까지 끝내둔다 - "다음 페이지에
    # 뭐가 있는지"를 알아야 병합 여부를 판단할 수 있어서 본문 순회보다 먼저 해야 함.
    pages_raw_tables, pages_table_bboxes = [], []
    for page in pdf:
        tables = page.find_tables().tables
        pages_raw_tables.append([(t.bbox, t.extract()) for t in tables])
        pages_table_bboxes.append([t.bbox for t in tables])
    merged_tables = _merge_page_spanning_tables(pages_raw_tables)
    tables_by_first_page = {}
    for mt in merged_tables:
        tables_by_first_page.setdefault(mt["first_page"], []).append(mt)

    # 문단/각주는 줄 단위로 즉시 Unit을 만들지 않고, 같은 종류(kind)의 줄이
    # 이어지는 동안 버퍼에 누적한 뒤 종류가 바뀌는 시점(문단↔각주, 표 등장)에만
    # 하나의 Unit으로 확정한다 - PDF의 "줄"은 페이지 폭 때문에 생기는 렌더링
    # 산물일 뿐 문장/문단 경계가 아니라서, 줄마다 Unit을 만들면 문장 중간이
    # 강제로 끊긴다(실측 발견: "...놀라운 일이" + "아니다..."가 한 문장인데 줄이
    # 갈려서 두 문장으로 오인됨). 이어붙일 때 구분자를 넣지 않고 원본 텍스트를
    # 그대로 잇는다 - PyMuPDF가 뽑은 줄 끝의 공백 유무가 이미 정확한 신호라서
    # (어절 경계면 공백 있음, 음절 중간 절단이면 공백 없음) 그대로 신뢰하면 된다.
    # 버퍼는 페이지 루프를 넘어 유지되므로 페이지 경계에 걸친 문단/각주도 자동
    # 병합된다(표의 페이지 경계 병합과 같은 이치).
    # 각주/참고문헌은 본문 흐름에 안 끼워넣고 다 모아뒀다가 문서 맨 끝에 원래
    # 순서대로 붙인다 - 페이지 경계에서 문장이 끊긴 채로 각주가 그 사이에 끼면
    # 본문이 더 이상해지므로(실측 발견), 처음부터 본문 스트림에서 완전히 빼는 게
    # 가장 단순하고 안전하다. 각주와 참고문헌을 구분하지 않는다(둘 다 본문보다
    # 작은 폰트를 쓰는 경우가 많아 같은 신호로 잡히고, 구분할 필요도 없다고 확인함).
    footnotes = []
    buf_para, buf_fn = "", ""

    def flush_para():
        nonlocal buf_para
        if buf_para.strip():
            b.add(buf_para.strip(), "para", {"style": "", "heading": False})
            log["paras"] += 1
        buf_para = ""

    def flush_fn():
        nonlocal buf_fn
        if buf_fn.strip():
            footnotes.append(buf_fn.strip())
            log["footnotes"] += 1
        buf_fn = ""

    for pi, page in enumerate(pdf):
        table_bboxes = pages_table_bboxes[pi]
        entries = []  # (y0, kind, text) - 페이지 안에서 위→아래 순서로 정렬할 것

        for block in page.get_text("dict")["blocks"]:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                text = "".join(s["text"] for s in line["spans"])
                if not text.strip():
                    continue
                bbox = line["bbox"]
                key = re.sub(r"\d+", "#", text.strip())
                if key in repeated:
                    continue
                if any(_bbox_overlaps(bbox, tb) for tb in table_bboxes):
                    continue  # 표 셀 텍스트 중복 방지
                # 폰트크기는 글자 수 가중평균 - 본문 줄 중간에 낀 위첨자 각주번호
                # (짧은 조각)가 span 개수 기준 단순평균을 끌어내려 본문 전체를
                # 각주로 오분류하는 버그가 있었음(실측 발견, 가중평균으로 해결).
                spans = [(s["size"], len(s["text"])) for s in line["spans"] if s["text"].strip()]
                total_chars = sum(n for _, n in spans)
                avg_size = sum(sz * n for sz, n in spans) / total_chars if total_chars else dominant_size
                kind = "footnote" if avg_size < footnote_threshold else "para"
                # normalize()가 아니라 _clean_pdf_line()을 쓴다 - normalize()의
                # 마지막 strip()이 줄 끝 공백까지 지워버리는데, 그 공백 유무가
                # 병합 시 어절 경계 판별의 유일한 신호라(실측 발견) 지우면 안 된다.
                entries.append((bbox[1], kind, _clean_pdf_line(text)))

        for mt in tables_by_first_page.get(pi, []):
            entries.append((mt["first_bbox"][1], "table", mt))
        entries.sort(key=lambda e: e[0])

        for _, kind, payload in entries:
            if kind == "para":
                if buf_fn:
                    flush_fn()
                if _needs_join_space(buf_para):
                    buf_para += " "
                buf_para += payload
            elif kind == "footnote":
                if buf_para:
                    flush_para()
                if _needs_join_space(buf_fn):
                    buf_fn += " "
                buf_fn += payload
            else:
                flush_para()
                flush_fn()
                log["tables"] += 1
                rows = payload["rows"]
                header = [h or f"col{i+1}" for i, h in enumerate(rows[0])]
                for ri, row in enumerate(rows[1:]):
                    pairs = {k: v for k, v in zip(header, row) if v}
                    if pairs:
                        b.add(json.dumps(pairs, ensure_ascii=False), "row",
                              {"table": log["tables"], "row": ri, "header": header})
                        log["rows"] += 1

    flush_para()
    flush_fn()
    for fn in footnotes:
        b.add(fn, "para", {"style": "", "heading": False})

    return b.done(Path(filepath).name, "pdf", log)

def load_file(filepath):
    ext = Path(filepath).suffix.lower()
    if ext == ".srt":
        return load_srt(str(filepath))
    if ext == ".docx":
        return load_docx(str(filepath))
    if ext == ".pdf":
        return load_pdf(str(filepath))
    raise ValueError("지원하지 않는 형식: " + ext)

def load_docx_bundle(paths):
    """여러 docx 파일(같은 큰 주제의 소주제 묶음이든, 서로 다른 주제 묶음이든)을
    각각 독립된 Doc으로 로드한다. 문서 경계를 넘어 청킹/검색하지 않도록
    묶지 않고 리스트로 반환 -> 상위 파이프라인이 문서별로 청킹하게 한다."""
    paths = sorted(Path(p) for p in paths)
    return [load_docx(str(p)) for p in paths]