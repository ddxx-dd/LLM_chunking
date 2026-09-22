import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

from langchain_core.documents import Document

SEP = "\n"

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

FOOTNOTE_SIZE_RATIO = 0.9  # 본문 폰트 크기의 90% 미만이면 각주/참고문헌급으로 간주

def _repeated_lines(pdf, threshold=0.3):
    """줄 단위 반복 빈도로 러닝헤더/푸터 탐지(블록 단위로 하면 블록/줄 불일치로 샘)."""
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
    """글자 수 가중 최빈 폰트크기 = 본문 크기(span 개수 기준이면 제목류가 왜곡함)."""
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
    """"다음 페이지 + 그 페이지 첫 표 + 컬럼수 동일"이면 표가 페이지 경계에서
    이어진 것으로 보고 합친다(헤더 재추출 없이 데이터 행으로 취급) - bbox 위치
    기반 판정은 실측상 안 맞는 경우가 있어 이 조건만 쓴다."""
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
    """PDF 한 줄 정리 - normalize()와 달리 좌우 공백을 보존한다(어절 경계 판별 신호)."""
    s = unicodedata.normalize("NFC", s)
    s = re.sub("[​‌‍‎‏﻿\xa0]", " ", s)
    s = re.sub(r"[ \t]+", " ", s)
    return s

def _needs_join_space(buf):
    """버퍼 끝이 문장부호+공백없음이면 문장경계 정규식이 못 잡으므로 공백을 보정한다.
    각주번호("1)")는 예외로 떼고 검사(한국 학술지 표기 관행)."""
    if not buf or buf.endswith((" ", "\n")):
        return False
    check = re.sub(r"\d+\)$", "", buf.rstrip())
    return bool(check) and check[-1] in ".!?"

def load_pdf(filepath):
    import fitz  # PyMuPDF - PDF를 안 쓰는 스크립트가 fitz 미설치 환경에서도 돌게 지연 임포트
    pdf = fitz.open(str(filepath))
    b = _Builder()
    log = {"paras": 0, "footnotes": 0, "tables": 0, "rows": 0}

    dominant_size = _dominant_body_size(pdf)
    footnote_threshold = dominant_size * FOOTNOTE_SIZE_RATIO
    repeated = _repeated_lines(pdf)

    # 표는 페이지경계 병합까지 미리 끝내둔다(다음 페이지 정보가 필요해서 본문 순회보다 먼저).
    pages_raw_tables, pages_table_bboxes = [], []
    for page in pdf:
        tables = page.find_tables().tables
        pages_raw_tables.append([(t.bbox, t.extract()) for t in tables])
        pages_table_bboxes.append([t.bbox for t in tables])
    merged_tables = _merge_page_spanning_tables(pages_raw_tables)
    tables_by_first_page = {}
    for mt in merged_tables:
        tables_by_first_page.setdefault(mt["first_page"], []).append(mt)

    # 문단/각주는 줄 단위로 즉시 Unit화하지 않고 같은 kind가 이어지는 동안 버퍼에
    # 누적한다 - PDF의 "줄"은 문장 경계가 아닌 렌더링 산물이라(줄마다 Unit을 만들면
    # 문장 중간이 강제로 끊김). 페이지 루프를 넘어 버퍼를 유지해 페이지 경계에 걸친
    # 문단/각주도 표처럼 자동 병합된다. 각주/참고문헌은 본문 흐름에 안 끼워넣고 모아뒀다가
    # 문서 끝에 붙인다(중간에 끼면 본문 문장이 더 이상해짐 - 실측 발견, 둘은 구분 안 함).
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
                # 글자수 가중평균 폰트크기 - 위첨자 각주번호(짧은 조각)가 단순평균을
                # 끌어내려 본문 줄 전체를 각주로 오분류하는 버그 방지(실측 발견).
                spans = [(s["size"], len(s["text"])) for s in line["spans"] if s["text"].strip()]
                total_chars = sum(n for _, n in spans)
                avg_size = sum(sz * n for sz, n in spans) / total_chars if total_chars else dominant_size
                kind = "footnote" if avg_size < footnote_threshold else "para"
                # normalize()는 줄 끝 공백까지 지워 병합 신호를 없애므로 안 쓴다.
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
