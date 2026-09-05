# 파일 -> Doc 정규화.
# 1) 지저분한 것을 치운다   (인코딩, <i> 태그, 빈 큐 등)
# 2) Doc.text 를 만든다     (두 청킹 방식이 똑같이 받는 입력)
# 3) Doc.units 를 남긴다    (타임스탬프·표 구조 = 병합의 스켈레톤)

import sys
import re
import json
import unicodedata
from pathlib import Path

from docx import Document as DocxDocument
from docx.table import Table
from docx.text.paragraph import Paragraph

sys.path.append(str(Path(__file__).resolve().parent.parent))
from core import Doc, Unit

SEP = "\n"
ENCODINGS = ("utf-8-sig", "utf-8", "cp949", "euc-kr", "utf-16")

# 인코딩을 순서대로 시도한다.
def read_text(filepath):
    raw = Path(filepath).read_bytes()
    for enc in ENCODINGS:
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    raise ValueError("디코딩 실패: " + str(filepath))
    

# 모든 로더가 공통으로 거치는 텍스트 정리.
# NFC 정규화가 중요한 이유:
#      macOS에서 만든 파일은 한글이 NFD(자모 분리)로 저장된다.
#      '한글' -> 'ㅎㅏㄴㄱㅡㄹ' 처럼 보이고, 길이도 2가 아니라 6이 되어
#      오프셋 계산이 전부 어긋난다.
def normalize(s):
    s = unicodedata.normalize("NFC", s)                      # NFD -> NFC
    s = s.replace("\r\n", "\n").replace("\r", "\n")          # 윈도우 개행 통일
    s = re.sub(r"[\u200b\u200e\u200f\ufeff\xa0]", " ", s)    # 폭0공백·BOM·nbsp
    s = re.sub(r"[ \t]+", " ", s)                            # 연속 공백 축약
    return s.strip()


#    예) b = _Builder()
#        b.add("아~뭐여~?", "cue", {"t_start":94.8})
#        b.add("또 돼지들...", "cue", {"t_start":96.6})
#        -> text  = "아~뭐여~?\n또 돼지들...\n"
#           units = [Unit(0,6,...), Unit(7,13,...)]
#                          ↑ 6 + len("\n") = 7 에서 다음 유닛 시작

class _Builder:
    def __init__(self):
        self.parts, self.units, self.pos = [], [] , 0
    
    def add(self, text, kind, meta=None):
        if not text:
            return
        self.units.append(Unit(self.pos, self.pos + len(text), kind, meta or {}))
        self.parts.append(text + SEP)
        self.pos += len(text) + len(SEP)
        
    def done(self,name,fmt,log):
        return Doc(name, "".join(self.parts), self.units, fmt, log)
    
    
    
#══════════════════════════════════════════════════════════════
# 1. 자막 (.srt)
# ══════════════════════════════════════════════════════════════
# 자막 한 블록의 구조:
#     7                                    <- 번호
#     00:01:34,750 --> 00:01:36,567        <- 시작 --> 끝 
#     아~뭐여~?                             <- 내용 (여러 줄일 수 있음)
#     (빈 줄)

CUE_RE = re.compile(
    r"(\d+)\s*\n"                                          #  번호
    r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*"               #  시작  (,와 . 둘 다 허용)
    r"(\d{2}:\d{2}:\d{2}[,.]\d{3})"                        #  끝
    r"[^\n]*\n"                                            #  좌표(X1:..) 등 흡수
    r"(.*?)(?=\n\s*\n|\Z)", re.S)

# 자막에서 걸러내야 하는 것들 
TAG_RE = re.complie(r"<[^>]{1,40}>|\{\\[^}]{0,40}\}")
#   <i>기울임</i>, {\an8}화면상단  같은 서식 태그
SPEAKER_RE = re.compile(r"^\s*[A-Z가-힣]{1,12}\s*[:：]\s*")
#   "지훈: 안녕하세요" -> "안녕하세요"   (화자 표시는 번역 대상이 아님)

DASH_RE    = re.compile(r"(^|\s)-\s*\S")
#   "-별거 아니에요~ - 접때도 그러두만…"  = 한 큐에 두 화자
#   ★ 쪼개지 않는다.  큐 1개 = 유닛 1개를 깨면 병합의 1:1 대응이 무너진다.
#     meta에 표시만 남기고, 번역 프롬프트에서 '-' 유지를 지시한다.


# '00:01:34,750' -> 94.75 (초)
def _to_sec(ts):
    ts = ts.replace(".", ",")
    h, m, rest = ts.split(":")
    s, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000
    

# 반환 예)
# doc.text  = "아~뭐여~?\n또 돼지들 싹 다 잡아다 파묻는 거여?\n..."
# doc.units = [Unit(0, 6,"cue",{"index":7, "t_start":94.8,"t_end":96.6}),
#                   Unit(7,28,"cue",{"index":8, "t_start":96.6,"t_end":99.6}), ...]
# doc.log   = {"cues_found":779, "dropped_bracket":18, "dropped_lyric":21, ...}
def load_srt(filepath, drop_bracket=True, drop_lyric=True, credit_head=5):
    raw = read_text(filepath)
    b = _Builder()
    log = {"cues_found": 0, "dropped_empty": 0, "dropped_bracket": 0,
           "dropped_lyric": 0, "dropped_credit": 0, "multi_speaker": 0,
           "tags_removed": 0, "malformed": 0}
    
    blocks = CUE_RE.findall(raw)
    log["cues_found"] = len(blocks)
    log["malformed"] = max(0, len(re.findall(r"^\d+\s*$", raw, re.M)) - len(blocks))
    
    for idx, (num, t0, t1, body) in enumerate(blocks):
        log["tags_removed"] += len(TAG_RE.findall(body))
        line = normalize(" ".join(TAG_RE.sub("", body).split()))
        line = SPEAKER_RE.sub("", line)

        if not line:
            log["dropped_empty"] += 1;  continue
        if idx < credit_head and CREDIT_RE.search(line):
            log["dropped_credit"] += 1;  continue
        if drop_bracket and BRACKET_RE.match(line):
            log["dropped_bracket"] += 1;  continue
        if drop_lyric and LYRIC_RE.search(line):
            log["dropped_lyric"] += 1;  continue

        meta = {"index": int(num), "t_start": _to_sec(t0), "t_end": _to_sec(t1)}
        if DASH_RE.search(line):
            meta["multi_speaker"] = True     
            log["multi_speaker"] += 1
        b.add(line, "cue", meta)

    return b.done(Path(filepath).name, "srt", log)
    
    
# ══════════════════════════════════════════════════════════════
# 2. 워드 (.docx)
# ══════════════════════════════════════════════════════════════

HEADING_PREFIX = ("Heading", "제목", "Title")

def _iter_body(doc):
    for child in doc.element.body.iterchildren():
        tag = child.tag.split("}")[-1]        
        if tag == "p":
            yield Paragraph(child, doc)
        elif tag == "tbl":
            yield Table(child, doc)
            
# 워드 파일 -> Doc
# 표를 Json 형식으로 변환한다.
# 반환 예)
# doc.text  = '포인터 기초\n포인터는 다른 변수의...\n{"연산자": "&", ...}\n...'
# doc.units = [Unit(0,  6,"para",{"style":"Heading 1","heading":True}),
#                   Unit(7, 38,"para",{"style":"Normal","heading":False}),
#                   Unit(65,93,"row", {"table":1,"row":0,"header":["연산자","의미"]})]
def load_docx(filepath, min_para_len = 2):
    doc = DocxDocument(filepath)
    b = _Builder()
    log = {"paras": 0, "headings": 0, "tables": 0, "rows": 0,
           "dropped_short": 0}
    tbl_no = 0
    
    for item in _iter_body(doc):
        # ── 문단 ──────────────────────────────────────────
        if isinstance(item, Paragraph):
            t = normalize(item.text)
            if len(t) < min_para_len:            # 빈 문단·한 글자 문단은 제외
                log["dropped_short"] += 1;  continue
            style = item.style.name if item.style else ""
            is_head = style.startswith(HEADING_PREFIX)
            # heading 플래그가 나중에 '정답 라벨'이 된다 (evaluation.py 참조)
            b.add(t, "para", {"style": style, "heading": is_head})
            log["paras"] += 1
            log["headings"] += int(is_head)
        # ── 표 ────────────────────────────────────────────
        else:
            tbl_no += 1
            log["tables"] += 1
            rows = sum(len(r.cells) for r in item.rows)      # 중복 포함 셀 수
            if not rows:
                continue

            header = rows[0]

            for ri, values in enumerate(rows[1:]):
                pairs = {k: v for k, v in zip(header, values) if k and v}
                if not pairs:
                    continue
                b.add(json.dumps(pairs, ensure_ascii=False), "row",
                      {"table": tbl_no, "row": ri, "header": header})
                log["rows"] += 1

    return b.done(Path(filepath).name, "docx", log)


# ══════════════════════════════════════════════════════════════
# 3. 평문 (LongBench 등)
# ══════════════════════════════════════════════════════════════
#  평문 -> Doc.  병합이 필요 없으므로 units는 문단 단위로만 둔다.
def load_text(text, name = "text"):
    b = _Builder()
    for para in re.split(r"\n\s*\n", text):     # 빈 줄로 문단 분리
        t = normalize(para)
        if t:
            b.add(t, "para", {})
    return b.done(name, "text", {"paras": len(b.units)})
            
        
def load_file(filepath):"
    ext = Path(filepath).suffix.lower()
    if ext == ".srt":
        return load_srt(filepath)
    if ext == ".docx":
        return load_docx(filepath)
    if ext in (".txt"):
        return load_text(read_text(filepath), Path(filepath).name)
    raise ValueError("지원하지 않는 형식: " + ext)
          

        
        
    
    
    

    
                    
        
    
        
        
    
    