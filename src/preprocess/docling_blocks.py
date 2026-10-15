"""Docling 변환과 Docling 문서 → 블록 목록 (pdf/docx 공용).

  pdf_converter()            PDF 변환기: OCR 끔, 표 FAST, cuDNN 끔(서버 충돌 우회), 수식 인식(생성 상한 768토큰, 한 번에 16개)
  docx_converter()           docx 변환기: Docling 기본
  prepare_pdf(src, workdir)  PyMuPDF로 PDF 내용 스트림을 정규화한 사본(실패하면 원본)
  drop_corner_items(doc)     쪽 위·아래 5% 안 항목(머리말·쪽 번호) 지우기
  PdfHeadingLevel            PDF 제목 수준: 제목 번호로 복원
  docx_heading_level(doc)    docx 제목 수준: Word 제목 스타일
  build_blocks(doc, level)   Docling 문서 → (blocks, log)

블록 변환 규칙
  SECTION_HEADER, TITLE      heading (level은 호출자가 넘기는 규칙으로 정함)
  TEXT, CAPTION 등 글         paragraph   (CAPTION은 role="caption", 표와 연결되면 for=<표 블록 id>)
  LIST_ITEM                  list_item   (항목마다 별도 블록)
  TABLE                      table       (캡션 없는 마크다운 표 - CAPTION이 별도 항목으로 나온다)
  FORMULA, CODE              formula, code (글이 비면 제외)
  FOOTNOTE                   footnote    (따로 모아 블록 목록 맨 끝에 붙인다. after_block_id에 원래 위치 기록)
  PICTURE, CHART, PAGE_HEADER, PAGE_FOOTER   제외
  표의 하위 항목(셀 안의 문단)은 이미 마크다운 표에 들어 있으면 따로 내보내지 않는다(같은 내용이 표와 문단으로 두 번 나온 실측).
  같은 inline 그룹 소속의 연속 항목은 원래 한 문장이므로 공백 하나로 이어 한 블록으로 만든다.
  3칸 이상 공백은 1칸으로 줄인다(목차 표의 정렬용 공백).
블록에는 id(b0, b1 ...), page(pdf만), heading_path(현재 제목 스택)를 채운다.
"""
import re
import time
from pathlib import Path

from docling_core.types.doc import DocItemLabel, GroupLabel

FORMULA_MAX_TOKENS = 768
FORMULA_BATCH = 16
CORNER_EDGE = 0.05                              # 쪽 위/아래 5% 안 항목은 머리말·바닥글·쪽 번호로 보고 지운다
CLEAN_OPTIONS = {"clean": True, "garbage": 4, "deflate": True}   # PyMuPDF 정제(mutool clean -s와 같은 내용 스트림 정규화)

SKIP_LABELS = frozenset({DocItemLabel.PICTURE, DocItemLabel.CHART, DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER})
HEADING_LABELS = (DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE)
_WS_COLLAPSE_RE = re.compile(r" {3,}")
_TABLE_NORM_RE = re.compile(r"[\s*]+")
_TYPE_BY_LABEL = {DocItemLabel.FOOTNOTE: "footnote", DocItemLabel.LIST_ITEM: "list_item",
                  DocItemLabel.FORMULA: "formula", DocItemLabel.CODE: "code"}
# "3" -> 1단계, "3.1" -> 2단계, "3.1.2" -> 3단계. 끝의 점은 무시.
_NUMBERED_HEADING_RE = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,2}){0,3})\.?\s+\S")
# 번호가 없어도 논문에서 항상 최상위 절인 제목들 - 1단계 고정.
_FIRST_LEVEL_WORDS_RE = re.compile(r"^(abstract|introduction|conclusions?|references?|acknowledge?ments?|appendix\w*)\b", re.I)


# ---------------------------------------------------------------- 변환기
def pdf_converter():
    """PDF 변환기. 수식 인식의 생성 상한·묶음 크기는 설치된 패키지를 고치지 않고 이 프로세스 안에서만 바꾼다."""
    import torch
    torch.backends.cudnn.enabled = False         # 서버의 cuDNN/드라이버 버전 불일치로 레이아웃 모델이 바로 죽는다(필수 우회)
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.models.inference_engines.vlm.transformers_engine import TransformersVlmEngine
    from docling.models.stages.code_formula.code_formula_vlm_model import CodeFormulaVlmModel

    orig = TransformersVlmEngine.predict_batch

    def capped(self, batch):                     # Docling 기본 2048토큰 → 768(반복 고장 수식이 끝없이 생성되는 것 방지)
        for x in batch:
            x.max_new_tokens = min(x.max_new_tokens or FORMULA_MAX_TOKENS, FORMULA_MAX_TOKENS)
        return orig(self, batch)

    TransformersVlmEngine.predict_batch = capped
    CodeFormulaVlmModel.elements_batch_size = FORMULA_BATCH
    opts = PdfPipelineOptions()
    opts.do_ocr = False                          # 디지털 텍스트 PDF라 OCR 불필요
    opts.table_structure_options.mode = TableFormerMode.FAST
    opts.do_formula_enrichment = True
    conv = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
    conv.initialize_pipeline(InputFormat.PDF)
    return conv


def docx_converter():
    from docling.document_converter import DocumentConverter
    return DocumentConverter()


def pdf_options(clean_applied):
    """1차 결과에 남기는 PDF 변환·블록 설정."""
    return {"do_ocr": False, "table_mode": "FAST", "corner_edge": CORNER_EDGE, "pdf_clean": dict(CLEAN_OPTIONS) if clean_applied else None,
            "formula": {"max_new_tokens": FORMULA_MAX_TOKENS, "batch": FORMULA_BATCH},
            "index_as_body": True, "fix_trailing": True, "fix_local_swaps": True, "line_geometry": True, "formula_text_layer": True,
            "pictures": True, "fake_spaces": True}


# ---------------------------------------------------------------- PDF 정제·구석 항목
def prepare_pdf(src, workdir):
    """변환에 쓸 PDF 경로 -> (경로, info). 일부 PDF(한글 문서 도구가 만든 것 등)는 docling-parse가 본문 한 줄을 통째로 못 읽는데,
    PyMuPDF로 내용 스트림을 정규화해 다시 저장하면 읽힌다. 정제에 실패하면(암호화·손상 등) 원본을 쓰고 info["error"]에 이유를 남긴다."""
    import pymupdf

    src = Path(src)
    info = {"enabled": True, "applied": False, "orig_bytes": src.stat().st_size}
    dst = Path(workdir) / f"{src.stem}.clean.pdf"
    t = time.time()
    try:
        with pymupdf.open(str(src)) as doc:
            n_pages = doc.page_count
            doc.save(str(dst), **CLEAN_OPTIONS)
        with pymupdf.open(str(dst)) as out:
            if out.page_count != n_pages:
                raise ValueError(f"정제본의 쪽 수가 다르다: {n_pages} -> {out.page_count}")
        info.update(applied=True, tool=f"pymupdf-{pymupdf.__version__}", clean_bytes=dst.stat().st_size, seconds=round(time.time() - t, 2))
        return dst, info
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        return src, info


def drop_corner_items(doc):
    """Docling이 놓친 구석 머리말/쪽 번호가 제목으로 잡히는 것을 막는다(prov bbox 기준, 그림·표 제외). 지운 개수를 돌려준다."""
    corner = []
    for item, _ in doc.iterate_items():
        if not item.prov or item.label in (DocItemLabel.PICTURE, DocItemLabel.TABLE):
            continue
        p = item.prov[0]
        height = doc.pages[p.page_no].size.height
        box = p.bbox.to_top_left_origin(height)
        if box.b <= height * CORNER_EDGE or box.t >= height * (1 - CORNER_EDGE):
            corner.append(item)
    if corner:
        doc.delete_items(node_items=corner)
    return len(corner)


# ---------------------------------------------------------------- 제목 수준
class PdfHeadingLevel:
    """PDF 제목 수준을 제목 번호로 복원한다(Docling이 PDF에서 주는 level은 글자 크기 추정이라 믿지 않는다 - 실측으로 전부 1단계).
    번호 없는 제목은 Abstract/Introduction/Conclusion(s)/References/Acknowledg(e)ments/Appendix 계열이면 1단계,
    그 외는 직전 '번호 있는' 제목의 한 단계 아래(최대 3단계, 앞에 번호 제목이 없으면 1단계). 문서 하나당 새로 만들 것."""

    def __init__(self):
        self._last_numbered_level = None

    def __call__(self, item):
        text = item.text.strip()
        m = _NUMBERED_HEADING_RE.match(text)
        if m:
            level = m.group(1).count(".") + 1
            self._last_numbered_level = level
            return level
        if _FIRST_LEVEL_WORDS_RE.match(text):
            return 1
        if self._last_numbered_level is not None:
            return min(self._last_numbered_level + 1, 3)
        return 1


def docx_heading_level(dl):
    """Docling이 docx에서 주는 제목 수준(Word 스타일 Heading 1~5). TITLE이 있는 문서는 TITLE을 1단계로 두고
    SECTION_HEADER를 한 단계 내린다(Docling 마크다운 내보내기와 같은 위계). 최대 6."""
    has_title = any(it.label == DocItemLabel.TITLE for it, _ in dl.iterate_items())

    def level(item):
        if item.label == DocItemLabel.TITLE:
            return 1
        lv = int(getattr(item, "level", 1) or 1)
        return min(lv + 1 if has_title else lv, 6)
    return level


# ---------------------------------------------------------------- Docling 문서 → 블록
def clean(text):
    return _WS_COLLAPSE_RE.sub(" ", text or "").strip()


def page_of(item):
    prov = getattr(item, "prov", None)
    return prov[0].page_no if prov else None


def inline_group_ref(item, doc):
    """item이 Docling의 'inline' 그룹(문장 하나가 서식 경계마다 여러 항목으로 쪼개진 것) 소속이면 그 그룹 참조, 아니면 None."""
    parent = getattr(item, "parent", None)
    if parent is None:
        return None
    try:
        group = parent.resolve(doc)
    except Exception:
        return None
    return parent.cref if getattr(group, "label", None) == GroupLabel.INLINE else None


def table_ancestor(item, doc):
    """item의 조상 중 표(TABLE)가 있으면 그 표 항목, 없으면 None. 부모를 못 풀면 None(표가 아닌 것으로 본다)."""
    cur = item
    for _ in range(50):
        parent = getattr(cur, "parent", None)
        if parent is None:
            return None
        try:
            cur = parent.resolve(doc)
        except Exception:
            return None
        if getattr(cur, "label", None) == DocItemLabel.TABLE:
            return cur
    return None


def in_table_text(text, table_md):
    """text가 표 마크다운 안에 (공백·굵게 표시 `*` 차이를 무시하고) 들어 있는가."""
    norm = lambda s: _TABLE_NORM_RE.sub(" ", s).strip()
    return bool(table_md) and norm(text) in norm(table_md)


def make_table_to_md(doc):
    """표 -> 캡션 없는 마크다운(캡션은 별도 CAPTION 항목으로도 나와 두 번 나오는 것을 막는다). compact_tables: 열 맞춤 공백 없앰."""
    from docling_core.transforms.serializer.markdown import MarkdownDocSerializer, MarkdownParams

    labels = set(MarkdownParams.model_fields["labels"].default) - {DocItemLabel.CAPTION}
    serializer = MarkdownDocSerializer(doc=doc, params=MarkdownParams(labels=labels, compact_tables=True))
    return lambda item: serializer.serialize(item=item).text


def build_blocks(doc, heading_level, table_to_md=None, page_of=page_of, items=None):
    """DoclingDocument -> (blocks, log). heading_level(item) -> int는 제목마다 문서 순서대로 호출된다.
    items: (항목, 수준) 목록(안 주면 doc.iterate_items() 순서). 1차가 읽기 순서를 고친 목록을 넘긴다.
    page_of(item)의 결과가 블록 page에 그대로 들어간다(1차가 위치 정보를 실어 보내는 데 쓴다)."""
    table_to_md = table_to_md or make_table_to_md(doc)
    blocks, footnotes, stack = [], [], []     # stack: [(level, text)]
    caption_items = []                        # [(docling self_ref, block)]
    table_captions = {}                       # caption self_ref -> 표 블록
    pending = {"v": None}                     # 이어 붙일 inline 그룹
    table_md = {}                             # 표 self_ref -> 내보낸 마크다운(하위 문단 중복 검사용)
    counter = {"n": 0}
    log = {"docling_items": 0, "docling_tables": 0, "empty_tables": 0, "skipped": {},
           "footnotes": 0, "captions": 0, "inline_groups_merged": 0, "empty_headings": 0, "table_cell_dups": 0}

    def new_id():
        counter["n"] += 1
        return f"t{counter['n']}"

    def path():
        return [t for _, t in stack]

    def emit(label, text, item):
        text = clean(text)
        if not text:
            return
        typ = _TYPE_BY_LABEL.get(label, "paragraph")
        b = {"id": new_id(), "type": typ, "text": text, "heading_path": path(), "page": page_of(item)}
        if label == DocItemLabel.CAPTION:
            b["role"] = "caption"
            caption_items.append((getattr(item, "self_ref", None), b))
            log["captions"] += 1
        if typ == "footnote":
            b["after_block_id"] = blocks[-1]["id"] if blocks else None
            footnotes.append(b)
            log["footnotes"] += 1
        else:
            blocks.append(b)

    def flush():
        p = pending["v"]
        if p is not None:
            if len(p["parts"]) > 1:
                log["inline_groups_merged"] += 1
            emit(p["label"], " ".join(p["parts"]), p["item"])
            pending["v"] = None

    for item, _ in (items if items is not None else doc.iterate_items()):
        label = item.label
        log["docling_items"] += 1
        if label in SKIP_LABELS:
            flush()
            log["skipped"][label.name] = log["skipped"].get(label.name, 0) + 1
            continue
        if label in HEADING_LABELS:
            flush()
            text = clean(item.text)
            if not text:
                log["empty_headings"] += 1
                continue
            level = int(heading_level(item))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, text))
            blocks.append({"id": new_id(), "type": "heading", "level": level, "text": text,
                           "heading_path": path(), "page": page_of(item)})
        elif label == DocItemLabel.TABLE:
            flush()
            log["docling_tables"] += 1
            md = clean(table_to_md(item))
            if not md:
                log["empty_tables"] += 1
                continue
            b = {"id": new_id(), "type": "table", "text": md, "heading_path": path(), "page": page_of(item)}
            blocks.append(b)
            table_md[getattr(item, "self_ref", None)] = md
            for c in getattr(item, "captions", None) or []:
                table_captions[c.cref] = b
        elif (getattr(item, "text", None) or "").strip():
            anc = table_ancestor(item, doc)
            if anc is not None and in_table_text(clean(item.text), table_md.get(getattr(anc, "self_ref", None))):
                flush()
                log["table_cell_dups"] += 1
                continue
            ref = inline_group_ref(item, doc)
            if ref is not None:
                p = pending["v"]
                if p is None or p["ref"] != ref:
                    flush()
                    p = pending["v"] = {"ref": ref, "label": label, "item": item, "parts": []}
                p["parts"].append(item.text)
            else:
                flush()
                emit(label, item.text, item)
    flush()

    for ref, b in caption_items:                 # 표 캡션 연결(캡션이 표 앞/뒤 어디에 있어도 된다)
        t = table_captions.get(ref)
        if t is not None:
            b["for"] = t["id"]

    final = blocks + footnotes                   # 각주는 맨 끝으로
    remap = {b["id"]: f"b{i}" for i, b in enumerate(final)}
    for b in final:
        b["id"] = remap[b["id"]]
        if b.get("after_block_id") is not None:
            b["after_block_id"] = remap[b["after_block_id"]]
        if b.get("for") is not None:
            b["for"] = remap[b["for"]]
    return final, log
