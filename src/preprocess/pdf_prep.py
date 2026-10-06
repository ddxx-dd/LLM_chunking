"""PDF -> 공통 JSON (docs/restructure_plan.md 4.2절). 기존 legacy/src/pdf_track/loader.py의 설정과
규칙을 그대로 옮긴다: OCR 끔, 표 FAST 모드, 서버 cuDNN 우회, 제목 번호로 수준 복원, 쪽 상/하단 5% 항목 제거.
"""
import re
import tempfile
import time
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

from docling_core.types.doc import DocItemLabel

from common.config import PROCESSED_DIR
from preprocess._docling import build_blocks

CORNER_EDGE = 0.05    # 쪽 위/아래 5% 안에 있는 항목은 머리말·바닥글·쪽번호로 보고 지운다

# PDF 정제(sanitize) 옵션 - docs/restructure_plan.md 4.2절.
# 일부 PDF(예: 한글 문서 도구가 만든 allganize journal-31-3-308)는 docling-parse가 본문 한 줄을 통째로 읽지
# 못한다. 원인은 docling-parse의 텍스트 추출 단계(레이아웃 모델이 아니다)이고 설정으로 끌 수 없다.
# PyMuPDF로 내용 스트림을 정규화해 다시 저장하면 같은 줄이 읽힌다(실측: 글자 +47, 표본 24편에서 표·제목 개수
# 변화 0, 1% 이상 감소 0). clean=True는 mutool clean -s 와 같은 내용 스트림 정규화다.
CLEAN_OPTIONS = {"clean": True, "garbage": 4, "deflate": True}
CLEAN_WORKDIR = PROCESSED_DIR / "_tmp_clean"      # 정제본을 잠깐 두는 곳(프로젝트 안, git 제외, 변환 직후 삭제)

# "3" -> 1단계, "3.1" -> 2단계, "3.1.2" -> 3단계. 끝의 점은 무시.
_NUMBERED_HEADING_RE = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,2}){0,3})\.?\s+\S")
# 번호가 없어도 논문에서 항상 최상위 절인 제목들 - 1단계 고정.
_FIRST_LEVEL_WORDS_RE = re.compile(
    r"^(abstract|introduction|conclusions?|references?|acknowledge?ments?|appendix\w*)\b", re.I)


class PdfHeadingLevel:
    """pdf는 제목 텍스트에 번호가 그대로 남아 있다는 것만 믿고 수준을 번호로 복원한다(Docling이 pdf에서
    주는 section_header level은 폰트 크기 추정이라 신뢰하지 않는다 - 실측으로 전부 1단계였다). 번호 없는
    제목은 Abstract/Introduction/Conclusion(s)/References/Acknowledg(e)ments/Appendix 계열이면 1단계,
    그 외는 직전 "번호 있는" 제목의 한 단계 아래(최대 3단계, 앞에 번호 제목이 없으면 1단계).
    문서 순서대로 호출되므로 직전 상태를 인스턴스에 들고 있는다. 문서 하나당 새로 만들 것."""

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


def drop_corner_items(doc):
    """Docling이 놓친 구석 머리말/쪽번호가 제목으로 잡히는 것을 막는다(prov bbox 기준). 그림과 표는 제외.
    (기존 실측: pdf 95개에서 지운 10건 전부 머리말·워터마크·쪽번호, 오탐 0.) 지운 개수를 돌려준다."""
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


@lru_cache(maxsize=1)
def get_converter():
    """모델 로딩 비용이 커서 프로세스당 한 번만 만든다."""
    import torch
    torch.backends.cudnn.enabled = False   # 서버의 cuDNN/드라이버 버전 불일치로 레이아웃 모델이 바로 죽는다(필수 우회)
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions()
    opts.do_ocr = False                    # 디지털 텍스트 PDF라 OCR 불필요(속도)
    opts.table_structure_options.mode = TableFormerMode.FAST   # 정확도보다 속도 우선
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})


def clean_pdf(src, dst):
    """PyMuPDF로 PDF 내용 스트림을 정규화(sanitize)해 dst에 저장한다. 쪽 수가 달라지면 예외."""
    import pymupdf

    doc = pymupdf.open(str(src))
    try:
        n_pages = doc.page_count
        doc.save(str(dst), **CLEAN_OPTIONS)
    finally:
        doc.close()
    out = pymupdf.open(str(dst))
    try:
        if out.page_count != n_pages:
            raise ValueError(f"정제본의 쪽 수가 다르다: {n_pages} -> {out.page_count}")
    finally:
        out.close()


def prepare_pdf(src, workdir, enabled=True):
    """변환에 쓸 PDF 경로를 정한다 -> (경로, info). 정제에 실패하면(암호화·손상 등) 원본을 그대로 쓰고
    info["error"]에 이유를 남긴다 - 정제 실패가 변환 실패로 번지지 않게 한다."""
    src = Path(src)
    info = {"enabled": bool(enabled), "applied": False, "orig_bytes": src.stat().st_size}
    if not enabled:
        return src, info
    dst = Path(workdir) / f"{src.stem}.clean.pdf"
    t = time.time()
    try:
        import pymupdf
        clean_pdf(src, dst)
        info.update(applied=True, tool=f"pymupdf-{pymupdf.__version__}", clean_bytes=dst.stat().st_size,
                    seconds=round(time.time() - t, 2))
        return dst, info
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        return src, info


def convert_pdf(entry, converter=None, clean=True, workdir=None):
    """entry: {doc_id, dataset, lang, path(절대경로), source_file(상대경로), title(선택)} -> 공통 JSON(dict).
    clean=True면 PDF를 정제본으로 바꿔 변환한다(CLEAN_OPTIONS). source_file은 항상 원본 경로다."""
    converter = converter or get_converter()
    workdir = Path(workdir or CLEAN_WORKDIR)
    workdir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:       # 변환이 끝나면 정제본은 지운다
        path, clean_info = prepare_pdf(entry["path"], tmp, enabled=clean)
        dl = converter.convert(str(path)).document
    corner = drop_corner_items(dl)
    blocks, log = build_blocks(dl, PdfHeadingLevel())
    log["corner_dropped"] = corner
    log["pdf_clean"] = clean_info
    first_heading = next((b["text"] for b in blocks if b["type"] == "heading"), None)
    return {
        "doc_id": entry["doc_id"], "dataset": entry["dataset"], "fmt": "pdf", "lang": entry["lang"],
        "title": entry.get("title") or first_heading or entry["doc_id"],
        "source_file": entry["source_file"],
        "parser": f"docling-{version('docling')}",
        "parser_options": {"do_ocr": False, "table_mode": "FAST", "corner_edge": CORNER_EDGE,
                           "pdf_clean": dict(CLEAN_OPTIONS) if clean_info["applied"] else None},
        "n_pages": len(dl.pages),
        "log": log,
        "blocks": blocks,
    }
