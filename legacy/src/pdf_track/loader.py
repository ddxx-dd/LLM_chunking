import re
from pathlib import Path

import torch
torch.backends.cudnn.enabled = False  # ★ 실측 확인 - 이 서버의 cuDNN/드라이버 버전 불일치로
# pdf 레이아웃 모델이 CUDNN_STATUS_NOT_INITIALIZED로 바로 죽는다(끌 수밖에 없는 필수 우회).

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling_core.types.doc import DocItemLabel, GroupLabel

from langchain_core.documents import Document

# 모델 로딩 비용이 커서 모듈 레벨에서 한 번만 생성해 재사용한다.
_pdf_options = PdfPipelineOptions()
_pdf_options.do_ocr = False  # vectara 코퍼스는 스캔본이 아니라 디지털 텍스트 PDF라 OCR 불필요(속도)
_pdf_options.table_structure_options.mode = TableFormerMode.FAST  # 정확도보다 속도 우선
_converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=_pdf_options)})

# "3" -> 1단계, "3.1" -> 2단계, "3.1.2" -> 3단계. 끝의 점은 무시("3.1"과 "3.1."은 같음
# - 그룹 1은 점 뒤의 마지막 "."을 절대 포함하지 않고, 그 "."은 뒤의 \.? 가 따로 먹는다).
_NUMBERED_HEADING_RE = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,2}){0,3})\.?\s+\S")
# 번호가 없어도 논문에서 항상 최상위 절인 제목들 - 1단계 고정.
_FIRST_LEVEL_WORDS_RE = re.compile(
    r"^(abstract|introduction|conclusions?|references?|acknowledge?ments?|appendix\w*)\b", re.I)


class _PdfHeadingLevel:
    """pdf는 마크다운 제목에 번호가 그대로 남아있다는 것만 믿고 수준을 번호로
    복원한다(Docling이 pdf에서 주는 section_header level은 폰트 크기 추정이라
    신뢰 안 함). 번호 없는 제목은 Abstract/Introduction/Conclusion(s)/References/
    Acknowledg(e)ments/Appendix 계열이면 1단계, 그 외는 직전 "번호 있는" 제목의
    한 단계 아래(최대 3단계, 앞에 번호 제목이 없으면 1단계) - 문서 순서대로
    호출되므로 직전 상태를 인스턴스에 들고 있는다. 문서 하나당 새로 만들 것."""

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


_WS_COLLAPSE_RE = re.compile(r" {3,}")
_SKIP_LABELS = (DocItemLabel.PICTURE, DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER)


def _inline_group_ref(item, doc):
    """item이 Docling의 "inline" 그룹(문장 하나가 서식 경계마다 여러 아이템으로
    쪼개졌을 때 "원래 한 줄"이라고 표시해두는 그룹) 소속이면 그 그룹 참조 문자열을,
    아니면 None을 반환한다 - 같은 그룹인지 비교하는 데 쓴다."""
    parent = getattr(item, "parent", None)
    if parent is None:
        return None
    try:
        group = parent.resolve(doc)
    except Exception:
        return None
    return parent.cref if getattr(group, "label", None) == GroupLabel.INLINE else None


def build_markdown(doc):
    """DoclingDocument -> 마크다운. export_to_markdown() 대신 iterate_items()로 직접
    조립한다 - 제목은 _PdfHeadingLevel 규칙으로 "#" 개수를 정하고, 목록·참고문헌은
    항목마다 빈 줄로 나눠서(export는 한 덩어리로 이어붙이고 "1. " 번호를 붙임, 실측)
    chunkers.detect_blocks가 항목 단위 블록을 얻게 한다. 같은 inline 그룹 소속 연속
    아이템은 원래 한 문장이므로 공백 하나로 이어붙인다(서식 경계의 원래 공백은 Docling이
    잃어버려 조사 앞에 공백이 남을 수 있음 - export_to_markdown도 똑같은 한계). 그림·
    머리글·바닥글은 뺀다."""
    heading_level = _PdfHeadingLevel()
    lines = []
    pending_ref = None
    pending_parts = []

    def flush():
        nonlocal pending_ref, pending_parts
        if pending_parts:
            lines.append(" ".join(pending_parts))
        pending_ref = None
        pending_parts = []

    for item, _ in doc.iterate_items():
        label = item.label
        if label in _SKIP_LABELS:
            flush()
            continue
        if label in (DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE):
            flush()
            lines.append(f"{'#' * heading_level(item)} {item.text}")
        elif label == DocItemLabel.TABLE:
            flush()
            lines.append(item.export_to_markdown(doc=doc))
        elif hasattr(item, "text") and item.text.strip():
            ref = _inline_group_ref(item, doc)
            if ref is not None:
                if ref != pending_ref:
                    flush()
                    pending_ref = ref
                pending_parts.append(item.text)
            else:
                flush()
                lines.append(item.text)
    flush()
    # 3칸 이상 공백을 1칸으로(목차 표 등의 열 정렬용 공백이 수천 자까지 이어지는 문제, 실측)
    return _WS_COLLAPSE_RE.sub(" ", "\n\n".join(lines))


_CORNER_EDGE = 0.05  # 쪽 위/아래 5% 안에 있는 항목은 머리말·바닥글·쪽번호로 보고 지운다


def _drop_corner_items(doc):
    """Docling이 놓친 구석 머리말/쪽번호가 제목(###)으로 잡히는 걸 막는다(prov bbox 기준).
    그림과 표는 제외 - 실측(95개 pdf): 지운 10건 전부 머리말·워터마크·쪽번호, 오탐 0."""
    corner = []
    for item, _ in doc.iterate_items():
        if not item.prov or item.label in (DocItemLabel.PICTURE, DocItemLabel.TABLE):
            continue
        p = item.prov[0]
        height = doc.pages[p.page_no].size.height
        box = p.bbox.to_top_left_origin(height)
        if box.b <= height * _CORNER_EDGE or box.t >= height * (1 - _CORNER_EDGE):
            corner.append(item)
    if corner:
        doc.delete_items(node_items=corner)


def load_pdf(filepath):
    """Docling으로 pdf -> 마크다운 변환. 표는 진짜 마크다운 표(|---|)로, 제목은
    번호(1/1.1/1.1.2 등)로 수준을 복원한 #/##/### 마크다운 헤더로 나온다(docx_track/
    loader.py와 같은 설계) - 기존 PyMuPDF 커스텀 로더는 헤더를 본문과 구분하지
    못하고 표도 행마다 JSON으로 저장해서 chunkers.py의 구조 인식이 사실상
    무력화됐었음(실측 확인) - Docling으로 교체해 해결. 복잡한 수식은
    <!-- formula-not-decoded -->로 빠지는데, 이는 fixed/semantic/smart 세 청커
    모두에 동일하게 영향을 주므로 비교의 공정성은 유지된다."""
    result = _converter.convert(str(filepath))
    _drop_corner_items(result.document)
    md = build_markdown(result.document)
    return Document(page_content=md, metadata={"name": Path(filepath).name, "fmt": "pdf"})


def load_pdf_bundle(paths):
    """여러 pdf를 각각 독립된 Document로 로드(문서 경계를 넘어 청킹/검색하지 않도록)."""
    paths = sorted(Path(p) for p in paths)
    return [load_pdf(str(p)) for p in paths]
