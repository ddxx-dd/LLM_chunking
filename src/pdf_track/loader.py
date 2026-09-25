from pathlib import Path

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling_core.types.doc.labels import DocItemLabel

from langchain_core.documents import Document

# 모델 로딩 비용이 커서 모듈 레벨에서 한 번만 생성해 재사용한다.
_pdf_options = PdfPipelineOptions()
_pdf_options.do_ocr = False  # vectara 코퍼스는 스캔본이 아니라 디지털 텍스트 PDF라 OCR 불필요(속도)
_pdf_options.table_structure_options.mode = TableFormerMode.FAST  # 정확도보다 속도 우선
_converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=_pdf_options)})
# 이미지는 청킹에 아무 의미가 없어(bge-m3는 텍스트만 임베딩 가능) 마크다운 변환에서
# 아예 제외한다 - PICTURE 라벨만 빼면 <!-- image --> placeholder 자체가 생기지 않는다.
_TEXT_LABELS = set(DocItemLabel) - {DocItemLabel.PICTURE}


def load_pdf(filepath):
    """Docling으로 pdf -> 마크다운 변환. 표는 진짜 마크다운 표(|---|)로, 헤더는
    #/## 마크다운 헤더로 나온다(docx_track/loader.py와 동일한 설계) - 기존 PyMuPDF
    커스텀 로더는 헤더를 본문과 구분하지 못하고 표도 행마다 JSON으로 저장해서
    smart_chunker.py의 구조 인식이 사실상 무력화됐었음(실측 확인) - Docling으로
    교체해 해결. 복잡한 수식은 <!-- formula-not-decoded -->로 빠지는데, 이는
    fixed/semantic/smart 세 청커 모두에 동일하게 영향을 주므로 비교의 공정성은
    유지된다."""
    result = _converter.convert(str(filepath))
    md = result.document.export_to_markdown(labels=_TEXT_LABELS)
    return Document(page_content=md, metadata={"name": Path(filepath).name, "fmt": "pdf"})


def load_pdf_bundle(paths):
    """여러 pdf를 각각 독립된 Document로 로드(문서 경계를 넘어 청킹/검색하지 않도록)."""
    paths = sorted(Path(p) for p in paths)
    return [load_pdf(str(p)) for p in paths]
