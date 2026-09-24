from pathlib import Path

from docling.document_converter import DocumentConverter
from docling_core.types.doc.labels import DocItemLabel

from langchain_core.documents import Document

# 모델 로딩 비용이 커서 모듈 레벨에서 한 번만 생성해 재사용한다.
_converter = DocumentConverter()
# 이미지는 청킹에 아무 의미가 없어(bge-m3는 텍스트만 임베딩 가능) 마크다운 변환에서
# 아예 제외한다 - PICTURE 라벨만 빼면 <!-- image --> placeholder 자체가 생기지 않는다.
_TEXT_LABELS = set(DocItemLabel) - {DocItemLabel.PICTURE}


def load_docx(filepath):
    """Docling으로 docx -> 마크다운 변환. 표는 진짜 마크다운 표(|---|)로, 헤더는
    #/## 마크다운 헤더로 나온다 - 구조를 아는 청커(스마트청킹)는 이 마크다운 문법을
    직접 파싱해서 표/섹션 경계를 찾고, fixed/semantic 청커는 그냥 문자열로 취급한다."""
    result = _converter.convert(str(filepath))
    md = result.document.export_to_markdown(labels=_TEXT_LABELS)
    return Document(page_content=md, metadata={"name": Path(filepath).name, "fmt": "docx"})


def load_docx_bundle(paths):
    """여러 docx를 각각 독립된 Document로 로드(문서 경계를 넘어 청킹/검색하지 않도록)."""
    paths = sorted(Path(p) for p in paths)
    return [load_docx(str(p)) for p in paths]
