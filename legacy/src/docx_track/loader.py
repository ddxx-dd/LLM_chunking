import re
from pathlib import Path

from docling.document_converter import DocumentConverter

from langchain_core.documents import Document

# 모델 로딩 비용이 커서 모듈 레벨에서 한 번만 생성해 재사용한다.
_converter = DocumentConverter()
_EMPTY_HEADING_RE = re.compile(r"^#+[ \t]*\n+", re.M)  # 글자 없이 #만 있는 제목 줄


def load_docx(filepath):
    """Docling으로 docx -> 마크다운 변환(Docling 기본 export_to_markdown 그대로,
    그림 표시와 빈 제목 줄만 뺀다). 표는 마크다운 표(|---|), 제목은 #/##/### 헤더로 나온다 -
    구조를 아는 청커(스마트청킹)는 이 마크다운 문법을 직접 파싱해서 표/섹션 경계를
    찾고, fixed/semantic 청커는 그냥 문자열로 취급한다."""
    result = _converter.convert(str(filepath))
    md = result.document.export_to_markdown(image_placeholder="")
    md = _EMPTY_HEADING_RE.sub("", md)
    return Document(page_content=md, metadata={"name": Path(filepath).name, "fmt": "docx"})


def load_docx_bundle(paths):
    """여러 docx를 각각 독립된 Document로 로드(문서 경계를 넘어 청킹/검색하지 않도록)."""
    paths = sorted(Path(p) for p in paths)
    return [load_docx(str(p)) for p in paths]
