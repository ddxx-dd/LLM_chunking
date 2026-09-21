"""DOCX 파이프라인: 텍스트/표 추출 -> 청킹 -> 검색(InMemoryVectorStore) -> LCEL 체인으로 요약 생성.
표는 검색된 청크에 실제로 걸친 부분만 원본 HTML 그대로 렌더링(LLM이 보정 안 함) -
청크가 표를 중간에 자르면 그 결손도 그대로 드러난다(구조 무시 청커의 정직한 약점)."""
import sys
from html.parser import HTMLParser
from pathlib import Path
from docx import Document as DocxDocument
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

sys.path.append(str(Path(__file__).resolve().parent.parent))
from mapper import chunk_bundle, covered_units
from indexing import build_retriever


class _TableHTMLParser(HTMLParser):
    """loaders.py: _table_rows_to_html()의 역변환 - <table><tr><td colspan= rowspan=>
    ...</td></tr></table> -> 행 리스트([[{"text","colspan","rowspan"}, ...], ...])."""

    def __init__(self):
        super().__init__()
        self.rows = []
        self._cur_row = None
        self._cur_cell = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "tr":
            self._cur_row = []
        elif tag == "td":
            self._cur_cell = {"text": "", "colspan": int(attrs.get("colspan", 1)), "rowspan": int(attrs.get("rowspan", 1))}

    def handle_data(self, data):
        if self._cur_cell is not None:
            self._cur_cell["text"] += data

    def handle_endtag(self, tag):
        if tag == "td" and self._cur_cell is not None:
            self._cur_row.append(self._cur_cell)
            self._cur_cell = None
        elif tag == "tr" and self._cur_row is not None:
            self.rows.append(self._cur_row)
            self._cur_row = None


def parse_table_html(html_text):
    p = _TableHTMLParser()
    p.feed(html_text)
    return p.rows


def _layout_grid(rows):
    """행 리스트(각 행엔 "그 행에서 시작하는" 셀만 있음, rowspan으로 이어지는 continuation
    셀은 생략돼있음 - _table_rows_to_html의 인코딩 방식)를 (r,c) 좌표 그리드로 복원."""
    n_rows = len(rows)
    occupied = [[] for _ in range(n_rows)]
    grid = {}
    max_cols = 0

    def is_occupied(r, c):
        return c < len(occupied[r]) and occupied[r][c]

    def mark(r, c):
        while len(occupied[r]) <= c:
            occupied[r].append(False)
        occupied[r][c] = True

    for r, row in enumerate(rows):
        c = 0
        for cell in row:
            while is_occupied(r, c):
                c += 1
            grid[(r, c)] = cell
            for dr in range(cell["rowspan"]):
                for dc in range(cell["colspan"]):
                    if r + dr < n_rows:
                        mark(r + dr, c + dc)
            c += cell["colspan"]
            max_cols = max(max_cols, c)
    return grid, n_rows, max_cols


def split_chunk_tables(chunk):
    """청크 안에서 표 유닛 구간을 raw HTML로 뽑아내고 나머지는 산문으로 남긴다.
    청크가 표를 중간에 자른 경우 HTML이 불완전할 수 있음 - 의도된 정직한 약점이라
    보정하지 않고, 렌더링 단계(_add_raw_table)에서 파싱 실패 시 건너뛴다.
    반환: (표_html_리스트, 산문_텍스트)."""
    c_start = chunk.metadata["start_index"]
    text = chunk.page_content
    table_spans = sorted(
        (max(u["start"], c_start) - c_start, min(u["end"], c_start + len(text)) - c_start)
        for u in covered_units(chunk) if u["kind"] == "table"
    )
    tables_html = [text[s:e] for s, e in table_spans]
    prose_parts, prev = [], 0
    for s, e in table_spans:
        prose_parts.append(text[prev:s])
        prev = e
    prose_parts.append(text[prev:])
    return tables_html, "".join(prose_parts)


_COMPOSE_INSTRUCTION = (
    "다음은 여러 문서에서 검색된, 아래 질문과 관련된 내용입니다. "
    "이 내용만 근거로 삼아 체계적으로 정리된 설명 문서를 작성하세요.\n"
    "형식 규칙: 소제목은 반드시 '## 소제목' 형식으로 쓰고, 일반 서술은 문단으로 쓰세요. "
    "표는 별도로 원본 그대로 첨부되니 여기서는 표를 새로 만들지 마세요. "
    "자료에 없는 내용은 지어내지 마세요.\n\n"
    "질문: {query}\n\n검색된 내용:\n{context}\n\n작성할 문서:"
)
_COMPOSE_PROMPT = PromptTemplate.from_template(_COMPOSE_INSTRUCTION)


def build_compose_chain(tokenizer, lc_llm):
    """prompt | llm | parser LCEL 체인. 모델이 Gemma 계열이라 순정 텍스트가 아니라
    tokenizer.apply_chat_template()을 거친 문자열이 필요 - RunnableLambda로 그 변환을
    체인 안에 끼워넣는다(generate_batch()가 쓰는 것과 동일한 템플릿 적용 방식)."""
    def _to_chat_text(prompt_value):
        messages = [{"role": "user", "content": prompt_value.to_string()}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    return _COMPOSE_PROMPT | RunnableLambda(_to_chat_text) | lc_llm | StrOutputParser()


def _md_table_row(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _is_md_table_separator(line):
    stripped = line.strip().strip("|")
    return bool(stripped) and "-" in stripped and all(c in " -:|" for c in stripped)


def _add_raw_table(d, doc_name, html_text):
    """검색된 표 HTML을 LLM 없이 그대로 워드 표(병합 셀 포함)로 추가.
    파싱 실패(청크 경계에서 잘린 불완전한 HTML)하면 조용히 건너뛴다 -
    이것도 구조 무시 청커의 약점을 그대로 드러내는 것이지 감출 대상이 아니다."""
    try:
        grid, n_rows, n_cols = _layout_grid(parse_table_html(html_text))
        if n_rows == 0 or n_cols == 0:
            return False
    except Exception:
        return False
    p = d.add_paragraph()
    run = p.add_run(f"[출처: {doc_name} · 검색된 조각에서 그대로 추출된 표]")
    run.italic = True
    table = d.add_table(rows=n_rows, cols=n_cols)
    table.style = "Table Grid"
    for (r, c), cell in grid.items():
        top_left = table.cell(r, c)
        top_left.text = cell["text"]
        colspan, rowspan = cell["colspan"], cell["rowspan"]
        if colspan > 1 or rowspan > 1:
            top_left.merge(table.cell(r + rowspan - 1, c + colspan - 1))
    return True


def write_docx_from_markdown(title, body_md, out_path, raw_tables=None):
    """LLM이 쓴 마크다운을 .docx로 변환. raw_tables는 원본 표 그대로 렌더링."""
    d = DocxDocument()
    d.add_heading(title, level=1)
    if raw_tables:
        d.add_heading("검색된 원본 표 데이터 (청킹·검색 결과 그대로, 보정 없음)", level=2)
        for doc_name, html_text in raw_tables:
            _add_raw_table(d, doc_name, html_text)
        d.add_heading("종합 설명", level=2)
    lines = body_md.split("\n")
    i, n = 0, len(lines)
    while i < n:
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("## "):
            d.add_heading(line[3:].strip(), level=2)
            i += 1
            continue
        if line.startswith("# "):
            d.add_heading(line[2:].strip(), level=2)
            i += 1
            continue
        if line.startswith("|") and i + 1 < n and _is_md_table_separator(lines[i + 1]):
            header = _md_table_row(line)
            i += 2
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                row = _md_table_row(lines[i])
                i += 1
                if len(row) == len(header) and "�" not in "".join(row):
                    rows.append(row)
            table = d.add_table(rows=1 + len(rows), cols=len(header))
            table.style = "Table Grid"
            for c, val in enumerate(header):
                cell = table.rows[0].cells[c]
                cell.text = val
                for run in cell.paragraphs[0].runs:
                    run.bold = True
            for r, row in enumerate(rows, start=1):
                for c, val in enumerate(row):
                    if c < len(header):
                        table.rows[r].cells[c].text = val
            continue
        d.add_paragraph(line)
        i += 1
    d.save(str(out_path))


def run_compose(label, splitter, bundle_docs, query, top_k, embed_model, tokenizer, lc_llm, out_dir):
    """청킹 -> 검색(InMemoryVectorStore) -> LCEL 체인 요약 생성 -> docx 저장까지 한 번에 실행."""
    print(f"\n[{label}] 청킹 및 검색 중...")
    bundle = chunk_bundle(bundle_docs, splitter)
    print(f"  총 청크 {len(bundle)}개")

    retriever = build_retriever([b["chunk"] for b in bundle], embed_model, k=top_k)
    retrieved = retriever.invoke(query)

    raw_tables = []
    prose_parts = []
    for chunk in retrieved:
        doc_name = chunk.metadata["name"]
        tables_html, prose = split_chunk_tables(chunk)
        for html_text in tables_html:
            raw_tables.append((doc_name, html_text))
            print(f"    표 검색됨 (출처: {doc_name})")
        if prose.strip():
            prose_parts.append(f"[출처: {doc_name}]\n{prose.strip()}")

    chain = build_compose_chain(tokenizer, lc_llm)
    body_md = chain.invoke({"query": query, "context": "\n\n".join(prose_parts)})

    out_path = Path(out_dir) / f"요약_{label}.docx"
    write_docx_from_markdown(f"{query} — {label} 청킹 기반", body_md, out_path, raw_tables=raw_tables)
    print(f"  ✅ 저장 완료 -> {out_path}")
    return out_path
