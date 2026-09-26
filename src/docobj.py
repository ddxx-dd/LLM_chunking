"""문서 객체 공통 스키마 - docx/pdf 파서가 만드는 elements를 저장/재사용한다.
elements: [{"id":, "label":, "level":, "text":, "loc":{...}, "span":(start,end)}, ...] -
한 번 파싱해두면 청킹/검색/요약/번역/병합이 전부 이 하나의 리스트만 보고 동작한다
(docx/pdf는 파서만 다름, 모양은 같음). smart_chunk.md 2·5·6절 참고."""
import json
import re
from pathlib import Path

# 표 셀 skip 판정(6절) - 숫자/기호만 있는 셀은 번역 대상에서 제외. "*"라 빈 문자열도 fullmatch됨
# (표 마크다운 빈 칸이 자동으로 skip 처리되는 것도 의도된 동작).
NUM_ONLY = re.compile(r"[\d\s.,%+\-−×/:()$€₩~]*")


def add_spans(elements):
    """flat_text = "\\n".join(e["text"] for e in elements)를 만들면서 각 요소의 "span"(flat_text
    안에서의 절대 위치)을 기록한다(2절). 표 요소는 cells의 "span_in_table"에 table.span[0]을
    더해 "span_abs"도 같이 계산한다(6절 - fixed/semantic이 표 중간을 자른 청크를 셀 단위로
    매핑할 때 씀). furniture는 애초에 elements에 없으므로 별도 필터링 불필요."""
    parts = []
    pos = 0
    for e in elements:
        text = e["text"]
        e["span"] = (pos, pos + len(text))
        parts.append(text)
        pos += len(text) + 1  # "\n" 구분자 1글자
        if e["label"] == "table":
            for cell in e.get("cells", []):
                cs, ct = cell["span_in_table"]
                cell["span_abs"] = (e["span"][0] + cs, e["span"][0] + ct)
    return "\n".join(parts)


def build_table_markdown(n_rows, n_cols, cell_texts):
    """세 청커가 공통으로 쓰는 표 마크다운 직렬화(6절) - 병합 셀은 시작 칸에만 텍스트,
    나머지 칸은 빈 칸으로 둔다(마크다운은 colspan/rowspan을 표현 못 하므로).
    cell_texts: dict[(row,col)] -> text (시작 칸 위치만). 반환: (markdown_text,
    dict[(row,col)] -> span_in_table)."""
    grid = [["" for _ in range(n_cols)] for _ in range(n_rows)]
    for (r, c), text in cell_texts.items():
        grid[r][c] = text.replace("\n", " ").replace("|", "\\|")
    header = "| " + " | ".join(grid[0]) + " |"
    sep = "|" + "|".join([" --- "] * n_cols) + "|"
    body_lines = ["| " + " | ".join(row) + " |" for row in grid[1:]]
    md = "\n".join([header, sep, *body_lines])

    spans = {}
    cursor = 0
    for r in range(n_rows):
        for c in range(n_cols):
            if (r, c) not in cell_texts:
                continue
            text = grid[r][c]
            if not text:
                spans[(r, c)] = (cursor, cursor)
                continue
            k = md.find(text, cursor)
            spans[(r, c)] = (k, k + len(text))
            cursor = k + len(text)
    return md, spans


def save_elements(doc_name, fmt, elements, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{doc_name}.json"
    path.write_text(
        json.dumps({"name": doc_name, "fmt": fmt, "elements": elements}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def load_elements(doc_name, out_dir):
    path = Path(out_dir) / f"{doc_name}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["elements"]
