"""문서 객체 공통 스키마 - docx/pdf 파서가 만드는 elements를 저장/재사용한다.
elements: [{"id":, "label":, "level":, "text":, "loc":{...}, "span":(start,end)}, ...] -
한 번 파싱해두면 청킹/검색/요약/번역/병합이 전부 이 하나의 리스트만 보고 동작한다
(docx/pdf는 파서만 다름, 모양은 같음). smart_chunk.md 2·5·6절 참고."""
import json
import re
from pathlib import Path

# ★ 실측 확인(2026-09-27, git push 준비 중 발견) - 상대경로라 스크립트를 src/에서
# 돌리면 src/results/checks/에, 저장소 루트에서 돌리면 results/checks/에 따로 쌓여서
# 항등/표시 테스트 pdf(문서당 수백MB~수GB, 폰트 임베드 버그와 겹쳐 총 67GB까지 쌓인 적
# 있음)가 두 곳에 중복 생성됐다. __file__ 기준 절대경로로 고정해서 CWD와 무관하게
# 항상 같은 곳(저장소 루트 results/checks/, src/data/processed/anchored/)에 쌓인다.
_SRC_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SRC_DIR.parent

# 중간 산출물 경로(원본 데이터 폴더에는 아무것도 안 씀) - anchored docx는 재사용을 위해
# 문서 이름별로 캐시, 항등/표시 테스트 출력은 결과물이라 results/checks/에 모은다.
ANCHORED_DIR = _SRC_DIR / "data/processed/anchored"
CHECKS_DIR = _REPO_ROOT / "results/checks"


def anchored_path_for(src_path):
    return ANCHORED_DIR / f"{Path(src_path).stem}.anchored.docx"


def checks_output_path(src_path, suffix):
    CHECKS_DIR.mkdir(parents=True, exist_ok=True)
    return CHECKS_DIR / f"{Path(src_path).stem}{suffix}"


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


PARSE_CACHE_DIR = _SRC_DIR / "data/processed"


def get_target(elements, owner):
    """owner 키(("el", id) 또는 ("cell", table_id, cell_id))로 실제 요소/셀 dict를 찾는다
    (5절) - pdf_track/writer.py에도 똑같은 함수가 있는데(이미 검증된 코드라 안 건드림),
    여긴 pipeline.py가 docx/pdf 공통으로 쓰는 용도라 여기 따로 둔다(의도된 소규모 중복)."""
    if owner[0] == "el":
        return elements[owner[1]]
    table = elements[owner[1]]
    return next(c for c in table["cells"] if c["cell_id"] == owner[2])


def is_skip_owner(elements, owner):
    return bool(get_target(elements, owner).get("skip"))


def owner_fragments_for_span(elements, start, end):
    """[start, end) 구간(flat_text 절대 좌표, add_spans가 매긴 span/span_abs 기준)과
    겹치는 요소/셀마다 (owner, 겹친 부분 텍스트만)을 문서 순서대로 반환한다(5절 -
    "조각은 청크와 겹친 구간만"). fixed/semantic은 flat_text를 문자 위치로 그냥
    자르므로 한 owner가 인접한 두 청크에 걸치는 일이 흔한데, ★ 실측 확인(2026-09-27,
    pdf dry-run에서 발견) - 처음엔 겹치기만 하면 owner의 "전체" 텍스트를 매 청크에
    넣었더니, 두 청크 다 그 owner를 번역 대상으로 잡아 결과가 그대로 중복(2배)됐다.
    이제 각 청크는 자기 쪽과 겹친 부분만 갖는다(smart는 owner를 절대 안 쪼개므로
    이 함수를 안 쓰고 group_elements가 준 owners에 전체 텍스트를 그대로 씀 - skip
    여부는 여기서 안 거르고 번역 루프 쪽에서 is_skip_owner로 균일하게 거른다)."""
    out = []
    for e in elements:
        if e["label"] == "table":
            for cell in e.get("cells", []):
                cs, ct = cell["span_abs"]
                s, t = max(cs, start), min(ct, end)
                if s < t:
                    out.append((("cell", e["id"], cell["cell_id"]), cell["text"][s - cs:t - cs]))
            continue
        es, et = e["span"]
        s, t = max(es, start), min(et, end)
        if s < t:
            out.append((("el", e["id"]), e["text"][s - es:t - es]))
    return out


def cached_parse(src_path, fmt, parse_fn, reparse=False):
    """Docling 파싱은 문서당 수십 초라 재실행마다 다시 돌리면 낭비 - data/processed/<이름>.json
    캐시가 있으면 그대로 불러오고, 없거나 reparse=True(--reparse 옵션)면 parse_fn(src_path)를
    실행해서 캐시를 새로 만든다. 캐시는 add_spans 적용 "전" 원본 elements(parse_docx/parse_pdf가
    직접 반환하는 것)만 저장 - span은 호출하는 쪽이 필요할 때 add_spans로 매번 계산(빠름,
    Docling과 무관)."""
    name = Path(src_path).stem
    if not reparse:
        try:
            return load_elements(name, PARSE_CACHE_DIR)
        except FileNotFoundError:
            pass
    elements = parse_fn(src_path)
    save_elements(name, fmt, elements, PARSE_CACHE_DIR)
    return elements
