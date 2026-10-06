"""Docling 문서(DoclingDocument) -> 공통 JSON 블록 (pdf/docx 공용, docs/restructure_plan.md 4.2~4.3절).

변환 규칙
  SECTION_HEADER, TITLE      heading (level은 호출자가 넘기는 규칙으로 정함)
  TEXT, CAPTION 등 글         paragraph   (CAPTION은 role="caption", 표와 연결되면 for=<표 블록 id>)
  LIST_ITEM                  list_item   (항목마다 별도 블록)
  TABLE                      table       (마크다운 표, caption 필드는 쓰지 않음 - CAPTION이 별도 항목으로 나온다)
  FORMULA, CODE              formula, code (글이 비면 제외)
  FOOTNOTE                   footnote    (A안: 따로 모아 블록 목록 맨 끝에 붙인다. after_block_id에 원래 위치 기록)
  PICTURE, CHART, PAGE_HEADER, PAGE_FOOTER   제외
  같은 inline 그룹 소속의 연속 항목은 원래 한 문장이므로 공백 하나로 이어 한 블록으로 만든다.
  3칸 이상 공백은 1칸으로 줄인다(목차 표의 정렬용 공백 문제, 기존 실측).
블록에는 id(b0, b1 ...), page(pdf만), heading_path(현재 제목 스택)를 채운다.
"""
import re

from docling_core.types.doc import DocItemLabel, GroupLabel

SKIP_LABELS = frozenset({DocItemLabel.PICTURE, DocItemLabel.CHART,
                         DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER})
HEADING_LABELS = (DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE)
_WS_COLLAPSE_RE = re.compile(r" {3,}")

_TYPE_BY_LABEL = {
    DocItemLabel.FOOTNOTE: "footnote",
    DocItemLabel.LIST_ITEM: "list_item",
    DocItemLabel.FORMULA: "formula",
    DocItemLabel.CODE: "code",
}


def clean(text):
    return _WS_COLLAPSE_RE.sub(" ", text or "").strip()


def page_of(item):
    prov = getattr(item, "prov", None)
    return prov[0].page_no if prov else None


def inline_group_ref(item, doc):
    """item이 Docling의 "inline" 그룹(문장 하나가 서식 경계마다 여러 항목으로 쪼개졌을 때 "원래 한 줄"
    이라고 표시해 둔 그룹) 소속이면 그 그룹 참조 문자열을, 아니면 None을 반환한다."""
    parent = getattr(item, "parent", None)
    if parent is None:
        return None
    try:
        group = parent.resolve(doc)
    except Exception:
        return None
    return parent.cref if getattr(group, "label", None) == GroupLabel.INLINE else None


def make_table_to_md(doc):
    """표 -> 마크다운. 캡션은 뺀다: Docling이 캡션을 별도 CAPTION 항목으로도 내놓기 때문에, 표 텍스트에
    캡션이 들어 있으면 캡션이 두 번 나온다(실측). 직렬화 설정의 labels에서 CAPTION을 뺀 serializer를 쓴다.
    compact_tables=True: 열 폭에 맞춘 공백·구분선 패딩을 없앤다(실측: 표 크기가 약 절반, 내용은 같음)."""
    from docling_core.transforms.serializer.markdown import MarkdownDocSerializer, MarkdownParams

    labels = set(MarkdownParams.model_fields["labels"].default) - {DocItemLabel.CAPTION}
    serializer = MarkdownDocSerializer(doc=doc, params=MarkdownParams(labels=labels, compact_tables=True))
    return lambda item: serializer.serialize(item=item).text


def build_blocks(doc, heading_level, table_to_md=None):
    """DoclingDocument -> (blocks, log). heading_level(item) -> int 은 제목마다 한 번, 문서 순서대로 호출된다.
    table_to_md(item) -> str 을 안 주면 make_table_to_md(doc)(캡션 없는 마크다운 표)를 쓴다(테스트는 가짜를 넘긴다)."""
    table_to_md = table_to_md or make_table_to_md(doc)
    blocks, footnotes, stack = [], [], []     # stack: [(level, text)]
    caption_items = []                        # [(docling self_ref, block)]
    table_captions = {}                       # caption self_ref -> 표 블록(임시 id)
    pending = {"v": None}                     # 이어 붙일 inline 그룹
    counter = {"n": 0}
    log = {"docling_items": 0, "docling_tables": 0, "empty_tables": 0, "skipped": {},
           "footnotes": 0, "captions": 0, "inline_groups_merged": 0, "empty_headings": 0}

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

    for item, _ in doc.iterate_items():
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
            for c in getattr(item, "captions", None) or []:
                table_captions[c.cref] = b
        elif (getattr(item, "text", None) or "").strip():
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

    for ref, b in caption_items:                 # 표 캡션 연결 (캡션이 표 앞/뒤 어디에 있어도 된다)
        t = table_captions.get(ref)
        if t is not None:
            b["for"] = t["id"]

    final = blocks + footnotes                   # A안: 각주는 맨 끝으로
    remap = {b["id"]: f"b{i}" for i, b in enumerate(final)}
    for b in final:
        b["id"] = remap[b["id"]]
        if b.get("after_block_id") is not None:
            b["after_block_id"] = remap[b["after_block_id"]]
        if b.get("for") is not None:
            b["for"] = remap[b["for"]]
    return final, log
