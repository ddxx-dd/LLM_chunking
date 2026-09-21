"""DOCX 파이프라인: 텍스트/표 추출 -> 청킹 -> top-k 검색 -> 요약 생성.
표는 검색된 청크에 실제로 들어있는 행만 원본 그대로 렌더링(LLM이 보정 안 함)."""
import json
import sys
from dataclasses import replace
from pathlib import Path
from docx import Document as DocxDocument

sys.path.append(str(Path(__file__).resolve().parent.parent))
from mapper import chunk_bundle
from retriever import retrieve_top_k_bundle
from llm import generate


def extract_row_dicts(text):
    """청크에 섞인 표 행(JSON 한 줄)을 파싱해서 꺼낸다."""
    rows = []
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and obj:
                    rows.append(obj)
            except Exception:
                pass
    return rows


def strip_row_json(text):
    """표 행(JSON) 줄을 빼고 순수 문단만 남긴다 - 표는 LLM한테 안 보여줌."""
    kept = []
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                json.loads(s)
                continue
            except Exception:
                pass
        kept.append(line)
    return "\n".join(kept)


def build_compose_prompt(query, retrieved):
    """검색된 조각으로 요약 프롬프트 구성."""
    body = "\n\n".join(f"[출처: {r['doc_name']}]\n{r['chunk'].text.strip()}" for r in retrieved if r["chunk"].text.strip())
    return (
        "다음은 여러 문서에서 검색된, 아래 질문과 관련된 내용입니다. "
        "이 내용만 근거로 삼아 체계적으로 정리된 설명 문서를 작성하세요.\n"
        "형식 규칙: 소제목은 반드시 '## 소제목' 형식으로 쓰고, 일반 서술은 문단으로 쓰세요. "
        "표는 별도로 원본 그대로 첨부되니 여기서는 표를 새로 만들지 마세요. "
        "자료에 없는 내용은 지어내지 마세요.\n\n"
        f"질문: {query}\n\n검색된 내용:\n{body}\n\n작성할 문서:"
    )


def _md_table_row(line):
    """마크다운 표 한 줄 -> 셀 문자열 리스트."""
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _is_md_table_separator(line):
    """'---|---' 같은 마크다운 표 구분선인지 확인."""
    stripped = line.strip().strip("|")
    return bool(stripped) and "-" in stripped and all(c in " -:|" for c in stripped)


def _add_raw_table(d, doc_name, rows):
    """검색된 표 행을 LLM 없이 그대로 워드 표로 추가."""
    p = d.add_paragraph()
    run = p.add_run(f"[출처: {doc_name} · 검색된 조각에서 그대로 추출된 {len(rows)}행]")
    run.italic = True
    headers = list(rows[0].keys())
    table = d.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    for c, h in enumerate(headers):
        cell = table.rows[0].cells[c]
        cell.text = h
        for run in cell.paragraphs[0].runs:
            run.bold = True
    for r, row in enumerate(rows, start=1):
        for c, h in enumerate(headers):
            table.rows[r].cells[c].text = str(row.get(h, ""))


def write_docx_from_markdown(title, body_md, out_path, raw_tables=None):
    """LLM이 쓴 마크다운을 .docx로 변환. raw_tables는 원본 표 그대로 렌더링."""
    d = DocxDocument()
    d.add_heading(title, level=1)
    if raw_tables:
        d.add_heading("검색된 원본 표 데이터 (청킹·검색 결과 그대로, 보정 없음)", level=2)
        for doc_name, rows in raw_tables:
            _add_raw_table(d, doc_name, rows)
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
            i += 2  # 헤더 줄 + 구분선(---) 줄 건너뜀
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                row = _md_table_row(lines[i])
                i += 1
                # 생성이 중간에 잘려 셀 수가 안 맞는 줄은 건너뜀
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


def run_compose(label, chunker_fn, bundle_docs, query, top_k, embed_model, tokenizer, model, device, out_dir):
    """청킹 -> top-k 검색 -> 요약 생성 -> docx 저장까지 한 번에 실행."""
    print(f"\n[{label}] 청킹 및 검색 중...")
    bundle_chunks = chunk_bundle(bundle_docs, chunker_fn)
    print(f"  총 청크 {len(bundle_chunks)}개")

    retrieved = retrieve_top_k_bundle(query, bundle_chunks, embed_model, k=top_k)

    raw_tables = []
    prose_retrieved = []
    for r in retrieved:
        rows = extract_row_dicts(r["chunk"].text)
        if rows:
            raw_tables.append((r["doc_name"], rows))
            print(f"    표 행 {len(rows)}개 검색됨 (출처: {r['doc_name']})")
        prose_text = strip_row_json(r["chunk"].text)
        if prose_text.strip():
            prose_retrieved.append({**r, "chunk": replace(r["chunk"], text=prose_text)})

    prompt = build_compose_prompt(query, prose_retrieved)
    body_md = generate(tokenizer, model, device, [{"role": "user", "content": prompt}], max_new_tokens=1400)

    out_path = Path(out_dir) / f"요약_{label}.docx"
    write_docx_from_markdown(f"{query} — {label} 청킹 기반", body_md, out_path, raw_tables=raw_tables)
    print(f"  ✅ 저장 완료 -> {out_path}")
    return out_path
