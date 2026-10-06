"""청크 구간 (start, end)에 블록·쪽·시각 메타데이터를 붙인다 (docs/restructure_plan.md 5.2절).

조각표(Piece 목록)와 청크 구간의 겹침을 이진 탐색으로 구해 아래를 채운다.
  block_ids     겹친 조각들의 블록 id (순서대로, 중복 없이)
  partial_start 청크가 첫 겹침 조각의 중간에서 시작하면 True (블록 단위 지표 - 문장 절단 여부는 stats의 sentence_cut_rate)
  partial_end   청크가 마지막 겹침 조각의 중간에서 끝나면 True
  page_start/end 겹친 블록의 쪽 번호 최솟값/최댓값 (pdf만, 없으면 None)
  t_start/t_end 겹친 첫 큐의 시작 / 마지막 큐의 끝 시각(초) (srt만)
  heading_path  첫 블록의 제목 경로 (저장만 한다, 청킹에는 쓰지 않음)
이 메타데이터는 검색 점수 계산에 쓰이지 않는다(쓰이는 것은 text뿐).
"""
from bisect import bisect_left, bisect_right


def attach_meta(doc, text, pieces, spans, count_tokens, chunker, params):
    """spans [(start, end)] -> 청크 dict 목록 (청크 파일 한 줄 = dict 하나, docs 3.2절)."""
    by_id = {b["id"]: b for b in doc["blocks"]}
    starts = [p.start for p in pieces]
    ends = [p.end for p in pieces]
    is_srt = doc.get("fmt") == "srt"
    out = []
    for i, (s, e) in enumerate(spans):
        first = bisect_right(ends, s)              # 끝이 s보다 큰 첫 조각
        last = bisect_left(starts, e) - 1          # 시작이 e보다 작은 마지막 조각
        hit = pieces[first:last + 1] if first <= last else []
        block_ids = []
        for p in hit:
            for bid in p.block_ids:
                if bid not in block_ids:
                    block_ids.append(bid)
        blocks = [by_id[b] for b in block_ids]
        pages = [b["page"] for b in blocks if b.get("page") is not None]
        rec = {
            "chunk_id": f"{doc['doc_id']}#{i:04d}", "doc_id": doc["doc_id"], "dataset": doc["dataset"],
            "chunker": chunker, "params": params,
            "start": s, "end": e, "text": text[s:e],
            "n_chars": e - s, "n_tokens": count_tokens(text[s:e]),
            "block_ids": block_ids,
            "partial_start": bool(hit) and s > hit[0].start,
            "partial_end": bool(hit) and e < hit[-1].end,
            "page_start": min(pages) if pages else None, "page_end": max(pages) if pages else None,
            "t_start": None, "t_end": None,
            "heading_path": list(blocks[0].get("heading_path") or []) if blocks else [],
        }
        if is_srt and hit:
            rec["t_start"] = by_id[hit[0].block_ids[0]]["start"]
            rec["t_end"] = by_id[hit[-1].block_ids[-1]]["end"]
        out.append(rec)
    return out
