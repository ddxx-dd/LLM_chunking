import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from chunking.semantic_chunker import split_sentences

MARK_RE = re.compile(r"\[(\d+)\]\s*(.*)")

# ══════════════════════════════════════════════════════════════
# 조각 = 청크와 유닛의 교집합
# ══════════════════════════════════════════════════════════════

def fragments(doc, chunk):
#청크가 덮는 유닛 조각들.  (unit_idx, start, end, is_whole)

#    is_whole=True  : 유닛 전체가 이 청크 안에 있다 (온전)
#    is_whole=False : 유닛이 청크 경계에서 잘렸다 (손상 지점)

    out = []
    for j, u in enumerate(doc.units):
        s, e = max(u.start, chunk.start), min(u.end, chunk.end)
        if s < e:                                   
            out.append((j, s, e, s == u.start and e == u.end))
    return out


def build_prompt(doc, chunk, instruction):
#조각마다 [n] 번호를 붙여 프롬프트를 만든다.  잘린 조각도 그대로 넣는다.


    frs = fragments(doc, chunk)
    body = "\n".join("[{}] {}".format(n + 1, doc.text[s:e])
                     for n, (j, s, e, w) in enumerate(frs))
    return frs, instruction.format(n=len(frs), body=body)

def parse_marked(raw, frs):
#[n] 번호가 붙은 LLM 출력을 (unit_idx, 번역문) 목록으로 되돌린다.


    got = {int(m.group(1)): m.group(2).strip() for m in MARK_RE.finditer(raw)}
    return [(j, got.get(n + 1, "")) for n, (j, s, e, w) in enumerate(frs)]

def merge_to_units(doc, all_pieces, joiner=""):
    buckets = [[] for _ in doc.units]
    for j, t in all_pieces:
        if t:
            buckets[j].append(t)
    return [joiner.join(b).strip() for b in buckets]

# ══════════════════════════════════════════════════════════════
# 지표
# ══════════════════════════════════════════════════════════════
def fragment_stats(doc, chunks):
#조각 관점의 손상 지표.  이게 자막 실험의 핵심 지표다.

#    fragments     : LLM 번역 호출 단위 수
#    broken_frags  : 유닛이 잘린 조각 수  -> 직접적 손상량
#    broken_rate   : 조각 중 잘린 비율
#    units_touched : 하나 이상의 조각이 닿은 유닛 수
#    units_missed  : 아예 안 덮인 유닛 수 (정상이면 0)
#    extra_frags   : 유닛 대비 늘어난 호출 수 -> 비용 증가

    total, broken = 0, 0
    hit = set()
    for c in chunks:
        for j, s, e, whole in fragments(doc, c):
            total += 1
            hit.add(j)
            if not whole:
                broken += 1
    return {
        "fragments":     total,
        "broken_frags":  broken,
        "broken_rate":   round(broken / max(1, total), 4),
        "units_touched": len(hit),
        "units_missed":  len(doc.units) - len(hit),
        "extra_frags":   total - len(doc.units),
    }

def build_map(doc, max_sentence_length=200):
#문장 <-> 유닛 매핑표.  분석·리포트용 (병합에는 쓰지 않는다).


    table = []
    for i, (s, a, b) in enumerate(split_sentences(doc.text, max_sentence_length)):
        hit = [j for j, u in enumerate(doc.units)
               if not (u.end <= a or u.start >= b)]      
        table.append({"i": i, "text": s, "start": a, "end": b, "units": hit})
    return table


# ══════════════════════════════════════════════════════════════
# 출력
# ══════════════════════════════════════════════════════════════
def _fmt(sec):
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return "%02d:%02d:%02d,%03d" % (h, m, int(s), round((s - int(s)) * 1000))


def write_srt(doc, unit_texts, out_path):
#번역된 유닛 텍스트로 .srt 를 쓴다.


    assert len(unit_texts) == len(doc.units), "유닛 개수 불일치"
    blocks = []
    for i, (u, txt) in enumerate(zip(doc.units, unit_texts), 1):
        blocks.append("%d\n%s --> %s\n%s\n" % (
            i, _fmt(u.meta["t_start"]), _fmt(u.meta["t_end"]), txt))
    Path(out_path).write_text("\n".join(blocks), encoding="utf-8")
    return out_path