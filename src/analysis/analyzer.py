import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                    
import matplotlib.pyplot as plt
import numpy as np

try:
    import koreanize_matplotlib          # 그래프에 한글이 깨지지 않게
except ImportError:
    pass

sys.path.append(str(Path(__file__).resolve().parent.parent))


def count_tokens(text, tokenizer):
    return len(tokenizer.encode(text))

# ══════════════════════════════════════════════════════════════
# 계산 
# ══════════════════════════════════════════════════════════════
def chunk_stats(doc, chunks, tokenizer=None):

#    지표 읽는 법:
#      char_cv   낮을수록 균일  -> 글자수 분할이 유리 (배치 처리 효율)
#      midword   낮을수록 좋음  -> 의미 분할이 유리 (문맥 보존)
    T = doc.text
    chars = [c.end - c.start for c in chunks]
    out = {
        "n_chunks":  len(chunks),
        "char_mean": round(float(np.mean(chars)), 1),
        "char_cv":   round(float(np.std(chars) / max(1e-9, np.mean(chars))), 3),
        "char_min":  int(min(chars)),
        "char_max":  int(max(chars)),
        "midword":       midword_cuts(T, chunks),
        "unit_straddle": unit_straddle(doc, chunks),
        "table_breaks":  table_breaks(doc, chunks),
        "lossless":  "".join(c.text for c in chunks) == T,
    }
    if tokenizer is not None:
        toks = [count_tokens(c.text, tokenizer) for c in chunks]
        out.update({
            "tok_mean":  round(float(np.mean(toks)), 1),
            "tok_cv":    round(float(np.std(toks) / max(1e-9, np.mean(toks))), 3),
            "tok_min":   int(min(toks)),
            "tok_max":   int(max(toks)),
            "tok_total": int(sum(toks)),      # LLM 비용에 직결
        })
    return out


def midword_cuts(text, chunks):
#경계가 단어 한가운데를 지나간 횟수.


    n = 0
    for c in chunks[1:]:                       
        i = c.start
        if 0 < i < len(text) and text[i-1].strip() and text[i].strip() \
           and text[i-1] not in " \n.!?…~":
            n += 1
    return n


def unit_straddle(doc, chunks):
#경계가 유닛(자막 큐 / 문단 / 표 행) 내부를 관통한 횟수.

    n = 0
    for c in chunks[1:]:
        for u in doc.units:
            if u.start < c.start < u.end:      # 경계가 유닛 '내부'에 있으면
                n += 1
                break                          # 유닛 하나당 한 번만 센다
    return n


def table_breaks(doc, chunks):
#경계가 표의 행(JSON 한 줄) 내부를 관통한 횟수.

    rows = [u for u in doc.units if u.kind == "row"]
    n = 0
    for c in chunks[1:]:
        for u in rows:
            if u.start < c.start < u.end:
                n += 1
                break
    return n


def topic_mix(doc, chunks):
#청크 하나가 걸쳐 있는 섹션(Heading) 수의 평균.  주제 혼재도.

#    1.0에 가까울수록 청크가 한 주제만 담고 있다는 뜻.
#    docx 요약 시나리오에서 검색 정확도와 직결된다.
    vals = []
    for c in chunks:
        h = sum(1 for u in doc.units
                if u.meta.get("heading") and c.start <= u.start and u.end <= c.end)
        vals.append(max(1, h))
    return round(float(np.mean(vals)), 3)


# ══════════════════════════════════════════════════════════════
# 
# ══════════════════════════════════════════════════════════════
def print_stats(stats, label=""):
    print("=" * 62)
    print(label)
    print("=" * 62)
    for k, v in stats.items():
        print("  {:15s} {}".format(k, v))
    print()


def print_chunks(doc, chunks, label="", preview=44, limit=None):
#청크 목록을 위치와 함께 나열한다.  limit로 앞 몇 개만 볼 수 있다."""
    print("=" * 70)
    print(label)
    print("=" * 70)
    for i, c in enumerate(chunks[:limit] if limit else chunks):
        head = c.text.strip().replace("\n", " / ")[:preview]   # 개행을 / 로 표시
        print("  [{:3d}] ({:6d},{:6d}) {:5d}자  {}".format(
            i, c.start, c.end, c.end - c.start, head))
    print()


# ══════════════════════════════════════════════════════════════
# 시각화
# ══════════════════════════════════════════════════════════════
def plot_boundaries(similarities, threshold, save_path, title=""):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.figure(figsize=(14, 5))
    plt.plot(similarities, linewidth=0.8, color="steelblue", label="similarity")
    plt.axhline(y=threshold, color="red", linestyle="--", linewidth=1.5,
                label="threshold = {:.3f}".format(threshold))

    cut_x = [i for i, s in enumerate(similarities) if s < threshold]
    cut_y = [similarities[i] for i in cut_x]
    plt.scatter(cut_x, cut_y, color="red", s=25, zorder=3,
                label="cut points ({})".format(len(cut_x)))

    plt.xlabel("sentence index")
    plt.ylabel("cosine similarity")
    plt.title(title)                         
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print("저장:", save_path)