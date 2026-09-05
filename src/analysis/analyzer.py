import os
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

def count_tokens(text, tokenizer):
    return len(tokenizer.encode(text))

def chunk_stats(doc, chunks, tokenizer=None):
    T = doc.text
    chars = [c.end - c.start for c in chunks]
    out = {
        "n_chunks": len(chunks),
        "char_mean": round(float(np.mean(chars)), 1),
        "char_cv": round(float(np.std(chars) / max(1e-9, np.mean(chars))), 3),
        "char_min": int(min(chars)),
        "char_max": int(max(chars)),
        "midword": midword_cuts(T, chunks),
        "unit_straddle": unit_straddle(doc, chunks),
        "table_breaks": table_breaks(doc, chunks),
        "lossless": "".join(c.text for c in chunks) == T,
    }
    if tokenizer is not None:
        toks = [count_tokens(c.text, tokenizer) for c in chunks]
        out.update({
            "tok_mean": round(float(np.mean(toks)), 1),
            "tok_cv": round(float(np.std(toks) / max(1e-9, np.mean(toks))), 3),
            "tok_min": int(min(toks)),
            "tok_max": int(max(toks)),
            "tok_total": int(sum(toks)),
        })
    return out

def midword_cuts(text, chunks):
    n = 0
    for c in chunks[1:]:
        i = c.start
        if 0 < i < len(text) and text[i-1].strip() and text[i].strip() and text[i-1] not in " \n.!?…~":
            n += 1
    return n

def unit_straddle(doc, chunks):
    n = 0
    for c in chunks[1:]:
        for u in doc.units:
            if u.start < c.start < u.end:
                n += 1
                break
    return n

def table_breaks(doc, chunks):
    rows = [u for u in doc.units if u.kind == "row"]
    n = 0
    for c in chunks[1:]:
        for u in rows:
            if u.start < c.start < u.end:
                n += 1
                break
    return n

def split_tables(doc, chunks):
    """표 하나(같은 meta['table'] id)의 행들이 서로 다른 청크에 나뉘어 담겼는지 확인.
    이 프로젝트의 docx 로더는 표의 행 하나하나를 이미 독립된 Unit으로 만들기 때문에,
    청크 경계는 대부분 행의 '중간'이 아니라 행과 행 '사이'에 떨어진다 - table_breaks가
    잡는 건 드문 경우(행 자체가 잘림)고, 이 함수가 실제로 흔한 경우("표는 하나인데
    청크는 여러 개로 갈림")를 직접 측정한다. 의도적으로 아무것도 고치지 않고
    "몇 번 테이블이 몇 개 청크로 쪼개졌는지"만 보고한다 - 분할 기법의 한계를
    감추지 않고 그대로 드러내기 위함."""
    def chunk_index_of(pos):
        for i, c in enumerate(chunks):
            if c.start <= pos < c.end:
                return i
        return None

    tables = {}
    for u in doc.units:
        if u.kind == "row":
            tables.setdefault(u.meta["table"], []).append(u)

    result = {}
    for table_id, rows in tables.items():
        chunk_ids = sorted({chunk_index_of(r.start) for r in rows})
        if len(chunk_ids) > 1:
            result[table_id] = chunk_ids
    return result

def topic_mix(doc, chunks):
    vals = []
    for c in chunks:
        h = sum(1 for u in doc.units if u.meta.get("heading") and c.start <= u.start and u.end <= c.end)
        vals.append(max(1, h))
    return round(float(np.mean(vals)), 3)

def print_stats(stats, label=""):
    print("=" * 62)
    print(label)
    print("=" * 62)
    for k, v in stats.items():
        print("  {:15s} {}".format(k, v))
    print()

def print_chunks(doc, chunks, label="", preview=44, limit=None):
    print("=" * 70)
    print(label)
    print("=" * 70)
    for i, c in enumerate(chunks[:limit] if limit else chunks):
        head = c.text.strip().replace("\n", " / ")[:preview]
        print("  [{:3d}] ({:6d},{:6d}) {:5d}자  {}".format(i, c.start, c.end, c.end - c.start, head))
    print()

def plot_boundaries(similarities, threshold, save_path, title=""):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.figure(figsize=(14, 5))
    plt.plot(similarities, linewidth=0.8, color="steelblue", label="similarity")
    plt.axhline(y=threshold, color="red", linestyle="--", linewidth=1.5, label="threshold = {:.3f}".format(threshold))

    cut_x = [i for i, s in enumerate(similarities) if s < threshold]
    cut_y = [similarities[i] for i in cut_x]
    plt.scatter(cut_x, cut_y, color="red", s=25, zorder=3, label="cut points ({})".format(len(cut_x)))

    plt.xlabel("sentence index")
    plt.ylabel("cosine similarity")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print("저장:", save_path)