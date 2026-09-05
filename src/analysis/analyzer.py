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