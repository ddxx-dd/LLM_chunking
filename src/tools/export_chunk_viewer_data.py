"""fixed_chunking vs semantic_chunking 결과를 원문과 나란히 눈으로 비교하기 위한
데이터 추출 스크립트. txt로 결과만 덤프하면 경계가 어디서 갈리는지 한눈에 안 보여서,
원문 + 두 방식의 청크 경계 + (semantic의) 문장별 유사도/임계값을 JSON으로 묶어
HTML 뷰어에 넘긴다. 파라미터는 실제 파이프라인 스크립트(subtitle_pipeline.py/
docx_pipeline.py)가 쓰는 값과 동일하게 맞춤 - 뷰어에 나오는 게 실제 실행 결과와
어긋나면 비교 도구로서 의미가 없음."""
import json
import sys
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import SRT_ENG_DIR, DOCX_ENG_DIR, DOCX_KOR_DIR, PDF_KOR_DIR, EMBED_MODEL, RESULTS_DIR
from loaders import load_file
from chunkers import fixed_chunking, split_sentences, calculate_similarities, calculate_threshold, split_at_boundaries

# (파일 경로, fixed_chunk_size, semantic kwargs) - 실제 스크립트에서 쓰는 값 그대로.
# Gutenberg/Wikipedia_설명문 경로는 코퍼스 개편으로 삭제되어 현재 코퍼스로 교체함
# (대용량문서/민법, Canada_Government 보고서 하나).
TARGETS = [
    (SRT_ENG_DIR / "Noah_Eng.srt", 500,
     dict(method="percentile", amount=15, min_chunk_tokens=128, max_chunk_tokens=1024)),
    (DOCX_KOR_DIR / "대용량문서" / "민법.docx", 600,
     dict(method="percentile", amount=15, min_chunk_tokens=128, max_chunk_tokens=500)),
    (DOCX_ENG_DIR / "RAG-Multi-Corpus" / "CloudWay-24" / "Baggage Allowance.docx", 600,
     dict(method="percentile", amount=15, min_chunk_tokens=128, max_chunk_tokens=500)),
    (DOCX_ENG_DIR / "Canada_Government" / "2020-2021-phthalates-in-ready-to-eat-meals-vegetable-fats-and-oils-overview.docx", 600,
     dict(method="percentile", amount=15, min_chunk_tokens=128, max_chunk_tokens=500)),
    (PDF_KOR_DIR / "한국어_논문" / "의사학" / "kjmh-31-2-181_Hygienic Masks in Colonial Korea.pdf", 600,
     dict(method="percentile", amount=15, min_chunk_tokens=128, max_chunk_tokens=500)),
]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("디바이스:", device)
    print("임베딩 모델 로드중...")
    embed_model = SentenceTransformer(EMBED_MODEL, device=device)
    print("로드 완료\n")

    docs_out = []
    for path, fixed_size, sem_kwargs in TARGETS:
        if not path.exists():
            print("건너뜀 (파일 없음):", path)
            continue
        print("처리중:", path.name)
        doc = load_file(path)
        text = doc.text

        fixed_chunks = fixed_chunking(text, chunk_size=fixed_size)

        sentences = split_sentences(text, 200)
        sentence_texts = [s[0] for s in sentences]
        vectors = embed_model.encode(sentence_texts, show_progress_bar=False)
        similarities = calculate_similarities(vectors)
        threshold = calculate_threshold(similarities, sem_kwargs["method"], sem_kwargs["amount"])
        semantic_chunks = split_at_boundaries(
            sentences, similarities, threshold, text, tokenizer=embed_model.tokenizer,
            max_chunk_tokens=sem_kwargs["max_chunk_tokens"], min_chunk_tokens=sem_kwargs["min_chunk_tokens"],
        )

        docs_out.append({
            "name": doc.name,
            "fmt": doc.fmt,
            "text": text,
            "units": [{"start": u.start, "end": u.end, "kind": u.kind} for u in doc.units],
            "fixed": {
                "chunk_size": fixed_size,
                "chunks": [[c.start, c.end] for c in fixed_chunks],
            },
            "semantic": {
                **sem_kwargs,
                "chunks": [[c.start, c.end] for c in semantic_chunks],
                "sentences": [[s[1], s[1] + len(s[0])] for s in sentences],
                "similarities": [round(s, 4) for s in similarities],
                "threshold": round(threshold, 4),
            },
        })
        print(f"  글자수={len(text)} fixed청크={len(fixed_chunks)}개 semantic청크={len(semantic_chunks)}개 문장={len(sentences)}개")

    out_path = RESULTS_DIR / "chunk_viewer_data.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(docs_out, f, ensure_ascii=False)
    print("\n저장 완료:", out_path, f"({out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
