"""통합 파이프라인 진입점(smart_chunk.md 10절 2단계) - fixed/semantic/smart 세 청커가
검색·요약·번역·병합에 미치는 영향을 같은 코드 경로로 비교하기 위한 단일 CLI.

    python pipeline.py --dataset {docx,pdf,srt} --task {retrieval,summary,translate} \\
                        --chunker {fixed,semantic,smart} --limit N [--fake-llm]

흐름: 파싱(캐시, docobj.cached_parse) -> 청킹(세 청커 모두 같은 flat_text/elements
입력, smart가 만든 N개에 fixed/semantic도 맞춤) -> 검색/요약/번역 -> (translate이고
smart일 때만) 원본 구조 그대로 병합(docx_track/pdf_track의 검증된 writer 재사용) ->
results/pipeline/에 저장.

retrieval/summary는 이번 단계에서 "정상 동작 확인" 수준으로만 구현한다 - 실제 채점
(Hit@k/MRR, QA 커버리지)과 그리드서치는 5단계("이후") 몫(원래 설계서 6절: "채점 코드는
연결만, 실행은 3·4번 범위에서만"). longbench는 아직 트랙 자체가 비어 있어 대상에서 뺐다.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from config import ALLGANIZE_DIR, RESULTS_DIR, SRT_ENG_DIR, SRT_KOR_DIR, SUBTITLE_DATASETS
from docobj import add_spans, cached_parse, get_target, is_skip_owner, owner_fragments_for_span
from smart_chunker import group_elements, make_fixed_chunks, make_semantic_chunks, section_for_owners, section_for_span

PIPELINE_RESULTS_DIR = RESULTS_DIR / "pipeline"

# 조각 태그로 [[n]](이중 괄호)를 쓴다 - ★ 실측 확인(2026-09-27, pdf dry-run에서 발견):
# 처음엔 [n](단일 괄호)을 썼는데, 논문 pdf 본문에 인용 번호("Cont and Tankov [8]")가
# 흔해서 그 [8]을 다음 조각 태그로 오인해 조각 뒷부분이 통째로 잘려나가는 진짜 버그가
# 있었다(strip_check로 발견 - "Cont and Tankov"에서 뚝 끊김). 논문 인용은 항상 단일
# 괄호라 이중 괄호는 절대 안 겹친다. docx는 한->영, pdf는 영->한(각 코퍼스가 이미 그
# 언어라 "번역"이 실제로 일어나는 방향으로 고정). srt는 인용 번호가 나올 일이 없는
# 자막이라 자기 것(mapper.py, 단일 괄호 [n])을 그대로 씀 - 안 건드림.
PROMPT_TEMPLATES = {
    "docx": "Translate each numbered fragment into natural English. "
            "Keep the [[n]] numbers exactly, one translation per line:\n{body}",
    "pdf": "다음 번호가 매겨진 조각들을 자연스러운 한국어로 번역하세요. "
           "[[n]] 번호를 그대로 유지하고, 한 줄에 하나씩 쓰세요:\n{body}",
}

NUM_TAG_RE = re.compile(r"\[\[(\d+)\]\]\s*(.*?)(?=\s*\[\[\d+\]\]|\Z)", re.S)


# ---------------------------------------------------------------------------
# 가짜/진짜 LLM 호출 - translate/summary 어느 쪽에서 부르든 함수 하나로 통일
# ---------------------------------------------------------------------------
def fake_llm(prompt):
    """3단계 dry-run용 - [n] 조각이 있으면 조각마다 표시 문자로 감싸서 그대로 돌려주고
    (번역 검증: 표시 문자 벗기면 원문과 같아야 함), 없으면(summary) 프롬프트 일부를
    표시만 한다. ★ 실측 확인(2026-09-27) - 처음 쓴 ⟦⟧(U+27E6/27E7)가 NanumGothic에
    없는 글자라 pdf 넣기 판정에서 착시성 실패를 크게 늘렸다(같은 pdf, A.원문그대로
    100%→B.⟦원문⟧ 29% 셀 성공률, 글꼴 있는 글자만 비교하면 문제 없어야 하는데 실제
    렌더링 시 없는 글자가 폭이 다른 대체 글자로 그려지며 넘침 유발). «»(U+00AB/00BB)는
    NanumGothic에 있는 글자로 교체(has_glyph 확인됨) - coverage_ratio 비교는 기존대로
    normalize_text가 기호를 다 걸러내므로 손볼 필요 없음."""
    if NUM_TAG_RE.search(prompt):
        return "\n".join(f"[[{m.group(1)}]] «{m.group(2).strip()}»" for m in NUM_TAG_RE.finditer(prompt))
    return f"«FAKE SUMMARY» {prompt[:80]}"


def parse_numbered(response):
    return {int(m.group(1)): m.group(2).strip() for m in NUM_TAG_RE.finditer(response)}


def fake_llm_srt(prompt):
    """★ 실측 확인(2026-09-27, srt dry-run에서 발견) - srt는 mapper.py의 자기
    태그 규칙(단일 괄호 [n], MARK_RE)을 그대로 쓰는데(자막엔 논문 인용번호가 없어
    겹칠 일이 없음 - 안 건드림), 위 fake_llm은 인용번호 충돌 수정 후 이중 괄호
    [[n]]만 인식해서 srt 프롬프트의 [n]을 못 찾고 엉뚱한 문자열을 돌려줬다(결과:
    거의 모든 조각이 "번호 누락"으로 집계됨 - missing_numbers가 이상하게 커서
    발견). srt 전용으로 단일 괄호를 인식하는 가짜 응답기를 따로 둔다."""
    from srt.mapper import MARK_RE as SRT_MARK_RE
    if SRT_MARK_RE.search(prompt):
        return "\n".join(f"[{m.group(1)}] «{m.group(2).strip()}»" for m in SRT_MARK_RE.finditer(prompt))
    return f"«FAKE SUMMARY» {prompt[:80]}"


def make_caller(fake, tokenizer=None, model=None, device=None):
    if fake:
        return fake_llm

    def call(prompt):
        from llm import generate
        return generate(tokenizer, model, device, [{"role": "user", "content": prompt}], max_new_tokens=1024)

    return call


# ---------------------------------------------------------------------------
# 모델 로딩 - embeddings(항상 필요, 청킹용)와 LLM(--fake-llm이면 생략)을 분리
# ---------------------------------------------------------------------------
def setup(fake_llm_flag, need_llm):
    """반환값에 srt_call_llm을 따로 둔다 - 진짜 LLM일 땐 call_llm과 같지만(모델 호출
    자체는 태그 규칙과 무관), 가짜일 땐 srt만의 단일 괄호 태그에 맞는 fake_llm_srt를
    쓴다(위 docstring 참고)."""
    from langchain_huggingface import HuggingFaceEmbeddings
    from transformers import AutoTokenizer

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", encode_kwargs={"normalize_embeddings": True})
    bge_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    count_tokens = lambda s: len(bge_tokenizer.encode(s, add_special_tokens=False))

    tokenizer = model = device = None
    if need_llm and not fake_llm_flag:
        from common import setup_models
        device, embed_model_unused, tokenizer, model = setup_models()

    call_llm = make_caller(fake_llm_flag, tokenizer, model, device)
    srt_call_llm = fake_llm_srt if fake_llm_flag else call_llm
    return embeddings, count_tokens, call_llm, bge_tokenizer, srt_call_llm


# ---------------------------------------------------------------------------
# docx/pdf 공통: elements -> 세 청커 공통 청크 리스트
# ---------------------------------------------------------------------------
def build_chunks(elements, flat_text, embeddings, count_tokens, chunker,
                  min_tokens=100, max_tokens=400, window=2):
    """세 청커 다 group_elements가 만든 N개에 맞춘다(5절 - 공정 비교의 핵심). 각 청크는
    "owners"(coverage 확인용, 전체 owner 키 목록)와 "fragments"(번역용, (owner, 청크와
    겹친 부분 텍스트만) 목록)를 둘 다 가진다 - smart는 owner를 절대 안 쪼개서 둘이
    사실상 같은 정보지만, fixed/semantic은 owner가 두 청크에 걸칠 수 있어(★ 실측 확인,
    2026-09-27) fragments가 전체 텍스트가 아니라 겹친 부분만 가져야 번역 결과가
    중복되지 않는다. 반환: ([{"owners":, "fragments":, "text":, "section":}], n_smart)."""
    smart_groups = group_elements(elements, embeddings._client, count_tokens, min_tokens, max_tokens, window)
    n = len(smart_groups)

    if chunker == "smart":
        chunks = smart_groups
        for c in chunks:
            c["section"] = section_for_owners(elements, c["owners"])
            c["fragments"] = [(o, text_for_owner(elements, o)) for o in c["owners"]]
        return chunks, n

    if chunker == "fixed":
        spans = make_fixed_chunks(flat_text, n)
    else:
        spans = make_semantic_chunks(flat_text, embeddings, n)

    chunks = []
    span_missing = 0
    for sp in spans:
        if sp is None:
            span_missing += 1
            continue
        s, e = sp
        frags = owner_fragments_for_span(elements, s, e)
        chunks.append({
            "owners": [o for o, _ in frags],
            "fragments": frags,
            "text": flat_text[s:e],
            "section": section_for_span(elements, s),
        })
    if span_missing:
        print(f"  경고: {chunker} recover_spans 실패 {span_missing}건(청크에서 제외됨)")
    return chunks, n


# ---------------------------------------------------------------------------
# 번역(docx/pdf 공통, 6절) - 청크 하나당 LLM 호출 1번, [[n]] 태그로 조각을 매핑
# ---------------------------------------------------------------------------
def text_for_owner(elements, owner):
    return get_target(elements, owner)["text"]


def translate_elements(elements, chunks, call_llm, prompt_template):
    """조각이 하나도 없는 청크는 LLM을 호출하지 않는다. 같은 owner에 여러 청크의
    조각이 걸치면(fixed/semantic 경계에서 흔함) 등장 순서대로 이어 붙인다(" ".join) -
    chunk["fragments"]가 이미 "겹친 부분만"이라 이어 붙이면 전체 텍스트가 복원된다.
    반환: ({owner: {"orig":, "translated":}}, stats)."""
    fragments_by_owner = {}
    pos = 0
    n_calls = 0
    n_skipped_empty = 0
    missing_numbers = 0
    for chunk in chunks:
        tagged = [(o, t) for o, t in chunk["fragments"] if not is_skip_owner(elements, o)]
        if not tagged:
            n_skipped_empty += 1
            continue
        body = "\n".join(f"[[{i + 1}]] {t}" for i, (_, t) in enumerate(tagged))
        response = call_llm(prompt_template.format(body=body))
        n_calls += 1
        parsed = parse_numbered(response)
        missing_numbers += sum(1 for i in range(len(tagged)) if (i + 1) not in parsed)
        for i, (owner, _) in enumerate(tagged):
            fragments_by_owner.setdefault(owner, []).append((pos, parsed.get(i + 1, "")))
            pos += 1

    results = {}
    for owner, frags in fragments_by_owner.items():
        frags.sort(key=lambda f: f[0])
        results[owner] = {
            "orig": text_for_owner(elements, owner),
            "translated": " ".join(t for _, t in frags if t).strip(),
        }
    stats = {"chunks": len(chunks), "llm_calls": n_calls, "chunks_no_fragments": n_skipped_empty,
              "missing_numbers": missing_numbers, "owners_translated": len(results)}
    return results, stats


# ---------------------------------------------------------------------------
# 병합(smart + translate 전용, 7절) - 이미 검증된 writer를 그대로 재사용
# ---------------------------------------------------------------------------
def merge_docx(src_path, elements, translations, out_path):
    from docx import Document as DocxDocument
    from docx_track.parse import add_para_ids
    from docx_track.writer import build_paraid_index, write_element_text

    anchored = add_para_ids(src_path)
    doc = DocxDocument(str(anchored))
    paraid_to_p = build_paraid_index(doc)
    written = skipped = 0
    for owner, r in translations.items():
        target = get_target(elements, owner)
        if target.get("merge_skip") or not r["translated"]:
            skipped += 1
            continue
        result = write_element_text(doc, paraid_to_p, target.get("loc"), r["translated"],
                                     auto_num=target.get("auto_num"))
        written += 1 if result == "ok" else 0
    doc.save(str(out_path))
    return {"written": written, "skipped": skipped}


def merge_pdf(src_path, elements, translations, out_path):
    import pymupdf
    from pdf_track.writer import write_translations

    pdf = pymupdf.open(str(src_path))
    trans_map = {o: r["translated"] for o, r in translations.items()
                 if r["translated"] and not get_target(elements, o).get("merge_skip")}
    stats = write_translations(pdf, elements, trans_map)
    # ★ 실측 확인(2026-09-27, git push 준비 중 발견) - insert_htmlbox(archive=...)가 호출마다
    # NanumGothic 전체를 새로 임베드해서(재사용 안 함) 일반 save()로는 원문 313KB짜리 pdf가
    # 900MB까지 불어났다(50개 합쳐 65GB). garbage=4가 중복 폰트 객체를 병합해서 없앤다
    # (같은 재현 테스트에서 20MB -> 57KB로 확인).
    pdf.ez_save(str(out_path), garbage=4, deflate=True)
    pdf.close()
    return stats


# ---------------------------------------------------------------------------
# retrieval/summary(docx/pdf 공통, 최소 동작 확인용 - 실제 채점은 5단계)
# ---------------------------------------------------------------------------
def run_retrieval(chunks, embeddings, doc_name):
    from indexing import build_retriever
    from langchain_core.documents import Document
    docs = [Document(page_content=c["text"], metadata={"name": doc_name, "section": c.get("section")})
            for c in chunks if c["text"].strip()]
    if not docs:
        return {"chunks": 0}
    retriever = build_retriever(docs, embeddings, k=min(5, len(docs)))
    probe = docs[len(docs) // 2].page_content[:60]  # 스모크 테스트용 임시 질의(실채점은 5단계)
    retrieved = retriever.invoke(probe)
    return {"chunks": len(docs), "probe_hit": any(d.page_content == docs[len(docs) // 2].page_content
                                                   for d in retrieved)}


def run_summary(chunks, call_llm, top_k=5):
    picked = sorted(chunks, key=lambda c: len(c["text"]), reverse=True)[:top_k]
    body = "\n\n".join(c["text"] for c in picked if c["text"].strip())
    if not body:
        return {"summary": "", "chunks_used": 0}
    prompt = f"다음 내용을 3문장으로 요약하세요:\n{body[:4000]}"
    return {"summary": call_llm(prompt), "chunks_used": len(picked)}


# ---------------------------------------------------------------------------
# docx/pdf 문서 하나 처리
# ---------------------------------------------------------------------------
def process_element_doc(src_path, fmt, parse_fn, task, chunker, embeddings, count_tokens, call_llm, reparse):
    t0 = time.monotonic()
    elements = cached_parse(src_path, fmt, parse_fn, reparse=reparse)
    flat_text = add_spans(elements)
    t_parse = time.monotonic() - t0

    t0 = time.monotonic()
    chunks, n_smart = build_chunks(elements, flat_text, embeddings, count_tokens, chunker)
    t_chunk = time.monotonic() - t0

    out = {"file": Path(src_path).name, "chunker": chunker, "n_chunks": len(chunks), "n_smart": n_smart,
           "time": {"parse": t_parse, "chunk": t_chunk}}

    if task == "retrieval":
        t0 = time.monotonic()
        out["retrieval"] = run_retrieval(chunks, embeddings, Path(src_path).name)
        out["time"]["task"] = time.monotonic() - t0
    elif task == "summary":
        t0 = time.monotonic()
        out["summary"] = run_summary(chunks, call_llm)
        out["time"]["task"] = time.monotonic() - t0
    elif task == "translate":
        t0 = time.monotonic()
        translations, stats = translate_elements(elements, chunks, call_llm, PROMPT_TEMPLATES[fmt])
        out["translate"] = stats
        out["time"]["task"] = time.monotonic() - t0

        if chunker == "smart":
            t0 = time.monotonic()
            out_dir = PIPELINE_RESULTS_DIR / fmt
            out_dir.mkdir(parents=True, exist_ok=True)
            merge_out = out_dir / f"{Path(src_path).stem}.translated{Path(src_path).suffix}"
            if fmt == "docx":
                out["merge"] = merge_docx(src_path, elements, translations, merge_out)
            else:
                out["merge"] = merge_pdf(src_path, elements, translations, merge_out)
            out["merge"]["out_path"] = str(merge_out)
            out["time"]["merge"] = time.monotonic() - t0

    return out


# ---------------------------------------------------------------------------
# srt - 기존 mapper.py/subtitle_pipeline.py primitives를 그대로 호출만 한다
# ---------------------------------------------------------------------------
def process_srt(movie_name, chunker, embeddings, bge_tokenizer, srt_call_llm, out_dir):
    """srt_call_llm: setup()이 준 srt 전용 caller(실제 LLM이면 call_llm과 같은 함수,
    가짜면 srt의 단일 괄호 [n] 태그에 맞는 fake_llm_srt) - mapper.py의 build_prompt/
    parse_marked(단일 괄호, 자기 fallback 로직 포함)를 그대로 쓴다(pipeline.py 자체의
    [[n]]/parse_numbered와 태그 형식이 달라서 안 섞음, 위 fake_llm_srt docstring 참고)."""
    from srt.loader import load_srt
    from srt.mapper import build_prompt, merge_to_units, parse_marked, write_srt
    from srt.subtitle_pipeline import default_subtitle_chunkers

    en_name, ko_name = SUBTITLE_DATASETS[movie_name]
    src_doc = load_srt(str(SRT_ENG_DIR / en_name))  # en2ko 고정(방향 그리드는 5단계 몫)

    t0 = time.monotonic()
    chunkers = default_subtitle_chunkers(embeddings._client, bge_tokenizer, "en")
    splitter = chunkers[chunker]
    chunks = splitter.split_documents([src_doc])
    t_chunk = time.monotonic() - t0

    t0 = time.monotonic()
    all_translated_pieces = []
    n_calls = 0
    for chunk in chunks:
        frs, prompt = build_prompt(src_doc, chunk,
                                    instruction="Translate each line to natural Korean. Keep the [n] numbers exactly:\n{body}")
        if not frs:
            continue
        response = srt_call_llm(prompt)
        n_calls += 1
        all_translated_pieces += parse_marked(response, frs)
    unit_texts = merge_to_units(src_doc, all_translated_pieces)
    out_path = out_dir / f"{movie_name}_en2ko_{chunker}.srt"
    write_srt(src_doc, unit_texts, out_path)
    t_task = time.monotonic() - t0

    return {"movie": movie_name, "chunker": chunker, "n_chunks": len(chunks), "llm_calls": n_calls,
            "out_path": str(out_path), "time": {"chunk": t_chunk, "task": t_task}}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def docx_files():
    files = sorted((ALLGANIZE_DIR / "docx").glob("**/*.docx"))
    return [f for f in files if "checkpoint" not in str(f)]


def pdf_files():
    from config import VECTARA_DIR
    return sorted(VECTARA_DIR.glob("*.pdf"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["docx", "pdf", "srt", "longbench"])
    ap.add_argument("--task", required=True, choices=["retrieval", "summary", "translate"])
    ap.add_argument("--chunker", required=True, choices=["fixed", "semantic", "smart"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fake-llm", action="store_true")
    ap.add_argument("--reparse", action="store_true", help="캐시 무시하고 Docling 다시 실행(docx/pdf만)")
    args = ap.parse_args()

    if args.dataset == "longbench":
        raise SystemExit("longbench 트랙은 아직 구현 전(src/longbench가 비어 있음) - 이번 파이프라인 대상 아님")

    need_llm = args.task in ("summary", "translate")
    print(f"모델 준비 중(fake_llm={args.fake_llm})...")
    embeddings, count_tokens, call_llm, bge_tokenizer, srt_call_llm = setup(args.fake_llm, need_llm)

    PIPELINE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = f"{args.dataset}_{args.task}_{args.chunker}"
    log_path = PIPELINE_RESULTS_DIR / f"{run_id}.json"

    results = []
    if args.dataset in ("docx", "pdf"):
        from docx_track.parse import parse_docx
        from pdf_track.parse import parse_pdf
        files = (docx_files() if args.dataset == "docx" else pdf_files())[: args.limit]
        parse_fn = parse_docx if args.dataset == "docx" else parse_pdf
        for i, f in enumerate(files, 1):
            print(f"[{i}/{len(files)}] {f.name}", flush=True)
            try:
                results.append(process_element_doc(f, args.dataset, parse_fn, args.task, args.chunker,
                                                     embeddings, count_tokens, call_llm, args.reparse))
            except Exception as e:
                results.append({"file": f.name, "error": str(e)})
                print(f"  예외: {e}", flush=True)
                import traceback
                traceback.print_exc()
    else:  # srt
        if args.task != "translate":
            raise SystemExit("srt 트랙은 현재 translate만 지원(retrieval/summary는 docx/pdf 대상)")
        movies = list(SUBTITLE_DATASETS)[: args.limit]
        out_dir = PIPELINE_RESULTS_DIR / "srt"
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, movie in enumerate(movies, 1):
            print(f"[{i}/{len(movies)}] {movie}", flush=True)
            try:
                results.append(process_srt(movie, args.chunker, embeddings, bge_tokenizer, srt_call_llm, out_dir))
            except Exception as e:
                results.append({"movie": movie, "error": str(e)})
                print(f"  예외: {e}", flush=True)
                import traceback
                traceback.print_exc()

    log_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    n_err = sum(1 for r in results if "error" in r)
    print(f"\n=== {run_id}: {len(results)}건 중 예외 {n_err}건, 로그 {log_path} ===")


if __name__ == "__main__":
    main()
