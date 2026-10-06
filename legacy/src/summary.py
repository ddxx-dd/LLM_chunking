"""요약(QA 기반 top-k) - retrieval이 찾은 top-k 청크로 Gemma가 질문에 답하고,
정답의 핵심 내용이 포함됐는지 Gemma가 O/X로 판정한다(QA 커버리지)."""
from llm import generate


def summarize_and_judge(question, target_answer, top_k_texts, tokenizer, model, device):
    context = "\n\n".join(f"[{i + 1}] {t}" for i, t in enumerate(top_k_texts))
    answer_prompt = f"다음 문서 조각들을 참고해서 질문에 답하세요.\n\n{context}\n\n질문: {question}\n답변:"
    n_in = len(tokenizer.encode(answer_prompt, add_special_tokens=False))
    generated = generate(tokenizer, model, device, [{"role": "user", "content": answer_prompt}],
                          max_new_tokens=n_in * 2 + 100)

    judge_prompt = (f"정답: {target_answer}\n생성된 답변: {generated}\n\n"
                     "생성된 답변이 정답의 핵심 내용을 포함하고 있습니까? O 또는 X로만 답하세요.")
    verdict_raw = generate(tokenizer, model, device, [{"role": "user", "content": judge_prompt}], max_new_tokens=10)
    if "O" in verdict_raw[:5].upper():
        verdict = "O"
    elif "X" in verdict_raw[:5].upper():
        verdict = "X"
    else:
        verdict = verdict_raw[:20]  # 예상 밖 응답 - 그대로 기록(집계에선 X 취급)
    return {"question": question, "generated": generated, "target_answer": target_answer, "verdict": verdict}


def run(retriever, qas, tokenizer, model, device, top_k=5):
    """qas: retrieval.run()이 준 것과 같은 리스트(question/target_answer 포함) -
    같은 retriever로 top-k를 다시 검색해서 요약·판정한다."""
    rows = []
    for i, qa in enumerate(qas, 1):
        retrieved = retriever.invoke(qa["question"])[:top_k]
        texts = [d.page_content for d in retrieved]
        r = summarize_and_judge(qa["question"], qa["target_answer"], texts, tokenizer, model, device)
        rows.append(r)
        if i % 10 == 0:
            print(f"  요약 {i}/{len(qas)} 완료", flush=True)
    coverage = sum(1 for r in rows if r["verdict"] == "O") / len(rows) if rows else None
    return {"coverage": coverage, "n": len(rows), "rows": rows}
