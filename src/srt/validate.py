"""srt 전처리 결과 점검 - preprocess/validate.validate_doc이 fmt가 srt일 때 호출한다(공통 점검: 블록 id 중복·빈 텍스트 등은 거기서 이미 끝남)."""


def validate_srt(blocks, types, log, errors, warnings, stats):
    """errors·warnings·stats를 그 자리에서 채운다."""
    expected = log.get("cues_found", 0) - log.get("dropped_empty", 0)
    if types.get("cue", 0) != expected:
        errors.append(f"큐 개수 불일치: 블록 {types.get('cue', 0)} vs 원본 {log.get('cues_found')} - 빈 큐 {log.get('dropped_empty')}")
    if set(types) - {"cue"}:
        errors.append("srt 문서에 cue가 아닌 블록이 있다")
    starts = [b["start"] for b in blocks if b.get("type") == "cue"]
    if any(b["end"] < b["start"] for b in blocks if b.get("type") == "cue"):
        errors.append("끝 시각이 시작 시각보다 빠른 큐가 있다")
    if any(a > b for a, b in zip(starts, starts[1:])):
        warnings.append("큐 시작 시각이 시간 순서가 아니다")
    stats["n_cues"] = types.get("cue", 0)
