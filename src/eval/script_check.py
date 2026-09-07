"""텍스트/토큰이 목표 스크립트(한글/라틴 등)로만 되어 있는지 확인하는 공용 유틸.
방향별 목표 스크립트 하나만 등록하면 되고, 그 외 알파벳은 뭐든 오염으로 취급한다."""
import sys
from pathlib import Path
import re

sys.path.append(str(Path(__file__).resolve().parent.parent))

SCRIPT_REGEX = {
    "hangul": re.compile(r"[가-힣]"),
    "latin": re.compile(r"[A-Za-z]"),
}


def has_target_script(s, target_script):
    """목표 스크립트 문자가 하나라도 있는지."""
    return bool(SCRIPT_REGEX[target_script].search(s))


def has_off_target_letters(s, target_script):
    """목표 스크립트가 아닌 알파벳이 섞여있는지."""
    target_re = SCRIPT_REGEX[target_script]
    return any(ch.isalpha() and not target_re.match(ch) for ch in s)


def is_translated_ok(s, target_script):
    """목표 언어로 온전히 번역됐는지 판단."""
    if not s.strip():
        return False
    if not any(ch.isalpha() for ch in s):
        return True  # 구두점뿐인 조각 - 번역 대상 자체가 없으니 통과
    return has_target_script(s, target_script) and not has_off_target_letters(s, target_script)


def build_off_target_ids(tokenizer, target_script):
    """vocab 전체를 디코딩해서, 목표 스크립트가 아닌 알파벳을 포함한 토큰 id 목록을
    만든다(bad_words_ids로 그대로 사용). Qwen 등 byte-level BPE는 vocab 원문 토큰
    문자열 자체가 바이트 치환 표현이라 decode()로 실제 텍스트를 복원해야 한다.
    한 방향(target_script)당 한 번만 계산해서 재사용할 것 - vocab 전체를 디코딩하는
    작업이라 몇 초 걸린다."""
    vocab = tokenizer.get_vocab()
    ids = list(vocab.values())
    decoded = tokenizer.batch_decode([[i] for i in ids])
    return [[tok_id] for tok_id, text in zip(ids, decoded) if text and has_off_target_letters(text, target_script)]
