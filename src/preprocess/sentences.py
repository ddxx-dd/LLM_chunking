"""문장 나누기 함수(공용): 영어 nltk punkt + 학술 약어 + 영어 보정, 한국어 Kiwi + 개조식 구조 표시 앞 자르기.

문장 단위 파일(processed_sen)은 더 만들지 않는다(2026-10-17: 청커가 clean_text 안에서 문장을 나눈다). 이 함수들은
LongBench 문장 나누기(preprocess/longbench.py)와 앞으로 만들 청커 공용 문장 함수가 쓴다.

도구 선택(2026-10-15 비교, tests/sent_split): 영어 nltk punkt + 학술 약어(F1 97.2%, 논문 96%),
  한국어 Kiwi + 개조식 구조 표시 앞 자르기(F1 87%; Kiwi만 79%, SaT 85%, spaCy 51%).
  - 글자가 하나도 없는 조각('(1)', '2')은 다음 문장 앞에 붙인다(마지막이면 앞 문장 뒤).
  - 영어 보정: 정리·절 라벨만 있는 문장('Proof.', 'Lemma 2.1.', '4.2.')은 다음 문장 앞에 붙이고,
    짧은 약어('i.i.d.', 'w.r.t.', 'etc.', 'P.C.')로 끝난 문장 뒤 소문자로 시작하는 문장은 합친다.
"""
import re

from common.config import ROOT

LETTER = re.compile(r"[A-Za-z0-9가-힣ㄱ-ㆎ一-鿿]")
# 한국어 개조식 구조 표시 앞에서 자른다: □ ■ ▪ ◦ ○ ● ㅇ ※ ▣ ◆ ◇, 주석 '* '(뒤에 조사가 오면 문장 안 각주 표시라 자르지 않음), ① ~ ⑳, '가. ', '1) '
KO_MARK = re.compile(r"\s(?=(?:[□■▪◦○●ㅇ※▣◆◇]\s?|\*\s(?![의을를이가은는에와과로도만]\s)|[①-⑳]|[가-하]\.\s|\d{1,2}\)\s)\S)")
_KIWI = None


# nltk 영어 punkt에 더하는 학술 약어(마침표 뒤에서 자르지 않음): 'Smith et al. (2020)', 'Fig. 3', 'Eq. (2)', 'Sec. 4', 'pp. 12'.
# 판정 자료(tests/sent_split/out/v2_labels.json)로 F1 95.7% → 97.2%, 재현율은 그대로(99.1%).
ABBREV = {"al", "et al", "fig", "figs", "eq", "eqs", "sec", "secs", "ref", "refs", "vol", "vols", "pp", "approx", "resp", "vs", "cf",
          "no", "nos", "ch", "app", "thm", "lem", "def", "prop", "cor", "tab", "e.g", "i.e", "ca", "ed", "eds"}
_PUNKT = None


def split_en(text):
    global _PUNKT
    if _PUNKT is None:
        import nltk
        from nltk.tokenize.punkt import PunktTokenizer
        path = str(ROOT / ".home" / "nltk_data")
        if path not in nltk.data.path:
            nltk.data.path.insert(0, path)
        _PUNKT = PunktTokenizer("english")
        _PUNKT._params.abbrev_types.update(ABBREV)
    return _PUNKT.tokenize(text)


def split_ko(text):
    global _KIWI
    if _KIWI is None:
        from kiwipiepy import Kiwi
        _KIWI = Kiwi()
    return [p for s in _KIWI.split_into_sents(text) for p in KO_MARK.split(s.text) if p and p.strip()]


# 영어 보정 1: 문장 전체가 정리·절 라벨('Proof.', 'Lemma 2.1.', 'Theorem 5 (Main).', '4.2.', 로마 숫자 'IV.')이면 다음 문장 앞에 붙인다.
EN_LABEL = re.compile(r"^(?:[IVXLC]{1,6}\.|\(?[A-Z]?\d{1,3}(?:\.\d{1,3})*\.?\)?|(?:Proof|Lemma|Theorem|Proposition|Corollary|Definition|Remark|Example|Assumption|Claim"
                      r"|Conjecture|Fact|Note|Step|Case|Problem|Exercise|Algorithm|Observation|Hypothesis|Notation|Condition|Property)"
                      r"(?: [A-Z]?\d{1,3}(?:\.\d{1,3})*[a-z]?)?(?: \([^)]{1,40}\))? ?[.:])$")
# 영어 보정 2: 앞 문장이 짧은 약어('etc.', 's.t.', 'a.k.a.', 'r.h.s.', 'e.g.', 한 글자+점)로 끝나고 다음 문장이 소문자로 시작하면 합친다.
EN_ABBR_END = re.compile(r"(?:\b(?:etc|i\.e|e\.g|a\.k\.a|s\.t|r\.h\.s|l\.h\.s|w\.r\.t|resp|viz|cf|vs|approx|al|ca|incl|esp|eqn?|figs?|secs?|refs?|no)\."
                         r"|(?:^|[\s(\[])(?:[A-Za-z]\.){1,3})[)\],]?$", re.I)


def fix_en(sents):
    out, carry = [], ""
    for s in sents:
        if carry:
            s, carry = carry + " " + s, ""
        if out and re.match(r"[a-z]", s) and EN_ABBR_END.search(out[-1]):
            out[-1] = out[-1] + " " + s
        elif EN_LABEL.match(s):
            carry = s
        else:
            out.append(s)
    if carry:
        out.append(carry)
    return out


def split(text, lang):
    """글 → 문장 목록(공백 정리, 글자 없는 조각은 다음 문장에 붙임)."""
    sents = [" ".join(s.split()) for s in (split_ko(text) if lang == "ko" else split_en(text))]
    if lang != "ko":
        sents = fix_en([s for s in sents if s])
    out, carry = [], ""
    for s in sents:
        if not s:
            continue
        if not LETTER.search(s):
            carry = (carry + " " + s).strip()
            continue
        out.append((carry + " " + s).strip() if carry else s)
        carry = ""
    if carry:
        if out:
            out[-1] = out[-1] + " " + carry
        else:
            out = [carry]
    return out
