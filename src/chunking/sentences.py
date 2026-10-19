"""청커 공용 문장 분리: 문서 텍스트 하나 → 단위 목록. semantic과 advanced_semantic만 쓴다(fixed·recursive는 텍스트를 그대로 받는다).
advanced_semantic은 split_structured로 문단 시작·제목 정보도 함께 받는다.

입력은 데이터셋마다 원래 텍스트 그대로다: pdf·docx는 data/clean_text의 text, LongBench는 원본 context.
  1 문단: 빈 줄('\\n\\n')은 항상 문단 경계(라이브러리는 빈 줄을 공백으로 보고 문단을 넘어 문장을 잇는다)
  2 표 문단('|'로 시작하는 줄이 2줄 이상)과 수식 문단('$$ … $$')은 나누지 않고 한 단위
  3 줄: 문단 안 '\\n'에서 줄을 나눈다(제목 줄·'Passage 1:'·참고문헌 항목이 본문 문장에 붙지 않게).
    단, 앞 줄이 문장 끝 기호 없이 끝나고 다음 줄이 소문자로 시작하면 접힌 줄로 보고 잇는다(NarrativeQA 원문은 한 줄 약 70자로 접혀 있다)
  4 문장: 줄 묶음마다 preprocess.sentences.split(영어 nltk punkt + 학술 약어 + 보정, 한국어 Kiwi + 개조식 구조 표시)
단위를 이으면 공백만 빼고 원래 텍스트와 같다(문장 안 공백만 정리된다). 줄 묶음 사이를 잇는 일은 하지 않는다.
"""
import re

from preprocess.sentences import split

PARA = re.compile(r"\n[^\S\n]*\n")
END = re.compile(r"[.!?:;)\]\"'”’…。]\s*$")


def is_table(p):
    return sum(1 for ln in p.split("\n") if ln.lstrip().startswith("|")) >= 2


def is_formula(p):
    p = p.strip()
    return p.startswith("$$") and p.endswith("$$") and len(p) >= 4


def paragraphs(text):
    """빈 줄로 나눈 문단(앞뒤 공백 정리, 빈 문단 제외)."""
    return [p.strip() for p in PARA.split(text) if p.strip()]


def lines(p):
    """문단 → 줄 묶음: '\\n'마다 나누되, 문장 끝 없이 끝난 줄 다음이 소문자로 시작하면(접힌 줄) 잇는다."""
    out = []
    for ln in p.split("\n"):
        ln = ln.strip()
        if not ln:
            continue
        if out and not END.search(out[-1]) and re.match(r"[a-z]", ln):
            out[-1] = out[-1] + " " + ln
        else:
            out.append(ln)
    return out


def split_units(text, lang="en"):
    """텍스트 → [(kind, 글)] (kind: sentence / table / formula)."""
    out = []
    for p in paragraphs(text):
        if is_table(p):
            out.append(("table", p))
        elif is_formula(p):
            out.append(("formula", p))
        else:
            for g in lines(p):
                out.extend(("sentence", s) for s in split(g, lang))
    return out


def split_structured(text, lang="en"):
    """텍스트 → [(글, 앞 구분, 제목)] (단위는 split_text와 같다). advanced_semantic이 쓴다.
    앞 구분: 원문에서 이 단위 앞에 있던 구분('\n\n' = 문단 시작, '\n' = 문단 안 새 줄, ' ' = 같은 줄, 첫 단위는 '').
    제목: 문단의 첫 줄이고 뒤에 줄이 더 있으며, 12낱말 이하이고 . ! ? 로 끝나지 않는 줄의 첫 단위(절 제목, 'Passage 1:' 등)."""
    out = []
    for p in paragraphs(text):
        if is_table(p) or is_formula(p):
            out.append((p, "\n\n", False))
            continue
        gs = lines(p)
        for gi, g in enumerate(gs):
            head = gi < len(gs) - 1 and len(g.split()) <= 12 and not re.search(r"[.!?]\s*$", g)
            for k, s in enumerate(split(g, lang)):
                out.append((s, ("\n\n" if gi == 0 else "\n") if k == 0 else " ", head and k == 0))
    if out:
        out[0] = (out[0][0], "", out[0][2])
    return out


def split_text(text, lang="en"):
    """텍스트 → 단위 글 목록(문장, 표, 수식). semantic·advanced_semantic이 쓰는 문장 분리."""
    return [t for _, t in split_units(text, lang)]
