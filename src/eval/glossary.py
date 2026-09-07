"""고유명사 용어집: 문서 전체에서 한 번만 만들어서 모든 청크에 동일하게 주입한다.
그래야 같은 이름이 청크마다 다르게 번역되는 걸 막을 수 있다.

확정 순서(이름 하나당):
  1) 위키피디아 조회 - 정확한 표준 표기를 얻을 가능성이 가장 높음
  2) 원문 속 실제 사용 문맥으로 재검증 - 동음이의어(예: 반려견 이름 Pluto가
     행성 "명왕성"으로 잘못 연결되는 경우)를 걸러냄
  3) 실패하면 폴백 - 여러 후보를 강제한글로 생성한 뒤, 역번역(back-translation)이
     원래 이름과 가장 가까운 것을 선택 (외부 정답 없이도 신뢰도를 가늠하는 방법)

전부 특정 이름/데이터셋에 의존하지 않는 일반적인 절차라 다른 자막에도 그대로 쓴다."""
import sys
import re
import json
import time
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path
from collections import Counter

sys.path.append(str(Path(__file__).resolve().parent.parent))
from llm.client import generate
from eval.script_check import is_translated_ok

# ── 1) 후보 이름 탐지 ──────────────────────────────────────────────
# 문장 중간(소문자/쉼표 뒤)에 나온 대문자 단어만 후보로 삼는다 - 문장 맨 앞의
# 흔한 단어(You, The, Okay...)는 그냥 문장 시작이라 대문자화된 것뿐이라 제외.
_NAME_RE = re.compile(r"(?<=[a-z,]\s)[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")

# 대사 중간에 대문자로 나와도 고유명사가 아닌 흔한 영어 단어들 - 데이터셋과
# 무관한 일반 불용어 목록이라 다른 자막에도 그대로 재사용 가능.
_STOPWORDS = {
    "i", "you", "he", "she", "it", "we", "they", "this", "that", "these", "those",
    "what", "who", "how", "why", "when", "where", "which",
    "the", "a", "an", "and", "but", "or", "so", "if", "as", "of", "in", "on", "at",
    "no", "yes", "okay", "ok", "well", "look", "see", "get", "got", "do", "did",
    "does", "come", "came", "good", "bad", "god", "dad", "mom", "mum", "stand",
    "sit", "sunday", "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "morning", "night", "today", "tomorrow", "yesterday", "now",
    "here", "there", "please", "thank", "thanks", "sorry", "hi", "hello", "hey",
    "mr", "mrs", "ms", "dr", "nothing", "something", "everything", "anything",
    "all", "some", "any", "none", "let", "can", "could", "will", "would", "should",
    "must", "may", "might", "not", "never", "always", "just", "still", "even",
}


def detect_candidate_names(text, min_count=2):
    """대문자 단어(연속)가 문장 중간에 min_count번 이상 등장하면 고유명사 후보."""
    counts = Counter(_NAME_RE.findall(text))
    return [
        name for name, c in counts.items()
        if c >= min_count and len(name) >= 3
        and not all(w.lower() in _STOPWORDS for w in name.split())
    ]


# ── 2) 위키피디아 조회 ──────────────────────────────────────────────
def _wiki_api_call(params, timeout=10, max_retries=8):
    """네트워크 문제로 "위키에 없다"고 잘못 판단해 폴백으로 새는 걸 최대한 줄이기
    위해, 일시적 오류(429, 타임아웃 등)는 지수 백오프로 꽤 끈질기게(최대 9번)
    재시도한다. 캐싱 덕분에 이 비용은 문서당 딱 한 번만 든다."""
    url = f"https://en.wikipedia.org/w/api.php?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "llm-chunking-glossary/1.0"})
    for attempt in range(max_retries + 1):
        try:
            time.sleep(0.4)  # 위키피디아 요청 제한(429) 방지용 자체 스로틀
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except Exception:
            if attempt < max_retries:
                time.sleep(min(2 ** attempt, 20))  # 1,2,4,8,16,20,20,20초로 점점 늘려가며 재시도
                continue
            return None
    return None


def wiki_lookup(en_title, lang_code="ko"):
    """영어 위키피디아 문서를 찾아 다른 언어판 링크 제목을 가져온다."""
    data = _wiki_api_call({
        "action": "query", "titles": en_title, "prop": "langlinks",
        "lllang": lang_code, "format": "json", "redirects": 1,
    })
    if not data:
        return None
    for page in data.get("query", {}).get("pages", {}).values():
        links = page.get("langlinks")
        if links:
            return links[0]["*"]
    return None


def extract_name_from_title(candidate, translated_title):
    """'The Truman Show' -> '트루먼 쇼'처럼 문서 제목 전체가 아니라, 후보 이름
    부분만 뽑아낸다. 단어 개수가 안 맞는 애매한 경우는 안전하게 전체를 반환."""
    c_words = candidate.split()
    t_words = translated_title.split()
    if len(c_words) == 1 and t_words:
        return t_words[0]
    return translated_title


# ── 3) 원문 문맥으로 재검증 ──────────────────────────────────────────
def get_context_sentences(name, doc, max_examples=3):
    """원문에서 이 이름이 실제로 등장하는 문장을 최대 max_examples개 뽑는다."""
    pat = re.compile(r"\b" + re.escape(name) + r"\b")
    out = []
    for u in doc.units:
        t = doc.text[u.start:u.end]
        if pat.search(t):
            out.append(t)
        if len(out) >= max_examples:
            break
    return out


def validate_sense(name, candidate_translation, examples, tokenizer, model, device):
    """위키에서 찾은 번역이 실제 문맥에서 쓰인 뜻과 맞는지 LLM에게 재확인시킨다.
    동음이의어(반려견 Pluto vs 행성 명왕성) 오류를 걸러내기 위함. 특정 이름을
    안 다루고 순수하게 "예문 + 후보 번역"만으로 판단하니 어떤 데이터셋에도 통한다."""
    if not examples:
        return True  # 검증할 예문이 없으면 그냥 통과시킴(안전한 기본값)
    ex_text = "\n".join(f"- {e}" for e in examples)
    prompt = (
        f'Here are example sentences from a script that use the word "{name}":\n'
        f"{ex_text}\n\n"
        f'A dictionary lookup suggests translating "{name}" into Korean as "{candidate_translation}".\n'
        f'Based ONLY on how "{name}" is actually used in the example sentences above, '
        f'does "{candidate_translation}" seem like a plausible, correct translation here '
        f"(not a completely different, unrelated meaning)? "
        f"Answer with exactly one word: YES or NO."
    )
    verdict = generate(tokenizer, model, device, [{"role": "user", "content": prompt}],
                        max_new_tokens=10, do_sample=False)
    return verdict.strip().upper().startswith("YES")


# ── 4) 폴백: 여러 후보 생성 + 역번역으로 최선 선택 ──────────────────────
def _edit_distance(a, b):
    a, b = a.lower(), b.lower()
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[len(a)][len(b)]


_TRANSLIT_SYSTEM = "You transliterate English proper nouns into natural Korean. Output ONLY the Korean transliteration, nothing else."
_BACKTRANS_SYSTEM = "Given a Korean transliteration of an English name, output your best guess of the original English spelling. Output ONLY the English word, nothing else."


def get_canonical_fallback(name, tokenizer, model, device, off_target_ids, target_script, n_candidates=8):
    """위키에 없거나(또는 문맥 재검증 실패) 했을 때의 폴백. 여러 후보(온도를 다르게)를
    강제한글(bad_words_ids)로 생성한 뒤, 각각을 다시 영어로 역번역해서 원래
    이름과 편집거리가 가장 가까운 후보를 채택한다 - 외부 정답 없이도 스스로
    신뢰도를 가늠하는 자기검증(self-consistency) 방식. 후보를 늘릴수록 "운 나쁘게
    첫 시도가 나쁜 값"으로 굳어질 위험이 줄어든다.

    반환: (선택된 후보, 편집거리) - 편집거리가 크면 호출부에서 "낮은 신뢰도"로
    표시할 수 있도록 그대로 넘겨준다(나쁜 값을 조용히 확정하지 않기 위함)."""
    temps = [None] + [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1][:max(0, n_candidates - 1)]
    candidates = []
    for t in temps:
        kwargs = {"do_sample": True, "temperature": t} if t else {}
        cand = generate(tokenizer, model, device, [
            {"role": "system", "content": _TRANSLIT_SYSTEM},
            {"role": "user", "content": "Spencer"},
            {"role": "assistant", "content": "스펜서"},
            {"role": "user", "content": name},
        ], max_new_tokens=20, bad_words_ids=off_target_ids, **kwargs)
        if cand and is_translated_ok(cand, target_script) and cand not in candidates:
            candidates.append(cand)
    if not candidates:
        return None, None

    scored = []
    for c in candidates:
        back = generate(tokenizer, model, device, [
            {"role": "system", "content": _BACKTRANS_SYSTEM},
            {"role": "user", "content": "스펜서"},
            {"role": "assistant", "content": "Spencer"},
            {"role": "user", "content": c},
        ], max_new_tokens=10, do_sample=False)
        scored.append((c, _edit_distance(name, back)))
    return min(scored, key=lambda x: x[1])


# ── 5) 오케스트레이션 ──────────────────────────────────────────────
def build_glossary(doc, tokenizer, model, device, off_target_ids, target_script, wiki_lang="ko", cache_path=None):
    """문서 전체에 대해 딱 한 번 호출한다. 반환: {영어이름: 확정번역}.

    cache_path를 주면 결과를 파일로 캐싱한다 - 위키피디아 조회는 네트워크
    상태에 따라, 폴백은 온도 샘플링 때문에 실행마다 살짝 다른 값이 나올 수
    있어(재현성 문제) 한 번 확정되면 그 값을 그대로 재사용한다. 같은 문서로
    다시 실행하면 네트워크/LLM 호출 없이 캐시를 즉시 읽어 완전히 동일한
    결과를 보장한다."""
    if cache_path and Path(cache_path).exists():
        return json.loads(Path(cache_path).read_text(encoding="utf-8"))

    LOW_CONFIDENCE_THRESHOLD = 2  # 역번역 편집거리가 이보다 크면 신뢰도 낮음으로 표시만 하고 그대로 사용
    candidates = detect_candidate_names(doc.text)
    glossary = {}
    for name in candidates:
        ko = wiki_lookup(name, wiki_lang)
        if ko:
            ko = extract_name_from_title(name, ko)
            examples = get_context_sentences(name, doc)
            if not validate_sense(name, ko, examples, tokenizer, model, device):
                ko = None  # 문맥과 안 맞으면 위키 결과 버리고 폴백으로
        if ko:
            if ko and is_translated_ok(ko, target_script):
                glossary[name] = ko
            continue

        ko, dist = get_canonical_fallback(name, tokenizer, model, device, off_target_ids, target_script)
        if ko and is_translated_ok(ko, target_script):
            glossary[name] = ko
            if dist is not None and dist > LOW_CONFIDENCE_THRESHOLD:
                # 나쁜 값을 조용히 확정하지 않고 정직하게 남긴다 - 그래도 캐싱은
                # 되므로(재현성), 이 로그를 보고 필요하면 사람이 수동으로 고칠 수 있다.
                print(f"    ⚠️ 용어집 낮은 신뢰도: {name} -> {ko} (역번역 편집거리 {dist})")

    if cache_path:
        Path(cache_path).write_text(json.dumps(glossary, ensure_ascii=False, indent=2), encoding="utf-8")
    return glossary


def inject_glossary(prompt, chunk_text, glossary):
    """청크 안에 실제로 등장하는 이름만 골라 프롬프트 맨 앞에 지시문으로 붙인다."""
    relevant = {k: v for k, v in glossary.items() if k in chunk_text}
    if not relevant:
        return prompt
    gloss_line = "다음 이름은 항상 이렇게 번역하세요: " + ", ".join(f"{k} → {v}" for k, v in relevant.items())
    return gloss_line + "\n" + prompt
