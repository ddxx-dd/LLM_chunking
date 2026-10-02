#!/usr/bin/env python
"""OPUS OpenSubtitles(v2024) 에서 영화 5편의 한/영 원본(raw XML)을 받아 .srt 로 변환한다. API 키 불필요.

1) en-ko 정렬 파일(xml/en-ko.xml.gz, 약 351MB)을 스트리밍으로 훑어 IMDb id 로 문서 쌍을 찾는다.
   한 영화에 쌍이 여러 개면 정렬 파일의 겹침 점수(score)가 가장 높은 쌍을 고른다.
2) raw/en.zip(35.8GB), raw/ko.zip(1.7GB) 은 통째로 받지 않고 HTTP Range 로 zip 목록과 해당 문서만 읽는다.
3) 원본 XML 을 srt 로 변환한다: 타임스탬프 그대로, UTF-8, 큐 번호 1부터.
4) 검증: EN 큐 중 KO 큐와 시간이 절반 이상 겹치는 비율, 큐 수, 마지막 큐 시간. 앞·중간·끝 큐 3개씩을 나란히 출력.
5) data/manifests/srt.json 에 OPUS 버전, 문서 경로, IMDb id, 점수를 기록한다.

저장 위치: data/raw/srt_eng, data/raw/srt_kor (git 제외), 받은 원본 XML/정렬 결과는 data/raw/opus/ (git 제외).
표준 라이브러리만 사용. 사용법: python scripts/download/get_srt_opus.py
"""
import bisect, datetime, gzip, io, json, os, re, sys, time, urllib.request, zipfile
import xml.etree.ElementTree as ET

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OPUS = os.path.join(ROOT, "data", "raw", "opus")
VERSION = "v2024"
BASE = f"https://object.pouta.csc.fi/OPUS-OpenSubtitles/{VERSION}"
UA = {"User-Agent": "LLMChunkingResearch/0.1"}

# (영화, 개봉연도, IMDb id, 영어 파일명, 한국어 파일명, 제목 확인용 키워드)
MOVIES = [
    ("Noah", 2014, 1959490, "Noah_Eng.srt", "노아.srt", ["noah", "methuselah"]),
    ("Deadpool", 2016, 1431045, "Deadpool_Eng.srt", "데드풀.srt", ["deadpool", "wade"]),
    ("Insidious: Chapter 2", 2013, 2226417, "Insidious_Chapter2_Eng.srt", "인시디어스2.srt", ["lambert", "elise"]),
    ("Doctor Strange", 2016, 1211837, "Doctor_Strange_Eng.srt", "닥터스트레인지.srt", ["kaecilius", "ancient one"]),
    ("Captain America: Civil War", 2016, 3498820, "Captain_America_Civil_War_Eng.srt", "캡틴아메리카시빌워.srt", ["accords", "sokovia"]),
]


class RangeFile(io.RawIOBase):
    """HTTP Range 로 읽는 파일 객체 (zipfile 이 목록과 개별 항목만 읽게 한다)."""
    def __init__(self, url):
        self.url, self.pos = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD", headers=UA), timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos
    def read(self, n=-1):
        if n < 0 or self.pos + n > self.size:
            n = self.size - self.pos
        if n <= 0:
            return b""
        for attempt in range(4):
            try:
                h = dict(UA, Range=f"bytes={self.pos}-{self.pos + n - 1}")
                with urllib.request.urlopen(urllib.request.Request(self.url, headers=h), timeout=300) as r:
                    data = r.read()
                self.pos += len(data)
                return data
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(3 * (attempt + 1))
    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)


def find_pairs():
    """정렬 파일에서 5편의 문서 쌍을 찾는다. 결과는 캐시한다."""
    cache = os.path.join(OPUS, "en-ko.pairs.json")
    if os.path.exists(cache):
        return json.load(open(cache, encoding="utf-8"))
    ids = {str(m[2]) for m in MOVIES}
    pat = re.compile(r'score="([^"]+)"[^>]*fromDoc="([^"]+)"\s+toDoc="([^"]+)"')
    found = {}
    url = f"{BASE}/xml/en-ko.xml.gz"
    print("정렬 파일 스트리밍:", url, flush=True)
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=600) as r:
        for line in gzip.GzipFile(fileobj=r):
            if b"<linkGrp" not in line:
                continue
            m = pat.search(line.decode("utf-8", "ignore"))
            if m and m.group(2).split("/")[-2] in ids:
                found.setdefault(m.group(2).split("/")[-2], []).append(
                    dict(score=float(m.group(1)), en_doc=m.group(2), ko_doc=m.group(3)))
    os.makedirs(OPUS, exist_ok=True)
    json.dump(found, open(cache, "w", encoding="utf-8"), indent=1)
    return found


def fetch_member(zip_obj, lang, doc):
    """doc 예: en/2013/2226417/1954573498.xml.gz -> zip 안 OpenSubtitles/raw/en/2013/2226417/1954573498.xml"""
    rel = doc.replace(".xml.gz", ".xml")
    dest = os.path.join(OPUS, "raw_xml", lang, os.path.basename(os.path.dirname(rel)), os.path.basename(rel))
    if not os.path.exists(dest):
        data = zip_obj.read(f"OpenSubtitles/raw/{rel}")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            f.write(data)
    return dest


def secs(t):
    h, m, s = t.replace(",", ".").split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def fmt(t):
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def parse_raw(xml_path):
    """raw XML -> [(start, end, text)]. S 태그에서 시작해 E 태그에서 끝나는 구간의 텍스트를 한 큐로 묶는다.

    한 큐가 여러 <s> 에 걸칠 수 있다(S 는 첫 문장, E 는 마지막 문장). 문장/줄 구분은 줄바꿈으로 둔다.
    """
    root = ET.parse(xml_path).getroot()
    cues, cur, orphans = [], None, 0

    def lines(txt):
        return [l.strip() for l in (txt or "").split("\n") if l.strip()]

    def feed(txt):
        nonlocal cur, orphans
        ls = lines(txt)
        if cur is not None:
            cur[2].extend(ls)
        elif ls:
            orphans += len(ls)

    for s in root.iter("s"):
        feed(s.text)
        for el in s:
            if el.tag == "time":
                kind, t = el.get("id", "")[-1:], secs(el.get("value"))
                if kind == "S":
                    if cur is not None and cur[2]:       # 앞 큐가 종료 시각 없이 끝남 -> 새 큐 직전에 닫는다
                        cues.append((cur[0], min(t, cur[0] + 7.0), cur[2]))
                    cur = [t, None, []]
                elif kind == "E" and cur is not None:
                    if cur[2]:
                        cues.append((cur[0], max(t, cur[0] + 0.001), cur[2]))
                    cur = None
            feed(el.tail)
    return cues, orphans


def write_srt(cues, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for i, (a, b, ls) in enumerate(cues, 1):
            f.write(f"{i}\n{fmt(a)} --> {fmt(b)}\n" + "\n".join(ls) + "\n\n")


def overlap_ratio(en, ko):
    """EN 큐 중 어떤 KO 큐와 (겹친 시간 / EN 큐 길이) >= 0.5 인 비율."""
    ks = sorted(ko)
    starts = [k[0] for k in ks]
    maxlen = max(k[1] - k[0] for k in ks)
    ok = 0
    for a, b, _ in en:
        i = bisect.bisect_left(starts, a - maxlen)
        best = 0.0
        while i < len(ks) and ks[i][0] < b:
            best = max(best, min(b, ks[i][1]) - max(a, ks[i][0]))
            i += 1
        ok += best >= 0.5 * (b - a)
    return ok / len(en)


def main():
    os.makedirs(OPUS, exist_ok=True)
    pairs = find_pairs()
    chosen = {}
    for name, year, imdb, en_f, ko_f, kw in MOVIES:
        cand = sorted(pairs.get(str(imdb), []), key=lambda p: -p["score"])
        if not cand:
            sys.exit(f"{name}: 정렬 파일에서 IMDb {imdb} 를 찾지 못했습니다.")
        chosen[imdb] = cand[0] | {"n_pairs": len(cand)}
    zips = {}
    for lang in ("ko", "en"):
        print(f"{lang} raw zip 목록 읽는 중 (Range)...", flush=True)
        zips[lang] = zipfile.ZipFile(RangeFile(f"{BASE}/raw/{lang}.zip"))
    manifest = dict(opus_version=VERSION, source="https://opus.nlpl.eu/OpenSubtitles/",
                    alignment=f"{BASE}/xml/en-ko.xml.gz", raw_en=f"{BASE}/raw/en.zip", raw_ko=f"{BASE}/raw/ko.zip",
                    created=datetime.date.today().isoformat(), movies=[])
    report = []
    for name, year, imdb, en_f, ko_f, kw in MOVIES:
        p = chosen[imdb]
        if f"/{year}/" not in p["en_doc"]:
            print(f"경고: {name} 의 문서 경로 연도가 {year} 가 아님: {p['en_doc']}")
        en_xml = fetch_member(zips["en"], "en", p["en_doc"])
        ko_xml = fetch_member(zips["ko"], "ko", p["ko_doc"])
        en, o_en = parse_raw(en_xml)
        ko, o_ko = parse_raw(ko_xml)
        write_srt(en, os.path.join(ROOT, "data", "raw", "srt_eng", en_f))
        write_srt(ko, os.path.join(ROOT, "data", "raw", "srt_kor", ko_f))
        text = " ".join(" ".join(c[2]) for c in en).lower()
        hits = {k: text.count(k) for k in kw}
        ratio = overlap_ratio(en, ko)
        row = dict(movie=name, year=year, imdb_id=imdb, score=p["score"], n_pairs=p["n_pairs"],
                   en_doc=p["en_doc"], ko_doc=p["ko_doc"], en_file=f"data/raw/srt_eng/{en_f}",
                   ko_file=f"data/raw/srt_kor/{ko_f}", cues_en=len(en), cues_ko=len(ko),
                   last_cue_en=fmt(en[-1][1]), last_cue_ko=fmt(ko[-1][1]), overlap_en_in_ko=round(ratio, 3),
                   orphan_lines_en=o_en, orphan_lines_ko=o_ko, title_keyword_hits=hits)
        manifest["movies"].append(row)
        report.append((row, en, ko))
    os.makedirs(os.path.join(ROOT, "data", "manifests"), exist_ok=True)
    with open(os.path.join(ROOT, "data", "manifests", "srt.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)

    print("\n| 영화 | IMDb | 점수 | EN 큐 | KO 큐 | EN 마지막 큐 | KO 마지막 큐 | EN 큐가 KO와 ≥50% 겹침 |")
    print("|---|---|---|---|---|---|---|---|")
    for r, _, _ in (x for x in report):
        print(f"| {r['movie']} | {r['imdb_id']} | {r['score']:.3f} | {r['cues_en']} | {r['cues_ko']} | "
              f"{r['last_cue_en']} | {r['last_cue_ko']} | {r['overlap_en_in_ko']:.1%} |")
    for r, en, ko in report:
        print(f"\n=== {r['movie']}  (제목 키워드 {r['title_keyword_hits']})")
        for label, idx in (("앞", [0, 1, 2]), ("중간", [len(en) // 2 - 1, len(en) // 2, len(en) // 2 + 1]),
                           ("끝", [len(en) - 3, len(en) - 2, len(en) - 1])):
            for i in idx:
                a, b, ls = en[i]
                j = bisect.bisect_left([k[0] for k in ko], a)
                j = min(max(j, 0), len(ko) - 1)
                ka, kb, kls = ko[j]
                print(f"[{label}] EN#{i + 1} {fmt(a)}-{fmt(b)} {' / '.join(ls)}")
                print(f"       KO#{j + 1} {fmt(ka)}-{fmt(kb)} {' / '.join(kls)}")


if __name__ == "__main__":
    main()
