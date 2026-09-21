"""캐나다 정부 오픈데이터 포털(open.canada.ca)에서 진짜 네이티브 .docx 문서를 수집한다.
CKAN API로 format=DOCX 리소스를 전수 조회한 뒤, 다음 3개 필터를 통과한 것만 채택:
  1. 언어 = 영어(langdetect) - 캐나다는 공용어 이중고시 의무라 프랑스어 버전도 항상 존재함
  2. 표 0~1개 - 그 이상은 대부분 통계/재무 보고서류라 표가 급격히 늘어남(실측 확인)
  3. 병합 셀(colspan/rowspan) 없음 - python-docx가 병합 셀 텍스트를 병합범위 전체에
     복제해서 반환하는데, 우리 load_docx()의 header-value zip 로직이 이걸 못 걸러내서
     서로 다른 행의 값이 엉뚱하게 짝지어지는 버그가 있음(실측 발견, 신청서/양식류에서
     흔함) - load_docx() 자체를 고치지 않고 수집 단계에서 배제하는 쪽을 택함(RAG-Multi-
     Corpus 선정 때와 같은 원칙: "병합 0건"인 소스만 채택).

실측 결과(2026-09): 전체 1,352개 리소스 중 1,205개 다운로드 성공, 그중 필터 통과 383개.
라이선스는 전부 캐나다 정부 계열 Open Government Licence 변형(연방 ca-ogl-lgo, 온타리오
on-oglo, 유콘 yk-oglyk, 퀘벡은 qc-cc-by로 명시 등) - CC-BY와 동등하게 허용적.
"""
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlparse

import langdetect
from docx import Document

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DOCX_ENG_DIR, DATA_DIR

API = "https://open.canada.ca/data/api/3/action/package_search"
OUT_DIR = DOCX_ENG_DIR / "Canada_Government"
MANIFEST_PATH = DATA_DIR / "canada_gov_docx_manifest.json"
TMP_DIR = Path("/tmp/canada_gov_docx_raw")

MAX_TABLES = 1


def curl_json(url, params):
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    out = subprocess.run(["curl", "-s", f"{url}?{qs}", "-H", "Referer: https://open.canada.ca/"],
                          capture_output=True, timeout=30)
    return json.loads(out.stdout)


def _en(v):
    """title/name 필드가 {"en":..,"fr":..} 대신 그냥 문자열로 올 수도 있음
    (CKAN API 응답 스키마가 바뀐 걸로 보임, 2026-09-20 실측 발견)."""
    return v.get("en", "") if isinstance(v, dict) else (v or "")


def list_docx_resources():
    """CKAN API로 format=DOCX 리소스를 전수 조회."""
    resources, start, rows, total = [], 0, 100, None
    while True:
        data = curl_json(API, {"fq": "res_format:DOCX", "rows": rows, "start": start})["result"]
        total = total or data["count"]
        if not data["results"]:
            break
        for pkg in data["results"]:
            for res in pkg.get("resources", []):
                if res.get("format", "").upper() == "DOCX":
                    resources.append({
                        "pkg_id": pkg["id"],
                        "pkg_title_en": _en(pkg.get("title")),
                        "res_id": res["id"],
                        "res_name": _en(res.get("name")),
                        "url": res.get("url", ""),
                        "license_id": pkg.get("license_id", ""),
                        "license_title": pkg.get("license_title", ""),
                    })
        start += rows
        print(f"  리소스 목록 조회: {start}/{total}")
        if start >= total:
            break
    return resources


def download(resources):
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(resources):
        url = r["url"]
        if not url:
            continue
        if url.startswith("/"):
            url = "https://open.canada.ca" + url  # API가 상대경로를 주는 경우가 있음(실측 발견)
        dest = TMP_DIR / f"{i:05d}_{r['res_id'][:8]}.docx"
        if dest.exists():
            continue
        subprocess.run(["curl", "-sL", "--max-time", "20", url,
                         "-H", "Referer: https://open.canada.ca/", "-o", str(dest)],
                        capture_output=True)
        if dest.exists() and dest.stat().st_size < 500:
            dest.unlink()
        if (i + 1) % 100 == 0:
            print(f"  다운로드 진행: {i + 1}/{len(resources)}")


def passes_filters(fp):
    if not zipfile.is_zipfile(fp):
        return None
    try:
        with zipfile.ZipFile(fp) as z:
            xml = z.read("word/document.xml").decode("utf-8", errors="ignore")
        doc = Document(str(fp))
    except Exception:
        return None
    if "<w:gridSpan" in xml or "<w:vMerge" in xml:
        return None
    n_tables = len(re.findall(r"<w:tbl>", xml))
    if n_tables > MAX_TABLES:
        return None
    n_paras = len([p for p in doc.paragraphs if p.text.strip()])
    # 앞부분(제목/목차)만 보면 이중언어 문서(캐나다는 공용어 이중고시 의무)가
    # 새서 통과함 - 실측 발견(23개). 전체 텍스트로 감지해야 함.
    full_text = " ".join(p.text for p in doc.paragraphs if p.text.strip())
    try:
        lang = langdetect.detect(full_text[:3000]) if full_text.strip() else "empty"
    except Exception:
        lang = "unknown"
    if lang != "en":
        return None
    return {"n_tables": n_tables, "n_paras": n_paras}


def safe_name(s, maxlen=80):
    return re.sub(r'[\\/:*?"<>|]', "_", s).strip()[:maxlen]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("1) 리소스 목록 조회 중...")
    resources = list_docx_resources()
    print(f"   전체 DOCX 리소스: {len(resources)}개")

    print("2) 다운로드 중...")
    download(resources)

    print("3) 필터링 및 정리 중...")
    manifest = []
    used_names = set()
    for i, r in enumerate(resources):
        fp = TMP_DIR / f"{i:05d}_{r['res_id'][:8]}.docx"
        if not fp.exists():
            continue
        info = passes_filters(fp)
        if info is None:
            continue
        url_basename = unquote(Path(urlparse(r["url"]).path).name)
        base = url_basename[:-5] if url_basename.lower().endswith(".docx") else (r["res_name"] or r["res_id"])
        name = safe_name(base) + ".docx"
        n = 1
        while name in used_names:
            n += 1
            name = safe_name(base) + f"_{n}.docx"
        used_names.add(name)
        (OUT_DIR / name).write_bytes(fp.read_bytes())
        manifest.append({"file": name, **r, **info})

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"\n완료: {len(manifest)}개 -> {OUT_DIR}")
    print(f"매니페스트: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
