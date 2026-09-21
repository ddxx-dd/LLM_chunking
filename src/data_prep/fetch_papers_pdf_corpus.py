"""논문(의사학/PLOS) 원본 PDF를 그대로 받아서 저장한다.

논문은 원래 PDF(또는 출판사 JATS-XML)가 네이티브 배포 포맷이고 docx는 아니다(직접 확인:
GROBID/science-parse 같은 논문 전용 파서도 PDF를 직접 입력받고, docx는 애초에 "저작 중"
포맷이지 출판된 논문의 배포 포맷이 아님). 그래서 지금까지 HTML→docx로 변환해서 쓰던
논문들을 원본 PDF 그대로 받는 형식으로 바꾼다. docx 변환(파싱 로직) 없이 파일을 그대로
저장만 함 - 실제 파싱(load_pdf())은 나중 단계에서 loader.py에 추가할 것.

PMC는 제외했다 - NCBI가 cloudpmc-viewer에 JS 기반 proof-of-work 챌린지를 걸어놔서
직접 다운로드가 안 되는데, 이건 우회하지 않는 게 맞다(봇 차단 회피는 이 프로젝트의
목적과 무관하게 하지 않을 것). PMC는 이미 갖고 있던 JATS-XML 소스(TomTBT/pmc_open_access_xml)
쪽이 오히려 그 데이터셋의 정식 프로그램적 접근 경로였는데, 사용자가 PMC 논문 장르
자체를 코퍼스에서 빼기로 결정해서 이 스크립트에서도 완전히 제외했다.

아동학회지(4066) / PLOS pone.0356173도 제외했다 - `load_pdf()`를 설계하다가 실제
PDF 레이아웃을 직접 확인해보니 이 둘만 2단(two-column) 편집이었다. PyMuPDF의
`get_text("blocks")`는 2단 레이아웃도 대체로 올바른 순서(왼쪽 단 전체→오른쪽 단 전체)로
뽑아주지만, `load_pdf()`의 표-문단 순서 재구성 로직(y좌표 정렬)이 2단에서는 왼쪽 단과
오른쪽 단을 섞어버리는 부작용이 있다. 컬럼 인식 로직을 추가하는 대신, `load_pdf()`를
"1단만 지원"으로 단순하게 유지하기로 하고(과설계 방지) 2단인 이 두 편을 코퍼스에서
제외했다.

`kjmh-31-1-35`도 제외했다 - `load_pdf()` 검증 스크립트로 실제 PDF를 뽑아보니 표가
11개나 나와서(원래 이 논문은 "표 0개"로 확인해서 골랐던 것) 직접 원인을 봤더니, 이
논문의 부록 표(고서 출처 비교표)가 정적 HTML에는 전혀 없고 JS로만 동적 로딩되는
콘텐츠였다(`#display-objects` 패널조차 없음) - 예전 HTML 기반 검증이 틀린 게 아니라
애초에 그 방법으론 접근 불가능한 정보였던 것. PDF는 완성 렌더링이라 이 표들이 다
들어있고, 결과적으로 이 논문은 "표 0~2개" 기준을 원래부터 만족 못 했던 것으로 확인돼
제외했다.

`pone.0334738`(Dynamical systems model of emotional contagion)도 제외했다 - 수식이
많은 논문이라 LaTeX 조판된 수식의 개별 기호(첨자·괄호·연산자 등)가 PDF 안에서 각각
따로 위치해 있어서, 줄 단위 추출이 수식 하나를 기호별로 잘게 쪼개 별도 "문단"으로
만들어버렸다(실측: 짧은 줄 698개, 대부분 "[", "]", "N", "X" 같은 수식 파편). 서술
문장 자체는 안 깨지지만 수식 부분이 노이즈로 흩어져서 품질이 떨어져 제외함 - 이건
이 프로젝트만의 문제가 아니라 문헌(PDFBoT 논문 등)에도 "디스플레이 모드 수식은
룰베이스 추출의 알려진 약점"으로 명시된 케이스임.

남은 14편(의사학 5 + PLOS 9)은 전부 1단·표 0~2개·수식 노이즈 없음을 PDF 원본에서
직접 재확인함.

각 소스의 PDF 링크는 논문 페이지 자체의 citation_pdf_url 메타태그(또는 PLOS의
article/file?type=printable 엔드포인트)에서 직접 확인한 것. PLOS는 Referer 헤더
없이 직접 GET하면 간헐적으로 404가 나는 걸 확인해서(HEAD 요청은 되는데 GET만 실패 -
hotlink 방지로 추정) 논문 페이지 URL을 Referer로 붙여서 요청한다.
"""
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import PDF_KOR_DIR, PDF_ENG_DIR

# (파일명, PDF URL)
MEDHIST = [
    ("kjmh-31-2-297_Dr. James Smith's Dream of Eradicating Smallpox",
     "https://www.medhist.or.kr/upload/pdf/kjmh-31-1-297.pdf"),
    ("kjmh-27-1-49_Korea-Japan Parasite Control Cooperation",
     "https://www.medhist.or.kr/upload/pdf/kjmh-27-1-49.pdf"),
    ("kjmh-31-2-181_Hygienic Masks in Colonial Korea",
     "https://www.medhist.or.kr/upload/pdf/kjmh-31-1-181.pdf"),
    ("kjmh-32-1-81_From Contact Lens to Dream Lens",
     "https://www.medhist.or.kr/upload/pdf/kjmh-32-1-81.pdf"),
    ("kjmh-34-2-209_Early HIV-AIDS Epidemic in Korea",
     "https://www.medhist.or.kr/upload/pdf/kjmh-34-1-209.pdf"),
]
# (파일명, doi, plos 저널 경로) - 표 0~2개, 병합 0, 교육학/환경과학/인지사회심리학 3개 분야
PLOS = [
    ("pone.0345347_AI awareness and education mixed methods", "10.1371/journal.pone.0345347", "plosone"),
    ("pone.0357589_High school science fair student communication", "10.1371/journal.pone.0357589", "plosone"),
    ("pone.0346386_The hypothesis becomes forced", "10.1371/journal.pone.0346386", "plosone"),
    ("pone.0347073_LASSO-based reduced-form CMAQ model", "10.1371/journal.pone.0347073", "plosone"),
    ("pone.0357404_Two applied approaches for treating aged DDT-contaminated soil", "10.1371/journal.pone.0357404", "plosone"),
    ("pbio.3003949_Seaweed carbon removal cannot keep up with climate-driven loss", "10.1371/journal.pbio.3003949", "plosbiology"),
    ("pone.0355610_Community-based climate knowledge", "10.1371/journal.pone.0355610", "plosone"),
    ("pmen.0000700_Human-centred resilience quantification", "10.1371/journal.pmen.0000700", "mentalhealth"),
    ("pcsy.0000115_Opinion polarization from compression-based decision making", "10.1371/journal.pcsy.0000115", "complexsystems"),
]


def fetch(url, referer=None, retries=3):
    # PLOS의 신설 저널(mentalhealth/complexsystems) 엔드포인트가 간헐적으로 404를
    # 내는 걸 실측함(같은 URL을 몇 초 뒤 재시도하면 됨 - 서버 쪽 캐시/워밍업 문제로 추정).
    headers = {"User-Agent": "Mozilla/5.0"}
    if referer:
        headers["Referer"] = referer
    req = urllib.request.Request(url, headers=headers)
    for attempt in range(retries):
        try:
            return urllib.request.urlopen(req, timeout=30).read()
        except urllib.error.HTTPError:
            if attempt == retries - 1:
                raise
            time.sleep(8)


def safe_name(name):
    return re.sub(r'[\\/:*?"<>|]', "", name)[:100]


def save(out_dir, name, url, referer=None):
    out = out_dir / f"{safe_name(name)}.pdf"
    if out.exists():
        return out, out.stat().st_size
    data = fetch(url, referer)
    out.write_bytes(data)
    return out, len(data)


def main():
    manifest = []

    mh_dir = PDF_KOR_DIR / "한국어_논문" / "의사학"
    mh_dir.mkdir(parents=True, exist_ok=True)
    for name, url in MEDHIST:
        out, n = save(mh_dir, name, url)
        manifest.append({"file": str(out), "source": "medhist.or.kr", "url": url, "bytes": n, "license": "CC BY-NC"})
        print(f"[의사학] {out.name}: {n:,} bytes")

    plos_dir = PDF_ENG_DIR / "PLOS_논문"
    plos_dir.mkdir(parents=True, exist_ok=True)
    for name, doi, journal in PLOS:
        url = f"https://journals.plos.org/{journal}/article/file?id={doi}&type=printable"
        referer = f"https://journals.plos.org/{journal}/article?id={doi}"
        out, n = save(plos_dir, name, url, referer=referer)
        manifest.append({"file": str(out), "source": f"PLOS ({journal})", "doi": doi, "url": url, "bytes": n, "license": "CC BY"})
        print(f"[PLOS] {out.name}: {n:,} bytes")
        time.sleep(2)  # journals.plos.org가 연속 요청에 간헐적으로 404를 내서 완충

    manifest_path = PDF_KOR_DIR.parent / "pdf_papers_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n완료: 총 {len(manifest)}편")
    print(f"매니페스트: {manifest_path}")


if __name__ == "__main__":
    main()
