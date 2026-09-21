"""udayallu/RAG-Multi-Corpus(GitHub, MIT License) -> data/docx_eng/RAG-Multi-Corpus/.

지금까지 이 프로젝트가 "실제 .docx 네이티브 문서"를 찾으려고 여러 경로(공공데이터포털/PRISM/
KOCW/공유마당/미국 정부기관 등)를 다 시도했지만 전부 실패했다 - 완성된 실제 문서는 거의
항상 PDF로 배포되고, .docx 자체가 원본으로 존재하는 오픈라이선스 코퍼스는 찾지 못했다.

`RAG-Multi-Corpus`는 다르다: 5개 가상 기업(항공사/은행/자동차/대학/기술기업)의 정책문서·
매뉴얼·보고서·안내서를 **저자가 직접 합성 생성한 콘텐츠**로, 처음부터 .docx가 진짜 원본
포맷이다(HTML을 우리가 변환한 게 아님). 저작권이 저자 본인에게 있고 리포 루트의 MIT
LICENSE가 전체 콘텐츠에 적용되므로, 지금까지 계속 걸렸던 "개별 문서 저작권 불분명" 문제가
전혀 없다. 직접 확인한 품질:
- 236개 docx 중 117개(49.6%)에 표가 있고 총 320개 표, **colspan/rowspan 병합 0건**
  (전수 확인, zipfile로 word/document.xml의 gridSpan/vMerge 문자열 검색).
- 문단이 진짜 "Heading 1"/"Heading 2" 워드 스타일을 써서 `load_docx()`의 헤딩 인식과
  바로 호환됨.

원본 그대로 받아서 회사(카테고리)별 하위 폴더로 저장한다(변환/파싱 없음 - 이미 .docx).
"""
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DOCX_ENG_DIR

REPO_URL = "https://github.com/udayallu/RAG-Multi-Corpus.git"
OUT_DIR = DOCX_ENG_DIR / "RAG-Multi-Corpus"


def main():
    clone_dir = Path("/tmp/RAG-Multi-Corpus-src")
    if clone_dir.exists():
        shutil.rmtree(clone_dir)
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(clone_dir)], check=True)

    datasets_dir = clone_dir / "datasets"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    for org_dir in sorted(datasets_dir.iterdir()):
        docx_src = org_dir / "docx"
        if not docx_src.is_dir():
            continue
        org_out = OUT_DIR / org_dir.name
        org_out.mkdir(parents=True, exist_ok=True)
        n = 0
        for f in sorted(docx_src.glob("*.docx")):
            shutil.copy2(f, org_out / f.name)
            n += 1
        total += n
        print(f"[{org_dir.name}] {n}개")

    shutil.rmtree(clone_dir)
    print(f"완료: 총 {total}개")
    print(f"저장 위치: {OUT_DIR}")


if __name__ == "__main__":
    main()
