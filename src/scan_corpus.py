"""10절 3단계: docx 45개·pdf 50개 전체에 항등 테스트(+docx는 표시 테스트)를 돌려서
문서별 결과와 실패 사유별 개수를 집계한다. 문서당 파싱은 1번만(elements 재사용,
§0 원칙 4번) + data/processed/<이름>.json 캐시 재사용(parse 코드가 바뀌었을 때만
--reparse로 다시 파싱). 단계별(파싱/항등/표시/병합지표) 소요 시간을 문서별로 기록한다."""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))


def scan_docx(files, reparse):
    from docobj import anchored_path_for, cached_parse
    from docx_track.identity_test import run_identity_test as docx_identity
    from docx_track.parse import parse_docx
    from merge_checks import count_mixed_formatting_paragraphs, run_mark_test

    results = []
    for i, f in enumerate(files, 1):
        name = Path(f).name
        t = {}
        try:
            t0 = time.monotonic()
            elements = cached_parse(f, "docx", parse_docx, reparse=reparse)
            t["parse"] = time.monotonic() - t0

            t0 = time.monotonic()
            idn = docx_identity(f, elements=elements)
            t["identity"] = time.monotonic() - t0

            t0 = time.monotonic()
            mark = run_mark_test(f, elements=elements)
            t["mark"] = time.monotonic() - t0

            t0 = time.monotonic()
            mixed = count_mixed_formatting_paragraphs(elements, anchored_path_for(f))
            t["merge_metrics"] = time.monotonic() - t0

            # [3] 최종 통과 = 항등 테스트(구조 100% + 텍스트 일치율 99.5%) AND 표시 테스트(미기록 0)
            passed = idn["passed"] and mark["unwritten"] == 0
            results.append({
                "file": name, "ok": True, "passed": passed,
                "text_match": idn["text_match"], "text_match_rate": idn["text_match_rate"],
                "struct_match": idn["struct_match"],
                "unwritten": mark["unwritten"], "unwritten_fallback": mark["unwritten_fallback"],
                "mixed": mixed["mixed_formatting_paragraphs"], "mixed_total": mixed["target_paragraphs"],
                "time": t,
            })
            print(f"[docx {i}/{len(files)}] {name}: 최종={'통과' if passed else '실패'} "
                  f"unwritten={mark['unwritten']} mixed={mixed['mixed_formatting_paragraphs']} "
                  f"시간(파싱/항등/표시/병합지표)={t['parse']:.1f}/{t['identity']:.1f}/{t['mark']:.1f}/{t['merge_metrics']:.1f}초",
                  flush=True)
        except Exception as e:
            results.append({"file": name, "ok": False, "error": str(e), "time": t})
            print(f"[docx {i}/{len(files)}] {name}: 예외 발생 - {e}", flush=True)
            traceback.print_exc()
    return results


def scan_pdf(files, reparse):
    from docobj import cached_parse
    from pdf_track.identity_test import run_identity_test as pdf_identity
    from pdf_track.parse import parse_pdf

    results = []
    for i, f in enumerate(files, 1):
        name = Path(f).name
        t = {}
        try:
            t0 = time.monotonic()
            elements = cached_parse(f, "pdf", parse_pdf, reparse=reparse)
            t["parse"] = time.monotonic() - t0

            t0 = time.monotonic()
            r = pdf_identity(f, elements=elements)
            t["identity"] = time.monotonic() - t0

            results.append({
                "file": name, "ok": True, "passed": r["passed"],
                "text_match_rate": r["text_match_rate"], "drawing_ok": r["drawing_ok"],
                "preserved_rate": r["preserved_rate"], "fit_rate": r["fit_rate"],
                "fit_by_kind": r["fit_by_kind"],
                "text_mismatches": r["text_mismatches"], "preserved_mismatches": r["preserved_mismatches"],
                "written": r["written"], "failed": r["failed"], "time": t,
            })
            print(f"[pdf {i}/{len(files)}] {name}: 항등={'통과' if r['passed'] else '실패'} "
                  f"text_match={r['text_match_rate']:.1%} preserved={r['preserved_rate']:.1%} "
                  f"fit_rate={r['fit_rate']:.1%} 시간(파싱/항등)={t['parse']:.1f}/{t['identity']:.1f}초", flush=True)
        except Exception as e:
            results.append({"file": name, "ok": False, "error": str(e), "time": t})
            print(f"[pdf {i}/{len(files)}] {name}: 예외 발생 - {e}", flush=True)
            traceback.print_exc()
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("track", choices=["docx", "pdf"])
    ap.add_argument("out_json")
    ap.add_argument("--reparse", action="store_true", help="캐시 무시하고 Docling 다시 실행")
    args = ap.parse_args()

    if args.track == "docx":
        files = sorted(Path("../data/allganize/docx").glob("**/*.docx"))
        files = [f for f in files if "checkpoint" not in str(f)]
        print(f"docx {len(files)}개 스캔 시작(reparse={args.reparse})", flush=True)
        results = scan_docx(files, args.reparse)
    else:
        import torch
        torch.backends.cudnn.enabled = False
        files = sorted(Path("../data/vectara_ragbench").glob("*.pdf"))
        print(f"pdf {len(files)}개 스캔 시작(reparse={args.reparse})", flush=True)
        results = scan_pdf(files, args.reparse)

    Path(args.out_json).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    n_ok = sum(1 for r in results if r.get("ok") and r.get("passed"))
    n_err = sum(1 for r in results if not r.get("ok"))
    total_time = sum(sum(r.get("time", {}).values()) for r in results)
    print(f"\n=== {args.track} 스캔 완료: {len(results)}개 중 {n_ok}개 통과, {n_err}개 예외, "
          f"총 {total_time/60:.1f}분 ===", flush=True)
