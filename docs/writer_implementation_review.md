# writer.py 정리 + 병합 지표 작업 결과 (2026-09-26)

> "지금 할 것" 6개 항목(pdf writer 신설, add_para_ids 중복 처리, skip/merge_skip 분리,
> caption 연결을 item.captions로, 중간 파일 경로 이동, 루트 정리) 반영 결과. docx는
> 전부 완료·검증됨. pdf는 writer.py를 새로 만들고 검증하는 과정에서 심각한 버그 2개를
> 찾아 고쳤고, 아직 못 고친 이슈가 남아있어 진행 여부 확인이 필요한 상태.

---

## 1. pdf_track/writer.py 신설 — 완료, 검증 중 버그 2개 발견·수정

identity_test.py에 있던 flatten_targets / test_fit / redact→apply→insert 로직을
`pdf_track/writer.py`로 옮기고, `write_translations(pdf, elements, translations)` 하나로
표준화했다(docx_track/writer.py와 동일 구조 - 표시/항등 테스트와 실제 병합이 같은 코드
경로를 타야 검증 의미가 있음). identity_test.py는 이 writer를 불러 쓰도록 교체.

### 버그 A(수정 완료): NotoSansKR 폰트 파일이 숫자를 깨뜨림

Google Fonts에서 받은 `NotoSansKR-Regular.ttf`(v39)로 렌더링하면, **숫자 바로 뒤에
공백 없이 알파벳이 붙는 특정 패턴**(예: `"(3.1)is"`)에서 숫자 글자가 무작위 한자로
깨진다(`"(埼.基)is"`). 격리 실험으로 확인:
- 기본 내장 폰트 → 정상(`"(3.1)is understood"`)
- NanumGothic(다른 한글 폰트) → 정상
- 다운로드한 NotoSansKR-Regular.ttf → 깨짐

즉 PyMuPDF나 한글 폰트 전반의 문제가 아니라 **그 특정 폰트 파일 자체의 결함**(글꼴
매핑/GSUB 테이블 문제로 추정). 실제 번역 결과에 숫자가 들어있으면 조용히 깨질 수 있는
심각한 문제라, **writer의 폰트를 NanumGothic으로 교체**(`fonts/NanumGothic.ttf`, 이미
로컬에 있는 패키지에서 복사, 추가 다운로드 없음). `fonts/NotoSansKR-Regular.ttf`는 삭제.

### 버그 B(수정 완료): test_fit용 스크래치 페이지가 대상끼리 오염됨

원래 스크래치 페이지를 **페이지 1개당 한 번만** 비우고, 그 안에서 같은 페이지의 대상
여러 개를 연달아 test_fit했다. 문제: 앞 대상이 (scale_low 한계로) 다 못 들어가서 넘친
내용이 스크래치 페이지에 남아있는 채로 다음 대상을 테스트하면 판정이 오염된다.

**실측**: 아주 길고 밀도 높은 문단 하나를 **격리된** 상태에서 테스트하면 `spare=-1`
(정확히 "안 들어감"으로 판정)인데, **같은 페이지의 다른 대상들과 스크래치 페이지를
공유**한 상태에서는 `spare>=0`("들어감")으로 잘못 판정되고, 실제로는 **내용의 약
70%가 잘려나간 채로 "성공"이라고 기록**되는 심각한 문제가 재현됨 — 번역 손실을 감지
못 하고 조용히 흘려보내는 버그.

**해결**: 대상마다 스크래치 페이지를 새로 만들도록 수정(`write_translations`).

### 아직 남은 문제(미해결, 확인 필요)

두 버그를 고친 뒤에도 샘플 논문(239개 대상) 기준 **33건이 여전히 텍스트 불일치**:
- 대부분은 **폰트 글자 대체**(NanumGothic에 정확한 수학 기호 글리프가 없어서 비슷하게
  생긴 다른 유니코드로 대체됨) - 예: `‖`→`∥`, `〈`→`⟨`, `·`→`・`. 사소하고 시각적으로만
  다른 문제라 비교 정규화로 처리 가능해 보임(아직 반영 안 함).
- el 17 케이스는 버그 B와 비슷한 "일부만 잘려서 들어감" 증상이 재현됐는데, 원인을
  마지막으로 확인하려던 진단 스크립트가 타임아웃으로 죽어서 **아직 확정 못 함**.
- 기준 (c)(모든 요소 bbox 바깥 영역 픽셀 동일)도 21개 페이지에서 여전히 실패 -
  원인 미진단(위 문제와 연관 가능성 있음, 겹치는 bbox 때문일 수도 있음).

---

## 2. add_para_ids 중복 paraId 처리 — 완료

기존엔 paraId가 **없는** `<w:p>`에만 새 ID를 부여했는데, **원본에 이미 있던 값이 중복**인
경우(예: 복사-붙여넣기로 같은 paraId가 두 문단에 남는 실제 Word 문서 흔한 케이스)는
처리하지 않았다. 이제 중복이면 첫 등장한 쪽은 유지하고 이후 등장은 새 ID로 재부여.

**샘플 문서 실측 결과**: 새로 부여 356개(원래 paraId가 하나도 없던 문서), **원본에
중복이라 재부여 0개** — 이 샘플엔 중복이 없었음.

---

## 3. skip / merge_skip 분리 — 완료

- `skip`(번역·채점 자체를 제외 - formula/picture/field_code/footnote_ref/tracked_change/
  math_font/num_only): §5 매핑 루프가 이 필드만 본다.
- `merge_skip`(위치를 못 믿어서 **쓰기만** 제외 - `table_mismatch`, `no_loc`): 번역·채점은
  정상적으로 하되, writer가 이 필드도 같이 확인해서 실제 문서에는 안 쓴다.

docx `_build_table_element`(table_mismatch), pdf/docx 둘 다 loc이 None인 경우(no_loc)에
반영. identity_test.py/merge_checks.py도 둘 다 확인하도록 수정.

---

## 4. 캡션 연결을 Docling item.captions로 — 완료(단, docx는 실효성 없음)

- **pdf**: 잘 작동함. 샘플 논문 9개 표 전부 `caption_ids`가 정확히 채워짐(부분 캡션
  "(a) Case BB" 같은 것까지 정확히 연결됨) - 이전의 인접 요소 휴리스틱보다 확실히 낫다.
- **docx**: `MsWordDocumentBackend` 소스를 확인한 결과 **캡션-표 연결 코드 자체가 없어서
  `item.captions`가 항상 빈 리스트**다. 지시대로 docx도 item.captions로 바꿨지만, 결과적으로
  docx는 `caption_ids`가 항상 `[]`가 된다 — 이전 인접 요소 휴리스틱(불완전하지만 가끔
  맞음)보다 **커버리지가 떨어지는 실질적 회귀**. docx만 예전 휴리스틱을 폴백으로 되살릴지
  결정이 필요함(아직 미반영).

---

## 5. 중간 파일 경로 이동 — 완료

- anchored docx: `data/processed/anchored/*.anchored.docx`
- 항등/표시 테스트 출력: `results/checks/*.identitytest.docx`, `*.marktest.docx`,
  `*.identitytest.pdf`
- 원본 데이터 폴더(`data/allganize/`, `data/vectara_ragbench/`)에는 아무것도 안 씀(확인됨).
- `.gitignore`에 `results/checks/` 추가.

## 6. 루트 정리 — 완료

- 루트 `smart_chunker.py` 삭제(`src/smart_chunker.py`와 byte-identical이었음, 확인 후 삭제).
- `README.md`에 `.venv311`/`.venv-eval` 설치 순서 + 실행 순서 추가.

---

## 결과 재실행

**docx**: `add_para_ids`/`identity_test`/`merge_checks` 전부 재실행, **항등 테스트
통과**(356→356 문단/도형 수, 텍스트 완전 일치, bold/italic 완전 보존), 표시 테스트
unwritten 2건(전부터 알려진 Docling 중복 콘텐츠 미인식 케이스, 새로운 문제 아님).

**pdf**: 버그 A·B 수정 후에도 **항등 테스트 아직 실패**(33/239 텍스트 불일치, 21페이지
바깥 영역 픽셀 불일치) - 위 "아직 남은 문제" 참고.

---

## 결정 필요한 것

1. pdf 항등 테스트 마무리(폰트 look-alike 정규화 + el 17류 문제 원인 확정 + 기준 c
   재조사)를 계속할지, 아니면 지금 상태를 "한계"로 기록하고 3단계(45/50 전체 스캔)로
   넘어갈지.
2. docx의 `caption_ids`가 item.captions 때문에 항상 빈 리스트가 되는 문제 - 예전
   인접 요소 휴리스틱을 docx 전용 폴백으로 되살릴지, 아니면 이대로 둘지.
