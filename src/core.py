# 파이프라인 전체가 공유하는 자료구조

# Doc.text -> 청커가 보는 유일한 입력.  두 청킹 방식이 똑같이 받는다.
# Doc.units -> 병합기·평가기만 보는 스켈레톤.
# Chunk     -> 청커의 출력.  text + (start, end)

from collections import namedtuple

# ── Unit : 원본 파일의 최소 조각 ──────────────────────────────
#   자막이면 큐 1개, 워드면 문단 1개 또는 표의 행 1개.
#
#   kind : "cue" | "para" | "row"
#   meta : cue -> {"index":7, "t_start":94.8, "t_end":96.6}
#          para -> {"style":"Heading 1", "heading":True}
#          row -> {"table":1, "row":0, "header":["연산자","의미"]}
#
#   예) Unit(start=7, end=28, kind="cue",
#            meta={"index":2, "t_start":96.6, "t_end":99.6})
#       -> doc.text[7:28] == "또 돼지들 싹 다 잡아다 파묻는 거여?"



Unit = namedtuple("Unit", ["start","end","kind","meta"])

# ── Doc : 로더의 출력 ────────────────────────────────────────
#   name : 파일 이름
#   text : 모든 유닛을 개행으로 이어붙인 평문. 
#   units: Unit 목록.  text 안의 위치와 원본 메타데이터를 이어준다.
#   fmt  : "srt" | "docx" | "text"
#   log  : 전처리가 무엇을 몇 개 정리했는지.

Doc = namedtuple("Doc",["name","text","units","fmt","log"])



# ── Chunk : 청커의 출력 ──────────────────────────────────────
#   text  : 청크 내용
#   start : Doc.text 안에서의 시작 위치(문자 인덱스)
#   end   : 끝 위치(exclusive)
#   불변식: chunk.text == doc.text[chunk.start:chunk.end]

Chunk = namedtuple("Chunk",["text","start","end"])

