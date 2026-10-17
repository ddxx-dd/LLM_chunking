"""전처리: 원본 PDF·docx → (1차, stage1) Docling 변환·블록(meta_text_v2) → (2차, stage2) 모든 청커의 입력 clean_text.
문서 목록은 sources, Docling 관련 공통 부분은 docling_blocks, clean_text 형식 정리는 clean_text, 문장 나누기 함수는 sentences,
LongBench 원본 읽기는 longbench(원문 그대로)."""
