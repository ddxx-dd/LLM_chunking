#!/bin/bash
# 실험 1: 세 데이터셋(vectara, allganize, docx_corpus) 전체에서 fixed / semantic / smart 비교 (docs/restructure_plan.md 6절, 평가는 top-3 고정)
#   source env.sh && bash scripts/legacy/exp1_main.sh [데이터셋...]
# 고정 파라미터(사용자 지정, 질문 점수로 고르지 않음): semantic·smart min 128 / max 512 / percentile 85, fixed 500자(overlap 0).
# 청크 -> data/chunks/<ds>/, 결과 -> results/<ds>/retrieval/<태그>.json (질문은 분할 없이 데이터셋 전체).
set -e
MIN=128; MAX=512; PCT=85; CHARS=500
DATASETS=${@:-vectara allganize docx_corpus}
PY=.venv311/bin/python
echo "[실험1 설정] semantic·smart min $MIN / max $MAX / percentile $PCT, fixed ${CHARS}자 / k=3 / 질문 전체(분할 없음) / 데이터셋 $DATASETS"
for ds in $DATASETS; do
  $PY -W ignore scripts/legacy/chunk_v1.py --dataset $ds --chunker fixed --chunk-size-chars $CHARS 2>&1 | grep -v "Loading weights" | tail -1
  for ch in semantic smart; do
    $PY -W ignore scripts/legacy/chunk_v1.py --dataset $ds --chunker $ch --min-tokens $MIN --max-tokens $MAX --percentile $PCT 2>&1 | grep -v "Loading weights" | tail -1
  done
  $PY -W ignore scripts/legacy/run_v1.py --task retrieval --dataset $ds \
      --tag "fixed__c${CHARS},semantic__p${PCT}-min${MIN}-max${MAX},smart__p${PCT}-min${MIN}-max${MAX}" 2>&1 | grep -v "Loading weights"
done
echo "[실험1 끝]"
