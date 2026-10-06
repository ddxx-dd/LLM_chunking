#!/bin/bash
# 실행 전에 `source env.sh`. 이 프로젝트는 .venv311/bin/python 만 쓴다(시스템 python3 사용 금지).
#   예) source env.sh && .venv311/bin/python scripts/preprocess.py --dataset vectara
export HF_HOME=/root/jupyter/LLM_chunking/.home/.cache/huggingface   # 모델 캐시를 프로젝트 안으로 고정
export HF_HUB_OFFLINE=1          # 실험 중 네트워크 접근 차단(재현성·속도). 모델을 새로 받을 때만 0으로
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/root/jupyter/LLM_chunking/src${PYTHONPATH:+:$PYTHONPATH}
