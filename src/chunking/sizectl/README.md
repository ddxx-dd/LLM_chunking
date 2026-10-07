# sizectl: min/max 크기 제어 의미 분할 엔진

"의미 경계에서 자르되 청크 크기를 min 이상 max 이하로 맞추는" 엔진. 프로젝트 요구사항의 토큰 카운팅·로드 밸런싱에 해당하며,
**지금은 srt와 `chunking/v1/`(실험 1)이 쓰고, 나중에 새 청커 7종에 min/max를 붙일 때 다시 쓸 부품**이다(옛 코드가 아니다).

| 파일 | 내용 |
|---|---|
| `semantic_core.py` | `SemanticEngine`(문장 임베딩 거리 → 임계값 → 크기 정리), `sentence_spans`(문장 나누기), `consecutive_distances` |
| `size_dp.py` | 순수 함수 `partition`: 크기 제약(max 하드, min 부족분 최소, 경계 이득 최대)을 사전식으로 푸는 DP |

새 청커(`chunking/`의 fixed, recursive, header, semantic, A, B, E)는 이 엔진을 쓰지 않는다(이번 실험은 min/max 없음).
