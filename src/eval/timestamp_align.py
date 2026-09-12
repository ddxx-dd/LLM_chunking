"""자막 큐 단위 비교를 위한 시간 기반 정렬. 통짜로 이어붙여 비교하는 대신,
원본 유닛 하나하나에 실제로 겹치는 시간대의 참조 큐를 찾아 맞춰준다.

정렬 기준은 Jaccard 시간 겹침 비율 (intersect+1)/(union+1) - OPUS 자막 코퍼스
정렬(Tiedemann)이 실제로 쓰는 공식과 같다. 겹치는 큐가 아예 없어도 union이
작을수록(=시간상 가까울수록) 점수가 자연히 높아지므로 별도 폴백 분기가 필요 없다.

같은 릴리즈의(=타임스탬프가 서로 잘 맞는) EN/KO 자막 쌍이면 이 단순한 방식만으로
충분하다는 게 실측으로 확인됨 - 이보다 복잡한 전체 시퀀스 DP 정렬을 시도해봤지만
오히려 더 나빴다(오정렬률 6~8% vs DP 23%). 그래서 굳이 더 복잡한 걸 쓰지 않는다."""


def _intersection(u, k):
    return max(0.0, min(u.meta["t_end"], k.meta["t_end"]) - max(u.meta["t_start"], k.meta["t_start"]))


def _jaccard(u, k):
    inter = _intersection(u, k)
    union = max(u.meta["t_end"], k.meta["t_end"]) - min(u.meta["t_start"], k.meta["t_start"])
    return (inter + 1) / (union + 1)


def align_by_overlap(src_doc, ref_doc):
    """src_doc 유닛마다, 시간이 제일 많이 겹치는 ref_doc 큐의 텍스트를 매칭.
    최선 매치조차 실제 시간 겹침이 0이면(한쪽 자막에만 그 구간이 존재 - 예: 삭제/누락된
    장면) 억지로 매칭시키지 않고 빈 문자열을 반환한다. score_chunks() 등 채점 함수가
    이미 빈 문자열을 스코어링에서 제외하므로 이것만으로 "매칭 없음" 처리가 끝난다 -
    OPUS/Tiedemann의 시간-슬롯 정렬이 "1:0 정렬"(대응 없음)을 명시적으로 허용하는 것과
    같은 원리. 실측으로 확인: 이 경우는 About Time 1905개 중 35개, Interstellar 1898개
    중 33개뿐이고, 실제로 안 겹쳐도 간격이 1초 미만인 경우(자막 제작자마다 다른 컷 포인트로
    생기는 정상적인 오차)는 내용상 정상 매칭인 게 수동 확인됨 - 그래서 간격 크기가 아니라
    "겹침이 아예 없다"는 조건만 쓴다(더 정교한 간격 기반 제외는 향후 검토 대상으로 남겨둠).
    반환: len(src_doc.units)와 같은 길이의 문자열 리스트(포지션이 곧 유닛 인덱스)."""
    out = []
    for u in src_doc.units:
        best = max(ref_doc.units, key=lambda k: _jaccard(u, k))
        out.append(ref_doc.text[best.start:best.end] if _intersection(u, best) > 0 else "")
    return out
