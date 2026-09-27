"""docx 병합 writer 핵심(smart_chunk.md 7절) - "문단에 직접 속한 w:t만 고친다" 방식
(그룹 도형을 run.text 대입으로 파괴하던 버그의 해결책, identity_test.py에서 실측 확인).
identity_test.py(자기 텍스트 되돌려쓰기)와 merge_checks.py(마킹 테스트)가 이 모듈을
공유한다."""
from docx.oxml import OxmlElement
from docx.oxml.ns import qn


def _local(tag):
    return tag.rsplit("}", 1)[-1]


def _append_text_run(para_xml, text):
    """para_xml에 w:r/w:t가 하나도 없을 때 새로 만들어서 붙인다 - ★ 실측 확인(2026-09-26,
    docx 3단계 재스캔에서 발견한 진짜 콘텐츠 손실 원인): 세로 병합 셀(vMerge)의 연속 행은
    OOXML 스펙상 문단은 있지만 run이 하나도 없는 빈 문단인 경우가 흔하다. 이런 문단이
    paraIds[0](번역문을 받을 자리)로 뽑히면, 기존 코드는 "쓸 w:t가 없다"며 그냥 넘어가고
    반환값도 실패였는데, 같은 셀의 "나머지" 문단(진짜 텍스트가 있던 paraIds[1:])은 그대로
    비워버려서 - 결과적으로 아무 데도 안 써진 채 원본 텍스트만 사라지는 실제 데이터 손실이
    있었다(실측: 병합 셀 3문단 중 1번째만 비어있고 3번째에 실제 텍스트가 있던 사례)."""
    r = OxmlElement("w:r")
    t = OxmlElement("w:t")
    t.text = text
    t.set(qn("xml:space"), "preserve")
    r.append(t)
    para_xml.append(r)
    return t


def direct_text_nodes(para_xml):
    """para_xml에 직접 속한 w:t만 반환 - 조건: t.iterancestors(w:p)의 첫 번째가
    이 문단 자신일 때만(중첩된 도형/텍스트박스 안 w:t는 제외)."""
    return [t for t in para_xml.iter(qn("w:t"))
            if next(t.iterancestors(qn("w:p")), None) is para_xml]


def build_paraid_index(doc):
    return {p.get(qn("w14:paraId")): p for p in doc.element.body.iter(qn("w:p"))}


def resolve_targets(doc, paraid_to_p, loc):
    if "paraId" in loc:
        p = paraid_to_p.get(loc["paraId"])
        return [p] if p is not None else []
    if "paraIds" in loc:
        return [paraid_to_p[pid] for pid in loc["paraIds"] if pid in paraid_to_p]
    cell = doc.tables[loc["table_index"]].cell(loc["row"], loc["col"])
    return [p._p for p in cell.paragraphs]


def fallback_sibling(para_xml):
    """★ 실측 확인(2026-09-26, 표시 테스트에서 발견) - mc:AlternateContent의 Choice(최신
    DrawingML) 쪽 문단만 Docling 요소가 되고 Fallback(구버전 VML) 쪽은 요소가 없어서
    한 번도 안 써진다(원문 그대로 남음 - 번역 후엔 원문·번역문이 섞인 상태가 됨). Choice
    쪽 문단이면 같은 AlternateContent 안 Fallback 쪽의 "같은 순번"(각 branch 안에서
    문서 순서 인덱스) 문단을 찾아 반환한다(없으면 None) - 실측: 인용구 카드 문단들이
    Choice/Fallback 양쪽에 순서대로 1:1 대응하는 것으로 확인됨."""
    choice = next((a for a in para_xml.iterancestors() if _local(a.tag) == "Choice"), None)
    if choice is None:
        return None
    alt = choice.getparent()
    if alt is None or _local(alt.tag) != "AlternateContent":
        return None
    fallback = next((c for c in alt if _local(c.tag) == "Fallback"), None)
    if fallback is None:
        return None
    choice_ps = list(choice.iter(qn("w:p")))
    try:
        idx = choice_ps.index(para_xml)
    except ValueError:
        return None
    fallback_ps = list(fallback.iter(qn("w:p")))
    if idx >= len(fallback_ps):
        return None
    return fallback_ps[idx]


def write_paragraph_text(para_xml, text, mirror_fallback=True):
    """이 문단(과, mirror_fallback이면 대응하는 Fallback 문단에도) 직접 속한 w:t에 text를
    쓴다. 반환: "ok" | "no_text_node"."""
    targets = [para_xml]
    if mirror_fallback:
        sib = fallback_sibling(para_xml)
        if sib is not None:
            targets.append(sib)

    wrote_any = False
    for p in targets:
        t_nodes = direct_text_nodes(p)
        if not t_nodes:
            if text:  # 쓸 내용이 있는데 w:t가 없으면 새로 만듦(위 _append_text_run 참고)
                _append_text_run(p, text)
                wrote_any = True
            continue
        t_nodes[0].text = text
        t_nodes[0].set(qn("xml:space"), "preserve")
        for extra in t_nodes[1:]:
            extra.text = ""
        wrote_any = True
    return "ok" if wrote_any else "no_text_node"


def write_element_text(doc, paraid_to_p, loc, text, auto_num=None, mirror_fallback=True):
    """요소/셀 하나의 loc에 text를 쓴다(paraId 우선, paraIds는 첫 문단만 쓰고 나머진 비움,
    table_index fallback). auto_num이 있으면 번역문 앞에서 뗀다(3절)."""
    if not loc:
        return "no_loc"
    if auto_num and text.startswith(auto_num):
        text = text[len(auto_num):]
    targets = resolve_targets(doc, paraid_to_p, loc)
    if not targets:
        return "no_loc"
    result = "ok"
    for i, para_xml in enumerate(targets):
        text_for_this_para = text if i == 0 else ""
        r = write_paragraph_text(para_xml, text_for_this_para, mirror_fallback=mirror_fallback)
        if i == 0 and r == "no_text_node":
            result = "no_text_node"
    return result
