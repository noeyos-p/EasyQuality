"""
최종 답변 생성 에이전트 (Answer Agent)
- 검색 에이전트가 제공한 [USE: ...] 태그를 수집하여 제거합니다.
- 마지막에 [참고 문서] 섹션을 자동 생성합니다.
"""

import re
from collections import defaultdict
from backend.agent import AgentState

def answer_agent_node(state: AgentState):
    """[서브] 답변 에이전트 - [USE: ...] 태그를 제거하고 [참고 문서] 섹션만 생성"""

    context_list = state.get("context", [])

    if not context_list:
        return {"messages": [{"role": "assistant", "content": "검색된 정보가 없습니다."}]}

    # context에서 검색 에이전트의 보고서 추출
    search_report = "\n\n".join(context_list)

    # "[검색 에이전트 조사 최종 보고]" 헤더 제거
    search_report = re.sub(r'###\s*\[검색 에이전트 조사 최종 보고\]\s*', '', search_report)

    # [DONE] 태그 제거 (나중에 [참고 문서] 뒤에 추가)
    search_report = re.sub(r'\s*\[DONE\]\s*$', '', search_report, flags=re.MULTILINE).strip()

    # [USE: ...] 태그 수집 (참고 문서 섹션 생성용)
    use_tags = re.findall(r'\[USE:\s*([^\|]+)\s*\|\s*([^\]]+)\]', search_report)
    print(f"[DEBUG answer.py] 총 {len(use_tags)}개 USE 태그 발견")
    doc_clauses = defaultdict(set)  # 문서별 조항 수집

    # [USE: 문서명 | 조항] 태그를 수집하고 제거
    def collect_and_remove_tag(match):
        """[USE: doc | clause] -> 태그 수집 후 제거 (조항 번호가 있는 것만)"""
        doc_name = match.group(1).strip()
        clause_info = match.group(2).strip()

        # 조항 정보에서 실제 조항 번호만 추출 (예: "5.1.3 제 3레벨(작업지침서(WI):" -> "5.1.3")
        clause_match = re.match(r'([\d\.]+)', clause_info)

        if clause_match:
            clean_clause = clause_match.group(1)
            # 조항 번호 형식 (숫자로 시작하고 점이 있음)
            if clean_clause and '.' in clean_clause and clean_clause[0].isdigit():
                doc_clauses[doc_name].add(clean_clause)
                print(f"[DEBUG answer.py] ✅ 수집: {doc_name} > {clean_clause}")
            else:
                # 숫자만 있는 경우도 포함 (예: "1", "2")
                doc_clauses[doc_name].add(clause_info.strip())
                print(f"[DEBUG answer.py] ✅ 수집: {doc_name} > {clause_info.strip()}")
        else:
            # 조항 번호가 아닌 메타 정보 (예: "문서 관계", "상위 참조", "v1.0 vs v2.0")
            # 이런 정보도 참고문서에 포함
            doc_clauses[doc_name].add(clause_info.strip())
            print(f"[DEBUG answer.py] ✅ 수집 (메타): {doc_name} > {clause_info.strip()}")

        # 인라인 인용 제거 - 빈 문자열 반환
        return ""

    # [USE: ...] 태그를 수집하고 제거 (인라인 인용 없이)
    converted = re.sub(
        r'\[USE:\s*([^\|]+)\s*\|\s*([^\]]+)\]',
        collect_and_remove_tag,
        search_report
    )

    # ========================================
    # [참고 문서] 섹션 자동 생성
    # ========================================
    if doc_clauses:
        print(f"[DEBUG answer.py] 최종 수집된 문서: {list(doc_clauses.keys())}")
        ref_section = "\n\n[참고 문서]\n"
        for doc_name in sorted(doc_clauses.keys()):
            clauses = doc_clauses[doc_name]
            # 조항 번호 정렬
            try:
                sorted_clauses = sorted(clauses, key=lambda x: [int(n) if n.isdigit() else n for n in re.split(r'\.', x)])
            except:
                sorted_clauses = sorted(clauses)

            ref_line = f"{doc_name}({', '.join(sorted_clauses)})\n"
            print(f"[DEBUG answer.py] 참고문서 라인: {ref_line.strip()}")
            ref_section += ref_line

        converted += ref_section
    else:
        print(f"[DEBUG answer.py] ⚠️ doc_clauses가 비어있음!")

    # [DONE] 태그를 마지막에 추가
    converted += "\n[DONE]"


    return {"messages": [{"role": "assistant", "content": converted}]}
