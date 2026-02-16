# import json
# import re
# from typing import Dict, List, Optional
# from backend.agent import get_openai_client, compare_versions_tool, AgentState

# def comparison_agent_node(state: AgentState):
#     """[서브] 비교 에이전트 (ID/버전 추출 및 리포트 생성)"""
#     client = get_openai_client()
#     query = state["query"]
#     model = state.get("worker_model") or state.get("model_name") or "gpt-4o-mini"
    
#     # 1. 문서명 및 버전 추출 (Regex)
#     # 문서명 패턴: ABC-DEF-12345 or ABC-12345
#     doc_pattern = r'([A-Z]{2,10}-[A-Z]{2,10}-\d{5})|([A-Z]{2,10}-\d{5})'
#     version_pattern = r'v?\d+(?:\.\d+)*'
    
#     doc_match = re.search(doc_pattern, query)
#     doc_name = doc_match.group(0) if doc_match else None
    
#     # 문서명이 추출되었다면 해당 부분을 query에서 제외하고 버전 검색 (ID 내 숫자가 버전으로 오인되는 것 방지)
#     query_for_versions = query.replace(doc_name, "") if doc_name else query
#     versions = re.findall(version_pattern, query_for_versions)
    
#     v1 = versions[0] if len(versions) >= 2 else None
#     v2 = versions[1] if len(versions) >= 2 else None
    
#     # 2. 추출 실패 시 LLM에게 요청
#     if not doc_name or not v1 or not v2:
#         prompt = f"""사용자의 질문에서 문서명(ID)과 비교할 두 가지 버전 번호를 추출하세요.
#         질문: {query}
        
#         [규칙]
#         - 반드시 JSON 형식으로만 응답하세요.
#         - 코드 블록(```json 등)을 사용하지 마세요.
#         - 큰따옴표(")만 사용하세요.
#         - 추출 실패 시 해당 값은 null로 채우세요.
        
#         예시: {{"doc_name": "EQ-SOP-00001", "v1": "1.0", "v2": "2.0"}}
#         """
#         try:
#             res = client.chat.completions.create(
#                 model=model,
#                 messages=[{"role": "user", "content": prompt}],
#                 temperature=0.0
#             )
#             content = res.choices[0].message.content.strip()
#             # 간단한 클렌징 (LLM이 규칙을 어기고 마크다운을 씌울 경우 대비)
#             content = re.sub(r'```json|```', '', content).strip()
            
#             data = json.loads(content)
#             doc_name = doc_name or data.get("doc_name")
#             v1 = v1 or data.get("v1")
#             v2 = v2 or data.get("v2")
#         except Exception as e:
#             return {"messages": [{"role": "assistant", "content": f"문서 ID 또는 버전 정보를 추출하는 데 실패했습니다. 파싱 에러: {e}"}]}

#     if not doc_name or not v1 or not v2:
#         return {"messages": [{"role": "assistant", "content": f"비교를 위해 문서명과 두 개의 버전이 필요합니다. (입력된 정보: {doc_name}, {v1}, {v2})"}]}

#     # 3. 데이터 조회 및 절차적 비교 (Chunk-based)
#     try:
#         from backend.agent import _sql_store
        
#         # 문서 정보 조회
#         doc1 = _sql_store.get_document_by_name(doc_name, v1)
#         doc2 = _sql_store.get_document_by_name(doc_name, v2)
        
#         if not doc1 or not doc2:
#             report = f"비교 실패: {doc_name}의 v{v1} 또는 v{v2}를 찾을 수 없습니다."
#             return {"context": [report]}

#         # 청크 조회
#         chunks1 = _sql_store.get_chunks_by_document(doc1['id'])
#         chunks2 = _sql_store.get_chunks_by_document(doc2['id'])

#         # 조항별 그룹화
#         def group_by_clause(chunks):
#             grouped = {}
#             for c in chunks:
#                 cl = c.get('clause') or 'Unknown'
#                 if cl not in grouped:
#                     grouped[cl] = []
#                 grouped[cl].append(c.get('content', '').strip())
#             return grouped

#         g1 = group_by_clause(chunks1)
#         g2 = group_by_clause(chunks2)

#         # 비교 수행
#         all_clauses = sorted(list(set(g1.keys()) | set(g2.keys())))
#         changes = []
        
#         for cl in all_clauses:
#             content1 = "\n".join(g1.get(cl, []))
#             content2 = "\n".join(g2.get(cl, []))
            
#             if cl not in g1:
#                 changes.append({"clause": cl, "type": "추가", "v1": "", "v2": content2})
#             elif cl not in g2:
#                 changes.append({"clause": cl, "type": "삭제", "v1": content1, "v2": ""})
#             elif content1 != content2:
#                 changes.append({"clause": cl, "type": "변경", "v1": content1, "v2": content2})

#         if not changes:
#             report = f"### [비교 에이전트 보고]\n문서: {doc_name} (v{v1} -> v{v2})\n\n두 버전 간에 조항 수준에서 변경된 내용이 없습니다."
#             return {"context": [report]}

#     except Exception as e:
#         report = f"비교 처리 중 오류가 발생했습니다: {e}"
#         return {"context": [report]}

#     # 4. 요약 리포트 생성 (Z.AI)
#     # 효율적인 토큰 사용을 위해 변경된 내용만 발췌하여 전달
#     changes_summary = []
#     for c in changes[:30]: # 너무 많으면 상위 30개만
#         changes_summary.append(f"[{c['type']}] 조항 {c['clause']}\n- 구버전: {c['v1'][:500]}\n- 신버전: {c['v2'][:500]}")
    
#     display_changes = "\n\n".join(changes_summary)
    
#     summary_prompt = f"""두 문서 버전 간의 조항별 변경 사항을 분석하여 요약 보고서를 작성하세요.
    
#     [문서 정보]
#     문서명: {doc_name}
#     버전: v{v1} -> v{v2}
    
#     [변경 사항 데이터]
#     {display_changes}
    
#     [작성 가이드]
#     1. 핵심 변경점 5개를 불릿 포인트로 정리하세요.
#     2. 변경 유형 태그를 부여하세요: [절차/역할/수치/기간/용어/기타] (중복 가능)
#     3. 위험도(낮음/중간/높음)를 판정하고 그 이유를 1~2문장으로 기술하세요.
#     4. 제공된 데이터를 바탕으로 구체적인 변경 내용을 설명하세요.
#     5. 어조는 전문적이고 분석적이어야 하며, 한국어로 작성하세요.
#     6. 답변 마지막에 [DONE]을 포함하세요.
#     """
    
#     try:
#         res = client.chat.completions.create(
#             model=model,
#             messages=[{"role": "user", "content": summary_prompt}],
#             temperature=0.1
#         )
#         report_content = res.choices[0].message.content
#     except Exception as e:
#         report_content = f"요약 리포트 생성 중 오류가 발생했습니다: {e}"

#     final_output = f"""### [비교 에이전트 보고]
# 문서: {doc_name} (v{v1} -> v{v2})

# {report_content}

# ---
# [조항별 상세 변경 내역 (발췌)]
# {display_changes[:3000] + (f"\n...(외 {len(changes)-30}건 더 있음)" if len(changes) > 30 else "")}
# """
#     return {"context": [final_output]}
