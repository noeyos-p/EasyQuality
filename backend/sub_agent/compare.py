"""
비교 에이전트 (Comparison Agent) - Refactored Version
- 두 문서 또는 같은 문서의 서로 다른 버전 간 차이점을 분석하거나 버전 목록을 조회합니다.
- 결과를 context에 보고서 형태로 저장합니다.
"""


import json
from typing import Optional
from backend.agent import get_zai_client, get_version_history_tool, AgentState, safe_json_loads
from backend.agent import get_openai_client, get_langchain_llm, AgentState, safe_json_loads
from langsmith import traceable
from backend.sql_store import SQLStore
from backend.graph_store import Neo4jGraphStore

def normalize_version(v: Optional[str]) -> Optional[str]:
    """버전 번호 정량화: '1' -> '1.0', 'v1.1' -> '1.1'"""
    if not v: return v
    v_str = str(v).strip().lower()
    if v_str.startswith('v'): v_str = v_str[1:]
    # 도트가 없고 숫자만 있는 경우 .0 부착
    if '.' not in v_str and v_str.replace('.', '').isdigit():
        return f"{v_str}.0"
    return v_str

@traceable(name="comparison_agent", run_type="chain")
def comparison_agent_node(state: AgentState):
    """[서브] 비교 에이전트 - 버전 목록 조회 또는 내용 비교 분석"""
    print(f"[COMPARISON AGENT] 진입! query={state.get('query')}")
    query = state["query"]
    model = state.get("worker_model") or state.get("model_name") or "gpt-4o"

    # 1. 의도 분석 (LangChain ChatOpenAI 사용 - LangSmith 자동 추적)
    # 사용자가 버전 목록을 보고 싶어하는지, 아니면 실제 내용 비교를 원하는지 구분
    intent_prompt = f"""Extract the intent and parameters from the user question.
Question: {query}

## Intent Classification

| Intent | Condition | Keyword Examples |
|--------|-----------|------------------|
| `list_history` | Retrieve version list/history of a specific document | "version types", "history", "revision log", "how many versions?" |
| `compare_versions` | Compare content between two versions | "differences", "changes", "what changed?", "compare with latest version" |

## Output (JSON only, no other text)

{{"intent": "list_history | compare_versions", "doc_id": "document ID", "v1": "version 1 or null", "v2": "version 2 or null"}}"""

    try:
        # LangChain ChatOpenAI 사용 (JSON 응답)
        llm = get_langchain_llm(model=model, temperature=0.0)
        llm = llm.bind(response_format={"type": "json_object"})

        res = llm.invoke([{"role": "user", "content": intent_prompt}])

        info = safe_json_loads(res.content)
        intent = info.get("intent")
        doc_id = info.get("doc_id")
        v1 = normalize_version(info.get("v1"))
        v2 = normalize_version(info.get("v2"))

        print(f"비교 의도: {intent}, 문서: {doc_id}, 버전: {v1} vs {v2}")

        # [CASE 1] 버전 목록 조회
        if intent == "list_history":
            print(f"[DEBUG] list_history 분기 실행")
            store = SQLStore()
            versions = store.get_document_versions(doc_id)
            if not versions:
                history = f"{doc_id} 문서의 버전을 찾을 수 없습니다."
            else:
                history = "\n".join([f"- v{v['version']} ({v['created_at']})" for v in versions])

            # [USE: ...] 태그 추가 (참고문서에 포함되도록)
            use_tag = f"[USE: {doc_id} | 버전 이력]"

            return {"context": [f"### [{doc_id} 버전 이력]\n{history}\n\n{use_tag} [DONE]"]}

        # [CASE 2] 버전 비교
        elif intent == "compare_versions":
            print(f"[DEBUG] compare_versions 분기 실행")
            # 버전 정보가 없으면 자동 추론 (최신 2개)
            if not v1 or not v2:
                print(f"[DEBUG] 버전 자동 선택 시작")
                store = SQLStore()
                versions = store.get_document_versions(doc_id)
                print(f"[DEBUG] 조회된 버전 개수: {len(versions) if versions else 0}")
                unique_versions = []
                seen = set()
                for v in versions:
                    if v['version'] not in seen:
                        unique_versions.append(v['version'])
                        seen.add(v['version'])
                
                print(f"[DEBUG] 고유 버전: {unique_versions}")
                
                if len(unique_versions) >= 2:
                    v1, v2 = unique_versions[1], unique_versions[0] # v1(이전), v2(최신)
                    print(f"     -> 자동 선택된 버전: {v1} vs {v2}")
                elif len(unique_versions) == 1:
                    return {"context": [f"### [{doc_id} 비교 불가]\n현재 문서의 버전이 하나({unique_versions[0]})뿐이라 비교할 대상이 없습니다. [DONE]"]}
                else:
                    return {"context": [f"### [{doc_id} 비교 불가]\n버전 목록을 가져올 수 없어 비교가 불가능합니다."]}

            # 실제 비교 데이터 조회 (SQL Diff) 및 검증
            print(f"[DEBUG] SQL Diff 조회 시작: {doc_id}, v1={v1}, v2={v2}")
            # [Hybrid] 실제 비교 데이터 조회 (SQL Store 직접 호출)
            print(f"[DEBUG] SQLStore 상세 Diff 조회 시작: {doc_id}, v1={v1}, v2={v2}")
            try:
                sql_store = SQLStore()
                diffs = sql_store.get_clause_diff(doc_id, v1, v2)

                # [Safety Check] "ADDED"로 표시된 항목이 v1.0 본문에 이미 있었는지 전수 조사
                # (도구 방식에서는 JSON 전달 문제로 누락될 수 있는 정밀 로직)
                added_items = [d for d in diffs if d.get('change_type') == 'ADDED']
                if added_items:
                    v1_doc = sql_store.get_document_by_name(doc_id, v1)
                    if v1_doc:
                        v1_chunks = sql_store.get_chunks_by_document(v1_doc['id'])
                        v1_full_text = "".join([c['content'] for c in v1_chunks]).replace(" ", "").replace("\n", "")
                        
                        for item in diffs:
                            if item.get('change_type') == 'ADDED':
                                item_content_norm = (item.get('v2_content') or "").replace(" ", "").replace("\n", "")
                                if len(item_content_norm) > 10 and item_content_norm in v1_full_text:
                                     item['change_type'] = 'UNCHANGED' 

                # 보고서용 데이터 포맷팅 (MODIFIED ONLY)
                comp_lines = []
                for item in diffs:
                    if item.get('change_type') == 'MODIFIED':
                        clause_id = item.get('clause') or "N/A"
                        v1_txt = (item.get('v1_content') or "").strip()
                        v2_txt = (item.get('v2_content') or "").strip()
                        # 공백 무시하고 실제 내용이 다른 경우만 포함
                        if v1_txt.replace(" ", "").replace("\n", "") != v2_txt.replace(" ", "").replace("\n", ""):
                            comp_lines.append(f"### 조항 {clause_id}\n[변경 전 원문]\n{v1_txt}\n\n[변경 후 원문]\n{v2_txt}\n")

                comp_data = "\n".join(comp_lines)
                if not comp_data:
                    comp_data = "데이터 비교 결과, 실제 텍스트 내용이 변경된 조항이 발견되지 않았습니다."
                    
            except Exception as e:
                 print(f"[DEBUG compare.py] SQL 상세 조회 실패: {e}")
                 comp_data = "데이터 비교 중 조항 정보를 가져오지 못했습니다."
            
            # [Hybrid] 영향 분석 조회 (Neo4j Store 직접 호출)
            graph_store = None
            try:
                graph_store = Neo4jGraphStore()
                graph_store.connect()
                impacts = graph_store.get_impact_analysis(doc_id)
                if not impacts:
                    impact_data = "이 문서의 변경으로 인해 영향을 받는 다른 문서는 발견되지 않았습니다."
                else:
                    impact_data = json.dumps(impacts, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[DEBUG compare.py] Graph 직접 조회 실패: {e}")
                impact_data = "영향 분석 데이터를 가져오는 중 오류가 발생했습니다."
            finally:
                if graph_store:
                    try:
                        graph_store.close()
                    except:
                        pass

            if not diffs: 
                 return {"context": [f"### [비교 에이전트 오류]\n{doc_id}의 지정된 버전({v1}, {v2}) 데이터를 가져올 수 없습니다. [DONE]"]}

            # 3. 종합 분석 (Z.AI)
            analysis_prompt = f"""당신은 GMP 규정 전문 분석가입니다. 아래의 변경 데이터(Diff)와 영향 분석 데이터(Impact)를 바탕으로 전문적이고 사실 중심적인 '문서 변경 비교 보고서'를 작성하세요.

[변경 조항 데이터 (Diff)]
{comp_data}

[영향 분석 데이터 (Impact)]
{impact_data}

## 보고서 작성 지침
1. **사실 근거**: 반드시 [변경 조항 데이터 (Diff)]에 제공된 원문 정보를 바탕으로 작성하세요.
2. **요약 섹션**: '1. 변경 핵심 요약'에는 전체 변경 사항의 취지와 핵심 내용을 전문적인 용어를 사용하여 2~3문장으로 명확하게 요약하세요.
3. **상태 기반 요약**: '2. 상세 비교' 섹션에서는 제공된 원문의 핵심 내용을 파악하여 요약된 형태로 기술하세요. [변경 전:]과 [변경 후:] 뒤에는 단순 원문 복사가 아닌, 해당 조항에서 무엇이 어떻게 달라졌는지 핵심을 요약하여 작성하세요.
4. **영향 평가**: '3. 영향 평가' 섹션에는 제공된 영향 분석 데이터를 바탕으로, 이 변경이 다른 문서나 프로세스에 미칠 구체적인 파급 효과를 분석하여 기술하세요.
5. **언어**: 모든 내용은 반드시 한국어로 작성하세요.
6. **가독성 및 레이아웃 (필수)**:
    - 모든 섹션 헤더는 대괄호(`[]`)를 사용하세요. (예: `[상세 비교]`, `[영향 평가]`)
    - `[상세 비교]` 섹션 내의 조항들은 반드시 `1. 조항 X.X`, `2. 조항 Y.Y`와 같이 번호를 매기세요.
    - 각 조항 하위의 `-변경 전:`과 `-변경 후:`는 반드시 **새로운 줄**에서 시작하고, 앞에 하이픈(`-`)과 들여쓰기를 적용하세요.
7. **종료 태그**: 답변의 가장 마지막 줄에는 반드시 `[DONE]` 태그만을 포함하세요.

## 출력 형식 (반드시 준수)
[변경 핵심 요약]
-(전체 내용을 관통하는 전문적인 요약 2~3문장)

[상세 비교]
1. 조항 4.1
  -변경 전: ...
  -변경 후: ...
2. 조항 4.2
  -변경 전: ...
  -변경 후: ...

[영향 평가]
-(영향을 받는 문서명과 구체적인 영향 사유 기술)
[DONE]"""
            
            try:
                # LangChain ChatOpenAI 사용
                llm = get_langchain_llm(model=model, temperature=0.0)
                res = llm.invoke([
                    {"role": "system", "content": "당신은 문서 변경 비교 보고서 작성기입니다. 가이드라인에 맞춰 가독성 있게 보고서를 작성하세요. 모든 답변은 반드시 한국어로만 작성해야 합니다."},
                    {"role": "user", "content": analysis_prompt}
                ])

                final_report = res.content

                # [USE: ...] 태그 생성 (참고문서에 포함되도록)
                use_tags = []

                # 메인 문서 (버전 정보 포함)
                use_tags.append(f"[USE: {doc_id} | v{v1} vs v{v2}]")

                # 변경된 조항들
                modified_count = 0
                for item in diffs:
                    if item.get('change_type') == 'MODIFIED':
                        clause_id = item.get('clause', 'N/A')
                        print(f"[DEBUG compare.py] 변경된 조항: {clause_id}")
                        if clause_id and clause_id != 'N/A':
                            use_tags.append(f"[USE: {doc_id} | {clause_id}]")
                            modified_count += 1

                print(f"[DEBUG compare.py] 총 {modified_count}개 조항 태그 생성")

                # 영향 받는 문서들 (있다면)
                if impacts and isinstance(impacts, list):
                    print(f"[DEBUG compare.py] 영향 문서 개수: {len(impacts)}")
                    for impact in impacts:
                        if isinstance(impact, dict):
                            impact_doc = impact.get('source_doc_id') or impact.get('doc_id')
                            impact_section = impact.get('citing_section', '')
                            if impact_doc:
                                if impact_section:
                                    use_tags.append(f"[USE: {impact_doc} | {impact_section}]")
                                else:
                                    use_tags.append(f"[USE: {impact_doc} | 영향 문서]")

                # USE 태그를 보고서에 추가 (hidden)
                use_tags_str = " ".join(use_tags)
                print(f"[DEBUG compare.py] 생성된 USE 태그: {use_tags_str[:200]}...")

                return {"context": [final_report + f"\n\n{use_tags_str} [DONE]"]}
            except Exception as e:
                return {"context": [f"### [비교 보고서 생성 실패]\nLLM 호출 중 오류가 발생했습니다: {e} [DONE]"]}

        else:
             return {"context": [f"### [이해 불가]\n죄송합니다, 의도를 파악하지 못했습니다. (Intent: {intent}) [DONE]"]}
    
    except Exception as e:
        print(f"Compare Agent Error: {e}")
        return {"context": [f"### [에이전트 처리 오류]\n{str(e)} [DONE]"]}
