import re
import json
import ast
# 하이브리드 방식: 참조 조회는 Tool, 정밀 영향 분석은 Store 직접 호출
from backend.graph_store import Neo4jGraphStore
from backend.agent import get_openai_client, get_langchain_llm, get_references_tool, AgentState, safe_json_loads, normalize_doc_id
from langsmith import traceable


# Graph Agent를 고도화하여 다음과 같은 기능을 추가했습니다.
# 질문 의도 분석: 영향 분석(Impact), 의존성 분석(Dependency) 등을 구분하여 답변합니다.
# 시각화 지원: 관계를 한눈에 볼 수 있도록 Mermaid 다이어그램을 생성합니다.
# 전문 보고서 형식: 참조 관계의 의미와 변경 시 주의사항을 포함한 상세 보고서를 제공합니다.
# 이제 에이전트에게 "SOP-xxx 변경 시 영향 알려줘"와 같이 질문하면 더욱 풍부한 답변을 받을 수 있습니다.
def generate_mermaid_flow(doc_id: str, refs: dict, impact_data: list = None) -> str:
    """Mermaid 다이어그램 코드 생성 (참조 및 영향 분석 통합)"""
    lines = ["graph LR"]
    safe_doc_id = doc_id.replace("-", "_")
    
    doc = refs.get("document") or {}
    title = (doc.get("title") or doc_id).replace('"', "'")
    
    # 메인 노드 스타일
    lines.append(f'    Main["{doc_id}<br/>({title})"]:::mainNode')
    
    # 1. 문서 간 참조 (References/Cited By)
    for ref in refs.get("references", []):
        # ref가 dict인 경우 doc_id 추출
        ref_doc = ref.get("doc_id") if isinstance(ref, dict) else ref
        if ref_doc:
            ref_id = ref_doc.replace("-", "_")
            lines.append(f'    Main --> {ref_id}["{ref_doc}"]')

    for cited in refs.get("cited_by", []):
        # cited가 dict인 경우 doc_id 추출
        cited_doc = cited.get("doc_id") if isinstance(cited, dict) else cited
        if cited_doc:
            cited_id = cited_doc.replace("-", "_")
            lines.append(f'    {cited_id}["{cited_doc}"] --> Main')

    # 2. 영향 분석 데이터가 있는 경우 추가 (정밀 관계)
    if impact_data:
        for idx, imp in enumerate(impact_data):
            src_id = imp.get("source_doc_id", "Unknown").replace("-", "_")
            section = imp.get("citing_section", "")
            lines.append(f'    Main -- "{section} 조항에서 언급" --> {src_id}')

    lines.append("    classDef mainNode fill:#f96,stroke:#333,stroke-width:4px,color:#000;")
    lines.append("    classDef default fill:#eee,stroke:#333,color:#000;")
    return "\n".join(lines)

@traceable(name="graph_agent", run_type="chain")
def graph_agent_node(state: AgentState):
    """[서브] 그래프 에이전트 (OpenAI) - 인텐트 분석 및 시각화 지원"""
    query = state["query"]
    model = state.get("worker_model") or state.get("model_name") or "gpt-4o"

    # 1. 의도 및 엔티티 추출 (LangChain ChatOpenAI 사용 - LangSmith 자동 추적)
    messages = state.get("messages", [])
    extraction_prompt = f"""Extract the target SOP ID and intent from the user question and conversation history.
Question: {query}

## Intent Classification

| Intent | Condition | Example |
|--------|-----------|---------|
| `impact_analysis` | Explore downstream documents/procedures affected when a document is changed | "If I change this, what gets affected?" |
| `dependency_analysis` | Explore upstream regulations/basis that a document references | "What is the basis regulation for this document?" |
| `relationship_check` | Check the connection between two documents | "What is the relationship between SOP-001 and SOP-004?" |
| `general_info` | Simple lookup of a document's reference list | "Show me the reference list" |

## Output (JSON only, no other text)

{{"doc_id": "EQ-SOP-001", "intent": "impact_analysis", "reason": "One-line justification"}}"""

    try:
        # LangChain ChatOpenAI 사용 (JSON 응답)
        llm = get_langchain_llm(model=model, temperature=0.0)
        llm = llm.bind(response_format={"type": "json_object"})

        # 메시지 형식 변환
        lc_messages = [{"role": "system", "content": "당신은 대화 맥락을 파석하여 엔티티를 추출하는 전문가입니다."}]
        lc_messages.extend(messages)
        lc_messages.append({"role": "user", "content": extraction_prompt})

        extraction_res = llm.invoke(lc_messages)

        # safe_json_loads를 통한 강인한 파싱
        info = safe_json_loads(extraction_res.content)
        doc_id = normalize_doc_id(info.get("doc_id")) # 정규화 적용
        intent = info.get("intent", "general_info")
    except:
        raw_match = re.search(r'([A-Za-z0-9_-]*SOP-\d+)', query.upper())
        doc_id = normalize_doc_id(raw_match.group(1)) if raw_match else None
        intent = "general_info"
    
    if not doc_id:
        return {"messages": [{"role": "assistant", "content": "[그래프 에이전트] 분석할 문서 ID를 찾지 못했습니다. 요청에 문서 번호를 포함해 주세요."}]}

    # 2. 데이터 조회
    refs_str = get_references_tool.invoke({"doc_id": doc_id})
    print(f"[DEBUG graph.py] refs_str: {refs_str[:200] if refs_str else 'None'}...")
    ref_data = safe_json_loads(refs_str) or {"document": {"doc_id": doc_id}, "references_to": [], "referenced_by": []}
    print(f"[DEBUG graph.py] ref_data keys: {ref_data.keys()}")

    # 키 이름 정규화 (references_to -> references, referenced_by -> cited_by)
    if "references_to" in ref_data:
        ref_data["references"] = ref_data["references_to"]
    if "referenced_by" in ref_data:
        ref_data["cited_by"] = ref_data["referenced_by"]

    print(f"[DEBUG graph.py] references: {ref_data.get('references', [])}")
    print(f"[DEBUG graph.py] cited_by: {ref_data.get('cited_by', [])}")

    # Neo4j 연결 (조항 정보 조회를 위해 항상 필요)
    graph_store = None
    impact_list = []

    print(f"[DEBUG graph.py] Neo4j 연결 시도 (intent={intent}, query={query})")
    try:
        graph_store = Neo4jGraphStore()
        graph_store.connect()

        # 영향 분석 (특정 조건에서만)
        if intent == "impact_analysis" or "영향" in query or "참조" in query or "관계" in query:
            print(f"[DEBUG graph.py] 영향 분석 실행")
            impact_list = graph_store.get_impact_analysis(doc_id)
            print(f"[DEBUG graph.py] impact_list 조회 성공: {len(impact_list)}개")

    except Exception as e:
        print(f"[DEBUG graph.py] Neo4j 연결 실패: {e}")
        impact_list = []
        # 실패해도 계속 진행 (조항 정보 없이)

    # 3. 시각화 (Mermaid) 생성
    mermaid_code = generate_mermaid_flow(doc_id, ref_data, impact_list)
    
    # 4. 심층 분석 (LangChain ChatOpenAI 사용)
    analysis_prompt = f"""다음 그래프 데이터를 바탕으로 질문에 대해 간단명료한 분석 보고서를 작성하세요.
    질문: {query}
    의도: {intent}
    데이터: {json.dumps(ref_data, ensure_ascii=False)}
    정밀 영향 분석: {json.dumps(impact_list, ensure_ascii=False)}

    [보고서 작성 규칙]
    - **FACTS ONLY**: 제공된 데이터(참조 관계, 영향 분석)에 없는 문서나 관계를 절대 언급하지 마세요.
    - **NO MARKDOWN EXCLUDING MERMAID**: 오직 머메이드 코드 블록( ```mermaid ) 외에는 마크다운 기호를 쓰지 마세요.
    - **PLAIN TEXT ONLY**: 줄글과 들여쓰기만 사용하세요.
    - 영향 분석 데이터가 없다면 "영향 분석 데이터가 없습니다"라고 명시하세요. 짐작해서 답변하지 마세요.

    [답변 형식]
    {doc_id} 관계 및 영향 분석 보고
    
    1. 참조 관계
    (참조하는/받는 문서 목록)
    
    2. 파급 효과 (영향 분석)
    (구체적인 영향 조항 및 내용 설명)
    
    3. 종합 의견
    (짧은 결론)
    """

    # LangChain ChatOpenAI 사용
    llm = get_langchain_llm(model=model, temperature=0.0)
    analysis_res = llm.invoke([{"role": "user", "content": analysis_prompt}])

    # Mermaid 다이어그램 추가 및 결과 조합
    llm_analysis = analysis_res.content.strip()

    # [USE: ...] 태그 생성 (참고문서에 포함되도록)
    use_tags = []

    # 메인 문서
    use_tags.append(f"[USE: {doc_id} | 문서 관계]")

    # 참조하는 문서들 (상위 문서) - 조항 정보 포함
    for ref_item in ref_data.get("references", []):
        # ref_item이 dict인 경우 doc_id 추출
        ref_doc = ref_item.get("doc_id") if isinstance(ref_item, dict) else ref_item
        if ref_doc:
            # 어느 조항에서 참조하는지 Neo4j에서 조회
            try:
                if graph_store and graph_store.driver:
                    with graph_store.driver.session(database=graph_store.database) as session:
                        result = session.run("""
                            MATCH (source:Document {doc_id: $source_id})-[:HAS_SECTION]->(section:Section)-[:MENTIONS]->(target:Document {doc_id: $target_id})
                            RETURN section.section_id as section_id
                            LIMIT 5
                        """, source_id=doc_id, target_id=ref_doc)

                        sections = [record["section_id"].split(":")[-1] if ":" in record["section_id"] else record["section_id"]
                                   for record in result]

                        if sections:
                            for section in sections:
                                use_tags.append(f"[USE: {doc_id} | {section}]")
                                print(f"[DEBUG graph.py] 상위 참조 태그 추가: {doc_id} > {section} (참조대상: {ref_doc})")
                        else:
                            use_tags.append(f"[USE: {ref_doc} | 상위 참조]")
                            print(f"[DEBUG graph.py] 상위 참조 태그 추가 (조항 정보 없음): {ref_doc}")
                else:
                    use_tags.append(f"[USE: {ref_doc} | 상위 참조]")
            except Exception as e:
                print(f"[DEBUG graph.py] 조항 정보 조회 실패: {e}")
                use_tags.append(f"[USE: {ref_doc} | 상위 참조]")

    # 참조받는 문서들 (하위 문서) - 조항 정보 포함
    for cited_item in ref_data.get("cited_by", []):
        # cited_item이 dict인 경우 doc_id 추출
        cited_doc = cited_item.get("doc_id") if isinstance(cited_item, dict) else cited_item
        if cited_doc:
            # 어느 조항에서 참조하는지 Neo4j에서 조회
            try:
                if graph_store and graph_store.driver:
                    with graph_store.driver.session(database=graph_store.database) as session:
                        result = session.run("""
                            MATCH (source:Document {doc_id: $source_id})-[:HAS_SECTION]->(section:Section)-[:MENTIONS]->(target:Document {doc_id: $target_id})
                            RETURN section.section_id as section_id
                            LIMIT 5
                        """, source_id=cited_doc, target_id=doc_id)

                        sections = [record["section_id"].split(":")[-1] if ":" in record["section_id"] else record["section_id"]
                                   for record in result]

                        if sections:
                            for section in sections:
                                use_tags.append(f"[USE: {cited_doc} | {section}]")
                                print(f"[DEBUG graph.py] 하위 참조 태그 추가: {cited_doc} > {section}")
                        else:
                            use_tags.append(f"[USE: {cited_doc} | 하위 참조]")
                            print(f"[DEBUG graph.py] 하위 참조 태그 추가 (조항 정보 없음): {cited_doc}")
                else:
                    use_tags.append(f"[USE: {cited_doc} | 하위 참조]")
            except Exception as e:
                print(f"[DEBUG graph.py] 조항 정보 조회 실패: {e}")
                use_tags.append(f"[USE: {cited_doc} | 하위 참조]")

    # 영향 분석 데이터 (조항 정보 포함)
    if impact_list:
        print(f"[DEBUG graph.py] impact_list 개수: {len(impact_list)}")
        for imp in impact_list:
            src_doc = imp.get("source_doc_id")
            section = imp.get("citing_section", "")
            print(f"[DEBUG graph.py] 영향 문서: {src_doc}, 조항: {section}")
            if src_doc:
                if section:
                    use_tags.append(f"[USE: {src_doc} | {section}]")
                else:
                    use_tags.append(f"[USE: {src_doc} | 영향 조항]")

    # USE 태그를 보고서에 추가 (hidden)
    use_tags_str = " ".join(use_tags)
    final_report = f"""### [그래프 에이전트 관계 분석 보고]

{llm_analysis}

#### 관계 시각화 (Mermaid)
```mermaid
{mermaid_code}
```

{use_tags_str}
[DONE]"""

    # Neo4j 연결 종료
    if graph_store:
        try:
            graph_store.close()
        except:
            pass

    return {"context": [final_report]}
