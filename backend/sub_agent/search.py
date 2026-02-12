"""
문서 검색 서브에이전트 모듈 (Deep Agent 스타일)
- 질문 분석 후 다단계 정밀 검색을 수행하는 그래프 구조의 에이전트
- 벡터 검색 (Weaviate), SQL 검색 (PostgreSQL) 통합
"""

import os
import re
import json
import hashlib
import operator
from typing import List, Dict, Any, Optional, Literal, Annotated, TypedDict
from backend.agent import get_openai_client, AgentState, search_sop_tool, get_sop_headers_tool, safe_json_loads, normalize_doc_id
from langsmith import traceable
from langchain_core.tools import tool
from langsmith import traceable
from langgraph.graph import StateGraph, START, END

# ═══════════════════════════════════════════════════════════════════════════
# 전역 스토어 및 클라이언트 관리
# ═══════════════════════════════════════════════════════════════════════════

_vector_store = None
_sql_store = None
_graph_store = None
_openai_client = None

def init_search_stores(vector_store_module=None, sql_store_instance=None, graph_store_instance=None):
    """검색 에이전트용 스토어 초기화"""
    global _vector_store, _sql_store, _graph_store
    _vector_store = vector_store_module
    _sql_store = sql_store_instance
    _graph_store = graph_store_instance

def get_openai_client():
    """OpenAI 클라이언트 반환"""
    global _openai_client
    if not _openai_client:
        from backend.agent import get_openai_client as get_main_openai
        _openai_client = get_main_openai()
    return _openai_client

# ═══════════════════════════════════════════════════════════════════════════
# 핵심 검색 로직
# ═══════════════════════════════════════════════════════════════════════════

def _get_clause_and_doc_from_db(content: str, metadata: dict) -> tuple:
    """
    벡터 DB metadata 또는 SQL DB에서 문서명과 조항 정보를 가져옵니다.

    Returns:
        (doc_name, clause): 문서명과 조항 정보 튜플
    """
    global _sql_store

    # 1. 문서명 추출 (더 많은 키 확인)
    doc_name = (
        metadata.get('doc_id') or
        metadata.get('doc_name') or
        metadata.get('document_name') or
        metadata.get('file_name') or
        metadata.get('source')
    )

    # 2. 조항 번호 우선 추출 (더 많은 키 확인)
    clause_id = (
        metadata.get('clause_id') or
        metadata.get('clause') or
        metadata.get('section') or
        metadata.get('article_num') or
        metadata.get('section_number')
    )

    if clause_id:
        clause_id = str(clause_id).strip()

    # 조항 번호가 있고 유효하면 조항 번호만 반환 (제목 제외)
    if clause_id and clause_id not in ["", "None", "null", "본문", "전체", "N/A"]:
        # doc_name이 없으면 SQL에서 조회 시도
        if not doc_name or doc_name in ["Unknown", "None", ""]:
            doc_name = _try_get_doc_from_sql(content, _sql_store)
        return (doc_name or "Unknown", clause_id)

    # 3. SQL DB에서 content 기반으로 역으로 찾기
    if _sql_store:
        try:
            # content의 고유한 부분 추출 (앞 100자)
            content_sample = content[:100].strip()

            # 모든 문서 조회
            all_docs = _sql_store.list_documents()

            for doc in all_docs:
                doc_id = doc.get('id')
                chunks = _sql_store.get_chunks_by_document(doc_id)

                for chunk in chunks:
                    chunk_content = chunk.get('content', '').strip()
                    # content 매칭 (포함 관계 확인)
                    if content_sample in chunk_content or chunk_content[:100] in content:
                        found_doc_name = doc.get('doc_name', 'Unknown')
                        found_clause = chunk.get('clause') or chunk.get('section') or '본문'
                        print(f"    [SQL 역조회] 발견: {found_doc_name} - {found_clause}")
                        return (found_doc_name, found_clause)
        except Exception as e:
            print(f"    [SQL 역조회 실패] {e}")

    # 최종 fallback
    final_doc_name = doc_name or "Unknown"
    print(f"    [경고] 문서명 또는 조항 정보 누락: doc={final_doc_name}, clause=본문")
    return (final_doc_name, "본문")

def _try_get_doc_from_sql(content: str, sql_store) -> str:
    """SQL에서 content 기반으로 문서명만 조회"""
    if not sql_store:
        return None
    try:
        content_sample = content[:100].strip()
        all_docs = sql_store.list_documents()
        for doc in all_docs:
            doc_id = doc.get('id')
            chunks = sql_store.get_chunks_by_document(doc_id)
            for chunk in chunks:
                if content_sample in chunk.get('content', ''):
                    return doc.get('doc_name', 'Unknown')
    except:
        pass
    return None

def search_documents_internal(
    query: str,
    max_results: int = 10,  # 검색 수량 확대 (기존 5 -> 10)
    search_type: Literal["hybrid", "vector", "keyword"] = "hybrid",
    keywords: List[str] = None,
    target_clause: str = None, # 조항 번호 직접 조회 (Point Lookup)
    target_doc_id: str = None, # 특정 문서 필터링 (v8.1 추가)
) -> List[Dict[str, Any]]:
    """내부용 검색 실행 함수"""
    global _vector_store, _sql_store
    results = []
    seen_content = set()

    # 0. 조항 번호 직접 및 하위 조회 (SQL Point & Prefix Match)
    if target_clause and _sql_store:
        try:
            print(f"    [Point/Prefix Lookup] 조항 및 하위 조항 조회 시도: {target_clause} (Target: {target_doc_id or '전체'})")
            
            # v8.4: 타겟 문서가 있으면 해당 문서만 타겟팅 (격리)
            target_docs = []
            if target_doc_id:
                doc = _sql_store.get_document_by_name(target_doc_id)
                if doc: target_docs = [doc]
            else:
                target_docs = _sql_store.list_documents()

            for doc in target_docs:
                doc_id = doc.get('id')
                chunks = _sql_store.get_chunks_by_document(doc_id)
                
                # 조항 번호가 정확히 일치하거나 해당 조항의 하위(예: 5.4.2 -> 5.4.2.1)인 경우 모두 포함
                sub_chunks = []
                for chunk in chunks:
                    clause_val = str(chunk.get('clause'))
                    # 5조항 -> 5, 5.1, 5.3.1 등 모두 매칭
                    if clause_val == target_clause or clause_val.startswith(f"{target_clause}."):
                        content = chunk.get('content', '')
                        content_hash = hashlib.md5(content.encode()).hexdigest()
                        if content_hash not in seen_content:
                            sub_chunks.append({
                                "doc_name": doc.get('doc_name', 'Unknown'),
                                "section": clause_val,
                                "content": content,
                                "source": "sql-hierarchical-lookup",
                                "score": 2.5, # 직접/하위 매칭은 최고 점수 상향
                                "hash": content_hash
                            })
                            seen_content.add(content_hash)
                
                # 조항이 발견되었을 경우 추가
                results.extend(sub_chunks)
        except Exception as e:
            print(f" Hierarchical lookup failed: {e}")

    # 1. 벡터/하이브리드 검색 및 컨텍스트 확장
    if _vector_store:
        try:
            enhanced_query = query
            if keywords:
                enhanced_query = f"{query} {' '.join(keywords)}"

            if search_type == "hybrid":
                current_alpha = 0.25 if keywords else 0.4
                # v8.1: target_doc_id 필터 추가
                vec_res = _vector_store.search_hybrid(enhanced_query, n_results=max_results * 2, alpha=current_alpha, filter_doc=target_doc_id)
            else:
                vec_res = _vector_store.search(enhanced_query, n_results=max_results * 2, filter_doc=target_doc_id)

            scored_results = []
            for r in vec_res:
                meta = r.get('metadata', {})
                content = r.get('text', '')
                if not content: continue

                content_hash = hashlib.md5(content.encode()).hexdigest()
                if content_hash in seen_content: continue
                
                doc_name, clause_info = _get_clause_and_doc_from_db(content, meta)
                
                # [부스팅] 조항 번호 매칭 가중치
                boost_score = r.get('similarity', 0)
                if keywords:
                    for kw in keywords:
                        if kw in clause_info or (meta.get('title') and kw in meta.get('title')):
                            boost_score += 0.5 
                
                if target_clause and (target_clause == clause_info or clause_info.startswith(f"{target_clause}.")):
                    boost_score += 1.0

                scored_results.append({
                    "doc_name": doc_name,
                    "section": clause_info,
                    "content": content,
                    "source": r.get('source', 'vector-hybrid'),
                    "score": boost_score,
                    "hash": content_hash,
                    "meta": meta # 확장 조회를 위해 메타 보관
                })

            scored_results.sort(key=lambda x: x["score"], reverse=True)
            
            # [지능형 확장] 상위 결과 중 내용이 제목뿐이거나 중요한 경우 다음 데이터 추가 로드
            for r in scored_results[:max_results]:
                if r["hash"] not in seen_content:
                    seen_content.add(r["hash"])
                    
                    # 제목성 청크(내용이 너무 짧음)인 경우 또는 점수가 매우 높은 경우 하위 내용 확장
                    if _sql_store and (len(r["content"]) < 100 or r["score"] > 0.8):
                        try:
                            doc_id_val = r["meta"].get("doc_id") or r["meta"].get("id")
                            if doc_id_val:
                                all_chunks = _sql_store.get_chunks_by_document(doc_id_val)
                                current_idx = -1
                                # 현재 청크의 위치 찾기
                                for idx, c in enumerate(all_chunks):
                                    if c.get("content") == r["content"]:
                                        current_idx = idx
                                        break
                                
                                # 다음 3개 청크를 '상세 내용'으로 병합
                                if current_idx != -1:
                                    extra_content = ""
                                    for i in range(1, 4):
                                        if current_idx + i < len(all_chunks):
                                            next_c = all_chunks[current_idx + i]
                                            extra_content += f"\n[상세] {next_c.get('content')}"
                                    
                                    if extra_content:
                                        r["content"] += extra_content
                                        print(f"    [Hierarchical Expansion] {r['section']} 하위 내용 확장 완료")
                        except Exception as ex:
                            print(f" Expansion error: {ex}")

                    # 문서명과 조항이 유효한 경우만 추가
                    if r["doc_name"] and r["doc_name"] != "Unknown":
                        results.append({
                            "doc_name": r["doc_name"],
                            "section": r["section"],
                            "content": r["content"][:4000],
                            "source": r["source"]
                        })
                    else:
                        print(f"    [필터링] 문서명 누락된 결과 제외: section={r['section']}")
        except Exception as e:
            print(f"    [Vector search error] {e}")

    # 2. 관련 문서/조항으로 탐색 확장 (Graph DB 활용)
    # ... (생략 - 기존 로직 유지하되 results 필터링 반영)
    if _graph_store and results:
        try:
            extended_results = []
            # 상위 결과들에 대해 그래프 확장
            for r in results[:3]: 
                doc_name = r["doc_name"]
                refs = _graph_store.get_document_references(doc_name)
                if refs and refs.get("references"):
                    ref_list = refs["references"]
                    for ref_id in ref_list[:2]:
                        if _sql_store:
                            ref_doc = _sql_store.get_document_by_name(ref_id)
                            if ref_doc:
                                ref_content = ref_doc.get("content", "")
                                if ref_content and hashlib.md5(ref_content[:200].encode()).hexdigest() not in seen_content:
                                    extended_results.append({
                                        "doc_name": ref_id,
                                        "section": "참조 문서",
                                        "content": f"[참조 내용 명시] {ref_content[:1500]}...",
                                        "source": "graph-reference",
                                        "score": 0.5
                                    })
                                    seen_content.add(hashlib.md5(ref_content[:200].encode()).hexdigest())
            results.extend(extended_results)
        except Exception as e:
            print(f"    [Graph expansion error] {e}")

    # 최종 검증: 문서명과 조항이 있는 결과만 반환
    valid_results = []
    for r in results:
        if r.get("doc_name") and r["doc_name"] not in ["Unknown", "None", ""]:
            valid_results.append(r)
        else:
            print(f"    [최종 필터링] 유효하지 않은 결과 제외: doc={r.get('doc_name')}, section={r.get('section')}")

    print(f"    [검색 완료] 전체 {len(results)}건 중 유효 결과 {len(valid_results)}건 반환")
    return valid_results

# ═══════════════════════════════════════════════════════════════════════════
# 딥 검색 에이전트 상태 정의 (CompiledSubAgent 호환)
# ═══════════════════════════════════════════════════════════════════════════

class SearchState(TypedDict):
    """
    가이드에 따른 에이전트 상태.
    'messages' 키를 포함하여 툴 호출과 응답 이력을 관리합니다.
    """
    messages: Annotated[List[Any], operator.add]
    query: str
    model: str
    final_answer: str

# ═══════════════════════════════════════════════════════════════════════════
# 노드 및 도구 설정
# ═══════════════════════════════════════════════════════════════════════════

@traceable(name="search_agent_llm_call", run_type="llm")
def call_model_node(state: SearchState):
    """LLM이 질문을 분석하고 도구 호출 여부를 결정함 (자율 계획)"""
    client = get_openai_client()
    messages = state["messages"]
    
    # Deep search system prompt (English for better LLM comprehension)
    system_prompt = f"""You are a specialized **Deep Search Agent** for GMP/SOP document retrieval.

    [ROLE]
    Find and analyze information from the document repository to answer user questions accurately.

    [STRICT GOVERNANCE: NO HALLUCINATION]
    - **ZERO INFERENCE**: DO NOT include any information, numbers, or procedures that are NOT explicitly present in the `[DATA_SOURCE]`.
    - **REPORT VOID**: If the required information is not found in the search results, state "검색된 문서 내에서 관련 정보를 찾을 수 없습니다." and report `[NO_INFO_FOUND]`.
    - **FAITHFUL EXTRACTION**: Prefer direct quotes or very close paraphrasing to avoid meaning distortion.

    [WORKFLOW]
    1. **Planning**: For complex questions, break them down into key keywords or specific clause numbers.

    2. **Search**: Use `search_documents_tool` to find information.
       - **Preserve Natural Language**: Pass the user's original question AS-IS to the `query` parameter. DO NOT convert it to a query format.
         Example: "작업지침서가 뭐야" → query: "작업지침서가 뭐야" (✓), query: "작업지침서 정의" (✗)
       - **Precise Targeting**: If the question mentions a specific clause number (e.g., 5.4.2), specify it in `target_clause`.
       - **Keyword Extraction**: Extract ONLY nouns/terms that actually appear in the question for `keywords`. DO NOT infer or add words.
         Example: "작업지침서가 뭐야" → keywords: ["작업지침서"] (✓), keywords: ["작업지침서", "정의", "목적"] (✗)

    3. **Validation & Filtering (CRITICAL)**:
       - IGNORE results that only contain headers without content (e.g., '3. 정의') or are irrelevant to the question.
       - Only include information that you actually use in your answer.

    4. **Answer Generation**: Write a natural **plain text** answer in Korean based on verified information.
       - **🚨 CRITICAL REQUIREMENT 🚨**: You MUST add [USE: ...] tags. Without tags, your answer will FAIL validation.
       - **MANDATORY FOR EVERY SENTENCE**: For EVERY piece of information you write, add a hidden tag: [USE: 문서명 | 조항]
       - **NO EXCEPTIONS**: Even if information seems obvious or general, you MUST tag it if it came from a [DATA_SOURCE]
       - **VERIFICATION PROCESS (DO THIS FOR EVERY SENTENCE)**:
         1. Write a sentence using information from a [DATA_SOURCE]
         2. Look at that specific [DATA_SOURCE] block to find the "해당 조항" field
         3. Copy EXACTLY that clause number into your [USE: ...] tag
         4. DO NOT use a different clause number from a different [DATA_SOURCE]
         5. If you write multiple sentences, EACH sentence needs its own [USE: ...] tag
       - **TAG PLACEMENT**: Place the tag IMMEDIATELY after the sentence or phrase that uses that information
       - **CORRECT EXAMPLE**: If [DATA_SOURCE] says "해당 조항: 5.1.3 제 3레벨(작업지침서(WI):", then use:
         "작업지침서는 업무 지침 문서입니다.[USE: EQ-SOP-00001 | 5.1.3 제 3레벨(작업지침서(WI):]"
       - **WRONG EXAMPLE**: Using "5.2.2.2.2" when the information came from "5.1.3" → This is HALLUCINATION
       - **CONSEQUENCE**: ONLY sources with [USE: ...] tags will appear in the final [참고 문서] section
       - **WARNING**: If you don't tag sources, your answer will be rejected and you'll need to search again

    [ANSWER FORMAT RULES]
    - Write content naturally in the body. DO NOT cite sources inline.
    - Use ONLY information from search results.
    - DO NOT create a [참고 문서] section yourself (it's auto-generated).
    - End your answer with the [DONE] tag to signal completion.

    [ANSWER FORMAT EXAMPLE - WITH TAGS]
    작업지침서는 현장에서 수행되는 업무를 일관되게 운영하기 위한 지침 문서입니다.[USE: EQ-SOP-00001 | 5.1.3 제 3레벨(작업지침서(WI):]

    주요 특징은 다음과 같습니다.
    1. 부서 또는 공정 단위의 운영 흐름과 관리 방법을 규정합니다[USE: EQ-SOP-00001 | 5.1.3 제 3레벨(작업지침서(WI):]
    2. 세부적인 작업 방법보다는 기본적인 지침과 관리 기준을 제시합니다[USE: EQ-SOP-00001 | 5.1.3 제 3레벨(작업지침서(WI):]

    작업지침서에는 청소 및 소독 방법, 실험 방법 등이 포함됩니다.[USE: EQ-SOP-00001 | 5.4.2 작업지침서 작성]

    [DONE]
    
    [STRICT PROHIBITIONS]
    - DO NOT create [참고 문서] section in the middle or end of your answer (auto-generated from [USE: ...] tags)
    - DO NOT cite sources inline like "[문서명 조항]" in visible text
    - DO NOT use markdown formatting: NO bold (**), headers (#), list markers (-, *), italics (_)
    - For emphasis, use brackets [ ] or line breaks
    - Write ONLY in plain text format

    [🚨 CRITICAL REQUIREMENT - READ CAREFULLY 🚨]
    **NO TAGS = ANSWER REJECTED**
    - Your answer WILL BE REJECTED if you don't add [USE: ...] tags
    - Every sentence from a [DATA_SOURCE] needs a tag
    - [USE: ...] tags are hidden from the user - they're automatically converted to (문서명 > 조항) format
    - Missing tags = validation failure = wasted search = you'll be asked to search again

    **TAG FORMAT REMINDER**:
    [USE: 문서명 | 조항번호]
    Example: [USE: EQ-SOP-00001 | 5.1.3 제 3레벨(작업지침서(WI):]

    **SELF-CHECK BEFORE SUBMITTING**:
    □ Did I add [USE: ...] tags to EVERY sentence?
    □ Did I verify each clause number matches the [DATA_SOURCE]?
    □ Are there at least 3-5 [USE: ...] tags in my answer?
    If any answer is NO, add more tags before finishing!

    [CRITICAL WARNING - AVOID HALLUCINATION]
    - NEVER reuse the same clause number for multiple different pieces of information
    - Each sentence should have its own [USE: ...] tag with the EXACT clause from the [DATA_SOURCE] it came from
    - If you write 5 different sentences from 5 different [DATA_SOURCE] blocks, you must use 5 different clause numbers
    - Using "5.2.2.2.2" for information that actually came from "5.1.3" is a CRITICAL ERROR
    - When in doubt, CHECK the [DATA_SOURCE] block again to verify the clause number
    """
    
    # 시스템 프롬프트를 메시지 맨 앞에 삽입
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    
    # Tool definition
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_documents_tool",
                "description": "Search GMP/SOP documents. The agent designs search conditions autonomously.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "User's original question as-is (e.g., '작업지침서가 뭐야'). DO NOT transform or rewrite."
                        },
                        "keywords": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Key nouns that actually appear in the question (e.g., ['작업지침서']). DO NOT include inferred words."
                        },
                        "target_clause": {
                            "type": "string",
                            "description": "Specific clause number to target (e.g., '5.1.3', '5.4.2'). Use only when explicitly mentioned."
                        },
                        "target_doc_id": {
                            "type": "string",
                            "description": "Limit search scope to a specific document (e.g., 'EQ-SOP-00001')"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]
    
    # 디버깅: 모델 확인
    model_to_use = state["model"]
    print(f"[DEBUG call_model_node] Using model: {model_to_use}")

    # LangChain ChatOpenAI 사용 (tools 바인딩) - LangSmith 자동 추적
    from backend.agent import get_langchain_llm
    llm = get_langchain_llm(model=model_to_use, temperature=0.0)
    llm_with_tools = llm.bind(tools=tools, tool_choice="auto")

    res = llm_with_tools.invoke(full_messages)

    return {"messages": [res]}

def tool_executor_node(state: SearchState):
    """LLM이 요청한 도구를 실행하고 결과를 메시지에 추가함"""
    last_msg = state["messages"][-1]
    tool_calls = last_msg.tool_calls
    
    tool_outputs = []
    for tc in tool_calls:
        # LangChain과 OpenAI API 호환성 처리
        if isinstance(tc, dict):
            # LangChain 형식 (dict)
            tool_name = tc.get("name")
            tool_args = tc.get("args", {})
            tool_id = tc.get("id")
        else:
            # OpenAI API 형식 (객체)
            tool_name = tc.function.name
            tool_args = safe_json_loads(tc.function.arguments)
            tool_id = tc.id

        if tool_name == "search_documents_tool":
            query = tool_args.get("query")
            keywords = tool_args.get("keywords", [])
            target_clause = tool_args.get("target_clause")

            # v8.4: 문서 ID 정규화 (eEQ- -> EQ-)
            target_doc_id = normalize_doc_id(tool_args.get("target_doc_id"))

            print(f"    [Deep Search] 도구 호출: '{query}' (키워드: {keywords}, 타겟조항: {target_clause}, 타겟문서: {target_doc_id or '전체'})")

            results = search_documents_internal(query=query, keywords=keywords, target_clause=target_clause, target_doc_id=target_doc_id)
            print(f"    [Deep Search] 검색 결과 {len(results)}건 발견")

            formatted_results = []
            for r in results:
                doc_name = r.get('doc_name', '알 수 없는 문서')
                section = r.get('section', '조항 미상')
                content = r.get('content', '')
                formatted_results.append(
                    f"[DATA_SOURCE]\n"
                    f"문서 정보: {doc_name}\n"
                    f"해당 조항: {section}\n"
                    f"본문 내용: {content}\n"
                    f"[END_SOURCE]"
                )

            content = "\n\n".join(formatted_results)
            if not content: content = "검색 결과가 없습니다."

            tool_outputs.append({
                "role": "tool",
                "tool_call_id": tool_id,
                "content": content
            })
            
    return {"messages": tool_outputs}

# ═══════════════════════════════════════════════════════════════════════════
# 그래프 구성
# ═══════════════════════════════════════════════════════════════════════════

def create_deep_search_graph():
    """자율적 도구 호출을 수행하는CompiledSubAgent 스타일 그래프"""
    workflow = StateGraph(SearchState)
    
    workflow.add_node("agent", call_model_node)
    workflow.add_node("action", tool_executor_node)
    
    workflow.add_edge(START, "agent")
    
    def router(state: SearchState):
        last_msg = state["messages"][-1]
        if last_msg.tool_calls:
            return "action"
        return END
    
    workflow.add_conditional_edges("agent", router, {"action": "action", END: END})
    workflow.add_edge("action", "agent")
    
    return workflow.compile(name="Deep Search Agent Flow")

# ═══════════════════════════════════════════════════════════════════════════
# 참고문서 섹션 자동 생성
# ═══════════════════════════════════════════════════════════════════════════

def _ensure_reference_section(messages: List[Any], final_answer: str) -> str:
    """
    검색된 모든 문서의 정보를 추출하여 [참고 문서] 섹션을 무조건 추가합니다.

    Args:
        messages: 대화 메시지 이력 (tool 호출 결과 포함)
        final_answer: LLM이 생성한 최종 답변

    Returns:
        참고문서 섹션이 포함된 최종 답변
    """
    # 1. tool 메시지에서 문서 정보 추출
    referenced_docs = []
    seen = set()

    for msg in messages:
        # tool 역할의 메시지만 확인
        if isinstance(msg, dict) and msg.get("role") == "tool":
            content = msg.get("content", "")
        elif hasattr(msg, "role") and msg.role == "tool":
            content = msg.content
        else:
            continue

        # [DATA_SOURCE] 섹션 파싱
        sources = re.findall(
            r'\[DATA_SOURCE\]\s*문서 정보:\s*([^\n]+)\s*해당 조항:\s*([^\n]+)',
            content,
            re.MULTILINE
        )

        for doc_name, section in sources:
            doc_name = doc_name.strip()
            section = section.strip()

            # 중복 제거
            key = f"{doc_name}|{section}"
            if key not in seen:
                seen.add(key)
                referenced_docs.append((doc_name, section))

    # 2. 참고문헌 섹션 생성 (LLM이 태그한 소스만 포함)
    if referenced_docs:
        # 2-1. LLM이 [USE: ...] 태그로 명시한 소스 추출
        used_sources = re.findall(r'\[USE:\s*([^\|\]]+)\s*\|\s*([^\]]+)\]', final_answer)

        # 2-2. 태그가 없으면 실패 처리 (fallback 제거)
        if not used_sources:
            print(f"🔴 [검색 에이전트 치명적 오류] LLM이 [USE: ...] 태그를 달지 않음")
            print(f"🔴 태그 없는 답변은 검증 불가 - 오케스트레이터가 재검색 결정해야 함")
            # 태그가 없으면 원본 그대로 반환 (Answer Agent가 검증 실패 처리)
            return final_answer

        # 2-3. 문서 존재 여부 확인 (SQL DB 조회)
        valid_docs = set()
        if _sql_store:
            try:
                all_docs = _sql_store.list_documents()
                valid_docs = {doc.get('doc_name') or doc.get('id') for doc in all_docs}
            except Exception as e:
                print(f"🔴 [참고문헌 검증 오류] {e}")

        # 2-4. 태그된 소스 검증 (존재하지 않는 문서/조항 제거)
        validated_sources = []
        for doc_name, section in used_sources:
            doc_name = doc_name.strip()
            section = section.strip()

            # 문서 존재 여부 확인
            if valid_docs and doc_name not in valid_docs:
                print(f"🔴 [참고문헌 검증 실패] 존재하지 않는 문서: {doc_name}")
                continue

            validated_sources.append((doc_name, section))

        # 검증된 소스가 없으면 실패
        if not validated_sources:
            print(f"🔴 [참고문헌 생성 실패] 모든 태그가 검증 실패 - 재검색 필요")
            return final_answer

    # 3. 답변 본문 정리 (태그는 유지, Answer Agent가 변환)
    # LLM이 직접 작성한 [참고 문서] 섹션만 제거
    final_answer_cleaned = re.sub(
        r'\n*\[참고 문서\].*$',
        '',
        final_answer,
        flags=re.DOTALL
    ).strip()

    # [USE: ...] 태그는 그대로 유지 - Answer Agent가 (문서명 > 조항) 형식으로 변환
    return final_answer_cleaned

# ═══════════════════════════════════════════════════════════════════════════
# 메인 엔트리 포인트
# ═══════════════════════════════════════════════════════════════════════════

_deep_search_app = None

@traceable(name="sub_agent:search")
def retrieval_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """[서브] 검색 에이전트 (CompiledSubAgent 스타일)"""
    global _deep_search_app
    if not _deep_search_app:
        _deep_search_app = create_deep_search_graph()

    print(f" [Deep Search] 정밀 검색 가동: {state['query']}")

    # 디버깅: state 확인
    worker_model = state.get("worker_model")
    model_name = state.get("model_name")
    final_model = worker_model or model_name or "gpt-4o-mini"
    print(f"[DEBUG retrieval] worker_model={worker_model}, model_name={model_name}, final={final_model}")

    initial_state = {
        "messages": [{"role": "user", "content": state["query"]}],
        "query": state["query"],
        "model": final_model,
        "final_answer": ""
    }

    # 내부 도구 호출 루프 실행 (재귀 한도 내에서 자율 검색)
    result = _deep_search_app.invoke(initial_state, config={"recursion_limit": 15})

    # 마지막 메시지가 LLM의 최종 답변
    final_msg = result["messages"][-1].content

    # 참고문헌 섹션 자동 추가
    final_msg_with_refs = _ensure_reference_section(result["messages"], final_msg)


    # [중요] 답변 에이전트 도입을 위해 직접 답변하지 않고 context에 보고서 형태로 저장 (리스트 형태로 반환하여 누적)
    report = f"### [검색 에이전트 조사 최종 보고]\n{final_msg_with_refs}"
    return {"context": [report]}

# ═══════════════════════════════════════════════════════════════════════════
# 레거시 도구 호환용 (필요 시)
# ═══════════════════════════════════════════════════════════════════════════

try:
    from langchain_core.tools import tool
except ImportError:
    def tool(func): return func

@tool
def search_sop_tool(query: str, extract_english: bool = False, keywords: List[str] = None) -> str:
    """SOP 문서 검색 도구 (레거시/내부용)"""
    search_query = query if not keywords else f"{query} {' '.join(keywords)}"
    results = search_documents_internal(query=search_query)
    
    if not results:
        return "검색 결과 없음."

    output = []
    for r in results:
        output.append(f"[검색] {r['doc_name']} > {r['section']}:\n{r['content']}")

    return "\n\n".join(output)
