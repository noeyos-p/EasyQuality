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
    max_results: int = 100,  # 에이전트 분석용 벡터 검색 수량 확대
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
                vec_res = _vector_store.search_hybrid(
                    enhanced_query,
                    n_results=max_results,
                    alpha=current_alpha,
                    filter_doc=target_doc_id
                )
            else:
                vec_res = _vector_store.search(
                    enhanced_query,
                    n_results=max_results,
                    filter_doc=target_doc_id
                )

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
                            # doc_id 메타는 문자열(예: "EQ-SOP-00001")이므로
                            # SQL의 numeric document_id로 변환 필요
                            doc_name_val = (
                                r["meta"].get("doc_id") or
                                r["meta"].get("doc_name") or
                                r.get("doc_name")
                            )
                            if doc_name_val:
                                doc_record = _sql_store.get_document_by_name(doc_name_val)
                                if doc_record:
                                    numeric_doc_id = doc_record['id']
                                    all_chunks = _sql_store.get_chunks_by_document(numeric_doc_id)

                                    current_section = r["section"]
                                    extra_content = ""
                                    expanded_clauses = []  # 확장에 사용된 하위 조항 번호 추적

                                    # 1차: 조항 번호 기반 하위 확장 (예: "1" → "1.1", "1.2" 등)
                                    for c in all_chunks:
                                        child_clause = str(c.get('clause', ''))
                                        child_content = c.get('content', '')
                                        if not child_clause or not child_content:
                                            continue
                                        # 현재 조항의 직접 하위 조항만 포함
                                        if child_clause.startswith(f"{current_section}.") and child_clause != current_section:
                                            content_hash = hashlib.md5(child_content.encode()).hexdigest()
                                            if content_hash not in seen_content:
                                                extra_content += f"\n[하위 조항 {child_clause}] {child_content}"
                                                expanded_clauses.append(child_clause)
                                                if len(extra_content) > 3000:
                                                    break

                                    # 2차: 조항 기반 확장 실패 시 위치 기반 폴백
                                    if not extra_content:
                                        current_idx = -1
                                        for idx, c in enumerate(all_chunks):
                                            if c.get("content") == r["content"]:
                                                current_idx = idx
                                                break
                                        if current_idx != -1:
                                            for i in range(1, 4):
                                                if current_idx + i < len(all_chunks):
                                                    next_c = all_chunks[current_idx + i]
                                                    next_clause = str(next_c.get('clause', ''))
                                                    extra_content += f"\n[하위 조항 {next_clause}] {next_c.get('content')}"
                                                    if next_clause:
                                                        expanded_clauses.append(next_clause)

                                    if extra_content:
                                        r["content"] += extra_content
                                        # section 필드에 확장된 하위 조항 번호도 포함
                                        if expanded_clauses:
                                            r["section"] = f"{current_section}, {', '.join(expanded_clauses)}"
                                        print(f"    [Hierarchical Expansion] {current_section} → {', '.join(expanded_clauses)} 확장 완료 (doc_id: {numeric_doc_id})")
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
    detected_doc_id: Optional[str] # v8.5 추가

# ═══════════════════════════════════════════════════════════════════════════
# 노드 및 도구 설정
# ═══════════════════════════════════════════════════════════════════════════

@traceable(name="search_agent_llm_call", run_type="llm")
def call_model_node(state: SearchState):
    """LLM이 질문을 분석하고 도구 호출 여부를 결정함 (자율 계획)"""
    client = get_openai_client()
    messages = state["messages"]
    
    system_prompt = f"""You are a specialized agent for GMP/SOP document retrieval.
Use the search tool to accurately answer user questions.
{f"**Priority search target**: {state.get('detected_doc_id')}" if state.get('detected_doc_id') else ""}

## Principles

- **No information without a source**: Never write content that is not in [DATA_SOURCE].
- **Specify document name**: Always use the document ID (e.g., EQ-SOP-00001) as the subject instead of pronouns like "this document" or "this regulation."
- **On search failure**: Return "No relevant information found within the searched documents." + [NO_INFO_FOUND].

## Search Method (search_documents_tool)

- `query`: Pass the user question **as-is in its original form**. Do not convert it into a query format.
  - O "What is a work instruction?" -> query: "What is a work instruction?"
  - X query: "work instruction definition"
- `target_clause`: Specify if a particular clause number is mentioned (e.g., "5.4.2").
- `keywords`: Extract only **nouns that actually appear** in the question. Do not infer or add additional terms.
  - O "What is a work instruction?" -> ["work instruction"]
  - X ["work instruction", "definition", "purpose"]

## Answer Writing Rules

**Format**: Korean plain text. No markdown (**, #, -, *). Use [ ] or line breaks for emphasis.

**Source tagging ([USE: ...] tags)**:
- Every sentence sourced from [DATA_SOURCE] must end with a `[USE: document name | clause]` tag.
- The clause number must be an **exact copy** of the "applicable clause" field from the corresponding [DATA_SOURCE].
- Information from different [DATA_SOURCE] entries must use their respective clause numbers.
- Answers without tags will be treated as verification failures.
- The [Reference Documents] section is auto-generated from tags; do not write it manually.

**Example**:
A work instruction is a guidance document for consistently operating on-site tasks.[USE: EQ-SOP-00001 | 5.1.3 Level 3 (Work Instruction (WI):]
The key characteristics are as follows.
1. It defines the operational flow and management methods at the department or process level.[USE: EQ-SOP-00001 | 5.1.3 Level 3 (Work Instruction (WI):]
2. It includes cleaning and disinfection methods, testing methods, etc.[USE: EQ-SOP-00001 | 5.4.2 Work Instruction Writing]
[DONE]

## Pre-submission Checklist

- Does every sentence have a [USE: ...] tag?
- Does each tag's clause number match the "applicable clause" of the corresponding [DATA_SOURCE]?
- Was the document ID used instead of pronouns like "this document"?
- Was the [Reference Documents] section not written manually?
- Is [DONE] appended at the end of the answer?"""
    
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
            # LLM이 target_doc_id를 누락해도, 질문에서 감지된 ID가 있으면 강제 주입
            if not target_doc_id and state.get("detected_doc_id"):
                target_doc_id = normalize_doc_id(state.get("detected_doc_id"))
                print(f"    [Deep Search] 감지된 문서 ID 강제 적용: {target_doc_id}")

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
    """자율적 도구 호출을 수행하는CompiledSubAgent 스타일 그래프
    - 검색은 1회만 수행, 재시도 결정은 외부 오케스트레이터가 담당
    """
    workflow = StateGraph(SearchState)

    workflow.add_node("agent", call_model_node)
    workflow.add_node("action", tool_executor_node)

    workflow.add_edge(START, "agent")

    def router(state: SearchState):
        last_msg = state["messages"][-1]
        # 이미 tool 결과가 1개 이상이면 → 답변 작성으로 강제 종료
        tool_count = sum(
            1 for m in state["messages"]
            if (isinstance(m, dict) and m.get("role") == "tool") or
               (hasattr(m, "role") and m.role == "tool")
        )
        if tool_count >= 1:
            return END
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

        # 2-2. 태그가 없으면 최소 태그를 자동 주입하여 파이프라인 단절 방지
        if not used_sources:
            print(f"🔴 [검색 에이전트 치명적 오류] LLM이 [USE: ...] 태그를 달지 않음")
            print(f"🟡 검색된 DATA_SOURCE 기반으로 [USE] 태그 자동 보강")
            fallback_tags = " ".join(
                [f"[USE: {doc} | {section}]" for doc, section in referenced_docs[:3]]
            )
            return f"{final_answer}\n{fallback_tags}".strip()

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

    # 질문 텍스트에서 SOP ID 자동 감지
    auto_doc_id = None
    sop_pattern = r'(EQ-(?:SOP|WI|FRM)-\d+)'
    match = re.search(sop_pattern, state['query'], re.IGNORECASE)
    if match:
        auto_doc_id = match.group(1).upper()
        print(f"    [Deep Search] 질문에서 문서 ID 감지: {auto_doc_id}")

    # 모델 정보 확인
    worker_model = state.get("worker_model")
    model_name = state.get("model_name")
    final_model = worker_model or model_name or "gpt-4o"

    # 오케스트레이터 Critic 피드백이 있으면 재시도 지시사항으로 추가
    critique_feedback = state.get("critique_feedback")
    initial_messages = [{"role": "user", "content": state["query"]}]
    if critique_feedback:
        print(f"    [Deep Search] Critic 피드백 반영: {critique_feedback}")
        initial_messages.append({
            "role": "user",
            "content": (
                f"[Orchestrator Critic Feedback] 이전 검색이 불충분했습니다. "
                f"다른 방식으로 재시도하세요:\n{critique_feedback}"
            ),
        })

    initial_state = {
        "messages": initial_messages,
        "query": state["query"],
        "model": final_model,
        "final_answer": "",
    }

    if auto_doc_id and not state.get("target_doc_id"):
        initial_state["detected_doc_id"] = auto_doc_id

    # 내부 도구 호출 (검색 1회 후 종료)
    result = _deep_search_app.invoke(initial_state, config={"recursion_limit": 10})

    # 마지막 메시지에서 실제 답변 추출
    # (router가 tool_calls 있는 상태에서 END 시킬 수 있으므로 content 있는 마지막 AI 메시지 탐색)
    final_msg = ""
    for msg in reversed(result["messages"]):
        content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else None)
        tool_calls = getattr(msg, "tool_calls", None) or (msg.get("tool_calls") if isinstance(msg, dict) else None)
        if content and not tool_calls:
            final_msg = content
            break

    # 답변이 없으면 (LLM이 tool_calls 만 했을 경우) tool_choice=none 으로 강제 답변 생성
    if not final_msg:
        from backend.agent import get_langchain_llm
        llm = get_langchain_llm(model=final_model, temperature=0.0)
        from langchain_core.tools import tool as lc_tool
        tools_def = [{"type": "function", "function": {"name": "search_documents_tool", "description": "Search", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}]
        llm_forced = llm.bind(tools=tools_def, tool_choice="none")
        forced_res = llm_forced.invoke(
            [{"role": "system", "content": "Write the final answer based on the search results above."}]
            + list(result["messages"])
        )
        final_msg = getattr(forced_res, "content", "") or "검색 결과를 찾을 수 없습니다."

    # 참고문헌 섹션 자동 추가
    final_msg_with_refs = _ensure_reference_section(result["messages"], final_msg)

    # [중요] 오케스트레이터가 결과를 비판(Critic)할 수 있도록 context에 보고서로 저장
    # has_no_info 복구 블록은 제거됨 - 오케스트레이터 Critic이 불충분 결과를 잡아 재시도 지시
    report = f"### [검색 에이전트 조사 최종 보고]\n{final_msg_with_refs}"
    return {"context": [report], "last_agent": "retrieval"}

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