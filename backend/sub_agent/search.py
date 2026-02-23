"""
문서 검색 서브에이전트 모듈
- 검색 → LLM 답변 단일 흐름 (내부 루프 없음)
- 벡터 검색 (Weaviate), SQL 검색 (PostgreSQL) 통합
"""

import os
import re
import json
import hashlib
from typing import List, Dict, Any, Optional, Literal
from backend.agent import get_openai_client, AgentState, search_sop_tool, get_sop_headers_tool, safe_json_loads, normalize_doc_id
from langsmith import traceable
from langchain_core.tools import tool

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

_SEARCH_SYSTEM = """You are a GMP/SOP document search specialist. You ONLY answer based on the [DATA_SOURCE] blocks provided. You have NO other knowledge. You cannot use your training data.

## Strict Rules

RULE 1 - SOURCES ONLY: Every sentence in your answer MUST come from a [DATA_SOURCE] block. If you cannot find the answer in the provided [DATA_SOURCE] blocks, you MUST say:
"검색된 문서 내에서 관련 정보를 찾을 수 없습니다.[NO_INFO_FOUND]"
Do NOT answer from general knowledge. Do NOT invent or infer information.

RULE 2 - TAG EVERY SOURCE: For every [DATA_SOURCE] block you use, end the sentence with [USE: document_name | clause].
- "document_name" = exact value from "문서 정보" field
- "clause" = exact value from "해당 조항" field (copy exactly, do not shorten)

RULE 3 - NO MERGING: Each [DATA_SOURCE] must be tagged separately. Do not merge multiple sources into one sentence.

RULE 4 - ALL RELEVANT SOURCES: If multiple [DATA_SOURCE] blocks are relevant, all of them must appear in your answer with their own [USE] tag. Do not pick just one.

RULE 5 - FORMAT: Korean plain text only. No markdown (no **, #, -, *). End with [DONE].

## Example
질문: 작업지침서가 뭐야?
→ 작업지침서는 현장 업무를 일관되게 운영하기 위한 문서입니다.[USE: EQ-SOP-00001 | 5.1.3]
부서 수준의 운영 흐름과 관리 방법을 정의합니다.[USE: EQ-SOP-00001 | 5.2.1]
세척 및 소독 방법, 시험 방법이 포함됩니다.[USE: EQ-SOP-00002 | 3.4]
[DONE]

## Example (no relevant results)
질문: 규격서가 뭐야?
→ 검색된 문서 내에서 관련 정보를 찾을 수 없습니다.[NO_INFO_FOUND]
[DONE]"""

@traceable(name="sub_agent:search")
def retrieval_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """[서브] 검색 에이전트 - 검색 1회 후 LLM 답변 생성"""
    query = state["query"]
    print(f"[Search] 검색 시작: {query}")

    # 문서 ID 감지
    auto_doc_id = None
    match = re.search(r'(EQ-(?:SOP|WI|FRM)-\d+)', query, re.IGNORECASE)
    if match:
        auto_doc_id = normalize_doc_id(match.group(1))
        print(f"[Search] 문서 ID 감지: {auto_doc_id}")

    # Critic 피드백이 있으면 쿼리에 추가
    critique_feedback = state.get("critique_feedback")
    search_query = query
    if critique_feedback:
        search_query = f"{query} {critique_feedback}"
        print(f"[Search] Critic 피드백 반영: {critique_feedback}")

    # 검색 실행
    final_model = state.get("worker_model") or state.get("model_name") or "gpt-4o"
    results = search_documents_internal(
        query=search_query,
        target_doc_id=auto_doc_id,
    )
    print(f"[Search] 검색 결과 {len(results)}건")

    # 결과 포매팅 (상위 20개, 내용 1000자 제한)
    formatted = []
    for r in results[:20]:
        formatted.append(
            f"[DATA_SOURCE]\n"
            f"문서 정보: {r.get('doc_name', 'Unknown')}\n"
            f"해당 조항: {r.get('section', '본문')}\n"
            f"본문 내용: {r.get('content', '')[:1000]}\n"
            f"[END_SOURCE]"
        )
    data_str = "\n\n".join(formatted) if formatted else "검색 결과 없음."

    # LLM 답변 생성 (1회)
    from backend.agent import get_langchain_llm
    llm = get_langchain_llm(model=final_model, temperature=0.0)
    messages = [
        {"role": "system", "content": _SEARCH_SYSTEM},
        {"role": "user", "content": f"질문: {query}\n\n{data_str}"},
    ]
    res = llm.invoke(messages)
    final_msg = getattr(res, "content", "").strip() or "검색 결과를 찾을 수 없습니다."

    # 참고문헌 섹션 자동 추가 (tool 메시지 없으므로 빈 리스트 전달)
    final_msg_with_refs = _ensure_reference_section([], final_msg)

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