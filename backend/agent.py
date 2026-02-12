"""
SOP 멀티 에이전트 시스템 v14.0
- Orchestrator (Main): OpenAI (GPT-4o-mini) - 질문 분석 및 라우팅, 최종 답변
- Specialized Sub-Agents: OpenAI (GPT-4o-mini) - 실행 및 데이터 처리
  1. Retrieval Agent: 문서 검색 및 추출
  2. Summarization Agent: 문서/조항 요약
  3. Comparison Agent: 버전 비교
  4. Graph Agent: 참조 관계 조회
"""

import os
import re
import json
import operator
import hashlib
import difflib
from typing import List, Dict, Optional, Any, Annotated, TypedDict, Literal
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════════
# 임포트 및 설정
# ═══════════════════════════════════════════════════════════════════════════

try:
    from openai import OpenAI
    from langchain_openai import ChatOpenAI
except ImportError:
    OpenAI = None
    ChatOpenAI = None

try:
    from zai import ZaiClient
    ZAI_AVAILABLE = True
except ImportError:
    ZAI_AVAILABLE = False
    pass

try:
    from langchain_core.tools import tool
    from langgraph.graph import StateGraph, START, END
    from langsmith import traceable
    LANGCHAIN_AVAILABLE = True
    LANGGRAPH_AGENT_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    LANGGRAPH_AGENT_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════
# 유틸리티: 안전한 파싱 및 정규화
# ═══════════════════════════════════════════════════════════════════════════

def safe_json_loads(text: str) -> dict:
    """마크다운 태그나 트레일링 콤마가 포함된 LLM의 JSON 응답을 안전하게 파싱"""
    if not text: return {}
    if isinstance(text, dict): return text
    
    try:
        # 1. 마크다운 코드 블록 제거
        clean_text = re.sub(r'^```(?:json)?\s*', '', text.strip())
        clean_text = re.sub(r'\s*```$', '', clean_text.strip())
        
        # 2. 트레일링 콤마 제거
        clean_text = re.sub(r',\s*}', '}', clean_text)
        
        return json.loads(clean_text)
    except:
        # 정규식으로 핵심 필드 추출 시도 (최후의 수단)
        res = {}
        for key in ["doc_id", "target_clause", "intent", "next_action", "plan", "mode"]:
            match = re.search(rf'"{key}"\s*:\s*"([^"]+)"', text)
            if match: res[key] = match.group(1)
        return res

def normalize_doc_id(text: Optional[str]) -> Optional[str]:
    """오타가 섞인 ID(eEQ-SOP-00009)를 정규화하여 실제 ID를 반환"""
    if not text: return None
    # SOP-00000 또는 SOP-000 형식 추출
    match = re.search(r'([A-Z0-9]+-SOP-\d+)', text.upper())
    if match:
        return match.group(1)
    return text.upper()

# ═══════════════════════════════════════════════════════════════════════════
# 전역 스토어 및 클라이언트
# ═══════════════════════════════════════════════════════════════════════════

_vector_store = None
_graph_store = None
_sql_store = None

_openai_client = None
_zai_client = None

def init_agent_tools(vector_store_module, graph_store_instance, sql_store_instance=None):
    global _vector_store, _graph_store, _sql_store
    _vector_store = vector_store_module
    _graph_store = graph_store_instance
    _sql_store = sql_store_instance
    
    # 서브 에이전트 스토어 초기화 (그래프 스토어 추가)
    try:
        from backend.sub_agent.search import init_search_stores
        init_search_stores(vector_store_module, sql_store_instance, graph_store_instance)
    except ImportError:
        pass

def get_openai_client():
    """OpenAI 클라이언트 반환 (직접 API 호출용)"""
    global _openai_client
    if not _openai_client:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            _openai_client = OpenAI(api_key=api_key)
    return _openai_client

_langchain_llm = None

def get_langchain_llm(model: str = "gpt-4o-mini", temperature: float = 0.0):
    """LangChain ChatOpenAI 반환 (LangSmith 추적용)"""
    if ChatOpenAI is None:
        raise ImportError("langchain-openai 패키지가 설치되지 않았거나 로드할 수 없습니다.")
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

def get_zai_client():
    global _zai_client
    if not _zai_client:
        api_key = os.getenv("ZAI_API_KEY")
        if api_key:
            _zai_client = ZaiClient(api_key=api_key)
    return _zai_client

# ═══════════════════════════════════════════════════════════════════════════
# 도구 정의 (Tools)
# ═══════════════════════════════════════════════════════════════════════════

@tool
def search_sop_tool(query: str, extract_english: bool = False, keywords: List[str] = None, target_doc_id: str = None) -> str:
    """SOP 문서 검색 도구.
    Hybrid Search(BM25 + Vector) 방식을 사용하여 키워드와 의미론적 연관성을 동시에 고려합니다.
    extract_english: True면 영문 내용 위주로 추출
    target_doc_id: 특정 문서 ID(예: EQ-SOP-00001)로 검색 범위를 한정할 때 사용
    """
    global _vector_store, _sql_store
    
    results = []
    seen_content = set() # 중복 내용 방지
    
    # 1. 벡터 스토어의 하이브리드 검색 활용 (v8.0+)
    if _vector_store:
        search_query = query
        if keywords:
            # 키워드가 있으면 쿼리에 보강하여 BM25 점수 가중치 부여
            search_query += " " + " ".join(keywords)
            
        # 하이브리드 검색 수행 (alpha=0.4: 키워드 비중 약간 높임)
        vec_res = []
        try:
            # vector_store 모듈에 구현된 search_hybrid 호출
            vec_res = _vector_store.search_hybrid(search_query, n_results=10, alpha=0.4)
        except AttributeError:
            # 만약 구현이 아직 안되었다면 기본 search 사용
            vec_res = _vector_store.search(search_query, n_results=10)
            
        for r in vec_res:
            meta = r.get('metadata', {})
            doc_id = meta.get('doc_id') or meta.get('doc_id') or meta.get('doc_name', 'Unknown')
            clause_id = meta.get('clause_id', '')
            title = meta.get('title', '')
            section = f"{clause_id} {title}" if clause_id and title else (meta.get('section') or meta.get('clause') or "본문")
            content = r.get('text', '')
            
            if target_doc_id and doc_id.upper() != target_doc_id.upper():
                continue
            
            if not content: continue
            
            # 해시로 중복 체크
            content_hash = hashlib.md5(content.encode()).hexdigest()
            if content_hash in seen_content: continue
            seen_content.add(content_hash)

            display_header = f"[검색] {doc_id} > {section}"
            
            # 요약용 정밀 검색(target_doc_id 지정) 시에는 글자 수 제한 대폭 완화
            limit = 8000 if target_doc_id else 1500
            
            if extract_english:
                # 영문 추출 로직: 알파벳 비율이 한글보다 높은 문단 필터링
                paragraphs = content.split('\n\n')
                eng_paras = []
                for p in paragraphs:
                    eng_count = len(re.findall(r'[a-zA-Z]', p))
                    kor_count = len(re.findall(r'[가-힣]', p))
                    if eng_count > kor_count and eng_count > 10:
                        eng_paras.append(p)
                
                if eng_paras:
                    results.append(f"{display_header} (영문):\n" + "\n\n".join(eng_paras[:3]))
                else:
                    results.append(f"{display_header}:\n{content[:limit]}...")
            else:
                results.append(f"{display_header}:\n{content[:limit]}")

    # 2. 결과가 전혀 없거나 매우 적을 경우 SQL 키워드 매칭 (보조/확정적 검색)
    if len(results) < 2 and _sql_store and keywords:
        all_docs = _sql_store.list_documents()
        for doc in all_docs:
            doc_name = doc.get('doc_name', '')
            # 문서명에 키워드가 포함된 경우
            if any(k.upper() in doc_name.upper() for k in keywords):
                doc_id = doc.get('id')
                sop_doc = _sql_store.get_document_by_id(doc_id)
                if sop_doc:
                    # 특정 문서 타겟팅 시에는 SQL에서도 더 많이 가져옴
                    sql_limit = 10000 if target_doc_id else 2000
                    full_content = sop_doc.get("content", "")
                    if full_content:
                        results.append(f"[문서 전체 가이드] {doc_name}:\n{full_content[:sql_limit]}...")
                
    return "\n\n".join(results) if results else "검색 결과 없음. 검색어나 키워드를 바꿔보세요."

@tool
def get_version_history_tool(doc_id: str) -> str:

    """특정 문서의 버전 히스토리를 조회"""
    global _sql_store
    if not _sql_store: return "SQL 저장소 연결 실패"
    versions = _sql_store.get_document_versions(doc_id)
    if not versions: return f"{doc_id} 문서의 버전을 찾을 수 없습니다."

    return "\n".join([f"- v{v['version']} ({v['created_at']})" for v in versions])

@tool
def compare_versions_tool(doc_id: str, v1: str, v2: str) -> str:

    """두 버전의 문서 내용을 비교하여 반환"""
    global _sql_store
    if not _sql_store: return ""
    

    doc1 = _sql_store.get_document_by_id(doc_id, v1)
    doc2 = _sql_store.get_document_by_id(doc_id, v2)
    
    if not doc1 or not doc2: return "비교할 버전을 찾을 수 없습니다."
    
    return f"=== v{v1} ===\n{doc1.get('content')[:2000]}\n\n=== v{v2} ===\n{doc2.get('content')[:2000]}"

@tool
def get_references_tool(doc_id: str) -> str:
    """참조 관계 조회"""
    import json
    from datetime import datetime

    global _graph_store
    if not _graph_store:
        return ""

    refs = _graph_store.get_document_relations(doc_id)

    if not refs:
        return ""

    # Neo4j DateTime 객체를 문자열로 변환
    def serialize_neo4j(obj):
        if hasattr(obj, 'to_native'):
            return obj.to_native().isoformat()
        elif isinstance(obj, dict):
            return {k: serialize_neo4j(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [serialize_neo4j(item) for item in obj]
        else:
            return obj

    refs_serialized = serialize_neo4j(refs)
    result = json.dumps(refs_serialized, ensure_ascii=False)
    return result

@tool
def get_sop_headers_tool(doc_id: str) -> str:
    """특정 문서의 실제 조항(Clause) 목록과 제목을 조회합니다.
    AI가 요약 계획을 세울 때 '짐작'하지 않고 실제 구조를 파악하기 위해 사용합니다.
    """
    global _sql_store
    if not _sql_store: return "SQL 저장소 연결 실패"
    
    doc = _sql_store.get_document_by_name(doc_id)
    if not doc: return f"'{doc_id}' 문서를 찾을 수 없습니다."
    
    chunks = _sql_store.get_chunks_by_document(doc['id'])
    if not chunks: return f"'{doc_id}' 문서의 조항 정보를 찾을 수 없습니다."
    
    headers = []
    seen_clauses = set()
    for c in chunks:
        clause = c.get('clause')
        if clause and clause not in seen_clauses:
            meta = c.get('metadata') or {}
            section = meta.get('section') or ""
            headers.append(f"- {clause}: {section}")
            seen_clauses.add(clause)
            
    return f"[{doc_id} 조항 목록]\n" + "\n".join(headers)

# ═══════════════════════════════════════════════════════════════════════════
# Agent State
# ═══════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    query: str
    messages: Annotated[List[Any], operator.add]

    next_agent: Literal["retrieval", "graph", "comparison", "answer", "end"]
    final_answer: str
    context: Annotated[List[str], operator.add]
    model_name: Optional[str]
    worker_model: Optional[str]
    orchestrator_model: Optional[str]
    loop_count: int
    # 추적 정보 (평가용)
    agent_calls: Optional[Dict[str, int]]  # 에이전트별 호출 횟수
    tool_calls_log: Optional[List[Dict[str, Any]]]  # 도구 호출 로그
    validation_results: Optional[Dict[str, Any]]  # 검증 결과

# ═══════════════════════════════════════════════════════════════════════════
# 노드 정의 (Nodes)
# ═══════════════════════════════════════════════════════════════════════════

def orchestrator_node(state: AgentState):
    """메인 에이전트 (OpenAI GPT-4o-mini) - 질문 분석 및 라우팅"""

    # 추적 정보 초기화
    agent_calls = state.get("agent_calls") or {}
    agent_calls["orchestrator"] = agent_calls.get("orchestrator", 0) + 1

    # 무한 루프 방지: 4번 이상 반복하면 강제 종료
    # (정상 흐름: retrieval -> orch -> graph -> orch -> answer 도 3회 필요)
    loop_count = state.get("loop_count", 0)
    if loop_count >= 4:
        print(f"🔴 루프 제한 도달 ({loop_count}회), 강제 종료")
        return {"next_agent": "answer", "loop_count": loop_count + 1, "agent_calls": agent_calls}
    
    client = get_openai_client()
    if not client:
        print("🔴 OpenAI 클라이언트 없음, retrieval로 라우팅")
        return {"next_agent": "retrieval", "loop_count": loop_count + 1}
    
    messages = state["messages"]
    
    system_prompt = """당신은 GMP 규정 시스템의 메인 오케스트레이터(Manager)입니다.
    사용자의 질문을 해결하기 위해 하위 전문가 에이전트들을 지휘하고, 그들의 보고를 검증하는 역할을 수행합니다.
    
    [작업 흐름]
    1. **History 분석**: 이전 대화 내용(History)을 보고, 이미 수행된 에이전트의 보고가 있는지 확인하세요.

    2. **판단(Judgement)**: 
       - 보고 내용이 충분하다면 -> 'finish'를 선택하여 서브 에이전트의 답변을 그대로 확정하세요. (오케스트레이터가 직접 답변을 재작성하거나 요약하지 않습니다)
       - 보고 내용이 부족하거나 오류가 있다면 -> 다른 에이전트를 호출하거나, 검색 조건을 바꿔서 다시 시도하게 하세요.
    
    [에이전트 목록 및 라우팅 가이드]
    1. retrieval: 규정 검색, 정보 조회. (어떤 문서가 있는지 모를 때 먼저 사용)
    2. comparison: 두 문서의 버전 차이 비교 또는 버전 히스토리(목록) 조회.
    3. graph: **참조/인용 관계(Reference), 상위/하위 규정 관계 확인**. "참조 목록 알려줘", "어떤 규정을 따르나?", "영향 분석해줘" 등의 질문은 반드시 이 에이전트가 처리해야 합니다.
    
    [라우팅 규칙]
    - 사용자가 "**버전**", "**이력**", "**History**", "**변경**", "**수정**", "**차이**", "**비교**" 등의 단어를 사용하여 특정 문서에 대해 질문하면 **무조건 `comparison`**을 호출하세요. (예: "SOP-001 버전 알려줘", "SOP-001 바뀐거 뭐야?", "1.0과 2.0 차이가 뭐야?")
    - 사용자가 "**참조 목록**", "**Reference**", "**연결된 문서**", "**상위문서**", "**하위문서**", "**근거 문서**", "**영향 분석**" 등을 물어보면 **무조건 `graph`**를 호출하세요. `retrieval`로 본문에서 찾으려 하지 마세요.
    - 이전 대화에서 이미 특정 문서(SOP ID)가 식별되었다면, 그 ID를 바탕으로 전문 에이전트(graph, comparison)를 즉시 호출하세요.
    - 만약 서브 에이전트가 "문서 ID를 찾지 못했다"고 보고한다면, `retrieval`을 통해 먼저 문서 ID를 찾은 후 다시 해당 에이전트를 부르세요.
    
    [중요 종료 조건]
    - 서브 에이전트가 답변 마지막에 `[DONE]`을 포함했거나, 답변 내용이 질문에 충분히 대답하고 있다면 **절대 다시 질문하거나 요약하지 말고 즉시 `finish`를 선택**하세요.
    - 이미 보고된 내용을 다듬기 위해 다른 에이전트를 호출하지 마세요.
    
    [출력 형식]
    JSON 형식으로 'next_action' (agent 이름 또는 'finish')과 'reason'을 반환하세요.
    - **중요(Termination)**: 서브 에이전트의 보고 내용에 이미 답변에 필요한 충분한 정보(예: 검색된 문단, 시각화 보고서, 요약 등)가 있다면 즉시 'finish'를 선택하세요.
    - **루프 방지(Loop Prevention)**: 
        1. 동일한 서브 에이전트({next_action})를 같은 목적({reason})으로 3회 이상 반복 호출하지 마세요. 
        2. 만약 `retrieval` 에이전트가 "검색 결과 없음"을 보고했다면, 똑같은 검색어로는 다시 호출하지 마세요. 검색어를 바꾸거나 실패를 인정하고 'finish'하세요.
        3. 이미 답변할 근거가 생겼음에도 불구하고 서브 에이전트를 계속 부르는 것은 금지됩니다.
    
    예: {"next_action": "retrieval", "reason": "규정 검색 결과가 부족하여 재검색 필요"}
    """
    
    # 현재까지 수집된 context 정보를 프롬프트에 추가하여 루프 방지
    current_context = state.get("context", [])
    combined_context_str = "\n".join([f"- {c[:1500]}..." for c in current_context]) if current_context else "없음"
    
    # [DONE] 태그 확인 (루프 강제 종료 조건 - 파이썬 레벨에서 하드코딩)
    has_done = any("[DONE]" in c for c in current_context)
    
    if has_done:
        print(f" [Orchestrator] [DONE] 신호 감지 -> 즉시 종료(finish) 결정")
        return {"next_agent": "answer"}

    orchestrator_input = f"""현재까지 수집된 에이전트들의 보고서 요약:
    {combined_context_str}
    
    위 보고서 내용을 바탕으로 다음 단계를 결정하세요. 만약 충분한 정보가 수집되었다면 'finish'를 선택하세요."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                *messages,
                {"role": "user", "content": orchestrator_input}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        decision = safe_json_loads(content)
        
        print(f"[DEBUG Orchestrator] LLM 응답: {content}")
        print(f"[DEBUG Orchestrator] 파싱된 결정: {decision}")

        next_agent = decision.get("next_action", "answer")  # LLM이 next_action을 반환함
        print(f"[DEBUG Orchestrator] next_agent 추출: {next_agent}")

        # 검증: 허용된 값만 통과 (state와 정확히 일치)
        ALLOWED_AGENTS = {"retrieval", "graph", "comparison", "answer"}
        if next_agent not in ALLOWED_AGENTS:
            print(f"🔴 잘못된 next_agent '{next_agent}' 감지, answer로 변경")
            next_agent = "answer"
        else:
            print(f"✅ next_agent '{next_agent}' 검증 통과")

        return {"next_agent": next_agent, "loop_count": loop_count + 1, "agent_calls": agent_calls}

    except Exception as e:
        print(f"Orchestrator Error: {e}")
        return {"next_agent": "answer", "final_answer": "오류가 발생했습니다.", "loop_count": loop_count + 1, "agent_calls": agent_calls}

# ═══════════════════════════════════════════════════════════════════════════
# 워크플로우 구성
# ═══════════════════════════════════════════════════════════════════════════

def create_workflow():
    # 서브 에이전트 노드들을 지연 임포트하여 순환 참조(Circular Import) 방지
    try:
        from backend.sub_agent.search import retrieval_agent_node as node_retrieval
        from backend.sub_agent.graph import graph_agent_node as node_graph
        from backend.sub_agent.answer import answer_agent_node as node_answer
        from backend.sub_agent.compare import comparison_agent_node as node_comparison
    except ImportError as e:
        error_msg = str(e)
        print(f" 서브 에이전트 로드 실패: {error_msg}")
        # 실패 시 기본 핸들러 정의 (에러 메시지 반환)
        def error_node(state): return {"messages": [{"role": "assistant", "content": f"에이전트 로딩 에러: {error_msg}"}]}
        node_retrieval = error_node
        node_comparison = error_node
        node_graph = error_node
        node_answer = error_node

    workflow = StateGraph(AgentState)

    # Nodes
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("retrieval", node_retrieval)
    workflow.add_node("comparison", node_comparison)
    workflow.add_node("graph", node_graph)
    workflow.add_node("answer", node_answer)
    
    # Edges
    workflow.add_edge(START, "orchestrator")
    
    # Router
    def router(state: AgentState):
        return state["next_agent"]
    
    workflow.add_conditional_edges(
        "orchestrator",
        router,
        {
            "retrieval": "retrieval",
            "comparison": "comparison",
            "graph": "graph",
            "answer": "answer",
            "end": END
        }
    )
    
    # 각 서브 에이전트는 다시 오케스트레이터로 돌아와서 결과를 보고함
    workflow.add_edge("retrieval", "orchestrator")
    workflow.add_edge("comparison", "orchestrator")
    workflow.add_edge("graph", "orchestrator")
    
    # 답변 에이전트가 생성한 답변은 최종 답변으로 종료
    workflow.add_edge("answer", END)
    
    return workflow.compile()

# ═══════════════════════════════════════════════════════════════════════════
# 실행 인터페이스
# ═══════════════════════════════════════════════════════════════════════════

_app = None

def run_agent(query: str, session_id: str = "default", model_name: str = None, embedding_model: str = None, chat_history: List[Dict] = None, **kwargs):
    global _app
    if not _app:
        _app = create_workflow()

    # 채팅 히스토리 반영
    messages = []
    if chat_history:
        messages.extend(chat_history)
    messages.append({"role": "user", "content": query})

    initial_state = {
        "query": query,
        "messages": messages,
        "next_agent": "orchestrator",
        "worker_model": model_name or "gpt-4o-mini",
        "orchestrator_model": "gpt-4o-mini",
        "model_name": model_name,
        "loop_count": 0,
        "agent_calls": {},  # 에이전트 호출 추적
        "tool_calls_log": [],  # Tool 호출 로그
        "validation_results": {}  # 검증 결과
    }

    # LangGraph 실행
    result = _app.invoke(initial_state, config={"recursion_limit": 10})

    # 최종 답변 추출
    final_answer = "답변을 생성하지 못했습니다."
    if "messages" in result and result["messages"]:
        last_msg = result["messages"][-1]
        if hasattr(last_msg, "content"):
            final_answer = last_msg.content
        elif isinstance(last_msg, dict):
            final_answer = last_msg.get("content", final_answer)

    # context 추출 (평가용)
    context = result.get("context", [])
    context_str = "\n\n".join(context) if isinstance(context, list) else context

    # ========================================
    # 평가 로그 생성 (agent_log)
    # ========================================

    # 1. 에이전트 호출 통계
    agent_calls = result.get("agent_calls", {})

    # 2. [USE: ...] 태그 분석
    use_tags = re.findall(r'\[USE:\s*([^\|]+)\s*\|\s*([^\]]+)\]', final_answer)
    use_tag_count = len(use_tags)

    # 3. Tool 호출 로그 추출 (messages에서)
    tool_calls_log = []
    for msg in result.get("messages", []):
        if isinstance(msg, dict) and msg.get("role") == "tool":
            tool_calls_log.append({
                "role": "tool",
                "content_preview": msg.get("content", "")[:200]
            })
        elif hasattr(msg, "role") and msg.role == "tool":
            content = msg.content if hasattr(msg, "content") else str(msg)
            tool_calls_log.append({
                "role": "tool",
                "content_preview": content[:200]
            })

    # 4. 검증 결과 분석
    validation_summary = {
        "grounding": "unknown",
        "format": "unknown",
        "has_use_tags": use_tag_count > 0,
        "no_info_found": False
    }

    # NO_INFO_FOUND 감지
    if "검색된 문서 내에서 관련 정보를 찾을 수 없" in final_answer or \
       "검색된 정보가 없습니다" in final_answer or \
       "[NO_INFO_FOUND]" in final_answer:
        validation_summary["no_info_found"] = True

    # 5. 검색 조건 추출 (context에서)
    search_conditions = []
    for ctx in context:
        # [Deep Search] 로그에서 검색 조건 추출 시도
        if "Deep Search" in ctx or "검색" in ctx:
            search_conditions.append({
                "query": query,
                "preview": ctx[:150]
            })

    return {
        "answer": final_answer,
        "agent_log": {
            # 기본 정보
            "query": query,
            "context": context_str,
            "next_agent": result.get("next_agent"),
            "loop_count": result.get("loop_count", 0),

            # 에이전트 호출 통계
            "agent_calls": agent_calls,
            "total_agent_calls": sum(agent_calls.values()) if agent_calls else 0,

            # Tool 호출 정보
            "tool_calls_count": len(tool_calls_log),
            "tool_calls_log": tool_calls_log[:5],  # 최대 5개만

            # 태그 분석
            "use_tag_count": use_tag_count,
            "use_tags_sample": use_tags[:3] if use_tags else [],  # 샘플 3개

            # 검증 결과
            "validation_summary": validation_summary,

            # 검색 조건
            "search_conditions": search_conditions[:3]  # 최대 3개
        },
        "wrapper": True
    }


# ═══════════════════════════════════════════════════════════════════════════
# 외부 노출 도구 목록
# ═══════════════════════════════════════════════════════════════════════════

AGENT_TOOLS = [
    search_sop_tool,
    get_version_history_tool,
    compare_versions_tool,
    get_references_tool,
    get_sop_headers_tool,
    compare_versions_tool
]
