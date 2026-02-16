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
            match = re.search(f'"{key}"\\s*:\\s*"([^"]+)"', text)
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
    
    system_prompt = """You are the orchestrator of the GMP regulatory system.
You direct sub-agents to resolve user questions and verify reported results.

## Routing (top-down, first match applies)

| Priority | Agent | Trigger Condition | Example |
|----------|-------|-------------------|---------|
| 1 | `comparison` | Questions about versions, changes, history, differences, or comparisons | "Show me the change history of SOP-001", "What changed?" |
| 2 | `graph` | References, citations, parent/child relationships, impact analysis | "Show me the reference list", "Find related regulations" |
| 3 | `chat` | Conversation context (History) questions or casual conversation | "What did I ask earlier?", "Hello", "Thanks" |
| 4 | `retrieval` | All regulation/knowledge questions not matching the above three | "What is the procedure when a deviation occurs?" |

> Note: `chat` is used only when asking about **conversation context**. "What is the purpose of SOP-001?" goes to `retrieval`.

## Workflow

1. Check History for any previously completed agent reports.
2. **If the report is sufficient** -> Proceed to `finish` immediately (confirm the sub-agent answer as-is; do not rewrite or summarize).
3. **If the report is insufficient** -> Call the appropriate agent.

## Core Rules

- **Immediate termination**: If the sub-agent answer contains `[DONE]` or sufficiently addresses the question, do not make any additional calls; proceed to `finish`.
- **When document ID is unconfirmed**: Before calling `comparison` or `graph`, first obtain the document ID via `retrieval`.
- **When results are excessive**: Do not ask the user for clarification; select the most relevant document and proceed.
- **Loop prevention**:
  - Do not repeat the same agent for the same purpose **more than 3 times**.
  - If `retrieval` reports "no results," do not re-call with the same search term -> change the search term or proceed to `finish`.
  - Do not make additional calls when sufficient evidence already exists.

## Output Format
```json
{"next_action": "retrieval | comparison | graph | chat | finish", "reason": "One-line justification"}
```"""
    
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

    # [Guardrail] 메타 인지 질문 강제 라우팅 (LLM 실수 방지)
    # "방금", "이전", "뭐라고", "내 질문" 등의 키워드가 있고, 아직 chat 에이전트를 부르지 않았다면
    last_user_msg = messages[-1]["content"] if messages else ""
    meta_keywords = ["방금", "뭐라고", "이전 질문", "내 질문", "무슨 말", "무슨 질문", "직전", "처음 질문", "첫 질문", "마지막 질문", "아까 질문"]
    is_meta_query = any(k in last_user_msg for k in meta_keywords)
    
    # 이미 chat을 다녀왔거나 루프 중이라면 무시
    if is_meta_query and "chat" not in agent_calls and loop_count == 0:
        print(f" [Guardrail] 메타 질문 감지 -> 'chat' 강제 라우팅 ('{last_user_msg}')")
        return {"next_agent": "chat", "loop_count": loop_count + 1, "agent_calls": agent_calls}

    # [Guardrail] 관계/참조/영향 질문은 graph 에이전트 우선 라우팅
    relation_keywords = [
        "관계", "참조", "인용", "연결", "상위문서", "하위문서", "근거 문서", "영향", "파급",
        "reference", "citation", "dependency", "impact", "relationship", "related regulation"
    ]
    is_relation_query = any(k.lower() in last_user_msg.lower() for k in relation_keywords)
    if is_relation_query and "graph" not in agent_calls and loop_count == 0:
        print(f" [Guardrail] 관계 질문 감지 -> 'graph' 강제 라우팅 ('{last_user_msg}')")
        return {"next_agent": "graph", "loop_count": loop_count + 1, "agent_calls": agent_calls}

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
        if next_agent == "finish":
            next_agent = "answer"
        print(f"[DEBUG Orchestrator] next_agent 추출: {next_agent}")

        # 검증: 허용된 값만 통과 (state와 정확히 일치)
        ALLOWED_AGENTS = {"retrieval", "graph", "comparison", "answer", "chat"}
        if next_agent not in ALLOWED_AGENTS:
            print(f"🔴 잘못된 next_agent '{next_agent}' 감지, answer로 변경")
            next_agent = "answer"
        else:
            print(f"✅ next_agent '{next_agent}' 검증 통과")

        return {"next_agent": next_agent, "loop_count": loop_count + 1, "agent_calls": agent_calls}

    except Exception as e:
        print(f"Orchestrator Error: {e}")
        return {"next_agent": "answer", "final_answer": "오류가 발생했습니다.", "loop_count": loop_count + 1, "agent_calls": agent_calls}

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
        from backend.sub_agent.chat import chat_agent_node as node_chat
    except ImportError as e:
        error_msg = str(e)
        print(f" 서브 에이전트 로드 실패: {error_msg}")
        # 실패 시 기본 핸들러 정의 (에러 메시지 반환)
        def error_node(state): return {"messages": [{"role": "assistant", "content": f"에이전트 로딩 에러: {error_msg}"}]}
        node_retrieval = error_node
        node_comparison = error_node
        node_graph = error_node
        node_answer = error_node
        node_chat = error_node

    workflow = StateGraph(AgentState)

    # Nodes
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("retrieval", node_retrieval)
    workflow.add_node("comparison", node_comparison)
    workflow.add_node("graph", node_graph)
    workflow.add_node("answer", node_answer)
    workflow.add_node("chat", node_chat)
    
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
            "chat": "chat",
            "end": END
        }
    )
    
    # 각 서브 에이전트는 다시 오케스트레이터로 돌아와서 결과를 보고함
    workflow.add_edge("retrieval", "orchestrator")
    workflow.add_edge("comparison", "orchestrator")
    workflow.add_edge("graph", "orchestrator")
    workflow.add_edge("chat", "orchestrator")
    
    # 답변 에이전트가 생성한 답변은 최종 답변으로 종료
    workflow.add_edge("answer", END)
    
    return workflow.compile()

# ═══════════════════════════════════════════════════════════════════════════
# 실행 인터페이스
# ═══════════════════════════════════════════════════════════════════════════

_app = None

def run_agent(query: str, session_id: str = "default", model_name: str = None, embedding_model: str = None, **kwargs):
    global _app
    if not _app:
        _app = create_workflow()

    # main.py -> run_agent(chat_history=...) 전달값 반영
    chat_history = kwargs.get("chat_history") or []
    messages = []
    if isinstance(chat_history, list):
        for msg in chat_history:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = msg.get("content")
            if role in {"system", "user", "assistant"} and content:
                messages.append({"role": role, "content": str(content)})
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
