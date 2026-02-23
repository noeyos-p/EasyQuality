"""
SOP 멀티 에이전트 시스템 v14.0
- Orchestrator (Main): OpenAI (GPT 계열) - 질문 분석 및 라우팅, 최종 답변
- Specialized Sub-Agents: OpenAI (GPT 계열) - 실행 및 데이터 처리
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

def get_langchain_llm(model: str = "gpt-4o", temperature: float = 0.0):
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
    벡터 검색은 search.py의 search_documents_internal을 통해서만 수행됩니다.
    extract_english: True면 영문 내용 위주로 추출
    target_doc_id: 특정 문서 ID(예: EQ-SOP-00001)로 검색 범위를 한정할 때 사용
    """
    # 벡터 검색은 search.py의 search_documents_internal에 위임 (lazy import으로 순환 참조 방지)
    from backend.sub_agent.search import search_documents_internal

    raw_results = search_documents_internal(
        query=query,
        keywords=keywords,
        target_doc_id=target_doc_id,
    )

    if not raw_results:
        return "검색 결과 없음. 검색어나 키워드를 바꿔보세요."

    results = []
    for r in raw_results:
        doc_name = r.get('doc_name', 'Unknown')
        section = r.get('section', '본문')
        content = r.get('content', '')
        if not content:
            continue

        display_header = f"[검색] {doc_name} > {section}"
        limit = 8000 if target_doc_id else 1500

        if extract_english:
            paragraphs = content.split('\n\n')
            eng_paras = [
                p for p in paragraphs
                if len(re.findall(r'[a-zA-Z]', p)) > len(re.findall(r'[가-힣]', p))
                and len(re.findall(r'[a-zA-Z]', p)) > 10
            ]
            if eng_paras:
                results.append(f"{display_header} (영문):\n" + "\n\n".join(eng_paras[:3]))
            else:
                results.append(f"{display_header}:\n{content[:limit]}...")
        else:
            results.append(f"{display_header}:\n{content[:limit]}")

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
    # Critic 루프 필드
    critique_feedback: Optional[str]      # 오케스트레이터가 서브에이전트에 전달하는 재시도 피드백
    last_agent: Optional[str]             # 직전에 실행된 서브에이전트 이름
    retry_per_agent: Optional[Dict[str, int]]  # 에이전트별 재시도 횟수
    # Planning 필드
    plan: Optional[str]                   # 질문 처음 수신 시 수립한 실행 계획

# ═══════════════════════════════════════════════════════════════════════════
# 노드 정의 (Nodes)
# ═══════════════════════════════════════════════════════════════════════════

_MAX_RETRY_PER_AGENT = 2  # 에이전트별 최대 재시도 횟수


def _plan_call(query: str, client) -> str:
    """질문을 분석해 실행 계획(Plan)을 수립. loop_count==0일 때 한 번만 호출됨.
    LangSmith에서 'orchestrator:plan' 스팬으로 추적됨."""
    plan_prompt = """You are the planning module of a GMP regulatory AI system.
Given a user query, output a concise step-by-step execution plan (3 lines max).
Each step must specify which agent to use and why.
Available agents: retrieval (document search), comparison (version diff), graph (reference relationships), chat (conversation history).

Output format (plain text, no markdown):
Step 1: [agent] - [reason]
Step 2: [agent] - [reason]
Step 3: (if needed) [agent] - [reason]"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": plan_prompt},
                {"role": "user", "content": f"Query: {query}"},
            ],
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Plan] 계획 수립 실패: {e}")
        return "Step 1: retrieval - Search for relevant documents"


def _critic_call(query: str, last_agent: str, report: str, client) -> dict:
    """서브에이전트 결과물을 비판적으로 평가하여 재시도 여부와 피드백 결정"""

    agent_criteria = {
        "retrieval": (
            "Contains specific, relevant information with [USE: doc | clause] citations. "
            "NOT acceptable: 'no results found', empty answer, or off-topic content. "
            "EXCEPTION: If the report clearly states the document/section genuinely does not exist in the system, mark as sufficient."
        ),
        "comparison": (
            "Contains actual version comparison with concrete changes listed (ADDED/DELETED/MODIFIED), OR "
            "a clear structural explanation why comparison is impossible (e.g., 'only one version exists', 'document not found'). "
            "Mark as SUFFICIENT if the agent correctly determined the task is structurally impossible. "
            "Mark as INSUFFICIENT only if the agent failed without trying (e.g., wrong document ID used, or no attempt made)."
        ),
        "graph": (
            "Contains actual reference/relationship data for the queried document, OR "
            "a clear explanation that no relationships exist. "
            "EXCEPTION: If the agent correctly determined there are no relationships, mark as sufficient."
        ),
        "chat": "Directly addresses the user's meta-question about conversation history.",
    }
    criteria = agent_criteria.get(last_agent, "Directly and specifically answers the user's query.")

    critic_prompt = f"""You are a strict quality evaluator for a GMP regulatory AI system.
Evaluate if the sub-agent report sufficiently answers the user query.

Sufficient criteria for '{last_agent}' agent:
{criteria}

Respond with JSON only:
{{"quality": "sufficient|insufficient", "feedback": "If insufficient: specific retry instructions (try different keywords, look in a specific document, change search scope, etc.)", "reason": "One-line explanation"}}"""

    user_content = f"User query: {query}\n\nSub-agent report:\n{report[:2500]}"

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": critic_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        result = safe_json_loads(resp.choices[0].message.content)
        print(f"[Critic] 평가 결과: quality={result.get('quality')}, reason={result.get('reason')}")
        return result
    except Exception as e:
        print(f"[Critic] 평가 실패 ({e}) → sufficient 처리")
        return {"quality": "sufficient", "feedback": "", "reason": "평가 오류로 sufficient 처리"}


try:
    from langsmith import traceable as _traceable
except ImportError:
    def _traceable(**kwargs):
        def decorator(fn): return fn
        return decorator


@_traceable(name="orchestrator", run_type="chain")
def orchestrator_node(state: AgentState):
    """메인 에이전트 (OpenAI GPT) - 실행 계획 수립 / 결과 비판 / 라우팅"""

    # 추적 정보 초기화
    agent_calls = state.get("agent_calls") or {}
    agent_calls["orchestrator"] = agent_calls.get("orchestrator", 0) + 1

    # 무한 루프 방지: critic 재시도 여유를 위해 8회로 상향
    loop_count = state.get("loop_count", 0)
    if loop_count >= 8:
        print(f"🔴 루프 제한 도달 ({loop_count}회), 강제 종료")
        return {"next_agent": "answer", "loop_count": loop_count + 1, "agent_calls": agent_calls}

    client = get_openai_client()
    if not client:
        print("🔴 OpenAI 클라이언트 없음, retrieval로 라우팅")
        return {"next_agent": "retrieval", "loop_count": loop_count + 1}

    context = state.get("context", [])

    # ─── Phase A: Critic ────────────────────────────────────────────────────
    # 직전에 서브에이전트가 실행됐다면 결과물을 비판적으로 평가하여 재시도 여부 결정
    last_agent = state.get("last_agent")
    retry_per_agent = dict(state.get("retry_per_agent") or {})

    if last_agent and last_agent not in ("orchestrator", "answer") and context:
        current_retries = retry_per_agent.get(last_agent, 0)

        if current_retries < _MAX_RETRY_PER_AGENT:
            print(f"[Critic] '{last_agent}' 결과 평가 중... (재시도 {current_retries}/{_MAX_RETRY_PER_AGENT})")
            critic_result = _critic_call(state["query"], last_agent, context[-1], client)

            if critic_result.get("quality") == "insufficient":
                retry_per_agent[last_agent] = current_retries + 1
                feedback = critic_result.get("feedback", "")
                print(f"[Critic] 결과 불충분 → '{last_agent}' 재시도 (피드백: {feedback})")
                return {
                    "next_agent": last_agent,
                    "critique_feedback": feedback,
                    "retry_per_agent": retry_per_agent,
                    "last_agent": last_agent,
                    "loop_count": loop_count + 1,
                    "agent_calls": agent_calls,
                }
            else:
                print(f"[Critic] '{last_agent}' 결과 충분 → 라우팅 계속")
        else:
            print(f"[Critic] '{last_agent}' 최대 재시도 도달 ({current_retries}회) → 라우팅 계속")

    # ─── Phase 0: Planning (첫 번째 진입 시 실행 계획 수립) ─────────────────────
    plan = state.get("plan")
    messages = state["messages"]
    last_user_msg = messages[-1]["content"] if messages else ""

    if loop_count == 0 and not plan:
        plan = _plan_call(state["query"], client)
        print(f"[Orchestrator] 실행 계획 수립:\n{plan}")

    # ─── Phase B: Router ────────────────────────────────────────────────────
    system_prompt = """You are the orchestrator of the GMP regulatory system.
You direct sub-agents to resolve user questions and verify reported results.
Follow the execution plan already established for this query.

## Routing (top-down, first match applies)

| Priority | Agent | Trigger Condition | Example |
|----------|-------|-------------------|---------|
| 1 | `comparison` | Questions about versions, changes, history, differences, or comparisons | "Show me the change history of SOP-001", "What changed?" |
| 2 | `graph` | References, citations, parent/child relationships, impact analysis | "Show me the reference list", "Find related regulations" |
| 3 | `chat` | Conversation context (History) questions or casual conversation | "What did I ask earlier?", "Hello", "Thanks" |
| 4 | `retrieval` | All regulation/knowledge questions not matching the above three | "What is the procedure when a deviation occurs?" |

> Note: `chat` is used only when asking about **conversation context**. "What is the purpose of SOP-001?" goes to `retrieval`.

## Core Rules

- **Immediate termination**: If the sub-agent answer contains `[DONE]` or sufficiently addresses the question, proceed to `finish`.
- **When document ID is unconfirmed**: Before calling `comparison` or `graph`, first obtain the document ID via `retrieval`.
- **When results are excessive**: Do not ask the user for clarification; select the most relevant document and proceed.
- **Loop prevention**: Do not repeat the same agent more than 3 times total.

## Output Format
```json
{"next_action": "retrieval | comparison | graph | chat | finish", "reason": "One-line justification"}
```"""

    combined_context_str = "\n".join([f"- {c[:1500]}..." for c in context]) if context else "없음"

    # [DONE] 태그 확인 (루프 강제 종료 조건)
    has_done = any("[DONE]" in c for c in context)
    if has_done:
        print(f"[Orchestrator] [DONE] 신호 감지 → 즉시 종료")
        return {
            "next_agent": "answer",
            "critique_feedback": None,
            "last_agent": "orchestrator",
            "plan": plan,
            "loop_count": loop_count + 1,
            "agent_calls": agent_calls,
        }

    orchestrator_input = f"""실행 계획:
{plan or '없음'}

현재까지 수집된 에이전트 보고서:
{combined_context_str}

위 내용을 바탕으로 다음 단계를 결정하세요. 충분한 정보가 수집됐다면 'finish'를 선택하세요."""

    # [Guardrail] 메타 인지 질문 강제 라우팅
    meta_keywords = ["방금", "뭐라고", "이전 질문", "내 질문", "무슨 말", "무슨 질문", "직전", "처음 질문", "첫 질문", "마지막 질문", "아까 질문"]
    is_meta_query = any(k in last_user_msg for k in meta_keywords)
    if is_meta_query and "chat" not in agent_calls and loop_count == 0:
        print(f"[Guardrail] 메타 질문 감지 → 'chat' 강제 라우팅")
        return {
            "next_agent": "chat",
            "last_agent": "chat",
            "critique_feedback": None,
            "plan": plan,
            "loop_count": loop_count + 1,
            "agent_calls": agent_calls,
        }

    # [Guardrail] 관계/참조/영향 질문
    relation_keywords = [
        "관계", "참조", "인용", "연결", "상위문서", "하위문서", "근거 문서", "영향", "파급",
        "reference", "citation", "dependency", "impact", "relationship", "related regulation"
    ]
    if any(k.lower() in last_user_msg.lower() for k in relation_keywords) and "graph" not in agent_calls and loop_count == 0:
        print(f"[Guardrail] 관계 질문 감지 → 'graph' 강제 라우팅")
        return {
            "next_agent": "graph",
            "last_agent": "graph",
            "critique_feedback": None,
            "plan": plan,
            "loop_count": loop_count + 1,
            "agent_calls": agent_calls,
        }

    # [Guardrail] 비교 질문
    compare_keywords = ["비교", "달라짐", "차이", "변경내역", "변경 내용", "버전", "v1", "v2", "계약", "문서 비교"]
    if any(k in last_user_msg for k in compare_keywords) and "comparison" not in agent_calls and loop_count == 0:
        print(f"[Guardrail] 비교 질문 감지 → 'comparison' 강제 라우팅")
        return {
            "next_agent": "comparison",
            "last_agent": "comparison",
            "critique_feedback": None,
            "plan": plan,
            "loop_count": loop_count + 1,
            "agent_calls": agent_calls,
        }

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
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

        print(f"[Orchestrator] LLM 결정: {content}")

        next_agent = decision.get("next_action", "answer")
        if next_agent == "finish":
            next_agent = "answer"

        ALLOWED_AGENTS = {"retrieval", "graph", "comparison", "answer", "chat"}
        if next_agent not in ALLOWED_AGENTS:
            print(f"🔴 잘못된 next_agent '{next_agent}' → answer로 변경")
            next_agent = "answer"

        # 서브에이전트로 라우팅 시 last_agent 설정, critique_feedback 초기화
        is_sub_agent = next_agent not in ("answer",)
        return {
            "next_agent": next_agent,
            "last_agent": next_agent if is_sub_agent else "orchestrator",
            "critique_feedback": None,
            "plan": plan,
            "loop_count": loop_count + 1,
            "agent_calls": agent_calls,
        }

    except Exception as e:
        print(f"Orchestrator Error: {e}")
        return {
            "next_agent": "answer",
            "critique_feedback": None,
            "last_agent": "orchestrator",
            "plan": plan,
            "loop_count": loop_count + 1,
            "agent_calls": agent_calls,
        }

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
# Deep Agent - 오케스트레이터가 서브에이전트를 툴로 직접 호출하는 ReAct 구조
# ═══════════════════════════════════════════════════════════════════════════

class DeepOrchestratorState(TypedDict):
    query: str
    messages: Annotated[List[Any], operator.add]
    context: Annotated[List[str], operator.add]
    worker_model: Optional[str]
    plan: Optional[str]
    plan_output: Optional[dict]  # LangSmith output용 파싱된 plan JSON


def _build_sub_agent_tools() -> list:
    """서브에이전트를 LangChain tool로 래핑. init_agent_tools() 이후에 호출해야 함."""

    @tool
    def retrieval(query: str, search_hint: str = "") -> str:
        """Search GMP/SOP regulatory documents for relevant information.
        Use search_hint to pass specific guidance (e.g., target document ID, clause, or retry instructions)."""
        from backend.sub_agent.search import retrieval_agent_node
        state = {
            "query": query,
            "worker_model": "gpt-4o",
            "model_name": "gpt-4o",
            "critique_feedback": search_hint or None,
            "messages": [{"role": "user", "content": query}],
            "context": [],
            "loop_count": 0,
            "agent_calls": {},
        }
        result = retrieval_agent_node(state)
        return (result.get("context") or ["검색 결과 없음"])[0]

    @tool
    def comparison(query: str) -> str:
        """Compare versions of a document or retrieve its version history."""
        from backend.sub_agent.compare import comparison_agent_node
        state = {
            "query": query,
            "worker_model": "gpt-4o",
            "model_name": "gpt-4o",
            "critique_feedback": None,
            "messages": [{"role": "user", "content": query}],
        }
        result = comparison_agent_node(state)
        return (result.get("context") or ["비교 결과 없음"])[0]

    @tool
    def graph(query: str) -> str:
        """Check document reference relationships and impact analysis."""
        from backend.sub_agent.graph import graph_agent_node
        state = {
            "query": query,
            "worker_model": "gpt-4o",
            "model_name": "gpt-4o",
            "critique_feedback": None,
            "messages": [{"role": "user", "content": query}],
        }
        result = graph_agent_node(state)
        return (result.get("context") or ["관계 데이터 없음"])[0]

    @tool
    def chat(query: str) -> str:
        """Handle questions about conversation history or casual conversation."""
        from backend.sub_agent.chat import chat_agent_node
        state = {
            "query": query,
            "worker_model": "gpt-4o",
            "model_name": "gpt-4o",
            "critique_feedback": None,
            "messages": [{"role": "user", "content": query}],
        }
        result = chat_agent_node(state)
        return (result.get("context") or ["대화 처리 실패"])[0]

    return [retrieval, comparison, graph, chat]


_deep_agent_tools_cache: Optional[list] = None

def _get_deep_agent_tools() -> list:
    global _deep_agent_tools_cache
    if _deep_agent_tools_cache is None:
        _deep_agent_tools_cache = _build_sub_agent_tools()
    return _deep_agent_tools_cache


_DEEP_ORCHESTRATOR_SYSTEM = """You are a Deep Agent orchestrator for a GMP regulatory document system.

## Tool Selection Examples

Use `graph` for questions like:
- "EQ-SOP-00001의 참조 문서가 뭐야?"
- "이 문서가 어떤 문서를 인용하고 있어?"
- "SOP-001을 변경하면 어디에 영향을 줘?"
- "이 문서의 상위 문서는 뭐야?"
- "어떤 문서가 이 SOP를 참조해?"
- "문서 간 관계를 보여줘"

Use `retrieval` for questions like:
- "SOP-001의 5.1 조항 내용은?"
- "장비 세척 절차가 뭐야?"
- "GMP에서 일탈은 어떻게 정의해?"
- "이 문서에서 책임자가 누구야?"

Use `comparison` for questions like:
- "SOP-001 버전 1과 버전 2의 차이는?"
- "이전 버전에서 뭐가 바뀌었어?"

Use `chat` for casual conversation or history questions.

If the [실행 계획] suggests `graph`, you MUST call `graph` — do not substitute `retrieval`.

## How to operate

- If the tool result contains ANY useful content → write the final answer immediately. Stop here.
- If the tool result is completely empty → call one more tool with a different query, then write the final answer regardless.

Never call a tool more than twice total.

## Answer format
- Korean plain text only. No markdown (no **, no #, no -).
- Every sentence must end with [USE: document_name | clause].
- End with [DONE]."""


_DEEP_PLAN_SYSTEM = """You are a planner for a GMP regulatory document agent.

Given the user query, available document headers, and detected document ID, output a JSON object:
{
  "reasoning": "<why you chose these sections>",
  "decision": {
    "doc_id": "<detected doc id or null>",
    "mode": "section" | "global",
    "plan": ["<clause_num or tool_name>", ...]
  }
}

Rules:
- FIRST classify the question intent:
  - Reference/relationship questions → mode="global", plan=["graph"]
    Examples: "참조 문서가 뭐야?", "어떤 문서를 인용해?", "변경 시 영향을 줘?", "상위 문서는?", "어떤 문서가 이걸 참조해?", "문서 관계 보여줘"
  - Content questions about a specific doc → mode="section", pick relevant clause numbers from available_headers
    Examples: "5.1 조항 내용은?", "책임자가 누구야?", "절차가 어떻게 돼?"
  - Version comparison questions → mode="global", plan=["comparison"]
    Examples: "버전 차이는?", "뭐가 바뀌었어?"
  - No specific doc → mode="global", plan=["retrieval"]
- Output JSON only. No markdown."""

@_traceable(name="deep_orchestrator:plan", run_type="chain")
def _deep_orchestrator_plan_node(state: DeepOrchestratorState):
    """Deep Agent 플래닝 노드 - @traceable이 plan_output dict를 LangSmith output으로 노출"""
    user_messages = [m for m in state["messages"] if (
        (isinstance(m, dict) and m.get("role") == "user") or
        (not isinstance(m, dict) and getattr(m, "type", "") == "human")
    )]
    if len(user_messages) != 1:
        return {"messages": [], "plan": None, "plan_output": None}

    query = state["query"]
    model = state.get("worker_model") or "gpt-4o"

    # 1. 쿼리에서 문서 ID 감지
    detected_doc_id = None
    sop_match = re.search(r'(EQ-(?:SOP|WI|FRM)-\d+)', query, re.IGNORECASE)
    if sop_match:
        detected_doc_id = sop_match.group(1).upper()

    # 2. 문서 헤더 조회
    available_headers = []
    headers_status = "skipped"
    if detected_doc_id:
        try:
            headers_raw = get_sop_headers_tool.invoke({"doc_id": detected_doc_id})
            available_headers = [l.strip().lstrip("- ") for l in headers_raw.splitlines() if l.strip() and not l.startswith("[")]
            headers_status = "success"
        except Exception:
            headers_status = "failed"

    # 3. LLM 플래닝 - response_format json_object 강제
    llm = get_langchain_llm(model=model, temperature=0.0)
    llm_json = llm.bind(response_format={"type": "json_object"})
    user_content = json.dumps({
        "query": query,
        "detected_doc_id": detected_doc_id,
        "available_headers": available_headers,
    }, ensure_ascii=False)
    plan_response = llm_json.invoke([
        {"role": "system", "content": _DEEP_PLAN_SYSTEM},
        {"role": "user", "content": user_content},
    ])
    plan_text = getattr(plan_response, "content", "") or ""

    try:
        llm_output = json.loads(plan_text)
    except Exception:
        llm_output = {"reasoning": "", "decision": {"doc_id": detected_doc_id, "mode": "global", "plan": ["retrieval"]}}

    plan_obj = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "node": "planner_node",
        "status": "success",
        "input": {"query": query, "detected_doc_id": detected_doc_id},
        "context_retrieval": {"headers_fetch": headers_status, "available_headers": available_headers},
        "llm_planning_output": llm_output,
    }

    print(f"[Deep Agent 계획]\n{json.dumps(plan_obj, ensure_ascii=False, indent=2)}")
    return {"messages": [], "plan": json.dumps(plan_obj, ensure_ascii=False), "plan_output": plan_obj}


@_traceable(name="deep_orchestrator:llm", run_type="llm")
def _deep_orchestrator_llm_node(state: DeepOrchestratorState):
    """Deep Agent 오케스트레이터 LLM - 툴 호출 여부 및 다음 액션 결정"""
    llm = get_langchain_llm(model=state.get("worker_model") or "gpt-4o", temperature=0.0)
    tools = _get_deep_agent_tools()
    llm_with_tools = llm.bind_tools(tools)

    # 지금까지 실행된 tool call 횟수 카운트
    tool_call_count = sum(
        1 for m in state["messages"]
        if (getattr(m, "type", None) == "tool" or
            (isinstance(m, dict) and m.get("role") == "tool"))
    )

    system_content = _DEEP_ORCHESTRATOR_SYSTEM

    # plan이 있으면 decision 정보를 system prompt에 참고용으로 추가
    plan = state.get("plan")
    if plan:
        try:
            plan_obj = json.loads(plan)
            decision = plan_obj.get("llm_planning_output", {}).get("decision", {})
            reasoning = plan_obj.get("llm_planning_output", {}).get("reasoning", "")
            doc_id = decision.get("doc_id")
            mode = decision.get("mode")
            plan_items = decision.get("plan", [])
            hint_lines = []
            if reasoning:
                hint_lines.append(f"reasoning: {reasoning}")
            if doc_id:
                hint_lines.append(f"target_doc: {doc_id}")
            if mode == "section" and plan_items:
                hint_lines.append(f"focus_clauses: {', '.join(plan_items)}")
            elif mode == "global" and plan_items:
                hint_lines.append(f"suggested_tools: {', '.join(plan_items)}")
            if hint_lines:
                system_content += "\n\n[실행 계획 (참고용)]\n" + "\n".join(hint_lines)
        except Exception:
            pass

    # 툴 호출이 2회 이상이면 추가 검색 금지 메시지 삽입
    if tool_call_count >= 2:
        system_content += "\n\n[IMPORTANT] You have already called tools. Do NOT call any more tools. Generate the final answer now using the information already collected."

    full_messages = [{"role": "system", "content": system_content}] + list(state["messages"])
    response = llm_with_tools.invoke(full_messages)
    return {"messages": [response]}


@_traceable(name="deep_orchestrator:tools", run_type="tool")
def _deep_orchestrator_tools_node(state: DeepOrchestratorState):
    """Deep Agent 툴 실행 노드 - LLM이 요청한 서브에이전트 툴을 실행"""
    last_msg = state["messages"][-1]
    tool_calls = last_msg.tool_calls

    tool_map = {t.name: t for t in _get_deep_agent_tools()}
    results = []
    context_entries = []

    for tc in tool_calls:
        if isinstance(tc, dict):
            name, args, tid = tc.get("name"), tc.get("args", {}), tc.get("id")
        else:
            name, args, tid = tc.name, tc.args, tc.id

        print(f"[Deep Orchestrator] 툴 실행: {name}({args})")
        try:
            output = tool_map[name].invoke(args) if name in tool_map else f"알 수 없는 툴: {name}"
        except Exception as e:
            output = f"툴 실행 오류 ({name}): {e}"

        output_str = str(output)
        results.append({"role": "tool", "tool_call_id": tid, "content": output_str[:4000]})
        context_entries.append(output_str)

    return {"messages": results, "context": context_entries}


def _deep_agent_router(state: DeepOrchestratorState):
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return END


def create_deep_agent_workflow():
    """Deep Agent 워크플로우: plan → orchestrator(ReAct) ↔ tools"""
    workflow = StateGraph(DeepOrchestratorState)
    workflow.add_node("plan", _deep_orchestrator_plan_node)
    workflow.add_node("orchestrator", _deep_orchestrator_llm_node)
    workflow.add_node("tools", _deep_orchestrator_tools_node)
    workflow.add_edge(START, "plan")
    workflow.add_edge("plan", "orchestrator")
    workflow.add_conditional_edges(
        "orchestrator",
        _deep_agent_router,
        {"tools": "tools", END: END},
    )
    workflow.add_edge("tools", "orchestrator")
    return workflow.compile(name="Deep Orchestrator")


# ═══════════════════════════════════════════════════════════════════════════
# 실행 인터페이스
# ═══════════════════════════════════════════════════════════════════════════

_app = None

@_traceable(name="run_agent", run_type="chain")
def run_agent(query: str, session_id: str = "default", model_name: str = None, embedding_model: str = None, **kwargs):
    global _app
    if not _app:
        _app = create_deep_agent_workflow()

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

    # Deep Agent 초기 상태 (DeepOrchestratorState)
    initial_state = {
        "query": query,
        "messages": messages,
        "context": [],
        "worker_model": model_name or "gpt-4o",
    }

    # LangGraph 실행 (recursion_limit: 툴 호출 왕복 최대 횟수)
    # LangSmith: @traceable 컨텍스트를 LangGraph 자식 span으로 연결
    invoke_config: Dict[str, Any] = {"recursion_limit": 20, "run_name": f"deep_agent:{session_id}"}
    try:
        from langsmith.run_helpers import get_current_run_tree
        from langchain_core.tracers.langchain import LangChainTracer
        rt = get_current_run_tree()
        if rt:
            tracer = LangChainTracer(project_name=os.getenv("LANGCHAIN_PROJECT", "default"))
            tracer.parent_run_id = rt.id
            invoke_config["callbacks"] = [tracer]
    except Exception:
        pass
    result = _app.invoke(initial_state, config=invoke_config)

    # 툴 결과(context)를 그대로 answer_agent_node로 전달 (LLM 재작성 없이)
    context_list = result.get("context", [])
    result_messages = result.get("messages", [])

    # 툴 호출이 전혀 없었으면(chat 등) 마지막 AI 메시지를 fallback으로 사용
    if not context_list:
        for msg in reversed(result_messages):
            content = getattr(msg, "content", None) or (msg.get("content", "") if isinstance(msg, dict) else "")
            tool_calls = getattr(msg, "tool_calls", None) or (msg.get("tool_calls") if isinstance(msg, dict) else None)
            msg_type = getattr(msg, "type", "") or (msg.get("role", "") if isinstance(msg, dict) else "")
            if msg_type in ("ai", "assistant") and not tool_calls and content:
                context_list = [content]
                break

    raw_answer = "\n\n".join(context_list) if context_list else "답변을 생성하지 못했습니다."

    # answer_agent_node로 [USE:] 태그 처리 및 [참고 문서] 섹션 생성
    from backend.sub_agent.answer import answer_agent_node
    answer_result = answer_agent_node({"context": context_list or [raw_answer]})
    final_answer_msg = answer_result.get("messages", [{}])[-1]
    final_answer = (
        final_answer_msg.get("content", raw_answer)
        if isinstance(final_answer_msg, dict)
        else getattr(final_answer_msg, "content", raw_answer)
    )

    # ── 평가 로그 생성 ────────────────────────────────────────────────────────

    # tool 호출 로그 (messages에서 role=tool인 것들)
    tool_calls_log = []
    agent_tool_counts: Dict[str, int] = {}
    for msg in result_messages:
        role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else None)
        if role == "tool":
            content = getattr(msg, "content", None) or (msg.get("content", "") if isinstance(msg, dict) else "")
            tool_calls_log.append({"role": "tool", "content_preview": str(content)[:200]})
        # AI 메시지에서 어떤 툴을 몇 번 호출했는지 집계
        if role in ("ai", "assistant"):
            tcs = getattr(msg, "tool_calls", None) or (msg.get("tool_calls") if isinstance(msg, dict) else None) or []
            for tc in tcs:
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "unknown")
                agent_tool_counts[name] = agent_tool_counts.get(name, 0) + 1

    use_tags = re.findall(r'\[USE:\s*([^\|]+)\s*\|\s*([^\]]+)\]', final_answer)
    use_tag_count = len(use_tags)

    validation_summary = {
        "has_use_tags": use_tag_count > 0,
        "no_info_found": any(kw in final_answer for kw in [
            "검색된 문서 내에서 관련 정보를 찾을 수 없",
            "검색된 정보가 없습니다",
            "[NO_INFO_FOUND]",
        ]),
    }

    return {
        "answer": final_answer,
        "agent_log": {
            # 기본 정보
            "query": query,
            "context": raw_answer[:2000],
            "loop_count": len(tool_calls_log),
            "plan": (lambda p: json.loads(p) if p else [])(result.get("plan")),

            # 에이전트 호출 통계 (툴 이름 기준)
            "agent_calls": agent_tool_counts,
            "total_agent_calls": sum(agent_tool_counts.values()),

            # Tool 호출 정보
            "tool_calls_count": len(tool_calls_log),
            "tool_calls_log": tool_calls_log[:5],  # 최대 5개만

            # 태그 분석
            "use_tag_count": use_tag_count,
            "use_tags_sample": use_tags[:3] if use_tags else [],  # 샘플 3개

            # 검증 결과
            "validation_summary": validation_summary,

            # 재시도 현황 (Deep Agent는 LLM 자체 판단으로 재시도)
            "retry_per_agent": agent_tool_counts,
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
