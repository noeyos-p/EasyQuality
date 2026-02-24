"""
비교 에이전트 (Comparison Agent) - ReAct Version
- 사용자의 질문에 따라 도구(버전 이력 조회, 버전 비교 등)를 사용하여 답변을 생성합니다.
- LangGraph의 ReAct 아키텍처를 사용하여 자율적으로 도구 호출 여부를 결정합니다.
"""

import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict, Literal

from backend.agent import (
    AgentState,
    get_langchain_llm,
    get_version_history_tool,
    compare_versions_tool,
    get_references_tool,
    normalize_doc_id,
    safe_json_loads
)
from langgraph.graph import StateGraph, START, END
from langsmith import traceable

# ═══════════════════════════════════════════════════════════════════════════
# 에이전트 상태 정의
# ═══════════════════════════════════════════════════════════════════════════

class CompareAgentState(TypedDict):
    query: str
    messages: Annotated[List[Any], operator.add]
    model: str
    context: List[str]
    doc_id: Optional[str]

# ═══════════════════════════════════════════════════════════════════════════
# 노드 정의
# ═══════════════════════════════════════════════════════════════════════════

def agent_node(state: CompareAgentState):
    """LLM이 도구 호출 여부를 결정하거나 최종 답변을 생성하는 노드"""
    llm = get_langchain_llm(model=state["model"], temperature=0.0)
    
    # 도구 바인딩
    tools = [get_version_history_tool, compare_versions_tool, get_references_tool]
    llm_with_tools = llm.bind_tools(tools)
    
    system_prompt = f"""당신은 GMP 규정 문서의 버전 비교 및 이력 분석 전문가입니다.
제공된 도구를 사용하여 사용자의 질문에 정확하고 전문적으로 답변하세요.
{f"**우선 조사 대상 문서**: {state.get('detected_doc_id')}" if state.get('detected_doc_id') else ""}

## 행동 지침
1. **버전 목록 조회**: `get_version_history_tool`을 사용하여 문서의 전체 버전 번호와 날짜를 확인하세요.
2. **변경 사항 비교**: 
   - `compare_versions_tool`을 사용하여 두 버전 간의 조항별 차이점(ADDED, DELETED, MODIFIED)을 분석하세요.
   - 버전이 명시되지 않았다면 히스토리를 먼저 조회한 뒤 최신 2개 버전을 비교하세요.
3. **영향 분석**: `get_references_tool`을 사용하여 변경된 문서가 참조하거나 인용하는 다른 문서(영향 범위)를 파악하세요.

## 보고서 작성 지침
1. **추가/삭제 우선 명시**: `[상세 비교]` 섹션의 가장 상단에 새롭게 추가되거나 삭제된 조항(`[상태: ADDED]` 또는 `[상태: DELETED]`)이 있다면, 해당 정보를 가장 먼저 기술하세요.
2. **변경 유의미성 판단 및 요약 전략**:
    - **단순 자구 수정/오타 교정**: 변경 내용이 규정의 의미나 절차에 유의미한 변화를 주지 않는 단순한 경우에는 상세 비교 대신 '담백하게 바뀐 핵심 부분'만 1문장으로 간략하게 요약하세요.
    - **실질적 변경**: 정책 변경, 조건 추가/완화, 책임 소재 변경 등 유의미한 변화가 있는 경우에는 아래 가이드라인(기존 방식)에 따라 상세하게 비교 분석하세요.
3. **사실 근거**: 반드시 [변경 조항 데이터 (Diff)]에 제공된 원문 정보를 바탕으로 작성하세요.
4. **요약 섹션**: '1. 변경 핵심 요약'에는 전체 변경 사항의 취지와 핵심 내용을 전문적인 용어를 사용하여 2~3문장으로 명확하게 요약하세요.
5. **상태 기반 요약**: '2. 상세 비교' 섹션에서는 제공된 원문의 핵심 내용을 파악하여 요약된 형태로 기술하세요. [변경 전:]과 [변경 후:] 뒤에는 단순 원문 복사가 아닌, 해당 조항에서 무엇이 어떻게 달라졌는지 핵심을 요약하여 작성하세요.
6. **영향 평가**: '3. 영향 평가' 섹션에는 제공된 영향 분석 데이터를 바탕으로, 이 변경이 다른 문서나 프로세스에 미칠 구체적인 파급 효과를 분석하여 기술하세요.
7. **언어**: 모든 내용은 반드시 한국어로 작성하세요.
8. **가독성 및 레이아웃 (필수)**:
    - 모든 섹션 헤더는 대괄호(`[]`)를 사용하세요. (예: `[상세 비교]`, `[영향 평가]`)
    - `[상세 비교]` 섹션 내의 조항들은 반드시 `1. 조항 X.X`, `2. 조항 Y.Y`와 같이 번호를 매기세요.
    - 각 조항 하위의 `-변경 전:`과 `-변경 후:`는 반드시 **새로운 줄**에서 시작하고, 앞에 하이픈(`-`)과 들여쓰기를 적용하세요.
    - **색상 적용 (필수)**: 
        - '변경 전'이라는 3글자는 반드시 `<span style="color: #ff4d4e">변경 전</span>`과 같이 HTML 태그를 사용하여 빨간색으로 표기하세요.
        - '변경 후'이라는 3글자는 반드시 `<span style="color: #2db7f5">변경 후</span>`와 같이 HTML 태그를 사용하여 파란색(청록색)으로 표기하세요.
        - **중요**: 전체 문장이 아닌 딱 이 '3글자'에만 색상을 적용해야 합니다.
    - **줄바꿈 고도화**:
        - `-변경 전:` 내용이 끝난 후, `-변경 후:`가 시작되기 전에 **최소 1줄 이상의 빈 줄**을 추가하여 두 블록이 절대 붙어있지 않게 하세요.
9. **종료 태그**: 답변의 가장 마지막 줄에는 반드시 `[DONE]` 태그만을 포함하세요.

## 출력 형식 (반드시 준수)
[변경 핵심 요약]
-(전체 내용을 관통하는 전문적인 요약 2~3문장)

[상세 비교]
1. 조항 4.1
  -<span style="color: #ff4d4e">변경 전</span>: (이전 내용 요약)

  -<span style="color: #2db7f5">변경 후</span>: (이후 내용 요약)

2. 조항 4.2
  -<span style="color: #ff4d4e">변경 전</span>: ...

  -<span style="color: #2db7f5">변경 후</span>: ...

[영향 평가]
-(영향을 받는 문서명과 구체적인 영향 사유 기술)

[참고 문서]
[USE: 문서ID | 조항/버전] 형식으로 나열

[DONE]
\"\"\"
    
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    response = llm_with_tools.invoke(messages)
    
    return {"messages": [response]}

def tool_node(state: CompareAgentState):
    """도구를 실행하는 노드"""
    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls
    
    tool_map = {
        "get_version_history_tool": get_version_history_tool,
        "compare_versions_tool": compare_versions_tool,
        "get_references_tool": get_references_tool
    }
    
    outputs = []
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]
        
        print(f"    [Comparison Tool] 호출: {tool_name}({tool_args})")
        
        if tool_name in tool_map:
            result = tool_map[tool_name].invoke(tool_args)
            outputs.append({
                "role": "tool",
                "tool_call_id": tool_id,
                "content": str(result)
            })
        else:
            outputs.append({
                "role": "tool",
                "tool_call_id": tool_id,
                "content": f"Error: Tool {tool_name} not found."
            })
            
    return {"messages": outputs}

# ═══════════════════════════════════════════════════════════════════════════
# 그래프 구성
# ═══════════════════════════════════════════════════════════════════════════

def create_compare_graph():
    workflow = StateGraph(CompareAgentState)
    
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    
    workflow.add_edge(START, "agent")
    
    def should_continue(state: CompareAgentState) -> Literal["tools", END]:
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "tools"
        return END
        
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()

# 전역 그래프 인스턴스 (캐싱)
_compare_graph = None

@traceable(name="comparison_agent", run_type="chain")
def comparison_agent_node(state: AgentState):
    """[서브] 비교 에이전트 메인 엔트리 포인트"""
    global _compare_graph
    if _compare_graph is None:
        _compare_graph = create_compare_graph()
        
    query = state["query"]
    model = state.get("worker_model") or state.get("model_name") or "gpt-4o"
    doc_id = normalize_doc_id(query)
    
    print(f"[COMPARISON AGENT] ReAct 가동! Query: {query}")
    
    # Critic 피드백 합산
    critique_feedback = state.get("critique_feedback")
    initial_user_msg = query
    if critique_feedback:
        initial_user_msg += f"\n\n[Orchestrator Feedback] {critique_feedback}"
        print(f"    [Comparison] 피드백 적용: {critique_feedback}")

    initial_state = {
        "query": query,
        "messages": [{"role": "user", "content": initial_user_msg}],
        "model": model,
        "context": [],
        "doc_id": doc_id
    }
    
    # 그래프 실행 (최대 10회 루프 제한)
    final_state = _compare_graph.invoke(initial_state, config={"recursion_limit": 15})
    
    # 최종 답변 추출
    last_ai_message = ""
    for msg in reversed(final_state["messages"]):
        if hasattr(msg, "content") and msg.content and not getattr(msg, "tool_calls", None):
            last_ai_message = msg.content
            break
            
    if not last_ai_message:
        last_ai_message = "비교 데이터를 분석하는 데 실패했습니다."

    return {
        "context": [f"### [비교 에이전트 조사 보고]\n{last_ai_message}"],
        "last_agent": "comparison"
    }
