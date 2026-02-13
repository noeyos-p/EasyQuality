"""
일반 대화 서브에이전트 모듈
- 문서 검색 없이 히스토리 기반의 일상 대화나 문맥 질문을 처리합니다.
- 오케스트레이터의 판단하에 'chat'으로 라우팅될 때 실행됩니다.
"""

from backend.agent import get_openai_client, AgentState

def chat_agent_node(state: AgentState):
    """[서브] 대화 에이전트 - 히스토리 기반 답변 생성"""
    query = state["query"]
    messages = state["messages"]
    
    # 시스템 프롬프트: 히스토리 기반 답변 유도
    # 시스템 프롬프트: 히스토리 기반 답변 유도
    system_instruction = """당신은 친절한 AI 어시스턴트입니다. 
    제공된 **대화 내역(History)**과 **[관련 과거 기억]**(시스템 메시지로 제공됨)을 바탕으로 사용자의 질문에 정확히 답변하세요.
    
    [핵심 역할]
    1. **메타 질문 처리**: "방금 내가 뭐라고 했어?", "이전 답변이 뭐였지?" 같은 과거 대화와 관련된 질문에 대해 History의 내용을 인용하여 정확히 답변하세요.
    2. **문맥 파악**: 사용자가 "그거", "저거" 등으로 이전 내용을 지칭하면, History에서 해당 내용을 찾아 답변하세요.
    3. **출처 언급**: 이전 답변이 특정 문서(SOP 등)를 인용했다면, 그 문서 정보도 함께 언급해주세요. (예: "방금 EQ-SOP-00001에 대해 질문하셨고, ~라고 답변드렸습니다.")
    
    [답변 형식]
    - 자연스러운 구어체(존댓말)로 답변하세요.
    - 답변 끝에 [DONE]을 붙여주세요. (오케스트레이터가 종료를 인식하기 위함)
    """
    
    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_instruction},
                *messages
            ],
            temperature=0.7
        )
        answer = response.choices[0].message.content
        
        # 보고서 형식으로 반환 (오케스트레이터가 재검토하도록)
        report = f"### [대화 에이전트 보고]\n{answer}"
        return {"context": [report]}
        
    except Exception as e:
        error_msg = f"대화 처리 중 오류가 발생했습니다: {str(e)} [DONE]"
        return {"context": [error_msg]}
