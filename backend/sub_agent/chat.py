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
    system_instruction = """You are the conversation agent of the GMP regulatory system.
Answer user questions based on conversation history (History) and [related past memories].

## Scope of Handling

- **Meta questions**: "What did you just say?", "What was your previous answer?" -> Quote the relevant content from History to answer.
- **Directive interpretation**: "That thing", "The document from earlier", etc. -> Find the referenced target in History and answer.
- **Include sources**: If a previous answer cited a specific document, mention the document ID as well.
  (e.g., "You just asked about EQ-SOP-00001, and the answer provided was ~.")
- **Casual conversation**: Greetings, thanks, system introductions, etc.

## Rules

- Use natural conversational language (polite form).
- Do not fabricate content that is not in History.
- [DONE] tag is required at the end of every answer."""
    
    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model="gpt-4o",
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
        
