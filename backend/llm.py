"""
LLM 모듈 v7.0 - Z.AI + Ollama + HuggingFace
"""

import os
import re
import torch
import requests
import time
import random
from typing import Dict, List, Optional, Any


device = "cuda" if torch.cuda.is_available() else "cpu"
_loaded_llm: Dict[str, Any] = {}

# ═══════════════════════════════════════════════════════════════════════════
# OpenAI 백엔드
# ═══════════════════════════════════════════════════════════════════════════

class OpenAILLM:
    """OpenAI API"""

    def __init__(self, model: str = "gpt-4o", api_key: str = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self._client = None
    
    def _get_client(self):
        """OpenAI Client 지연 로딩"""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai 패키지가 필요합니다: pip install openai")
        return self._client

    def generate(
        self,
        prompt: str,
        system: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> str:
        """OpenAI API를 사용하여 응답 생성"""
        if not self.api_key:
            print(" OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
            return " 오류: OPENAI_API_KEY가 설정되지 않았습니다."

        client = self._get_client()
        
        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            # print(f" OpenAI API 호출 중... (모델: {self.model})")
            
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response.choices[0].message.content or ""

        except Exception as e:
            print(f" OpenAI 호출 오류: {e}")
            return f" OpenAI 호출 오류: {str(e)}"

    def generate_stream(
        self,
        prompt: str,
        system: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ):
        """스트리밍 생성"""
        client = self._get_client()
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        stream = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    @staticmethod
    def is_available() -> bool:
        return bool(os.getenv("OPENAI_API_KEY"))


# ═══════════════════════════════════════════════════════════════════════════
# Z.AI 백엔드 (GLM-4.7-Flash)
# ═══════════════════════════════════════════════════════════════════════════

class ZaiLLM:
    """Z.AI GLM-4.7 API"""

    def __init__(self, model: str = "glm-4.7-flash", api_key: str = None):
        self.model = model
        self.api_key = api_key or os.getenv("ZAI_API_KEY", "")
        self._client = None
    
    def _get_client(self):
        """ZaiClient 지연 로딩"""
        if self._client is None:
            try:
                from zai import ZaiClient
                self._client = ZaiClient(api_key=self.api_key)
            except ImportError:
                raise ImportError("zai-sdk 패키지가 필요합니다: pip install zai-sdk")
        return self._client

    def generate(
        self,
        prompt: str,
        system: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2048  #  기본 토큰 상향
    ) -> str:
        """Z.AI API를 사용하여 응답 생성 (재시도 로직 포함)"""
        # API 키 확인
        if not self.api_key or "your-api-key" in self.api_key:
            print(" ZAI_API_KEY가 설정되지 않았거나 기본값입니다. .env 파일을 확인하세요.")
            return " 오류: ZAI_API_KEY가 설정되지 않았습니다. .env 파일에 실제 API 키를 입력해주세요."

        client = self._get_client()
        
        max_retries = 3
        base_delay = 2  # 초
        
        for attempt in range(max_retries):
            try:
                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": prompt})

                print(f" Z.AI API 호출 중... (모델: {self.model}, MaxTokens: {max_tokens}, 시도: {attempt+1})")
                
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                if not response.choices:
                    print(f" Z.AI 응답에 choices가 없습니다: {response}")
                    return " 오류: Z.AI로부터 적절한 응답을 받지 못했습니다."
                    
                msg_obj = response.choices[0].message
                content = getattr(msg_obj, 'content', "") or ""
                reasoning = getattr(msg_obj, 'reasoning_content', "") or ""
                
                # 본문(content)이 비어있는데 reasoning_content만 있는 경우
                if not content and reasoning:
                    #  방법 1: 완전한 JSON 블록 찾기 ({ ... })
                    json_match = re.search(r'(\{.*\})', reasoning, re.DOTALL)
                    if json_match:
                        content = json_match.group(1)
                        print(f" [Recall] Reasoning에서 완전한 JSON 복구 ({len(content)}자)")
                    else:
                        #  방법 2: 잘린 JSON이라도 시작 부분이라도 찾기
                        start_idx = reasoning.find('{')
                        if start_idx != -1:
                            content = reasoning[start_idx:]
                            if not content.strip().endswith('}'):
                                content = content.strip() + '"}'
                            print(f" [Recall] 잘린 JSON 강제 복구 시도")
                        else:
                            content = " 답변 생성 중 토큰 한도에 도달했습니다."
                
                return content

            except Exception as e:
                error_msg = str(e)
                # 할당량 초과(429) 또는 높은 동시성 오류(1302) 시 재시도
                if "429" in error_msg or "1302" in error_msg or "Rate limit" in error_msg or "too many" in error_msg.lower():
                    # 지수 백오프 + 지터 (서버 부하 분산)
                    delay = (base_delay * (2 ** attempt)) + random.uniform(1, 3)
                    print(f" API 한도 초과/과부하 감지. {delay:.1f}초 후 자동 재시도... ({attempt+1}/{max_retries})")
                    time.sleep(delay)
                    continue
                else:
                    print(f" Z.AI 호출 오류: {e}")
                    return f" AI 호출 오류: {str(e)}"
        
        return " 오류: 여러 번의 재시도 후에도 AI 응답을 받지 못했습니다. (API 할당량 초과)"

    def generate_stream(
        self,
        prompt: str,
        system: str = None,
        temperature: float = 0.7,
        max_tokens: int = 512
    ):
        """스트리밍 생성"""
        client = self._get_client()
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    @staticmethod
    def is_available() -> bool:
        """API 키 설정 여부"""
        return bool(os.getenv("ZAI_API_KEY"))


# ═══════════════════════════════════════════════════════════════════════════
# Ollama 백엔드
# ═══════════════════════════════════════════════════════════════════════════

class OllamaLLM:
    """Ollama 네이티브 API"""

    def __init__(self, model: str = "qwen2.5:3b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def generate(
        self,
        prompt: str,
        system: str = None,
        temperature: float = 0.1,
        max_tokens: int = 256
    ) -> str:
        """텍스트 생성"""
        try:
            return self._call_chat_api(prompt, system, temperature, max_tokens)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return self._call_generate_api(prompt, system, temperature, max_tokens)
            raise
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Ollama 서버 연결 실패. 'ollama serve' 실행 필요")

    def _call_chat_api(
        self,
        prompt: str,
        system: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """/api/chat 엔드포인트"""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})

        # Qwen3 thinking 모드 비활성화
        final_prompt = f"/no_think {prompt}" if "qwen3" in self.model.lower() else prompt
        messages.append({"role": "user", "content": final_prompt})

        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
                "think": False,
            },
            timeout=120
        )
        response.raise_for_status()
        data = response.json()

        message = data.get("message", {})
        content = message.get("content", "") if isinstance(message, dict) else str(message)

        # thinking fallback
        if not content and isinstance(message, dict) and message.get("thinking"):
            content = message.get("thinking", "")

        return content

    def _call_generate_api(
        self,
        prompt: str,
        system: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """/api/generate 엔드포인트 (구버전)"""
        full_prompt = f"{system}\n\n{prompt}" if system else prompt

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json().get("response", "")

    @staticmethod
    def list_models(base_url: str = "http://localhost:11434") -> List[str]:
        """사용 가능한 모델 목록"""
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            if response.ok:
                return [m["name"] for m in response.json().get("models", [])]
        except Exception:
            pass
        return []

    @staticmethod
    def is_available(base_url: str = "http://localhost:11434") -> bool:
        """서버 실행 여부"""
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=3)
            return response.ok
        except Exception:
            return False


# ═══════════════════════════════════════════════════════════════════════════
# HuggingFace 백엔드
# ═══════════════════════════════════════════════════════════════════════════

def load_llm(model_name: str):
    """HuggingFace LLM 로드"""
    if model_name in _loaded_llm:
        return _loaded_llm[model_name]

    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f" Loading LLM: {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    dtype = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None
    )

    if device == "cpu":
        model = model.to(device)
    model.eval()

    _loaded_llm[model_name] = (tokenizer, model)
    print(f" Loaded: {model_name}")
    return tokenizer, model


def generate_with_hf(
    prompt: str,
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    max_new_tokens: int = 256,
    temperature: float = 0.1
) -> str:
    """HuggingFace 생성"""
    tokenizer, model = load_llm(model_name)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            repetition_penalty=1.15,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )

    return tokenizer.decode(
        outputs[0][len(inputs["input_ids"][0]):],
        skip_special_tokens=True
    ).strip()


# ═══════════════════════════════════════════════════════════════════════════
# 통합 인터페이스
# ═══════════════════════════════════════════════════════════════════════════

def get_llm_response(
    prompt: str,
    llm_model: str = "gpt-4o", # 기본값: OpenAI
    llm_backend: str = "openai", # openai | zai | ollama
    max_tokens: int = 1024,
    temperature: float = 0.7
) -> str:
    """통합 LLM 응답 (Hybrid)
    - 메타데이터 추출 등에는 기본적으로 OpenAI 사용
    - 테스트 시 llm_backend="zai" 로 변경 가능
    """
    if llm_backend == "zai":
        llm = ZaiLLM(llm_model if llm_model.startswith("glm") else "glm-4.7-flash")
        return llm.generate(prompt, temperature=temperature, max_tokens=max_tokens)
    elif llm_backend == "ollama":
        llm = OllamaLLM(llm_model)
        return llm.generate(prompt, temperature=temperature, max_tokens=max_tokens)
    elif llm_backend == "openai":
         # OpenAI는 별도 클래스 없이 간단하게 처리하거나 OpenAILLM 클래스 활용
        llm = OpenAILLM(llm_model if not llm_model.startswith("glm") else "gpt-4o")
        return llm.generate(prompt, temperature=temperature, max_tokens=max_tokens)
    else:
        return generate_with_hf(prompt, llm_model, max_tokens, temperature)


# ═══════════════════════════════════════════════════════════════════════════
# 검색 결과 분석 (에이전트용)
# ═══════════════════════════════════════════════════════════════════════════

def analyze_search_results(results: List[Dict]) -> Dict:
    """검색 결과 분석 - 되묻기 여부 판단"""
    if not results:
        return {'needs_clarification': False, 'options': [], 'unique_documents': []}

    doc_groups = {}

    for r in results:
        meta = r.get('metadata', {})
        doc_name = meta.get('doc_name', 'unknown')
        doc_title = meta.get('doc_title', doc_name)
        article_num = meta.get('article_num')
        article_type = meta.get('article_type', 'article')
        section = meta.get('section')  # 제N조 형식
        score = r.get('similarity', 0)

        # 섹션 표시
        section_display = section or ""
        if not section_display and article_num:
            if article_type == 'article':
                section_display = f"제{article_num}조"
            elif article_type == 'chapter':
                section_display = f"제{article_num}장"
            elif article_type == 'section':
                section_display = f"제{article_num}절"
            else:
                section_display = str(article_num)

        if doc_name not in doc_groups:
            doc_groups[doc_name] = {
                'title': doc_title,
                'max_score': score,
                'sections': {section_display} if section_display else set(),
                'count': 1
            }
        else:
            doc_groups[doc_name]['max_score'] = max(doc_groups[doc_name]['max_score'], score)
            if section_display:
                doc_groups[doc_name]['sections'].add(section_display)
            doc_groups[doc_name]['count'] += 1

    unique_docs = list(doc_groups.keys())

    # 되묻기 판별: 여러 문서에서 비슷한 점수
    needs_clarification = False
    if len(unique_docs) > 1:
        scores = sorted([info['max_score'] for info in doc_groups.values()], reverse=True)
        if len(scores) >= 2 and (scores[0] - scores[1]) < 0.15:
            needs_clarification = True

    # 선택지
    options = []
    for d_name in unique_docs:
        info = doc_groups[d_name]
        sections_list = sorted(list(info['sections']))[:3]
        sections_str = f" ({', '.join(sections_list)})" if sections_list else ""

        options.append({
            "doc_name": d_name,
            "doc_title": info['title'],
            "display_text": f"{info['title']}{sections_str}",
            "score": info['max_score'],
            "sections": sections_list,
        })

    return {
        'needs_clarification': needs_clarification,
        'options': options,
        'unique_documents': unique_docs
    }


def generate_clarification_question(
    query: str,
    options: List[Dict],
    llm_model: str = "qwen2.5:3b",
    llm_backend: str = "ollama"
) -> str:
    """되묻기 질문 생성"""
    options_text = "\n".join([
        f"- {opt['display_text']} (관련도: {opt['score']:.0%})"
        for opt in options
    ])

    prompt = f"""사용자가 "{query}"에 대해 질문했습니다.
관련 문서들:
{options_text}

어떤 문서의 내용을 바탕으로 답변할지 정중하게 물어보세요.
한국어로 짧게 응답하세요."""

    try:
        return get_llm_response(prompt, llm_model, llm_backend, max_tokens=200)
    except Exception:
        return f"'{query}'에 대해 여러 규정(SOP)이 검색되었습니다. 어떤 문서를 확인할까요?\n\n{options_text}"


# ═══════════════════════════════════════════════════════════════════════════
# 모델 프리셋
# ═══════════════════════════════════════════════════════════════════════════

OLLAMA_MODELS = [
    {"key": "qwen2.5:0.5b", "name": "Qwen2.5-0.5B", "desc": "초경량 (1GB)", "vram": "1GB"},
    {"key": "qwen2.5:1.5b", "name": "Qwen2.5-1.5B", "desc": "경량 (2GB)", "vram": "2GB"},
    {"key": "qwen2.5:3b", "name": "Qwen2.5-3B", "desc": "추천 (3GB)", "vram": "3GB"},
    {"key": "qwen2.5:7b", "name": "Qwen2.5-7B", "desc": "고성능 (5GB)", "vram": "5GB"},
    {"key": "qwen3:4b", "name": "Qwen3-4B", "desc": "최신 (4GB)", "vram": "4GB"},
    {"key": "llama3.2:3b", "name": "Llama3.2-3B", "desc": "경량 (3GB)", "vram": "3GB"},
    {"key": "gemma2:2b", "name": "Gemma2-2B", "desc": "경량 (2GB)", "vram": "2GB"},
    {"key": "mistral:7b", "name": "Mistral-7B", "desc": "영어 특화 (5GB)", "vram": "5GB"},
]

HUGGINGFACE_MODELS = [
    {"key": "Qwen/Qwen2.5-0.5B-Instruct", "name": "Qwen2.5-0.5B", "desc": "초경량"},
    {"key": "Qwen/Qwen2.5-1.5B-Instruct", "name": "Qwen2.5-1.5B", "desc": "경량"},
    {"key": "Qwen/Qwen2.5-3B-Instruct", "name": "Qwen2.5-3B", "desc": "VRAM 6GB+"},
    {"key": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "name": "TinyLlama", "desc": "영어"},
]
