"""
Weaviate 벡터 스토어 - v8.0 (v4 Client 기반)
- Weaviate Python Client v4 적용
- 지능형 스키마 관리 및 고성능 Batch 지원
- gRPC 기반의 빠른 검색 구현
"""

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='weaviate')

import weaviate
import weaviate.classes as wvc
from weaviate.classes.query import MetadataQuery, Filter
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import hashlib
import torch
from transformers import AutoTokenizer, AutoModel
from dataclasses import dataclass
import re
import json
import os


# ═══════════════════════════════════════════════════════════════════════════
# 설정 상수
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_COLLECTION = "documents"
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "192.168.0.79")
WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT", "8080"))
WEAVIATE_SCHEME = os.getenv("WEAVIATE_SCHEME", "http")

# 검색 품질 설정
DEFAULT_SIMILARITY_THRESHOLD = 0.35
HIGH_CONFIDENCE_THRESHOLD = 0.65
MEDIUM_CONFIDENCE_THRESHOLD = 0.45
MIN_RESULTS_BEFORE_FILTER = 1

# 임베딩 모델 필터링 기준
MAX_EMBEDDING_DIM = 1024
MAX_MEMORY_MB = 1300

# 전역 캐시
_client: Optional[weaviate.WeaviateClient] = None
_embed_models: Dict = {}
_device: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════
# 검색 결과 데이터 클래스
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SearchResult:
    """검색 결과 단일 항목"""
    text: str
    similarity: float
    metadata: Dict
    id: str
    confidence: str

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "similarity": self.similarity,
            "metadata": self.metadata,
            "id": self.id,
            "confidence": self.confidence,
        }


@dataclass
class SearchResponse:
    """검색 응답 전체"""
    results: List[SearchResult]
    query: str
    total_found: int
    filtered_count: int
    quality_summary: Dict

    def to_dict(self) -> Dict:
        return {
            "results": [r.to_dict() for r in self.results],
            "query": self.query,
            "total_found": self.total_found,
            "filtered_count": self.filtered_count,
            "quality_summary": self.quality_summary,
        }


# ═══════════════════════════════════════════════════════════════════════════
# 유틸리티 함수
# ═══════════════════════════════════════════════════════════════════════════

def get_device() -> str:
    global _device
    if _device is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    return _device


def get_client() -> weaviate.WeaviateClient:
    """Weaviate v4 persistent client - ngrok 우선 (원격 팀원 지원)"""
    global _client
    if _client is None:
        # ngrok 설정 (환경변수에서 읽기)
        GRPC_HOST = os.getenv("WEAVIATE_GRPC_HOST", "0.tcp.jp.ngrok.io")
        GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "15979"))

        # 1. ngrok 터널 먼저 시도 (원격 접속 가능)
        if WEAVIATE_SCHEME == "https":
            try:
                from weaviate import WeaviateClient
                from weaviate.connect import ConnectionParams

                print(f" ngrok 터널 연결 시도 중...")
                print(f"   HTTP: {WEAVIATE_HOST}:{WEAVIATE_PORT}")
                print(f"   gRPC: {GRPC_HOST}:{GRPC_PORT}")

                connection_params = ConnectionParams.from_params(
                    http_host=WEAVIATE_HOST,
                    http_port=WEAVIATE_PORT,
                    http_secure=True,
                    grpc_host=GRPC_HOST,
                    grpc_port=GRPC_PORT,
                    grpc_secure=False
                )

                _client = WeaviateClient(
                    connection_params=connection_params,
                    skip_init_checks=True,
                    additional_config=wvc.init.AdditionalConfig(
                        timeout=wvc.init.Timeout(init=30, query=60, insert=120)
                    ),
                    additional_headers={
                        "ngrok-skip-browser-warning": "true"
                    }
                )
                _client.connect()
                _client.collections.list_all()
                print(f" Weaviate v4 연결 (ngrok)")
            except Exception as e:
                print(f" ngrok 연결 실패: {e}")
                raise
        else:
            # HTTP인 경우 로컬 연결
            try:
                _client = weaviate.connect_to_local(
                    host=WEAVIATE_HOST,
                    port=WEAVIATE_PORT,
                    grpc_port=50051,
                    additional_config=wvc.init.AdditionalConfig(
                        timeout=wvc.init.Timeout(init=10, query=60, insert=120)
                    )
                )
                _client.collections.list_all()
                print(f" Weaviate v4 로컬 연결 ({WEAVIATE_HOST}:{WEAVIATE_PORT})")
            except Exception as e:
                print(f" 연결 실패: {e}")
                raise

    if not _client.is_connected():
        _client.connect()

    return _client


def get_collection_name_for_model(base_name: str, model_name: str) -> str:
    """v4에서도 PascalCase 권장되므로 규칙 유지"""
    safe_base = re.sub(r'[^a-zA-Z0-9]', '', base_name)
    if not safe_base:
        safe_base = "Collection"
    
    model_hash = hashlib.md5(model_name.encode()).hexdigest()[:8]
    class_name = safe_base[0].upper() + safe_base[1:] + "V4" + model_hash
    return class_name


def calculate_confidence(similarity: float) -> str:
    """유사도 기반 신뢰도 계산"""
    if similarity >= HIGH_CONFIDENCE_THRESHOLD:
        return "high"
    elif similarity >= MEDIUM_CONFIDENCE_THRESHOLD:
        return "medium"
    return "low"


# ═══════════════════════════════════════════════════════════════════════════
# 임베딩 모델
# ═══════════════════════════════════════════════════════════════════════════

def get_embedding_model(model_name: str = "intfloat/multilingual-e5-small"):
    """임베딩 모델 로드"""
    global _embed_models
    if model_name in _embed_models:
        return _embed_models[model_name]

    print(f" Loading embedding model: {model_name}...")
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    model.eval()

    _embed_models[model_name] = (tokenizer, model)
    return tokenizer, model


def embed_text(text: str, model_name: str = "intfloat/multilingual-e5-small") -> List[float]:
    """텍스트 임베딩 (e5 prefix 지원)"""
    tokenizer, model = get_embedding_model(model_name)
    device = get_device()

    # e5 모델 권장 프리픽스 (검색 쿼리 시)
    # if "e5" in model_name.lower():
    #     text = f"query: {text}"

    if len(text) > 1500: text = text[:1500]

    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    mask = inputs['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
    sum_embeddings = torch.sum(outputs.last_hidden_state * mask, 1)
    sum_mask = torch.clamp(mask.sum(1), min=1e-9)
    embedding = (sum_embeddings / sum_mask).cpu().numpy()[0]
    return embedding.tolist()


# ═══════════════════════════════════════════════════════════════════════════
# 컬렉션 (v4) 관리
# ═══════════════════════════════════════════════════════════════════════════

def list_collections() -> List[str]:
    """v4: client.collections.list_all() 사용"""
    client = get_client()
    cols = client.collections.list_all()
    return list(cols.keys())


def get_collection_info(collection_name: str) -> Dict:
    """v4: 클러스터 통계 또는 aggregate 사용"""
    try:
        client = get_client()
        collection = client.collections.get(collection_name)
        res = collection.aggregate.over_all(total_count=True)
        return {"name": collection_name, "count": res.total_count}
    except Exception as e:
        return {"name": collection_name, "count": 0, "error": str(e)}


def delete_collection(collection_name: str) -> bool:
    """v4: client.collections.delete()"""
    try:
        client = get_client()
        client.collections.delete(collection_name)
        return True
    except Exception:
        return False


def close_client():
    """Weaviate 클라이언트 연결 종료"""
    global _client
    if _client is not None:
        try:
            _client.close()
            print(" Weaviate 연결 종료됨")
        except Exception as e:
            print(f" Weaviate 종료 중 오류: {e}")
        finally:
            _client = None


def ensure_collection(client: weaviate.WeaviateClient, collection_name: str):
    """v4: 스키마 생성 (Properties + Config)"""
    if not client.collections.exists(collection_name):
        client.collections.create(
            name=collection_name,
            properties=[
                wvc.config.Property(name="text", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="metadata_json", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="doc_name", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="doc_id", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="clause", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="title", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="model", data_type=wvc.config.DataType.TEXT),
            ]
        )
        print(f"Collection 생성: {collection_name}")


# ═══════════════════════════════════════════════════════════════════════════
# 데이터 생성/수정/삭제
# ═══════════════════════════════════════════════════════════════════════════

def add_documents(
    texts: List[str],
    metadatas: List[Dict],
    collection_name: str = DEFAULT_COLLECTION,
    model_name: str = "intfloat/multilingual-e5-small",
) -> Dict:
    """v4: collection.data.insert_many() 활용"""
    actual_name = get_collection_name_for_model(collection_name, model_name)
    client = get_client()
    ensure_collection(client, actual_name)
    
    collection = client.collections.get(actual_name)
    
    data_objects = []
    for i, text in enumerate(texts):
        meta = metadatas[i]
        vector = embed_text(text, model_name)
        
        data_objects.append(
            wvc.data.DataObject(
                properties={
                    "text": text,
                    "metadata_json": json.dumps(meta),
                    "doc_name": str(meta.get("doc_name", "")),
                    "doc_id": str(meta.get("doc_id", "")),
                    "clause": str(meta.get("clause_id") or meta.get("clause", "")), # 추가
                    "title": str(meta.get("title", "")), # 추가
                    "model": model_name
                },
                vector=vector
            )
        )
    
    # 배치 삽입
    res = collection.data.insert_many(data_objects)

    if res.has_errors:
        print(f"🔴 일부 데이터 삽입 실패: {res.errors}")

    return {
        "success": not res.has_errors,
        "added": len(texts) - len(res.errors),
        "collection": actual_name,
    }


def add_single_text(
    text: str,
    metadata: Dict,
    collection_name: str = DEFAULT_COLLECTION,
    model_name: str = "intfloat/multilingual-e5-small",
) -> Dict:
    """단일 텍스트 추가"""
    return add_documents([text], [metadata], collection_name, model_name)


def search(
    query: str,
    collection_name: str = DEFAULT_COLLECTION,
    n_results: int = 5,
    model_name: str = "intfloat/multilingual-e5-small",
    filter_doc: Optional[str] = None,
    similarity_threshold: Optional[float] = None,
    return_low_confidence: bool = False,
) -> List[Dict]:
    """v4: collection.query.near_vector() 활용"""
    actual_name = get_collection_name_for_model(collection_name, model_name)
    client = get_client()

    if not client.collections.exists(actual_name): return []
    
    vector = embed_text(query, model_name)
    collection = client.collections.get(actual_name)
    
    # 필터 구성 (doc_id 사용)
    filters = None
    if filter_doc:
        filters = Filter.by_property("doc_id").equal(filter_doc)
    
    # 쿼리 실행
    res = collection.query.near_vector(
        near_vector=vector,
        limit=max(n_results * 2, 10),
        filters=filters,
        return_metadata=MetadataQuery(certainty=True, distance=True)
    )

    search_results = []
    threshold = similarity_threshold or DEFAULT_SIMILARITY_THRESHOLD
    
    for obj in res.objects:
        # v4 certainty는 0~1 (Cosine 기준 1-distance/2 또는 유사)
        certainty = obj.metadata.certainty if obj.metadata.certainty is not None else 0.0
        similarity = certainty # Weaviate v4의 certainty는 이미 정규화되어 있음
        
        if not return_low_confidence and similarity < threshold: continue
        
        try:
            meta = json.loads(obj.properties.get('metadata_json', '{}'))
        except:
            meta = obj.properties

        top_doc_id = obj.properties.get('doc_id', '')
        if top_doc_id and not meta.get('doc_id'):
            meta['doc_id'] = top_doc_id

        search_results.append({
            "text": obj.properties.get('text', ""),
            "similarity": round(similarity, 4),
            "metadata": meta,
            "id": str(obj.uuid),
            "confidence": calculate_confidence(similarity),
        })

    # 최소 결과 보장 루틴
    if len(search_results) < MIN_RESULTS_BEFORE_FILTER and res.objects:
        search_results = []
        for obj in res.objects[:n_results]:
            similarity = obj.metadata.certainty or 0.0
            try: meta = json.loads(obj.properties.get('metadata_json', '{}'))
            except: meta = obj.properties
            top_doc_id = obj.properties.get('doc_id', '')
            if top_doc_id and not meta.get('doc_id'):
                meta['doc_id'] = top_doc_id
            search_results.append({
                "text": obj.properties.get('text', ""),
                "similarity": round(similarity, 4),
                "metadata": meta,
                "id": str(obj.uuid),
                "confidence": calculate_confidence(similarity),
            })

    return search_results[:n_results]


def search_hybrid(
    query: str,
    collection_name: str = DEFAULT_COLLECTION,
    n_results: int = 5,
    model_name: str = "intfloat/multilingual-e5-small",
    alpha: float = 0.5,
    filter_doc: Optional[str] = None,
) -> List[Dict]:
    """v4: collection.query.hybrid() 활용 (Hybrid Search)"""
    actual_name = get_collection_name_for_model(collection_name, model_name)
    client = get_client()

    if not client.collections.exists(actual_name): return []
    
    vector = embed_text(query, model_name)
    collection = client.collections.get(actual_name)
    
    # 필터 구성 (doc_id 사용)
    filters = None
    if filter_doc:
        filters = Filter.by_property("doc_id").equal(filter_doc)
    
    # 쿼리 실행
    try:
        res = collection.query.hybrid(
            query=query,
            vector=vector,
            alpha=alpha,
            limit=n_results,
            filters=filters,
            return_metadata=MetadataQuery(score=True, explain_score=True)
        )

        search_results = []
        for obj in res.objects:
            score = obj.metadata.score if obj.metadata.score is not None else 0.0

            try:
                meta = json.loads(obj.properties.get('metadata_json', '{}'))
            except:
                meta = obj.properties

            # top-level doc_id 프로퍼티를 metadata에 병합 (metadata_json에 없는 경우 대비)
            top_doc_id = obj.properties.get('doc_id', '')
            if top_doc_id and not meta.get('doc_id'):
                meta['doc_id'] = top_doc_id

            search_results.append({
                "text": obj.properties.get('text', ""),
                "similarity": round(score, 4),
                "metadata": meta,
                "id": str(obj.uuid),
                "confidence": "high" if score > 0.7 else "medium" if score > 0.4 else "low",
            })
        return search_results
    except Exception as e:
        print(f" Hybrid search failed: {e}")
        # Fallback to normal search
        return search(query, collection_name, n_results, model_name, filter_doc)


def search_advanced(
    query: str,
    collection_name: str = DEFAULT_COLLECTION,
    n_results: int = 5,
    model_name: str = "intfloat/multilingual-e5-small",
    filter_doc: Optional[str] = None,
    similarity_threshold: Optional[float] = None,
    return_low_confidence: bool = False,
) -> SearchResponse:
    """확장 검색 (SearchResponse 객체 반환)"""
    results = search(
        query, collection_name, n_results, model_name, 
        filter_doc, similarity_threshold, return_low_confidence
    )
    
    # 데이터 클래스 변환
    structured_results = [SearchResult(**r) for r in results]
    
    return SearchResponse(
        results=structured_results,
        query=query,
        total_found=len(structured_results),
        filtered_count=0,
        quality_summary={
            "high": len([r for r in structured_results if r.confidence == "high"]),
            "medium": len([r for r in structured_results if r.confidence == "medium"]),
            "low": len([r for r in structured_results if r.confidence == "low"]),
        }
    )


# ═══════════════════════════════════════════════════════════════════════════
# 호환성/관리 함수들
# ═══════════════════════════════════════════════════════════════════════════

def delete_by_doc_name(doc_name: str, collection_name: str = DEFAULT_COLLECTION, model_name: Optional[str] = None) -> Dict:
    """doc_id 또는 doc_name 프로퍼티로 벡터 삭제"""
    actual_classes = [get_collection_name_for_model(collection_name, model_name)] if model_name else [c for c in list_collections() if c.startswith(collection_name)]

    client = get_client()
    deleted_total = 0

    for cls in actual_classes:
        col = client.collections.get(cls)
        # 청크 메타데이터에는 doc_id 필드로 저장됨 → doc_id로 우선 삭제
        res = col.data.delete_many(where=Filter.by_property("doc_id").equal(doc_name))
        deleted_total += res.successful
        # doc_name 필드에도 혹시 있을 경우 추가 삭제
        res2 = col.data.delete_many(where=Filter.by_property("doc_name").equal(doc_name))
        deleted_total += res2.successful

    return {"success": deleted_total > 0, "deleted": deleted_total}


def delete_all(collection_name: str = DEFAULT_COLLECTION, model_name: Optional[str] = None) -> Dict:
    """전체 삭제"""
    actual_classes = [get_collection_name_for_model(collection_name, model_name)] if model_name else [c for c in list_collections() if c.startswith(collection_name)]
    deleted = []
    for cls in actual_classes:
        if delete_collection(cls): deleted.append(cls)
    return {"success": len(deleted) > 0, "deleted_collections": deleted}


def list_documents(collection_name: str = DEFAULT_COLLECTION, model_name: Optional[str] = None) -> List[Dict]:
    """문서 목록 조회 (v4 iterator 활용)"""
    # v4에서는 PascalCase를 사용하므로 대소문자 무시하고 매칭
    all_cols = list_collections()
    
    if model_name:
        actual_classes = [get_collection_name_for_model(collection_name, model_name)]
    else:
        # prefix 매칭 (대소문자 무시)
        prefix = collection_name.lower()
        actual_classes = [c for c in all_cols if c.lower().startswith(prefix)]
    
    docs = {}
    client = get_client()
    
    for cls in actual_classes:
        col = client.collections.get(cls)
        # 1000개만 샘플링하여 목록화
        for obj in col.iterator(include_vector=False):
            doc_name = obj.properties.get('doc_name', 'unknown')
            model = obj.properties.get('model', 'unknown')
            key = f"{doc_name}|{model}"
            
            if key not in docs:
                try: meta = json.loads(obj.properties.get('metadata_json', '{}'))
                except: meta = {}

                doc_title = meta.get('doc_title') or meta.get('title') or ''

                # doc_id 추출 (우선순위: metadata.doc_id > doc_name > doc_title에서 추출)
                doc_id = meta.get('doc_id') or doc_name

                # doc_id가 비어있거나 'unknown'이면 doc_title에서 추출 시도
                if not doc_id or doc_id == 'unknown':
                    import re
                    # EQ-SOP-00010, EQ-WI-00012 같은 패턴 추출
                    match = re.search(r'(EQ-[A-Z]+-\d+)', doc_title)
                    if match:
                        doc_id = match.group(1)
                    else:
                        doc_id = doc_title.split('_')[0] if '_' in doc_title else doc_title

                docs[key] = {
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "doc_title": doc_title,
                    "chunk_count": 0,
                    "model": model,
                    "collection": cls
                }
            docs[key]["chunk_count"] += 1
            if len(docs) > 500: break # 성능 방어
            
    return list(docs.values())


# search_advanced, search_with_context 등은 v3와 로직이 유사하므로 생략하거나 
# search()를 내부적으로 활용하도록 유지하면 됩니다.
# ═══════════════════════════════════════════════════════════════════════════

def search_with_context(query: str, collection_name: str = DEFAULT_COLLECTION, n_results: int = 3, model_name: str = "intfloat/multilingual-e5-small", filter_doc: Optional[str] = None, similarity_threshold: Optional[float] = None) -> Tuple[List[Dict], str]:
    results = search(query, collection_name, n_results, model_name, filter_doc, similarity_threshold)
    context_parts = []
    for i, r in enumerate(results):
        meta = r.get('metadata', {})
        header = f"[{meta.get('doc_name', 'Unknown')} > {meta.get('title', 'No Title')}] (유사도: {r['similarity']:.1%})"
        context_parts.append(f"{header}\n{r['text']}")
    return results, "\n\n---\n\n".join(context_parts)

EMBEDDING_MODEL_SPECS = {
    "intfloat/multilingual-e5-small": {
        "name": "multilingual-e5-small", "dim": 384, "memory_mb": 120, "lang": "multi",
    },
}


def filter_compatible_models() -> List[Dict]:
    """호환 가능한 모델 목록"""
    return [
        {"path": path, **spec}
        for path, spec in EMBEDDING_MODEL_SPECS.items()
        if spec['dim'] <= MAX_EMBEDDING_DIM and spec['memory_mb'] <= MAX_MEMORY_MB
    ]
def get_embedding_model_info(model_path: str) -> Dict:
    """모델 정보 조회"""
    return EMBEDDING_MODEL_SPECS.get(model_path, {"name": "unknown", "dim": 0})

def is_model_compatible(model_path: str) -> bool:
    """모델이 호환되는지 확인"""
    spec = EMBEDDING_MODEL_SPECS.get(model_path)
    if not spec: return False
    return spec['dim'] <= MAX_EMBEDDING_DIM and spec['memory_mb'] <= MAX_MEMORY_MB
