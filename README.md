# Easy Quality

GMP/SOP 문서 관리 및 분석을 위한 RAG 기반 AI 챗봇 플랫폼. 문서 업로드, 시맨틱 검색, AI 에이전트 기반 비교 분석, 지식 그래프 시각화 기능을 제공합니다.

---

## 주요 기능

- **RAG 기반 Q&A**: 업로드된 문서를 기반으로 한 맥락 인식 질의응답
- **AI 에이전트 시스템**: LangGraph 기반 복합 작업 수행 (비교, 영향 분석, 그래프 생성)
- **문서 관리**: PDF, DOCX, MD, TXT 문서 업로드 및 버전 관리
- **문서 비교**: 문서 버전 간 차이 분석 및 변경 이력 추적
- **지식 그래프**: Neo4j 기반 문서 관계 시각화 (Mermaid, Force-Graph)
- **OnlyOffice 연동**: 브라우저 내 문서 편집
- **품질 평가**: LLM-as-Judge 기반 답변 품질 평가
- **사용자 인증**: JWT 기반 로그인/회원가입

---

## 기술 스택

### Frontend
| 항목 | 기술 |
|------|------|
| Framework | React 19, TypeScript |
| Build Tool | Vite |
| Styling | Tailwind CSS |
| Visualization | React Force Graph 2D, Mermaid |
| Markdown | React-Markdown (GFM, Rehype Raw) |

### Backend
| 항목 | 기술 |
|------|------|
| Framework | FastAPI (Python 3.11) |
| LLM | Anthropic Claude, OpenAI GPT-4o |
| Agent | LangChain + LangGraph |
| Embeddings | Sentence-Transformers (multilingual-e5-small) |
| Auth | JWT (python-jose), bcrypt |

### Database & Storage
| 항목 | 기술 |
|------|------|
| Vector Store | Weaviate 1.27.0 |
| Graph DB | Neo4j (Aura) |
| RDB | PostgreSQL (Neon) + pgvector |
| File Storage | AWS S3 |
| Document Server | OnlyOffice DocumentServer |

### Infrastructure
| 항목 | 기술 |
|------|------|
| Container | Docker, Docker Compose |
| Cloud | AWS EC2 |
| Registry | Docker Hub |
| Monitoring | LangSmith |

---

## 프로젝트 구조

```
easy_quality/
├── main.py                # FastAPI 앱 엔트리포인트
├── requirements.txt       # Python 의존성
├── docker-compose.yml     # 로컬 개발 환경
├── docker-compose.prod.yml # 프로덕션 환경
├── Dockerfile             # Backend 이미지
├── backend/               # Python 백엔드 패키지
└── frontend/              # React + TypeScript 프론트엔드
```

### Backend (`backend/`)
```
├── app_state.py           # 공유 상태, 상수
├── agent.py               # 에이전트 오케스트레이션
├── llm.py                 # LLM 통합 (Anthropic, OpenAI, Ollama)
├── vector_store.py        # Weaviate 연동
├── graph_store.py         # Neo4j 연동
├── sql_store.py           # PostgreSQL 연동
├── evaluation.py          # LLM-as-Judge 평가
├── document_pipeline.py   # 문서 처리 및 청킹
├── s3_store.py            # AWS S3 연동
├── onlyoffice_service.py  # OnlyOffice 연동
├── routers/               # API 엔드포인트
│   ├── chat.py            # 채팅 및 큐 워커
│   ├── documents.py       # 문서 업로드/다운로드/삭제
│   ├── graph.py           # 그래프 시각화
│   ├── auth.py            # 인증 (로그인/회원가입)
│   ├── agent_router.py    # 에이전트 채팅
│   └── onlyoffice.py      # OnlyOffice 콜백
└── sub_agent/             # 특화 서브 에이전트
    ├── answer.py          # 답변 생성
    ├── compare.py         # 문서 비교
    ├── graph.py           # 그래프 생성
    └── search.py          # ReAct 기반 고급 검색
```

### Frontend (`frontend/src/`)
```
├── components/
│   ├── chat/              # 채팅 인터페이스
│   ├── document/          # 문서 관리 및 뷰어
│   ├── graph/             # 그래프 시각화
│   ├── history/           # 변경 이력 및 Diff
│   ├── auth/              # 인증 모달
│   └── layout/            # Sidebar, 알림, Toast
├── hooks/
│   └── useAuth.ts         # 인증 훅
└── types.ts               # TypeScript 타입 정의
```

---

## 로컬 개발 환경 설정

### 사전 요구사항
- Python 3.11+
- Node.js 18+
- Docker & Docker Compose

### 1. Docker Compose로 실행 (권장)

```bash
docker-compose up -d
```

> - Frontend: `http://localhost`
> - Backend API: `http://localhost:8000`
> - API Docs: `http://localhost:8000/docs`
> - Weaviate: `http://localhost:8079`
> - OnlyOffice: `http://localhost:8080`

### 2. Backend 수동 실행

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python main.py
```

> Backend 서버가 `http://localhost:8000` 에서 실행됩니다.

### 3. Frontend 실행

```bash
cd frontend
npm install
npm run dev
```

> 개발 서버가 `http://localhost:3000` 에서 실행됩니다.

---

## 환경 변수

### LLM API 키 (`.env`)
| 변수 | 설명 |
|------|------|
| `ANTHROPIC_API_KEY` | Anthropic Claude API 키 |
| `OPENAI_API_KEY` | OpenAI API 키 |

### LangChain
| 변수 | 설명 |
|------|------|
| `LANGCHAIN_TRACING_V2` | LangSmith 트레이싱 활성화 |
| `LANGCHAIN_API_KEY` | LangSmith API 키 |
| `LANGCHAIN_PROJECT` | LangSmith 프로젝트 이름 |

### PostgreSQL (Neon)
| 변수 | 설명 |
|------|------|
| `PG_HOST` | PostgreSQL 호스트 |
| `PG_DATABASE` | 데이터베이스 이름 |
| `PG_USER` | DB 사용자명 |
| `PG_PASSWORD` | DB 비밀번호 |
| `PG_PORT` | DB 포트 |

### Neo4j
| 변수 | 설명 |
|------|------|
| `NEO4J_URI` | Neo4j 연결 URI |
| `NEO4J_USER` | Neo4j 사용자명 |
| `NEO4J_PASSWORD` | Neo4j 비밀번호 |

### Weaviate
| 변수 | 설명 |
|------|------|
| `WEAVIATE_HOST` | Weaviate 호스트 |
| `WEAVIATE_PORT` | Weaviate HTTP 포트 |
| `WEAVIATE_GRPC_HOST` | Weaviate gRPC 호스트 |
| `WEAVIATE_GRPC_PORT` | Weaviate gRPC 포트 |

### AWS S3
| 변수 | 설명 |
|------|------|
| `AWS_ACCESS_KEY_ID` | AWS 접근 키 |
| `AWS_SECRET_ACCESS_KEY` | AWS 시크릿 키 |
| `AWS_REGION` | AWS 리전 |
| `S3_BUCKET_NAME` | S3 버킷 이름 |

### OnlyOffice
| 변수 | 설명 |
|------|------|
| `ONLYOFFICE_SERVER_URL` | OnlyOffice 서버 URL |
| `ONLYOFFICE_SECRET_KEY` | OnlyOffice JWT 시크릿 |
