import psycopg2
from psycopg2.extras import RealDictCursor
import json
import os
import re
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# DB 접속 정보 (환경변수 또는 요쳥된 기본값)
DB_CONFIG = {
    "host": os.getenv("PG_HOST", "localhost"),
    "database": os.getenv("PG_DATABASE", "postgres"),
    "user": os.getenv("PG_USER", "postgres"),
    "password": os.getenv("PG_PASSWORD", "1111"),
    "port": os.getenv("PG_PORT", "5432")
}

class SQLStore:
    """PostgreSQL 기반 원본 문서 및 메타데이터 저장소"""
    
    def __init__(self, config: Dict = None):
        self.config = config or DB_CONFIG
        
    def _get_connection(self):
        conn = psycopg2.connect(**self.config)
        try:
            from pgvector.psycopg2 import register_vector
            register_vector(conn)
        except ImportError:
            pass # pgvector 라이브러리가 없거나, DB에 vector 확장이 없는 경우 등
        except Exception:
            pass
        return conn

    def init_db(self):
        """스키마 초기화: 문서 기반 통합 관리 테이블 생성"""
        # 연결 정보 출력
        host = self.config.get("host", "localhost")
        if host in ["localhost", "127.0.0.1"]:
            print(f"[SQLStore] PostgreSQL: 로컬호스트 연결 중 ({host})")
        else:
            print(f"[SQLStore] PostgreSQL: 원격 DB 연결 중 ({host})")
            
        # sop_id의 UNIQUE 제약조건을 제거하고 (sop_id, version) 복합 유니크를 권장하지만,
        query = """
        -- users 테이블
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            rank TEXT,
            dept TEXT
        );

        -- doc_name 테이블 (문서명 마스터)
        CREATE TABLE IF NOT EXISTS doc_name (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL UNIQUE     -- 문서명 (ex: EQ-SOP-00001)
        );

        -- document 테이블
        CREATE TABLE IF NOT EXISTS document (
            id SERIAL PRIMARY KEY,
            doc_name_id INTEGER REFERENCES doc_name(id) ON DELETE RESTRICT,  -- FK: 문서명
            content TEXT,                 -- 원본 전체 마크다운 또는 텍스트
            doc_type TEXT,                -- 문서 타입 (.pdf, .docx 등)
            version TEXT DEFAULT '1.0',   -- 문서 버전
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            modified_at TIMESTAMP,        -- 문서수정일자
            approved_at TIMESTAMP,        -- 승인일자
            effective_at TIMESTAMP,       -- 발효일자
            deprecated_at TIMESTAMP,      -- 폐기일자
            status TEXT DEFAULT '사용중' CHECK (status IN ('폐기', '사용중', '승인대기중'))  -- 문서상태
        );

        -- chunk 테이블
        CREATE TABLE IF NOT EXISTS chunk (
            id SERIAL PRIMARY KEY,
            clause TEXT,                  -- 조항 번호 (ex 1.1, 5.1.2)
            content TEXT NOT NULL,        -- 청크 내용
            metadata JSONB,               -- 청크 메타데이터 (헤더, 섹션 등)
            document_id INTEGER REFERENCES document(id) ON DELETE CASCADE
        );

        -- memory 테이블
        CREATE TABLE IF NOT EXISTS memory (
            id SERIAL PRIMARY KEY,
            answer TEXT,
            question TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            users_id INTEGER REFERENCES users(id) ON DELETE SET NULL
        );
        ALTER TABLE memory ADD COLUMN IF NOT EXISTS session_id TEXT DEFAULT 'default';
        ALTER TABLE memory ADD COLUMN IF NOT EXISTS embedding VECTOR(384);

        -- 인덱스 생성
        CREATE INDEX IF NOT EXISTS idx_chunk_document_id ON chunk(document_id);
        CREATE INDEX IF NOT EXISTS idx_chunk_clause ON chunk(clause);
        CREATE INDEX IF NOT EXISTS idx_chunk_metadata ON chunk USING GIN (metadata);
        CREATE INDEX IF NOT EXISTS idx_memory_users_id ON memory(users_id);
        CREATE INDEX IF NOT EXISTS idx_memory_users_session ON memory(users_id, session_id);
        CREATE INDEX IF NOT EXISTS idx_document_doc_name_id ON document(doc_name_id);

        -- [Migration] doc_name 컬럼이 존재할 경우 삭제 (v2 전환 완료 후)
        DO $$ 
        BEGIN 
            IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='document' AND column_name='doc_name_id') THEN
                ALTER TABLE document ALTER COLUMN doc_name_id DROP NOT NULL;
            END IF;
        END $$;
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                    conn.commit()
            print(" [SQLStore] PostgreSQL 테이블이 준비되었습니다 (document, chunk, users, memory).")
        except Exception as e:
            print(f"🔴 [SQLStore] DB 초기화 실패: {e}")

    # ═══════════════════════════════════════════════════════════════════════════
    # doc_name 테이블 관련 메서드
    # ═══════════════════════════════════════════════════════════════════════════

    def get_or_create_doc_name(self, name: str) -> Optional[int]:
        """문서명 조회 또는 생성 (없으면 생성)"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    # 먼저 조회
                    cur.execute("SELECT id FROM doc_name WHERE name = %s", (name,))
                    row = cur.fetchone()
                    if row:
                        return row[0]
                    # 없으면 생성
                    cur.execute("INSERT INTO doc_name (name) VALUES (%s) RETURNING id", (name,))
                    doc_name_id = cur.fetchone()[0]
                    conn.commit()
                    return doc_name_id
        except Exception as e:
            print(f"🔴 [SQLStore] 문서명 처리 실패: {e}")
            return None

    def list_doc_names(self) -> List[Dict]:
        """모든 문서명 목록 조회"""
        query = "SELECT id, name FROM doc_name ORDER BY name"
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query)
                    return cur.fetchall()
        except Exception:
            return []

    # ═══════════════════════════════════════════════════════════════════════════
    # document 테이블 관련 메서드
    # ═══════════════════════════════════════════════════════════════════════════

    def delete_document_by_name(self, doc_name: str) -> bool:
        """문서명으로 모든 버전 삭제 (chunk는 CASCADE로 자동 삭제)"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT id FROM doc_name WHERE name = %s", (doc_name,))
                    row = cur.fetchone()
                    if not row:
                        print(f" [SQLStore] 문서 없음: {doc_name}")
                        return False
                    doc_name_id = row[0]
                    cur.execute("DELETE FROM document WHERE doc_name_id = %s", (doc_name_id,))
                    cur.execute("DELETE FROM doc_name WHERE id = %s", (doc_name_id,))
                    conn.commit()
            print(f" [SQLStore] 문서 삭제 완료: {doc_name}")
            return True
        except Exception as e:
            print(f" [SQLStore] 문서 삭제 실패: {e}")
            return False

    def save_document(
        self,
        doc_name: str,
        content: str,
        doc_type: str = None,
        version: str = "1.0",
        status: str = "사용중",
        modified_at: str = None,
        approved_at: str = None,
        effective_at: str = None,
        deprecated_at: str = None
    ) -> Optional[int]:
        """문서 정보를 저장합니다. (doc_name은 자동으로 doc_name 테이블에 등록)"""
        # 1. doc_name_id 조회 또는 생성
        doc_name_id = self.get_or_create_doc_name(doc_name)
        if not doc_name_id:
            return None

        # 2. 같은 doc_name + version이 이미 있으면 덮어쓰기(UPDATE), 없으면 새로 삽입(INSERT)
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    # 1. 만약 신규 버전의 상태가 "사용중"이라면, 기존의 "사용중"인 버전들을 "폐기"로 변경
                    if status == "사용중":
                        cur.execute(
                            "UPDATE document SET status = '폐기', deprecated_at = NOW() WHERE doc_name_id = %s AND status = '사용중' AND version != %s",
                            (doc_name_id, version)
                        )

                    cur.execute(
                        "SELECT id FROM document WHERE doc_name_id = %s AND version = %s",
                        (doc_name_id, version)
                    )
                    existing = cur.fetchone()

                    if existing:
                        cur.execute(
                            """
                            UPDATE document
                            SET content = %s, doc_type = %s, status = %s, modified_at = NOW()
                            WHERE doc_name_id = %s AND version = %s
                            RETURNING id;
                            """,
                            (content, doc_type, status, doc_name_id, version)
                        )
                        doc_id = cur.fetchone()[0]
                        conn.commit()
                        print(f" [SQLStore] 문서 덮어쓰기: {doc_name} v{version} (ID: {doc_id})")
                    else:
                        cur.execute(
                            """
                            INSERT INTO document (doc_name_id, content, doc_type, version, status, modified_at, approved_at, effective_at, deprecated_at)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                            RETURNING id;
                            """,
                            (doc_name_id, content, doc_type, version, status,
                             modified_at, approved_at, effective_at, deprecated_at)
                        )
                        doc_id = cur.fetchone()[0]
                        conn.commit()
                        print(f" [SQLStore] 문서 저장 성공: {doc_name} v{version} (ID: {doc_id})")

            return doc_id
        except Exception as e:
            print(f" [SQLStore] 문서 저장 실패: {e}")
            return None

    def save_chunk(
        self,
        document_id: int,
        clause: str,
        content: str,
        metadata: Dict = None
    ) -> Optional[int]:
        """청크를 저장합니다."""
        insert_query = """
            INSERT INTO chunk (clause, content, metadata, document_id)
            VALUES (%s, %s, %s, %s)
            RETURNING id;
        """

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(insert_query, (
                        clause,
                        content,
                        json.dumps(metadata or {}),
                        document_id
                    ))
                    chunk_id = cur.fetchone()[0]
                    conn.commit()
            return chunk_id
        except Exception as e:
            print(f" [SQLStore] 청크 저장 실패: {e}")
            return None

    def save_chunks_batch(
        self,
        document_id: int,
        chunks: List[Dict]
    ):
        """여러 청크를 일괄 저장합니다."""
        insert_query = """
            INSERT INTO chunk (clause, content, metadata, document_id)
            VALUES (%s, %s, %s, %s);
        """

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    for chunk in chunks:
                        cur.execute(insert_query, (
                            chunk.get('clause'),
                            chunk.get('content'),
                            json.dumps(chunk.get('metadata', {})),
                            document_id
                        ))
                    conn.commit()
            print(f" [SQLStore] {len(chunks)}개 청크 저장 성공 (document_id: {document_id})")
        except Exception as e:
            print(f" [SQLStore] 청크 일괄 저장 실패: {e}")

    def get_document_by_id(self, document_id: int) -> Optional[Dict]:
        """문서 ID로 문서 조회"""
        query = """
            SELECT d.id, dn.name as doc_name, d.content, d.doc_type, d.version, d.created_at,
                   d.modified_at, d.approved_at, d.effective_at, d.deprecated_at, d.status
            FROM document d
            JOIN doc_name dn ON d.doc_name_id = dn.id
            WHERE d.id = %s
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, (document_id,))
                    return cur.fetchone()
        except Exception as e:
            return None

    def get_document_by_name(self, doc_name: str, version: str = None) -> Optional[Dict]:
        """문서명으로 문서 조회 (버전 미지정 시 최신 버전)"""
        base_query = """
            SELECT d.id, dn.name as doc_name, d.content, d.doc_type, d.version, d.created_at,
                   d.modified_at, d.approved_at, d.effective_at, d.deprecated_at, d.status
            FROM document d
            JOIN doc_name dn ON d.doc_name_id = dn.id
            WHERE dn.name = %s
        """
        if version:
            query = base_query + " AND d.version = %s"
            params = (doc_name, version)
        else:
            # 버전이 여러 개일 경우 버전을 내림차순 정렬하여 가장 높은 버전을 선택 (문자열 정렬이므로 10.0이 2.0보다 작을 수 있음에 주의, 여기서는 우선 기본 DESC 사용)
            query = base_query + " ORDER BY d.version DESC, d.created_at DESC LIMIT 1"
            params = (doc_name,)

        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, params)
                    return cur.fetchone()
        except Exception as e:
            print(f"🔴 [SQLStore] 문서 조회 실패: {e}")
            return None

    def get_document_versions(self, doc_name: str) -> List[Dict]:
        """문서의 버전 히스토리 조회"""
        query = """
            SELECT d.id, dn.name as doc_name_id, d.version, d.created_at, d.approved_at, 
                   d.effective_at, d.deprecated_at, d.status, d.doc_type
            FROM document d
            JOIN doc_name dn ON d.doc_name_id = dn.id
            WHERE dn.name = %s
            ORDER BY d.created_at DESC
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, (doc_name,))
                    return cur.fetchall()
        except Exception as e:
            print(f"❌ [SQLStore] 버전 히스토리 조회 실패: {e}")
            return []

    def get_chunks_by_document(self, document_id: int) -> List[Dict]:
        """특정 문서의 모든 청크 조회"""
        query = "SELECT id, clause, content, metadata FROM chunk WHERE document_id = %s ORDER BY id"
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, (document_id,))
                    return cur.fetchall()
        except Exception:
            return []

    def get_document_versions(self, doc_name: str) -> List[Dict]:
        """특정 문서의 모든 버전 목록 조회"""
        query = """
            SELECT d.id, dn.name as doc_name, d.version, d.status, d.created_at, d.effective_at
            FROM document d
            JOIN doc_name dn ON d.doc_name_id = dn.id
            WHERE dn.name = %s
            ORDER BY d.version DESC, d.created_at DESC
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, (doc_name,))
                    return cur.fetchall()
        except Exception:
            return []

    def get_all_documents(self) -> List[Dict]:
        """모든 문서 목록 조회"""
        query = """
            SELECT d.id, dn.name as doc_name, d.version, d.status, d.doc_type,
                   d.created_at, d.effective_at
            FROM document d
            JOIN doc_name dn ON d.doc_name_id = dn.id
            ORDER BY dn.name, d.version DESC
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query)
                    return cur.fetchall()
        except Exception as e:
            print(f"🔴 문서 목록 조회 실패: {e}")
            return []

    def get_clause_diff(self, doc_name: str, v1: str, v2: str) -> List[Dict]:
        """두 버전 간의 조항 단위 비교 (Added, Deleted, Modified)"""
        print(f"[SQLStore] 조항 비교 시작: {doc_name} v{v1} vs {v2}")

        def _normalize_for_diff(text: str) -> str:
            """문서 비교용 정규화: 포맷 노이즈만 최소한으로 제거"""
            if not text:
                return ""
            raw = str(text)
            t = raw.lower()

            # 문서 본문 이후 붙는 승인/개정 로그는 비교 대상에서 제외
            tail_markers = [
                "*end of document*",
                "document revision history",
                "문서개정이력",
                "document approvals",
                "approved date",
            ]
            cut_pos = -1
            for marker in tail_markers:
                p = t.find(marker)
                if p != -1 and (cut_pos == -1 or p < cut_pos):
                    cut_pos = p
            if cut_pos != -1:
                t = t[:cut_pos]

            # 전자결재/서명/이메일/타임스탬프류 노이즈 제거
            t = re.sub(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', ' ', t)
            t = re.sub(r'\b\d{4}-\d{2}-\d{2}\b', ' ', t)
            t = re.sub(r'\b\d{1,2}[-/]\w{3}[-/]\d{4}\b', ' ', t)
            t = re.sub(r'\b\d{1,2}:\d{2}:\d{2}\b', ' ', t)
            t = re.sub(r'gmt[+\-]?\d*', ' ', t)

            # 승인 워크플로우/결재 로그 라인 제거
            t = re.sub(r'\btask\s*:\s*approval task\b.*', ' ', t)
            t = re.sub(r'\bverdict\s*:\s*approve\b.*', ' ', t)
            t = re.sub(r'\bproduction manager\b.*', ' ', t)
            t = re.sub(r'\bquality manager\b.*', ' ', t)
            t = re.sub(r'본 문서의 전자서명 날짜.*', ' ', t)

            # 공백/기호 정규화
            t = re.sub(r'\s+', ' ', t)
            return t.strip()

        def _token_set(text: str) -> set:
            """비교용 토큰 집합 (한글 1글자 포함)"""
            if not text:
                return set()
            cleaned = re.sub(r'[^0-9a-z가-힣]+', ' ', text.lower())
            return {tok for tok in cleaned.split() if tok}

        # v1 문서 ID 조회
        doc1 = self.get_document_by_name(doc_name, v1)
        if not doc1: return [{"error": f"v{v1} 버전을 찾을 수 없습니다."}]

        # v2 문서 ID 조회
        doc2 = self.get_document_by_name(doc_name, v2)
        if not doc2: return [{"error": f"v{v2} 버전을 찾을 수 없습니다."}]

        # 조항별로 content를 병합한 후 비교
        query = """
            WITH v1_clauses_raw AS (
                SELECT
                    clause,
                    -- 조항 매핑 키:
                    -- 1) 숫자 계층 조항(예: 4.1.2) 우선 매핑
                    -- 2) 비숫자 조항은 공백 정규화 후 텍스트 매핑
                    -- 3) 빈값/일반 라벨(N/A 등)은 id 기반 고유 키로 분리
                    CASE
                        WHEN clause IS NULL OR TRIM(clause) = '' THEN CONCAT('__NOCLAUSE__:', id::text)
                        WHEN REGEXP_MATCH(clause, '(\\d+(?:\\.\\d+)*)') IS NOT NULL THEN (REGEXP_MATCH(clause, '(\\d+(?:\\.\\d+)*)'))[1]
                        WHEN LOWER(TRIM(clause)) IN ('n/a', 'na', 'none', 'null', '-', 'clause', 'section', 'article', 'body', 'text') THEN CONCAT('__GENERIC__:', id::text)
                        ELSE LOWER(REGEXP_REPLACE(TRIM(clause), '\\s+', ' ', 'g'))
                    END AS clause_key,
                    content,
                    id
                FROM chunk
                WHERE document_id = %s AND clause IS NOT NULL
            ),
            v1_clauses AS (
                SELECT
                    MIN(clause) AS clause,
                    clause_key,
                    STRING_AGG(content, ' ' ORDER BY id) as content
                FROM v1_clauses_raw
                GROUP BY clause_key
            ),
            v2_clauses_raw AS (
                SELECT
                    clause,
                    CASE
                        WHEN clause IS NULL OR TRIM(clause) = '' THEN CONCAT('__NOCLAUSE__:', id::text)
                        WHEN REGEXP_MATCH(clause, '(\\d+(?:\\.\\d+)*)') IS NOT NULL THEN (REGEXP_MATCH(clause, '(\\d+(?:\\.\\d+)*)'))[1]
                        WHEN LOWER(TRIM(clause)) IN ('n/a', 'na', 'none', 'null', '-', 'clause', 'section', 'article', 'body', 'text') THEN CONCAT('__GENERIC__:', id::text)
                        ELSE LOWER(REGEXP_REPLACE(TRIM(clause), '\\s+', ' ', 'g'))
                    END AS clause_key,
                    content,
                    id
                FROM chunk
                WHERE document_id = %s AND clause IS NOT NULL
            ),
            v2_clauses AS (
                SELECT
                    MIN(clause) AS clause,
                    clause_key,
                    STRING_AGG(content, ' ' ORDER BY id) as content
                FROM v2_clauses_raw
                GROUP BY clause_key
            )
            SELECT
                COALESCE(v2.clause, v1.clause) as clause,
                CASE
                    WHEN v1.clause IS NULL THEN 'ADDED'
                    WHEN v2.clause IS NULL THEN 'DELETED'
                    WHEN REGEXP_REPLACE(v1.content, '\\s+', '', 'g') <> REGEXP_REPLACE(v2.content, '\\s+', '', 'g') THEN 'MODIFIED'
                    ELSE 'UNCHANGED'
                END as change_type,
                v1.content as v1_content,
                v2.content as v2_content
            FROM v1_clauses v1
            FULL OUTER JOIN v2_clauses v2 ON v1.clause_key = v2.clause_key
            ORDER BY COALESCE(v2.clause, v1.clause)
        """

        diffs = []
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, (doc1['id'], doc2['id']))
                    rows = cur.fetchall()

                    for row in rows:
                        if row['change_type'] == 'UNCHANGED':
                            continue

                        item = dict(row)
                        change_type = item.get('change_type')

                        # ADDED/DELETED 판정 보정:
                        # 본문 없이 placeholder(N/A 등)만 있는 경우는 제외
                        if change_type in ('ADDED', 'DELETED'):
                            raw_text = item.get('v2_content') if change_type == 'ADDED' else item.get('v1_content')
                            normalized = _normalize_for_diff(raw_text or "")
                            if normalized in {"", "n/a", "na", "none", "null", "-"}:
                                continue

                        # MODIFIED 판정 보정:
                        # 포맷/개행/서명 등 비본질 차이는 변경으로 보지 않음
                        if change_type == 'MODIFIED':
                            v1_raw = item.get('v1_content') or ""
                            v2_raw = item.get('v2_content') or ""
                            n1 = _normalize_for_diff(v1_raw)
                            n2 = _normalize_for_diff(v2_raw)

                            if n1 == n2:
                                continue

                            # 중복 chunk/순서 차이/OCR 분할 차이 보정
                            t1 = _token_set(n1)
                            t2 = _token_set(n2)
                            if t1 and t2:
                                only1 = t1 - t2
                                only2 = t2 - t1

                                # 토큰 집합이 동일하면 내용 동일로 간주
                                if not only1 and not only2:
                                    continue

                                # 미세한 꼬리 잘림/기호 노이즈(1토큰 이내)는 무시
                                inter = len(t1 & t2)
                                union = len(t1 | t2) or 1
                                jaccard = inter / union
                                if jaccard >= 0.94 and (len(only1) + len(only2)) <= 1:
                                    continue

                        diffs.append(item)

            return diffs
        except Exception as e:
            print(f"🔴 [SQLStore] 조항 비교 실패: {e}")
            return [{"error": str(e)}]

    def list_documents(self) -> List[Dict]:
        """모든 문서 목록 조회"""
        query = """
            SELECT d.id, dn.name as doc_name, d.doc_type, d.version, d.created_at,
                   d.modified_at, d.approved_at, d.effective_at, d.deprecated_at, d.status
            FROM document d
            JOIN doc_name dn ON d.doc_name_id = dn.id
            ORDER BY dn.name ASC, d.version DESC, d.created_at DESC
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query)
                    return cur.fetchall()
        except Exception as e:
            print(f" [SQLStore] 목록 조회 실패: {e}")
            return []

    def update_document_status(
        self,
        document_id: int,
        status: str,
        approved_at: str = None,
        effective_at: str = None,
        deprecated_at: str = None
    ) -> bool:
        """문서 상태 업데이트 (폐기, 사용중, 승인대기중)"""
        query = """
            UPDATE document
            SET status = %s,
                modified_at = CURRENT_TIMESTAMP,
                approved_at = COALESCE(%s, approved_at),
                effective_at = COALESCE(%s, effective_at),
                deprecated_at = COALESCE(%s, deprecated_at)
            WHERE id = %s
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (status, approved_at, effective_at, deprecated_at, document_id))
                    conn.commit()
            print(f" [SQLStore] 문서 상태 변경: ID {document_id} → {status}")
            return True
        except Exception as e:
            print(f" [SQLStore] 문서 상태 변경 실패: {e}")
            return False

    def execute_data_correction(self) -> bool:
        """기존 중복 '사용중' 문서들을 정리 (최신만 남기고 '폐기')"""
        query = """
            UPDATE document 
            SET status = '폐기', deprecated_at = NOW()
            WHERE status = '사용중' 
              AND id NOT IN (
                SELECT id FROM (
                  SELECT id, ROW_NUMBER() OVER(PARTITION BY doc_name_id ORDER BY created_at DESC, version DESC) as rn 
                  FROM document 
                  WHERE status = '사용중'
                ) t WHERE rn = 1
              );
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                    count = cur.rowcount
                    conn.commit()
            print(f" [SQLStore] 데이터 보정 완료: {count}개 버전을 '폐기'로 변경")
            return True
        except Exception as e:
            print(f" [SQLStore] 데이터 보정 실패: {e}")
            return False

    # Users 테이블 관련 메서드
    def save_user(self, name: str, rank: str = None, dept: str = None) -> Optional[int]:
        """사용자 저장"""
        insert_query = "INSERT INTO users (name, rank, dept) VALUES (%s, %s, %s) RETURNING id;"
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(insert_query, (name, rank, dept))
                    user_id = cur.fetchone()[0]
                    conn.commit()
            return user_id
        except Exception as e:
            print(f" [SQLStore] 사용자 저장 실패: {e}")
            return None

    def get_user(self, user_id: int) -> Optional[Dict]:
        """사용자 조회"""
        query = "SELECT id, name, rank, dept FROM users WHERE id = %s"
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, (user_id,))
                    return cur.fetchone()
        except Exception:
            return None

    # Memory 테이블 관련 메서드
    def save_memory(self, question: str, answer: str, users_id: int, session_id: str, embedding: List[float] = None) -> Optional[int]:
        """대화 기록 저장 (세션 기반, 세션별 최대 100개 유지)"""
        if embedding:
            insert_query = "INSERT INTO memory (question, answer, users_id, embedding, session_id) VALUES (%s, %s, %s, %s, %s) RETURNING id;"
            params = (question, answer, users_id, embedding, session_id)
        else:
            insert_query = "INSERT INTO memory (question, answer, users_id, session_id) VALUES (%s, %s, %s, %s) RETURNING id;"
            params = (question, answer, users_id, session_id)
            
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(insert_query, params)
                    memory_id = cur.fetchone()[0]

                    # 세션별 최근 100개만 유지
                    if users_id and session_id:
                        delete_query = """
                            DELETE FROM memory
                            WHERE id IN (
                                SELECT id
                                FROM memory
                                WHERE users_id = %s AND session_id = %s
                                ORDER BY created_at DESC
                                OFFSET 100
                            );
                        """
                        cur.execute(delete_query, (users_id, session_id))

                    conn.commit()
            return memory_id
        except Exception as e:
            print(f" [SQLStore] 대화 기록 저장 실패: {e}")
            return None

    def get_memory_by_user(self, users_id: int, limit: int = 10) -> List[Dict]:
        """사용자별 대화 기록 조회"""
        query = """
            SELECT id, question, answer, created_at
            FROM memory
            WHERE users_id = %s
            ORDER BY created_at DESC
            LIMIT %s
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, (users_id, limit))
                    return cur.fetchall()
        except Exception:
            return []

    def get_conversation_history(self, user_id: int, limit: int = 10) -> List[Dict]:
        """대화 기록 조회 (Agent용 포맷)"""
        memories = self.get_memory_by_user(user_id, limit)
        history = []
        # 최신순으로 가져왔으므로 역순으로 정렬하여 시간순으로 배치
        for mem in reversed(memories):
            history.append({"role": "user", "content": mem["question"]})
            history.append({"role": "assistant", "content": mem["answer"]})
        return history

    def get_memory_by_session(self, users_id: int, session_id: str, limit: int = 10) -> List[Dict]:
        """사용자 + 세션별 대화 기록 조회"""
        query = """
            SELECT id, question, answer, created_at
            FROM memory
            WHERE users_id = %s AND session_id = %s
            ORDER BY created_at DESC
            LIMIT %s
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, (users_id, session_id, limit))
                    return cur.fetchall()
        except Exception:
            return []

    def get_conversation_history_by_session(self, user_id: int, session_id: str, limit: int = 10) -> List[Dict]:
        """세션별 대화 기록 조회 (Agent용 포맷)"""
        memories = self.get_memory_by_session(user_id, session_id, limit)
        history = []
        for mem in reversed(memories):
            history.append({"role": "user", "content": mem["question"]})
            history.append({"role": "assistant", "content": mem["answer"]})
        return history

    def search_memory_similar(self, users_id: int, query_embedding: List[float], limit: int = 3) -> List[Dict]:
        """
        pgvector 기반 유사도 기억 검색
        - embedding <-> query_embedding (거리 기준 오름차순)
        """
        sql = """
            SELECT id, question, answer, created_at
            FROM memory
            WHERE users_id = %s AND embedding IS NOT NULL
            ORDER BY embedding <-> %s
            LIMIT %s
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(sql, (users_id, str(query_embedding), limit))
                    return cur.fetchall()
        except Exception as e:
            print(f" [SQLStore] 유사 기억 검색 실패: {e}")
            return []

    # ═══════════════════════════════════════════════════════════════════════════
    # 마이그레이션 함수
    # ═══════════════════════════════════════════════════════════════════════════

    def migrate_v2(self) -> bool:
        """
        v2 스키마 마이그레이션:
        - doc_name 테이블 생성
        - document 테이블에 새 컬럼 추가 (modified_at, approved_at, effective_at, deprecated_at, status)
        - document.doc_name → doc_name 테이블로 이관 및 FK 연결
        """
        print("[SQLStore] 마이그레이션 v2 시작...")

        migration_queries = [
            # 1. doc_name 테이블 생성
            """
            CREATE TABLE IF NOT EXISTS doc_name (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL UNIQUE
            );
            """,

            # 2. document 테이블에 새 컬럼 추가
            "ALTER TABLE document ADD COLUMN IF NOT EXISTS modified_at TIMESTAMP;",
            "ALTER TABLE document ADD COLUMN IF NOT EXISTS approved_at TIMESTAMP;",
            "ALTER TABLE document ADD COLUMN IF NOT EXISTS effective_at TIMESTAMP;",
            "ALTER TABLE document ADD COLUMN IF NOT EXISTS deprecated_at TIMESTAMP;",
            "ALTER TABLE document ADD COLUMN IF NOT EXISTS status TEXT DEFAULT '사용중';",
            "ALTER TABLE document ADD COLUMN IF NOT EXISTS doc_name_id INTEGER;",

            # 3. 기존 doc_name 데이터를 doc_name 테이블로 이관
            """
            INSERT INTO doc_name (name)
            SELECT DISTINCT doc_name FROM document
            WHERE doc_name IS NOT NULL
            ON CONFLICT (name) DO NOTHING;
            """,

            # 4. doc_name_id 값 업데이트
            """
            UPDATE document d
            SET doc_name_id = dn.id
            FROM doc_name dn
            WHERE d.doc_name = dn.name AND d.doc_name_id IS NULL;
            """,

            # 5. doc_name 컬럼의 NOT NULL 제약조건 제거
            "ALTER TABLE document ALTER COLUMN doc_name DROP NOT NULL;",

            # 6. 인덱스 생성
            "CREATE INDEX IF NOT EXISTS idx_document_doc_name_id ON document(doc_name_id);",
            # 7. 레거시 doc_name 컬럼 삭제
            "ALTER TABLE document DROP COLUMN IF EXISTS doc_name CASCADE;",
        ]

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    for i, query in enumerate(migration_queries, 1):
                        try:
                            cur.execute(query)
                            print(f"    단계 {i} 완료")
                        except Exception as e:
                            # 이미 존재하는 컬럼 등 무시
                            if "already exists" in str(e) or "does not exist" in str(e):
                                print(f"   ⏩ 단계 {i} 스킵 (이미 적용됨)")
                            else:
                                print(f"    단계 {i} 경고: {e}")
                    conn.commit()

            print(" [SQLStore] 마이그레이션 v2 및 컬럼 정제 완료!")
            return True

        except Exception as e:
            print(f" [SQLStore] 마이그레이션 실패: {e}")
            return False

    def check_migration_status(self) -> Dict:
        """현재 스키마 상태 확인"""
        status = {
            "doc_name_table": False,
            "document_new_columns": False,
            "doc_name_id_populated": False
        }

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    # doc_name 테이블 존재 확인
                    cur.execute("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'doc_name')")
                    status["doc_name_table"] = cur.fetchone()[0]

                    # document 테이블 새 컬럼 확인
                    cur.execute("""
                        SELECT column_name FROM information_schema.columns
                        WHERE table_name = 'document' AND column_name IN ('status', 'doc_name_id')
                    """)
                    columns = [row[0] for row in cur.fetchall()]
                    status["document_new_columns"] = 'status' in columns and 'doc_name_id' in columns

                    # doc_name_id 채워졌는지 확인
                    if status["document_new_columns"]:
                        cur.execute("SELECT COUNT(*) FROM document WHERE doc_name_id IS NOT NULL")
                        count = cur.fetchone()[0]
                        cur.execute("SELECT COUNT(*) FROM document")
                        total = cur.fetchone()[0]
                        status["doc_name_id_populated"] = (count == total) if total > 0 else True

            return status
        except Exception as e:
            print(f" 상태 확인 실패: {e}")
            return status

    # ═══════════════════════════════════════════════════════════════════════════
    # Auth 관련 메서드
    # ═══════════════════════════════════════════════════════════════════════════

    def register_user(self, username: str, password_hash: str, name: str, email: str = None, rank: str = None, dept: str = None) -> Optional[int]:
        """[Auth] 신규 사용자 가입 (비밀번호 해시 포함)"""
        insert_query = """
            INSERT INTO users (username, password_hash, name, email, rank, dept, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Seoul')
            RETURNING id;
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(insert_query, (username, password_hash, name, email, rank, dept))
                    user_id = cur.fetchone()[0]
                    conn.commit()
            return user_id
        except Exception as e:
            print(f" [SQLStore] 사용자 가입 실패: {e}")
            return None

    def get_user_by_username(self, username: str) -> Optional[Dict]:
        """[Auth] 로그인 ID로 사용자 조회 (비밀번호 검증용)"""
        from psycopg2.extras import RealDictCursor
        query = "SELECT id, username, password_hash, name, email, rank, dept, last_login, created_at FROM users WHERE username = %s"
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, (username,))
                    return cur.fetchone()
        except Exception:
            return None

    def update_last_login(self, user_id: int):
        """[Auth] 마지막 로그인 시간 갱신"""
        query = "UPDATE users SET last_login = (CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Seoul') WHERE id = %s"
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (user_id,))
                    conn.commit()
        except Exception as e:
            print(f" [SQLStore] 마지막 로그인 갱신 실패: {e}")

    def get_user(self, user_id: int) -> Optional[Dict]:
        """사용자 조회"""
        from psycopg2.extras import RealDictCursor
        query = "SELECT id, username, name, rank, dept, email, last_login FROM users WHERE id = %s"
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, (user_id,))
                    return cur.fetchone()
        except Exception:
            return None


if __name__ == "__main__":
    # 테스트
    store = SQLStore()

    # 마이그레이션 상태 확인
    print("\n 현재 스키마 상태:")
    status = store.check_migration_status()
    for key, value in status.items():
        print(f"   {key}: {'' if value else ''}")

    # 마이그레이션 필요 시 실행
    if not all(status.values()):
        print("\n 마이그레이션이 필요합니다.")
        user_input = input("마이그레이션을 실행하시겠습니까? (y/n): ")
        if user_input.lower() == 'y':
            store.migrate_v2()
    else:
        print("\n 스키마가 최신 상태입니다.")

    store.init_db()

    # 사용자 생성 테스트
    user_id = store.save_user("홍길동", "사원", "품질관리팀")
    print(f"생성된 사용자 ID: {user_id}")

    # 문서 생성 테스트
    doc_id = store.save_document(
        doc_name="EQ-SOP-00010",
        content="# 품질관리기준서\n\n## 1. 목적\n본 기준서는...",
        doc_type=".md",
        version="1.0"
    )
    print(f"생성된 문서 ID: {doc_id}")

    # 청크 생성 테스트
    if doc_id:
        chunk_id = store.save_chunk(
            document_id=doc_id,
            clause="1.1",
            content="본 기준서는 품질관리기준서의 작성, 검토, 승인에 관한 기준을 정한다.",
            metadata={"section": "목적", "H2": "1. 목적"}
        )
        print(f"생성된 청크 ID: {chunk_id}")

    # 대화 기록 저장 테스트
    if user_id:
        memory_id = store.save_memory(
            question="품질관리기준서는 어떻게 작성하나요?",
            answer="품질관리기준서는 작성, 검토, 승인 절차를 따릅니다.",
            users_id=user_id,
            session_id="default"
        )
        print(f"생성된 대화 기록 ID: {memory_id}")
