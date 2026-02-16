"""
Neo4j 그래프 저장소

노드 타입:
- Document: 문서 관리
- Section: 조항 관리 + LLM 메타데이터
- DocumentType: 문서 유형 (SOP, WI, FORM 등)
- Concept: 관리 영역 (user_account, document_lifecycle, training 등)

관계 타입:
- HAS_SECTION: Document -> Section
- PARENT_OF: Section -> Section (계층)
- REFERENCES: Document -> Document (문서 간 참조)
- IS_TYPE: Document -> DocumentType
- MENTIONS: Section -> Document (조항 내 타 문서 언급)
- BELONGS_TO_CONCEPT: Section -> Concept
"""

from neo4j import GraphDatabase
from typing import List, Dict, Optional
import re
import uuid
import os

class Neo4jGraphStore:
    """Neo4j 그래프 저장소"""

    def __init__(
        self,
        uri: str = "neo4j+s://d00efa60.databases.neo4j.io",
        user: str = "neo4j",
        password: str = "4Qs45al1Coz_NwZDSMcFV9JIFjU7zXPjdKyptQloS6c",
        database: str = "neo4j"
    ):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver = None

    def connect(self):
        """Neo4j 연결"""
        if not self.driver:
            try:
                # 연결 타임아웃 설정 (10초)
                self.driver = GraphDatabase.driver(
                    self.uri,
                    auth=(self.user, self.password),
                    connection_timeout=10.0,
                    max_connection_lifetime=300
                )
                if self.test_connection():
                    print(f"🟢 Neo4j 연결 성공")
            except Exception as e:
                print(f"🔴 Neo4j 연결 실패: {e}")
                self.driver = None
        return self

    def close(self):
        if self.driver:
            try:
                self.driver.close()
            except Exception as e:
                print(f"🔴 Neo4j 연결 종료 중 오류 (무시): {e}")
            finally:
                self.driver = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def test_connection(self) -> bool:
        try:
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1")
                return True
        except:
            return False

    # ═══════════════════════════════════════════════════════════════════════════
    # 스키마 초기화
    # ═══════════════════════════════════════════════════════════════════════════

    def init_schema(self):
        """인덱스 및 제약조건"""
        constraints = [
            "CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",
            "CREATE CONSTRAINT section_id IF NOT EXISTS FOR (s:Section) REQUIRE s.section_id IS UNIQUE",
            "CREATE CONSTRAINT doc_type_code IF NOT EXISTS FOR (dt:DocumentType) REQUIRE dt.code IS UNIQUE",
            "CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (c:Concept) REQUIRE c.concept_id IS UNIQUE",
            "CREATE INDEX doc_title IF NOT EXISTS FOR (d:Document) ON (d.title)",
            "CREATE INDEX section_title IF NOT EXISTS FOR (s:Section) ON (s.title)",
            "CREATE INDEX section_intent_scope IF NOT EXISTS FOR (s:Section) ON (s.intent_scope)",
        ]

        with self.driver.session(database=self.database) as session:
            for c in constraints:
                try:
                    session.run(c)
                except:
                    pass
        print("🟢 스키마 초기화 완료")

    def clear_all(self):
        """모든 데이터 삭제"""
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("🟢 모든 데이터 삭제 완료")

    # ═══════════════════════════════════════════════════════════════════════════
    # Document 관리
    # ═══════════════════════════════════════════════════════════════════════════

    def create_document(self, doc_id: str, title: str, version: str = "1.0",
                       effective_date: str = "", owning_dept: str = "", **metadata):
        """문서 생성"""
        with self.driver.session(database=self.database) as session:
            session.run("""
                MERGE (d:Document {doc_id: $doc_id})
                SET d.title = $title,
                    d.version = $version,
                    d.effective_date = $effective_date,
                    d.owning_dept = $owning_dept,
                    d.updated_at = datetime()
            """, doc_id=doc_id, title=title, version=version,
                effective_date=effective_date, owning_dept=owning_dept)

    def create_document_type(self, code: str, name_kr: str, name_en: str):
        """DocumentType 노드 생성"""
        with self.driver.session(database=self.database) as session:
            session.run("""
                MERGE (dt:DocumentType {code: $code})
                SET dt.name_kr = $name_kr, dt.name_en = $name_en
            """, code=code, name_kr=name_kr, name_en=name_en)

    def link_document_type(self, doc_id: str, type_code: str):
        """Document -[:IS_TYPE]-> DocumentType 관계"""
        with self.driver.session(database=self.database) as session:
            session.run("""
                MATCH (d:Document {doc_id: $doc_id})
                MATCH (dt:DocumentType {code: $type_code})
                MERGE (d)-[:IS_TYPE]->(dt)
            """, doc_id=doc_id, type_code=type_code)

    def init_type_hierarchy(self):
        """문서 타입 간 계층 관계(SOP > WI > FORM) 보장"""
        with self.driver.session(database=self.database) as session:
            session.run("""
                MATCH (sop:DocumentType {code: 'SOP'})
                MATCH (wi:DocumentType {code: 'WI'})
                MATCH (frm:DocumentType) WHERE frm.code IN ['FORM', 'FRM']
                MERGE (sop)-[:SUPERIOR_TO]->(wi)
                MERGE (wi)-[:SUPERIOR_TO]->(frm)
            """)
        print("🟢 문서 타입 계층 구조(SUPERIOR_TO) 초기화 완료")

    def create_concept(self, concept_id: str, name_kr: str, name_en: str, description: str = ""):
        """Concept 노드 생성 (관리 영역)"""
        with self.driver.session(database=self.database) as session:
            session.run("""
                MERGE (c:Concept {concept_id: $concept_id})
                SET c.name_kr = $name_kr,
                    c.name_en = $name_en,
                    c.description = $description
            """, concept_id=concept_id, name_kr=name_kr, name_en=name_en, description=description)

    def link_section_concept(self, section_id: str, concept_id: str):
        """Section -[:BELONGS_TO_CONCEPT]-> Concept 관계"""
        with self.driver.session(database=self.database) as session:
            session.run("""
                MATCH (s:Section {section_id: $section_id})
                MATCH (c:Concept {concept_id: $concept_id})
                MERGE (s)-[:BELONGS_TO_CONCEPT]->(c)
            """, section_id=section_id, concept_id=concept_id)

    def get_document(self, doc_id: str) -> Optional[Dict]:
        """문서 조회"""
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (d:Document {doc_id: $doc_id})
                OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)
                RETURN d, count(s) as section_count
            """, doc_id=doc_id)
            record = result.single()
            if record:
                return {**dict(record["d"]), "section_count": record["section_count"]}
            return None

    def get_all_documents(self) -> List[Dict]:
        """모든 문서 목록"""
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (d:Document)
                OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)
                RETURN d, count(s) as section_count
                ORDER BY d.doc_id
            """)
            return [{**dict(r["d"]), "section_count": r["section_count"]} for r in result]

    def delete_document(self, doc_id: str):
        """문서 및 관련 섹션 삭제"""
        with self.driver.session(database=self.database) as session:
            session.run("""
                MATCH (d:Document {doc_id: $doc_id})
                OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)
                DETACH DELETE d, s
            """, doc_id=doc_id)
        print(f"🟢 문서 삭제: {doc_id}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 관리
    # ═══════════════════════════════════════════════════════════════════════════

    def create_section(self, doc_id: str, section_id: str, title: str, content: str,
                      clause_level: int = 0, main_section: str = None, llm_meta: Dict = None, **kwargs):
        """섹션 생성 + LLM 메타데이터 (evaluate_gmp_unified 호환)"""
        meta = llm_meta or {}

        # main_section 기본값
        if not main_section:
            main_section = section_id.split('.')[0] if '.' in section_id else section_id

        with self.driver.session(database=self.database) as session:
            session.run("""
                MATCH (d:Document {doc_id: $doc_id})
                MERGE (s:Section {section_id: $section_id})
                SET s.doc_id = $doc_id,
                    s.title = $title,
                    s.content = $content,
                    s.clause_level = $clause_level,
                    s.main_section = $main_section,
                    s.content_type = $content_type,
                    s.main_topic = $main_topic,
                    s.sub_topics = $sub_topics,
                    s.actors = $actors,
                    s.actions = $actions,
                    s.conditions = $conditions,
                    s.summary = $summary,
                    s.intent_scope = $intent_scope,
                    s.intent_summary = $intent_summary,
                    s.language = $language
                MERGE (d)-[:HAS_SECTION]->(s)
            """,
            doc_id=doc_id,
            section_id=section_id,
            title=title,
            content=content,
            clause_level=clause_level,
            main_section=main_section,
            content_type=meta.get("content_type", ""),
            main_topic=meta.get("main_topic", ""),
            sub_topics=str(meta.get("sub_topics", [])),
            actors=str(meta.get("actors", [])),
            actions=str(meta.get("actions", [])),
            conditions=str(meta.get("conditions", [])),
            summary=meta.get("summary", ""),
            intent_scope=meta.get("intent_scope", ""),
            intent_summary=meta.get("intent_summary", ""),
            language=meta.get("language", "ko")
            )

    def create_section_hierarchy(self, parent_id: str, child_id: str):
        """섹션 계층 관계 (같은 문서 내에서만)"""
        with self.driver.session(database=self.database) as session:
            session.run("""
                MATCH (p:Section {section_id: $parent})
                MATCH (c:Section {section_id: $child})
                WHERE p.doc_id = c.doc_id
                MERGE (p)-[:PARENT_OF]->(c)
            """, parent=parent_id, child=child_id)

    def get_section_hierarchy(self, doc_id: str) -> List[Dict]:
        """문서의 섹션 계층"""
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (d:Document {doc_id: $doc_id})-[:HAS_SECTION]->(s:Section)
                OPTIONAL MATCH (s)-[:PARENT_OF]->(child:Section)
                RETURN s, collect(child.section_id) as children
                ORDER BY s.section_id
            """, doc_id=doc_id)
            return [{"section": dict(r["s"]), "children": r["children"]} for r in result]

    def get_subsections_recursive(self, doc_id: str, parent_section_id: str) -> List[str]:
        """특정 조항의 모든 하위 조항을 재귀적으로 조회

        Args:
            doc_id: 문서 ID (예: "EQ-SOP-00001")
            parent_section_id: 부모 조항 번호 (예: "1", "2.1")

        Returns:
            부모 조항 + 모든 하위 조항의 section_id 리스트
            예: ["1", "1.1", "1.2", "1.2.1", "1.2.2", "1.3"]
        """
        with self.driver.session(database=self.database) as session:
            # Cypher의 경로 탐색을 사용하여 재귀적으로 모든 하위 조항 조회
            result = session.run("""
                MATCH (d:Document {doc_id: $doc_id})-[:HAS_SECTION]->(parent:Section {section_id: $section_id})
                OPTIONAL MATCH path = (parent)-[:PARENT_OF*]->(child:Section)
                WITH parent, collect(DISTINCT child.section_id) as descendants
                RETURN parent.section_id as root, descendants
            """, doc_id=doc_id, section_id=parent_section_id)

            record = result.single()
            if not record:
                return []

            # 부모 + 자식들을 하나의 리스트로
            all_sections = [record["root"]] + [s for s in record["descendants"] if s]
            return all_sections

    def get_section_content(self, section_id: str) -> Optional[Dict]:
        """특정 섹션의 상세 정보 조회

        Args:
            section_id: 섹션 ID (예: "EQ-SOP-00001:1", "EQ-SOP-00001:5.1.1")

        Returns:
            섹션 정보 딕셔너리 (section_id, title, content 등)
        """
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (s:Section {section_id: $section_id})
                RETURN s
            """, section_id=section_id)

            record = result.single()
            if record:
                return dict(record["s"])
            return None

    def search_sections(self, keyword: str, doc_id: str = None) -> List[Dict]:
        """섹션 검색"""
        query = """
            MATCH (s:Section)
            WHERE toLower(s.content) CONTAINS toLower($keyword)
               OR toLower(s.title) CONTAINS toLower($keyword)
        """
        if doc_id:
            query += " AND s.doc_id = $doc_id"
        query += " RETURN s LIMIT 10"

        with self.driver.session(database=self.database) as session:
            result = session.run(query, keyword=keyword, doc_id=doc_id)
            return [dict(r["s"]) for r in result]

    # ═══════════════════════════════════════════════════════════════════════════
    # 문서 간 참조
    # ═══════════════════════════════════════════════════════════════════════════

    def create_reference(self, from_doc: str, to_doc: str):
        """문서 간 참조 관계 (참조된 문서가 없으면 자동 생성)"""
        with self.driver.session(database=self.database) as session:
            session.run("""
                MERGE (from:Document {doc_id: $from_doc})
                ON CREATE SET from.title = $from_doc, from.version = "", from.effective_date = "", from.owning_dept = ""
                MERGE (to:Document {doc_id: $to_doc})
                ON CREATE SET to.title = $to_doc, to.version = "", to.effective_date = "", to.owning_dept = ""
                MERGE (from)-[:REFERENCES]->(to)
            """, from_doc=from_doc, to_doc=to_doc)

    def link_section_mentions(self, section_id: str, mentioned_docs: List[str]):
        """섹션에서 언급한 문서들 연결 (Section -[:MENTIONS]-> Document)"""
        with self.driver.session(database=self.database) as session:
            for doc_id in mentioned_docs:
                # MERGE: 참조된 Document가 없으면 자동 생성
                session.run("""
                    MATCH (s:Section {section_id: $section})
                    MERGE (d:Document {doc_id: $doc})
                    ON CREATE SET d.title = $doc, d.version = "", d.effective_date = "", d.owning_dept = ""
                    MERGE (s)-[:MENTIONS]->(d)
                """, section=section_id, doc=doc_id)

    def get_document_references(self, doc_id: str) -> Dict:
        """문서 참조 관계 (MENTIONS 기반)"""
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (d:Document {doc_id: $doc_id})

                // 이 문서의 섹션들이 MENTIONS하는 문서들
                OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)-[:MENTIONS]->(ref:Document)

                // 다른 문서의 섹션들이 이 문서를 MENTIONS하는 경우
                OPTIONAL MATCH (citing_section:Section)-[:MENTIONS]->(d)
                OPTIONAL MATCH (citing_doc:Document)-[:HAS_SECTION]->(citing_section)

                RETURN d,
                       collect(DISTINCT ref.doc_id) as references,
                       collect(DISTINCT citing_doc.doc_id) as cited_by
            """, doc_id=doc_id)
            record = result.single()
            if record:
                # null 값 제거
                references = [r for r in record["references"] if r]
                cited_by = [c for c in record["cited_by"] if c]
                return {
                    "document": dict(record["d"]),
                    "references": references,
                    "cited_by": cited_by
                }
            return None

    def get_document_relations(self, doc_id: str) -> Dict:
        """문서 관계 조회 (REFERENCES + MENTIONS 통합)

        Returns:
            {
                "doc_id": str,
                "title": str,
                "references_to": [{"doc_id": str, "title": str}],  # 이 문서가 참조하는 상위 문서들
                "referenced_by": [{"doc_id": str, "title": str}]   # 이 문서를 참조하는 하위 문서들
            }
        """
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (d:Document {doc_id: $doc_id})

                // REFERENCES: 이 문서가 참조하는 문서들 (상위 문서)
                OPTIONAL MATCH (d)-[:REFERENCES]->(ref_doc:Document)

                // MENTIONS: 이 문서의 섹션이 언급하는 문서들
                OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)-[:MENTIONS]->(mention_doc:Document)

                // REFERENCES: 이 문서를 참조하는 문서들 (하위 문서)
                OPTIONAL MATCH (citing_doc:Document)-[:REFERENCES]->(d)

                // MENTIONS: 다른 문서의 섹션이 이 문서를 언급
                OPTIONAL MATCH (citing_section:Section)-[:MENTIONS]->(d)
                OPTIONAL MATCH (citing_via_mention:Document)-[:HAS_SECTION]->(citing_section)

                RETURN d.doc_id as doc_id,
                       d.title as title,
                       collect(DISTINCT {doc_id: ref_doc.doc_id, title: ref_doc.title}) +
                       collect(DISTINCT {doc_id: mention_doc.doc_id, title: mention_doc.title}) as references_to,
                       collect(DISTINCT {doc_id: citing_doc.doc_id, title: citing_doc.title}) +
                       collect(DISTINCT {doc_id: citing_via_mention.doc_id, title: citing_via_mention.title}) as referenced_by
            """, doc_id=doc_id)
            record = result.single()
            if record:
                # null 값 제거 및 중복 제거
                references_to = [
                    r for r in record["references_to"]
                    if r.get("doc_id") and r.get("doc_id") != doc_id
                ]
                referenced_by = [
                    r for r in record["referenced_by"]
                    if r.get("doc_id") and r.get("doc_id") != doc_id
                ]

                # 중복 제거 (doc_id 기준)
                seen_refs = set()
                unique_refs = []
                for r in references_to:
                    if r["doc_id"] not in seen_refs:
                        seen_refs.add(r["doc_id"])
                        unique_refs.append(r)

                seen_cited = set()
                unique_cited = []
                for r in referenced_by:
                    if r["doc_id"] not in seen_cited:
                        seen_cited.add(r["doc_id"])
                        unique_cited.append(r)

                return {
                    "doc_id": record["doc_id"],
                    "title": record["title"],
                    "references_to": unique_refs,
                    "referenced_by": unique_cited
                }
            return None

    def get_impact_analysis(self, target_doc_id: str) -> List[Dict]:
        """특정 문서가 변경되었을 때, 이를 참조하고 있는 다른 문서(파급 효과) 조회

        Returns:
            List[Dict]: [
                {
                    "source_doc_id": "EQ-WI-0001",
                    "source_doc_title": "작업지침...",
                    "citing_section": "EQ-WI-0001:3.2",
                    "citing_content": "본 작업은 EQ-SOP-0001에 따른다..."
                }, ...
            ]
        """
        if not self.driver:
            print(f"🔴 Neo4j 드라이버가 초기화되지 않음")
            return []

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("""
                    MATCH (citing_section:Section)-[:MENTIONS]->(target:Document {doc_id: $doc_id})
                    MATCH (citing_doc:Document)-[:HAS_SECTION]->(citing_section)
                    RETURN citing_doc.doc_id as doc_id,
                           citing_doc.title as title,
                           citing_section.section_id as section_id,
                           citing_section.content as content
                """, doc_id=target_doc_id)

                impact_list = []
                for record in result:
                    impact_list.append({
                        "source_doc_id": record["doc_id"],
                        "source_doc_title": record["title"],
                        "citing_section": record["section_id"].split(":")[-1] if ":" in record["section_id"] else record["section_id"],
                        "citing_content": record["content"][:200] + "..." # 너무 길면 자름
                    })
                return impact_list
        except Exception as e:
            print(f"🔴 영향 분석 조회 실패: {e}")
            return []

    # ═══════════════════════════════════════════════════════════════════════════
    # 통계
    # ═══════════════════════════════════════════════════════════════════════════

    def get_graph_stats(self) -> Dict:
        """그래프 통계"""
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                OPTIONAL MATCH (d:Document) WITH count(d) as docs
                OPTIONAL MATCH (s:Section) WITH docs, count(s) as sections
                OPTIONAL MATCH (dt:DocumentType) WITH docs, sections, count(dt) as doc_types
                OPTIONAL MATCH (c:Concept) WITH docs, sections, doc_types, count(c) as concepts
                OPTIONAL MATCH ()-[r]->() WITH docs, sections, doc_types, concepts, count(r) as rels
                RETURN docs, sections, doc_types, concepts, rels
            """)
            record = result.single()
            return {
                "documents": record["docs"] or 0,
                "sections": record["sections"] or 0,
                "document_types": record["doc_types"] or 0,
                "concepts": record["concepts"] or 0,
                "relationships": record["rels"] or 0
            }

    def get_full_graph(self, include_sections: bool = False, doc_id: Optional[str] = None) -> Dict:
        """전체 문서 그래프 가져오기 (시각화용)

        Args:
            include_sections: True면 Section 노드/관계까지 포함
            doc_id: include_sections=True일 때 특정 문서 섹션만 포함하고 싶을 때 사용
        """
        with self.driver.session(database=self.database) as session:
            # 모든 Document 노드 가져오기 (doc_id 필터가 있으면 기준 문서 + 직접 연결 문서만)
            if doc_id:
                nodes_result = session.run("""
                    MATCH (d:Document {doc_id: $doc_id})
                    OPTIONAL MATCH (d)-[:REFERENCES|HAS_SECTION|MENTIONS*1..2]-(linked:Document)
                    WITH collect(DISTINCT d) + collect(DISTINCT linked) as docs
                    UNWIND docs as doc
                    OPTIONAL MATCH (doc)-[:IS_TYPE]->(dt:DocumentType)
                    RETURN DISTINCT doc.doc_id as id, doc.title as title, doc.version as version,
                           dt.code as doc_type, dt.name_kr as type_name
                """, doc_id=doc_id)
            else:
                nodes_result = session.run("""
                    MATCH (d:Document)
                    OPTIONAL MATCH (d)-[:IS_TYPE]->(dt:DocumentType)
                    RETURN d.doc_id as id, d.title as title, d.version as version,
                           dt.code as doc_type, dt.name_kr as type_name
                """)

            nodes = []
            node_ids = set()
            for record in nodes_result:
                node = {
                    "id": record["id"],
                    "kind": "document",
                    "title": record["title"],
                    "version": record["version"],
                    "doc_type": record["doc_type"],
                    "type_name": record["type_name"]
                }
                nodes.append(node)
                node_ids.add(record["id"])

            # 모든 REFERENCES와 MENTIONS 관계 가져오기
            if doc_id:
                links_result = session.run("""
                    MATCH (d:Document {doc_id: $doc_id})
                    OPTIONAL MATCH (d)-[:REFERENCES]->(to_doc:Document)
                    WITH d, collect(DISTINCT {source: d.doc_id, target: to_doc.doc_id, type: 'REFERENCES'}) as refs
                    OPTIONAL MATCH (d)-[:HAS_SECTION]->(:Section)-[:MENTIONS]->(mentioned:Document)
                    WITH refs + collect(DISTINCT {source: d.doc_id, target: mentioned.doc_id, type: 'MENTIONS_DOC'}) as rels
                    UNWIND rels as rel
                    RETURN rel.source as source, rel.target as target, rel.type as type
                """, doc_id=doc_id)
            else:
                links_result = session.run("""
                    MATCH (d1:Document)-[:REFERENCES]->(d2:Document)
                    RETURN DISTINCT d1.doc_id as source, d2.doc_id as target, 'REFERENCES' as type
                    UNION
                    MATCH (d1:Document)-[:HAS_SECTION]->(s:Section)-[:MENTIONS]->(d2:Document)
                    RETURN DISTINCT d1.doc_id as source, d2.doc_id as target, 'MENTIONS_DOC' as type
                """)

            links = []
            seen_links = set()
            for record in links_result:
                if not record["source"] or not record["target"]:
                    continue
                key = (record["source"], record["target"], record["type"])
                if key in seen_links:
                    continue
                seen_links.add(key)
                links.append({
                    "source": record["source"],
                    "target": record["target"],
                    "type": record["type"],
                })

            # Section 노드/관계 확장
            if include_sections:
                # 문서-문서 연결에 실제로 기여하는 섹션(MENTIONS 보유)만 포함
                if doc_id:
                    section_nodes_result = session.run("""
                        MATCH (d:Document {doc_id: $doc_id})-[:HAS_SECTION]->(s:Section)-[:MENTIONS]->(target:Document)
                        WHERE target.doc_id <> d.doc_id
                        RETURN s.section_id as id,
                               s.title as title,
                               s.doc_id as doc_id,
                               s.main_section as main_section,
                               s.clause_level as clause_level
                        ORDER BY s.section_id
                    """, doc_id=doc_id)
                else:
                    section_nodes_result = session.run("""
                        MATCH (d:Document)-[:HAS_SECTION]->(s:Section)-[:MENTIONS]->(target:Document)
                        WHERE target.doc_id <> d.doc_id
                        RETURN s.section_id as id,
                               s.title as title,
                               s.doc_id as doc_id,
                               s.main_section as main_section,
                               s.clause_level as clause_level
                        ORDER BY s.section_id
                    """)

                for record in section_nodes_result:
                    sid = record["id"]
                    if not sid or sid in node_ids:
                        continue
                    nodes.append({
                        "id": sid,
                        "kind": "section",
                        "doc_id": record["doc_id"],
                        "title": record["title"],
                        "main_section": record["main_section"],
                        "clause_level": record["clause_level"] or 0,
                    })
                    node_ids.add(sid)

                # Document -> Section
                if doc_id:
                    has_section_result = session.run("""
                        MATCH (d:Document {doc_id: $doc_id})-[:HAS_SECTION]->(s:Section)-[:MENTIONS]->(target:Document)
                        WHERE target.doc_id <> d.doc_id
                        RETURN d.doc_id as source, s.section_id as target
                    """, doc_id=doc_id)
                else:
                    has_section_result = session.run("""
                        MATCH (d:Document)-[:HAS_SECTION]->(s:Section)-[:MENTIONS]->(target:Document)
                        WHERE target.doc_id <> d.doc_id
                        RETURN d.doc_id as source, s.section_id as target
                    """)

                for record in has_section_result:
                    key = (record["source"], record["target"], "HAS_SECTION")
                    if key in seen_links:
                        continue
                    seen_links.add(key)
                    links.append({
                        "source": record["source"],
                        "target": record["target"],
                        "type": "HAS_SECTION"
                    })

                # Section -> Document (언급)
                if doc_id:
                    mention_result = session.run("""
                        MATCH (dsrc:Document {doc_id: $doc_id})-[:HAS_SECTION]->(s:Section)-[:MENTIONS]->(d:Document)
                        WHERE d.doc_id <> dsrc.doc_id
                        RETURN s.section_id as source, d.doc_id as target
                    """, doc_id=doc_id)
                else:
                    mention_result = session.run("""
                        MATCH (dsrc:Document)-[:HAS_SECTION]->(s:Section)-[:MENTIONS]->(d:Document)
                        WHERE d.doc_id <> dsrc.doc_id
                        RETURN s.section_id as source, d.doc_id as target
                    """)

                for record in mention_result:
                    key = (record["source"], record["target"], "MENTIONS")
                    if key in seen_links:
                        continue
                    seen_links.add(key)
                    links.append({
                        "source": record["source"],
                        "target": record["target"],
                        "type": "MENTIONS"
                    })

            return {
                "nodes": nodes,
                "links": links
            }


# ═══════════════════════════════════════════════════════════════════════════
# 유틸리티: 문서 업로드 헬퍼
# ═══════════════════════════════════════════════════════════════════════════

def upload_document_to_graph(graph: Neo4jGraphStore, result: dict, filename: str):
    """document_pipeline 결과를 Neo4j에 업로드 (evaluate_gmp_unified 호환)"""
    doc_id = result.get("doc_id") or "UNKNOWN"
    title = result.get("doc_title") or filename

    # 문서 타입 추출 (EQ-SOP, EQ-WI 등)
    doc_type_code = ""
    doc_type_kr = ""
    doc_type_en = ""
    if doc_id.startswith("EQ-SOP"):
        doc_type_code = "SOP"
        doc_type_kr = "표준운영절차서"
        doc_type_en = "Standard Operating Procedure"
    elif doc_id.startswith("EQ-WI"):
        doc_type_code = "WI"
        doc_type_kr = "작업지침서"
        doc_type_en = "Work Instruction"
    elif doc_id.startswith("EQ-FORM") or doc_id.startswith("EQ-FRM"):
        doc_type_code = "FRM"
        doc_type_kr = "양식"
        doc_type_en = "Form"

    # Document 생성 (버전 정보는 result에서 추출)
    version = result.get("version") or "1.0"
    graph.create_document(doc_id=doc_id, title=title, version=version)

    # DocumentType 생성 및 연결
    if doc_type_code:
        graph.create_document_type(doc_type_code, doc_type_kr, doc_type_en)
        graph.link_document_type(doc_id, doc_type_code)

    # 기본 DocumentType 노드들 초기화 (MERGE이므로 중복 없음)
    doc_types = [
        ("SOP", "표준운영절차서", "Standard Operating Procedure"),
        ("WI", "작업지침서", "Work Instruction"),
        ("FRM", "양식", "Form"),
        ("MBR", "제조기록서", "Master Batch Record"),
        ("SPEC", "규격서", "Specification"),
    ]
    for code, name_kr, name_en in doc_types:
        graph.create_document_type(code, name_kr, name_en)

    # 기본 Concept 노드들 초기화 (MERGE이므로 중복 없음)
    concepts = [
        ("user_account", "사용자 접근 관리", "User Access Management", "사용자 계정, 권한, 역할 관리"),
        ("document_lifecycle", "문서 수명주기", "Document Lifecycle", "문서 작성, 승인, 개정, 폐기 등"),
        ("training", "교육 및 자격", "Training and Qualification", "교육, 훈련, 자격 관리"),
        ("system_configuration", "시스템 설정", "System Configuration", "시스템 구성 및 설정"),
        ("audit_evidence", "감사 증적", "Audit Evidence", "감사 대응 자료"),
    ]
    for concept_id, name_kr, name_en, description in concepts:
        graph.create_concept(concept_id, name_kr, name_en, description)

    # Section 생성 및 멘션 수집
    # v8.7: 같은 clause_id의 청크들을 먼저 병합
    from collections import defaultdict

    # 1. clause_id별로 청크 그룹화
    sections_by_clause = defaultdict(list)
    for chunk in result.get("chunks", []):
        meta = chunk.get("metadata", {})
        clause_id = meta.get("clause_id")
        if not clause_id:
            continue
        sections_by_clause[clause_id].append(chunk)

    # 2. 각 clause_id별로 하나의 Section 생성
    all_mentions = set()
    for clause_id, chunks in sections_by_clause.items():
        # 첫 번째 청크의 메타데이터 사용 (모두 동일하므로)
        first_chunk = chunks[0]
        meta = first_chunk.get("metadata", {})

        # section_id는 문서ID:조항번호 형식으로 전역 고유하게
        section_id = f"{doc_id}:{clause_id}"
        main_section = clause_id.split('.')[0] if '.' in clause_id else clause_id

        # 모든 LLM 메타데이터 필드 포함
        llm_meta = {
            "content_type": meta.get("content_type", ""),
            "main_topic": meta.get("main_topic", ""),
            "sub_topics": meta.get("sub_topics", []),
            "actors": meta.get("actors", []),
            "actions": meta.get("actions", []),
            "conditions": meta.get("conditions", []),
            "summary": meta.get("summary", ""),
            "intent_scope": meta.get("intent_scope", ""),
            "intent_summary": meta.get("intent_summary", ""),
            "language": meta.get("language", "ko"),
        }

        # 같은 clause_id의 모든 청크 content 병합
        # title은 모든 청크에 동일하므로 한 번만, content는 모두 합치기
        title = meta.get("title", "")

        # 각 청크에서 title을 제거하고 content만 추출
        content_parts = []
        for chunk in chunks:
            text = chunk.get("text", "")
            # title 제거 (title + "\n\n" + content 형식)
            if text.startswith(title):
                content_only = text[len(title):].strip()
                if content_only.startswith("\n\n"):
                    content_only = content_only[2:].strip()
                content_parts.append(content_only)
            else:
                content_parts.append(text)

        # 전체 content = title + 모든 content_parts
        merged_content = f"{title}\n\n" + "\n\n".join(content_parts)

        graph.create_section(
            doc_id=doc_id,
            section_id=section_id,
            title=title,
            content=merged_content,
            clause_level=meta.get("clause_level", 0),
            main_section=main_section,
            llm_meta=llm_meta
        )

        # Concept 연결 (intent_scope가 있으면)
        intent_scope = llm_meta.get("intent_scope", "")
        if intent_scope:
            graph.link_section_concept(section_id, intent_scope)

        # 계층 관계 (부모도 문서ID 포함)
        if '.' in clause_id:
            parent_clause = '.'.join(clause_id.split('.')[:-1])
            parent_section_id = f"{doc_id}:{parent_clause}"
            graph.create_section_hierarchy(parent_section_id, section_id)

        # 타 문서/조항 언급 추출 (모든 청크에서)
        for chunk in chunks:
            content = chunk.get("text", "")
            # 문서 ID 패턴 (EQ-SOP-00009, EQ-WI-00012 등)
            doc_mentions = re.findall(r'(EQ-[A-Z]+-\d{5})', content, re.IGNORECASE)
            if doc_mentions:
                unique_mentions = list(set([m.upper() for m in doc_mentions]))
                graph.link_section_mentions(section_id, unique_mentions)
                all_mentions.update(unique_mentions)

    # [중요] 문서 단위의 REFERENCES 관계 생성
    # 조항 레벨의 MENTIONS뿐만 아니라 문서 자체의 관계를 맺어 거시적 영향도 분석 지원
    for mentioned_doc in all_mentions:
        if mentioned_doc != doc_id: # 자기 자신 참조 제외
            graph.create_reference(doc_id, mentioned_doc)
            print(f"  🔗 문서 레퍼런스 생성: {doc_id} -> {mentioned_doc}")

    # [중요] 계층 구조 초기화 호출
    # 매 업로드마다 실행하여 계층 관계(SOP > WI > FORM)를 보장함
    graph.init_type_hierarchy()

