"""
문서 처리 파이프라인 v2.0 - PDF 조항 단위 파싱

단계:
1. PDF → 마크다운 변환
2. 조항 파싱 (정규식, evaluate_gmp_unified 방식)
3. 조항별 메타데이터 추출 (LLM)
4. 청크 생성
"""

import os
import re
import json
from typing import List, Dict
from pathlib import Path
from io import BytesIO
from .llm import get_llm_response


# ═══════════════════════════════════════════════════════════════════════════
# 1단계: PDF → 마크다운 변환
# ═══════════════════════════════════════════════════════════════════════════

def pdf_to_markdown(pdf_content: bytes) -> str:
    """
    PDF를 마크다운으로 변환 (다중 파서 폴백)

    Returns:
        str: 마크다운 텍스트
    """
    # 1순위: pdfplumber (가장 안정적)
    try:
        import pdfplumber
        md_lines = []
        with pdfplumber.open(BytesIO(pdf_content)) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ''
                if text.strip():
                    md_lines.append(f"<!-- PAGE:{i + 1} -->")
                    md_lines.append(text)

        markdown = '\n'.join(md_lines)
        if len(markdown.strip()) > 100:
            print("    ✓ pdfplumber로 변환 완료")
            return markdown
    except Exception as e:
        print(f"    pdfplumber 실패: {e}")

    # 2순위: PyMuPDF
    try:
        import fitz
        pdf = fitz.open(stream=pdf_content, filetype="pdf")
        md_lines = []
        for page_num, page in enumerate(pdf):
            text = page.get_text()
            if text.strip():
                md_lines.append(f"<!-- PAGE:{page_num + 1} -->")
                md_lines.append(text)

        markdown = '\n'.join(md_lines)
        if len(markdown.strip()) > 100:
            print("    ✓ PyMuPDF로 변환 완료")
            return markdown
    except Exception as e:
        print(f"    PyMuPDF 실패: {e}")

    raise Exception("PDF 변환 실패 (모든 파서 실패)")


# ═══════════════════════════════════════════════════════════════════════════
# 유틸리티: PDF 노이즈 제거
# ═══════════════════════════════════════════════════════════════════════════

def _clean_noise_globally(markdown: str) -> str:
    """
    전체 문서에서 반복되는 행(헤더/푸터)을 동적으로 탐지하고 제거합니다.
    (하드코딩된 정규식을 최소화하고 빈도 분석 기반으로 동작)
    """
    if not markdown:
        return ""

    # 1. 페이지 단위로 분할하여 컨텍스트 수집
    # markers는 보존하면서 분할
    parts = re.split(r'(<!-- PAGE:\d+ -->)', markdown)
    page_data = []
    for i in range(1, len(parts), 2):
        marker = parts[i]
        content = parts[i+1] if i+1 < len(parts) else ""
        page_num_match = re.search(r'\d+', marker)
        page_num = int(page_num_match.group()) if page_num_match else 0
        page_data.append({"marker": marker, "page": page_num, "content": content})

    if not page_data:
        # 마커가 없으면 최소한의 페까지만 처리
        return re.sub(r'<!-- PAGE:\d+ -->', '', markdown).strip()

    # 2. 행 빈도 분석 (상단/하단 5행 중심)
    # 숫자를 #으로 치환하여 "Page 1 of 10"과 "Page 2 of 10"을 동일한 템플릿으로 인식
    line_templates = {}
    for p in page_data:
        lines = [l.strip() for l in p["content"].split('\n') if l.strip()]
        candidates = set(lines[:5] + lines[-5:])
        for line in candidates:
            # 특수 기호나 숫자가 포함된 문패턴을 템플릿화
            template = re.sub(r'\d+', '#', line)
            line_templates[template] = line_templates.get(template, 0) + 1

    # 3. 30% 이상의 페이지에서 나타나는 템플릿을 노이즈로 간주
    total_pages = len(page_data)
    noise_threshold = max(2, total_pages * 0.3)
    noise_templates = {t for t, count in line_templates.items() if count >= noise_threshold}

    # 디버그: 노이즈로 판정된 템플릿 출력 (필요시 주석 해제)
    # if noise_templates:
    #     print(f"    [노이즈 제거] {len(noise_templates)}개 템플릿 제거 (임계값: {noise_threshold:.1f}페이지)")
    #     for template in list(noise_templates)[:10]:  # 처음 10개
    #         print(f"      - {template[:70]}")

    # 4. 노이즈 제거 및 본문 재구성
    # 조항 패턴을 가진 라인은 보존 (단, "1 of 11" 같은 페이지 번호는 제외)
    # "1." (섹션 헤더), "1.1" (조항), 조항만 있는 경우 모두 매칭
    clause_pattern = re.compile(r'^\s*(\d+(?:\.\d+)*\.?)(?:\s+(?!of\s+\d)|$)')

    preserved_clauses = 0
    removed_lines = 0

    cleaned_parts = []
    for p in page_data:
        lines = p["content"].split('\n')
        cleaned_lines = []
        for l in lines:
            stripped = l.strip()
            if not stripped:
                cleaned_lines.append("")
                continue

            # 조항 패턴이 있으면 무조건 보존
            if clause_pattern.match(stripped):
                cleaned_lines.append(l)
                preserved_clauses += 1
                continue

            template = re.sub(r'\d+', '#', stripped)
            if template in noise_templates:
                removed_lines += 1
                continue
            cleaned_lines.append(l)

        # 페이지 마커는 정보 추출을 위해 일단 유지
        cleaned_parts.append(p["marker"] + "\n" + "\n".join(cleaned_lines))

    # 디버그 로깅 (필요시 주석 해제)
    # print(f"    [노이즈 필터] 조항 보존: {preserved_clauses}개, 노이즈 제거: {removed_lines}개")

    result = "\n".join(cleaned_parts)
    
    # 5. 전형적인 문서 종료 마커 등 최소한의 정적 정제
    result = re.sub(r'\*{3,}\s*END OF DOCUMENT\s*\*{3,}', '', result, flags=re.IGNORECASE)
    
    return result.strip()


def _normalize_text(text: str) -> str:
    """텍스트 정규화: 불필요한 줄바꿈 제거, 한영 문단 구분 유지"""

    # 1. 연속된 줄바꿈 정리 (3개 이상 -> 2개)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 2. 문단별 처리
    paragraphs = text.split('\n\n')
    normalized_paragraphs = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # 한글 비율 체크 (한글 문단 vs 영어 문단 구분)
        korean_chars = len(re.findall(r'[가-힣]', para))
        total_chars = len(re.sub(r'\s', '', para))
        is_korean = korean_chars > total_chars * 0.3  # 30% 이상이 한글

        # 같은 언어 문단 내에서만 줄바꿈 제거
        # 단, 문장 끝(마침표 등)은 유지
        if '\n' in para:
            # 줄바꿈을 공백으로 치환
            normalized = para.replace('\n', ' ')
            # 연속 공백 제거
            normalized = re.sub(r'\s+', ' ', normalized)
            normalized_paragraphs.append(normalized.strip())
        else:
            normalized_paragraphs.append(para)

    # 3. 문단 재결합 (한영 구분은 \n\n 유지)
    return '\n\n'.join(normalized_paragraphs)


def _split_recursive(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """긴 텍스트를 재귀적으로 분할 (문장, 줄바꿈 기준)"""
    if len(text) <= chunk_size:
        return [text]

    separators = ["\n\n", "\n", ". ", " ", ""]
    chunks = []

    parts = []
    for sep in separators:
        if sep in text:
            parts = text.split(sep)
            break

    if not parts:
        return [text[:chunk_size]]

    current_chunk = ""
    for p in parts:
        if len(current_chunk) + len(p) < chunk_size:
            current_chunk += p + (sep if sep != "" else "")
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = p + (sep if sep != "" else "")

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# ═══════════════════════════════════════════════════════════════════════════
# 2단계: 조항 파싱 (evaluate_gmp_unified 방식)
# ═══════════════════════════════════════════════════════════════════════════

def parse_clauses(markdown: str, max_level: int = None) -> List[Dict]:
    """
    마크다운에서 조항 추출 (evaluate_gmp_unified.py의 load_pdf_by_clause 방식)

    조항 패턴: 5.1.2 형식의 숫자 번호

    Args:
        markdown: 마크다운 텍스트
        max_level: 최대 조항 깊이 (None이면 모든 레벨 포함)

    Returns:
        List[Dict]: 조항 리스트
            - clause: 조항 번호 (예: "5.1.2")
            - title: 조항 제목
            - content: 조항 내용
            - level: 깊이 (0부터 시작)
    """
    # 볼드 마커 제거 (노이즈 분석 전에 먼저 실행)
    # **1** **목적 Purpose** → 1 목적 Purpose
    # 이렇게 하면 노이즈 템플릿 매칭에서 섹션 헤더가 제거되지 않음
    markdown = re.sub(r'\*\*', '', markdown)  # 모든 볼드 마커 제거

    # 동적 노이즈 제거 (빈도 분석 기반)
    markdown = _clean_noise_globally(markdown)

    # 페이지 번호 패턴 제거 (조항으로 잘못 인식되는 것 방지)
    # 예: "1 of 11", "2 of 11", "Page 1", "페이지 1" 등
    page_number_patterns = [
        r'(?m)^\s*(\d+)\s+of\s+\d+.*?$',  # "1 of 11 Number: ..." 형식
        r'(?m)^\s*(?:Page|페이지)\s+\d+.*?$',  # "Page 1", "페이지 1" 형식
        r'(?m)^\s*\d+\s*/\s*\d+.*?$',  # "1/11" 형식
    ]
    for pattern in page_number_patterns:
        markdown = re.sub(pattern, '', markdown, flags=re.MULTILINE)

    # 개정이력/변경이력 섹션 제거 (제목 행만 제거)
    # 주의: 이전에 re.DOTALL을 사용하면 문서 끝까지 모두 제거되므로 제거함
    revision_patterns = [
        r'(?im)^(개정\s*이력|변경\s*이력|Revision\s+History|Change\s+History)[^\n]*\n?',
        r'(?im)^(\d+\.\d+\s+전체\s+변경관리)[^\n]*\n?'
    ]
    for pattern in revision_patterns:
        markdown = re.sub(pattern, '', markdown)

    # 조항 번호 패턴 (매우 관대하게)
    # "1." (섹션 헤더) 및 "1.1" (조항) 모두 매칭하도록 trailing dot 허용
    pattern = r'(?:^|\n)\s*(\d+(?:\.\d+)*\.?)\s+'
    all_matches = list(re.finditer(pattern, markdown))

    print(f"    디버그: {len(all_matches)}개 패턴 발견")

    clauses = []
    filtered_count = 0
    filtered_details = []  # 제외된 조항 상세 정보

    for i, m in enumerate(all_matches):
        clause_num = m.group(1)
        level = clause_num.count(".")

        # 해당 조항이 위치한 페이지 찾기
        page_num = 1
        page_marker_before = re.findall(r'<!-- PAGE:(\d+) -->', markdown[:m.start()])
        if page_marker_before:
            page_num = int(page_marker_before[-1])

        # 레벨 필터링
        if max_level is not None and level > max_level:
            filtered_count += 1
            filtered_details.append(f"{clause_num}: 레벨 초과")
            continue

        # 내용 추출
        start = m.end()
        end = all_matches[i+1].start() if i+1 < len(all_matches) else len(markdown)
        content = markdown[start:end]

        # 본문 내의 페이지 마커 제거
        content = re.sub(r'<!-- PAGE:\d+ -->', '', content).strip()

        # 최소 길이 체크
        if len(content) < 5:
            filtered_count += 1
            filtered_details.append(f"{clause_num}: 내용 너무 짧음 ({len(content)}자)")
            continue

        # 제목과 본문 분리
        # v8.5: 제목이 여러 줄인 경우 대비 - 첫 문장 또는 200자까지

        # 1. 마침표나 콜론으로 끝나는 첫 문장 찾기
        sentence_ends = ['. ', '.\n', ': ', ':\n']
        title_end = -1

        for end_marker in sentence_ends:
            pos = content.find(end_marker)
            if pos > 0 and pos < 300:  # 300자 이내의 마침표/콜론
                title_end = pos + 1
                break

        if title_end > 0:
            # 마침표/콜론으로 끝나는 첫 문장을 제목으로
            title = content[:title_end].replace('\n', ' ').strip()
            body = content[title_end:].strip()
        else:
            # 마침표가 없으면 첫 200자를 제목으로
            first_newline = content.find('\n\n')  # 문단 구분
            if 0 < first_newline < 200:
                title = content[:first_newline].replace('\n', ' ').strip()
                body = content[first_newline:].strip()
            else:
                title = content[:200].replace('\n', ' ').strip()
                body = content[200:].strip() if len(content) > 200 else ""

        # 본문에서 노이즈 제거
        # body = _clean_pdf_noise(body) # 전역 정제로 대체됨

        # 필터링: 잘못된 매칭 제거
        skip = False
        skip_reason = ""

        # 1. 제목이 너무 짧으면 제외
        if len(title) < 2:
            filtered_count += 1
            skip = True
            skip_reason = f"제목 너무 짧음: '{title}'"

        # 2. 버전 번호 (1.0 전체 변경...) 제외 - 더 정밀한 조건
        elif re.match(r'^\d+\.\d+$', clause_num) and level == 1:
            # 제목이 "버전", "개정 내역", "변경 이력" 등으로 시작하는 경우만 제외
            if re.match(r'^(버전|개정\s*내역|변경\s*이력|Revision\s+History|Change\s+Log|Version)\b', title, re.IGNORECASE):
                filtered_count += 1
                skip = True
                skip_reason = f"버전 번호 패턴: '{title}'"

        # 3. 괄호로만 구성된 제목 제외
        elif re.match(r'^\([^\)]+\)\s*$', title):
            filtered_count += 1
            skip = True
            skip_reason = f"괄호만: '{title}'"

        # 4. 숫자와 기호만 있는 제목 제외
        elif re.match(r'^[\d\s\(\)\-\.,]+$', title):
            filtered_count += 1
            skip = True
            skip_reason = f"숫자/기호만: '{title}'"

        # 5. "characters", "numbers" 같은 설명 텍스트 제외
        elif re.match(r'^(characters|numbers|digits|letters|Level)\b', title, re.IGNORECASE):
            filtered_count += 1
            skip = True
            skip_reason = f"설명 텍스트: '{title}'"

        # 6. "of" 나 페이지 번호로 시작하는 경우 제외
        elif re.match(r'^(of\s+\d+|page\s+\d+)', title, re.IGNORECASE):
            filtered_count += 1
            skip = True
            skip_reason = f"페이지 번호: '{title}'"

        if skip:
            filtered_details.append(f"{clause_num}: {skip_reason}")
            continue

        clauses.append({
            "clause": clause_num,
            "title": title,
            "content": body,
            "level": level,
            "page": page_num
        })

    print(f"    ✓ {len(clauses)}개 조항 추출")

    if filtered_count > 0:
        print(f"    ⚠ {filtered_count}개 조항 제외됨:")
        for detail in filtered_details[:10]:  # 최대 10개만 출력
            print(f"      - {detail}")
        if len(filtered_details) > 10:
            print(f"      ... 외 {len(filtered_details) - 10}개")

    return clauses


def parse_docx_clauses(docx_content: bytes) -> List[Dict]:
    """
    DOCX 파일의 다단계 목록 구조(ilvl)를 읽어 조항 구조 생성.

    Normal 스타일 + numPr ilvl 값으로 계층 판단:
      ilvl=0 → 대분류 (level 0)
      ilvl=1 → 중분류 (level 1)
      ilvl=2 → 소분류 (level 2)
    List Paragraph / ilvl 없는 Normal → 본문 content

    Heading 스타일도 함께 지원 (혼합 문서 대응)
    """
    from docx import Document as DocxDocument
    from docx.oxml.ns import qn
    from io import BytesIO

    doc = DocxDocument(BytesIO(docx_content))

    def get_ilvl(para) -> int | None:
        """numPr에서 ilvl 값 추출"""
        numPr = para._element.find('.//' + qn('w:numPr'))
        if numPr is not None:
            ilvl_elem = numPr.find(qn('w:ilvl'))
            if ilvl_elem is not None:
                try:
                    return int(ilvl_elem.get(qn('w:val'), -1))
                except (ValueError, TypeError):
                    pass
        return None

    def get_heading_level(style_name: str) -> int | None:
        """Heading 1~6 / 제목 1~6 스타일에서 레벨 추출"""
        for prefix in ("Heading ", "제목 "):
            if style_name.startswith(prefix):
                try:
                    return int(style_name[len(prefix):]) - 1
                except ValueError:
                    pass
        return None

    counters = [0] * 9
    clauses = []
    current_clause = None
    content_lines = []

    def flush_clause():
        nonlocal current_clause, content_lines
        if current_clause is not None:
            current_clause["content"] = "\n".join(content_lines).strip()
            if current_clause["title"] or len(current_clause["content"]) >= 3:
                clauses.append(current_clause)
        current_clause = None
        content_lines = []

    def make_clause(level: int, title: str):
        nonlocal current_clause, content_lines
        flush_clause()
        counters[level] += 1
        for i in range(level + 1, len(counters)):
            counters[i] = 0
        clause_num = ".".join(str(counters[i]) for i in range(level + 1))
        current_clause = {
            "clause": clause_num,
            "title": title,
            "content": "",
            "level": level,
            "page": 1,
        }
        content_lines = []

    for para in doc.paragraphs:
        style_name = para.style.name if para.style else "Normal"
        text = para.text.strip()
        if not text:
            continue

        ilvl = get_ilvl(para)
        heading_level = get_heading_level(style_name)

        if heading_level is not None:
            # Word 헤딩 스타일 우선
            make_clause(heading_level, text)
        elif ilvl is not None and style_name == "Normal":
            # Normal + ilvl → 구조적 항목
            make_clause(ilvl, text)
        else:
            # List Paragraph 또는 ilvl 없는 일반 본문
            if current_clause is not None:
                content_lines.append(text)
            else:
                # 조항 없이 시작하는 본문 → 가상 최상위 조항
                make_clause(0, text[:100])

    flush_clause()

    print(f"    ✓ Word 다단계 목록 기반 {len(clauses)}개 조항 추출")
    return clauses


# ═══════════════════════════════════════════════════════════════════════════
# 4단계: 메타데이터 추출 (evaluate_gmp_unified 방식)
# ═══════════════════════════════════════════════════════════════════════════

def extract_metadata(content: str, clause_id: str, title: str,
                     embed_model=None) -> Dict:
    """
    조항별 메타데이터 추출 (LLM 사용)

    Returns:
        Dict: 메타데이터
            - content_type, main_topic, actors, actions, ...
            - intent_summary, intent_embedding
    """
    if len(content.strip()) < 30:
        return _default_metadata()

    prompt = f"""당신은 GMP 문서 분석 전문가입니다.
다음 문서 청크를 분석해서 검색에 유용한 메타데이터를 추출하세요.

## 청크 정보
- 조항번호: {clause_id}
- 제목: {title}
- 내용:
{content[:1000]}

## 추출할 메타데이터
다음 항목들을 JSON 형식으로 추출하세요:

1. content_type: 이 청크가 설명하는 내용의 유형 (예: 목적, 정의, 책임, 절차, 기준, 기록, 참고문헌 등)
2. main_topic: 이 청크의 핵심 주제
3. sub_topics: 관련 세부 주제들 (리스트, 최대 3개)
4. actors: 언급된 역할자/담당자 (리스트)
5. actions: 수행해야 하는 행위/절차 (리스트, 최대 3개)
6. conditions: 특수 조건이나 상황 (리스트)
7. summary: 한 문장 요약 (30자 이내)
8. intent_scope: 이 조항이 다루는 관리 영역 (다음 중 1개만 선택)
   - user_account: 사용자 계정·권한·역할 관리
   - document_lifecycle: 문서의 수명주기 (작성, 승인, 개정, 폐기 등)
   - system_configuration: 시스템 설정 or 구조 변경
   - audit_evidence: 감사대응에 관련한 자료
   - training: 교육, 훈련, 자격
9. intent_summary: 이 조항이 어떤 질문에 답하는지를 영어 1문장으로 요약 (예: "What is the purpose of Level 1 quality manual?")
10. language: 이 청크의 주요 언어 ("ko" 또는 "en")

## 출력
JSON만 출력:
{{"content_type": "...", "main_topic": "...", "sub_topics": [...], "actors": [...], "actions": [...], "conditions": [...], "summary": "...", "intent_scope": "...", "intent_summary": "...", "language": "..."}}
"""

    try:
        response = get_llm_response(prompt, llm_model="gpt-4o-mini", llm_backend="openai", max_tokens=4096, temperature=0)
        result = response.strip()

        # JSON 블록 파싱
        if "```" in result:
            json_str = result.split("```")[1]
            if json_str.startswith("json"):
                json_str = json_str[4:]
            result = json_str.strip()

        metadata = json.loads(result)

        # intent_embedding 생성
        if embed_model and metadata.get("intent_summary"):
            try:
                embedding = embed_model.encode([metadata["intent_summary"]])[0].tolist()
                metadata["intent_embedding"] = embedding
            except:
                metadata["intent_embedding"] = []
        else:
            metadata["intent_embedding"] = []

        return metadata

    except Exception as e:
        print(f"    메타데이터 추출 실패: {e}")
        return _default_metadata()


def _default_metadata() -> Dict:
    """기본 메타데이터"""
    return {
        "content_type": "",
        "main_topic": "",
        "sub_topics": [],
        "actors": [],
        "actions": [],
        "conditions": [],
        "summary": "",
        "intent_scope": "",
        "intent_summary": "",
        "language": "ko",
        "intent_embedding": []
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5단계: 청크 생성
# ═══════════════════════════════════════════════════════════════════════════

def create_chunks(clauses: List[Dict], doc_id: str, doc_title: str,
                 use_llm_metadata: bool = False, embed_model=None,
                 chunk_size: int = 800, chunk_overlap: int = 100) -> List[Dict]:
    """
    조항을 청크로 변환 (긴 조항은 재분할)

    Returns:
        List[Dict]: 청크 리스트
    """
    chunks = []
    global_idx = 0

    if use_llm_metadata:
        print(f"    LLM 메타데이터 추출: gpt-4o-mini")

    for idx, clause in enumerate(clauses):
        clause_num = clause["clause"]
        title = clause["title"]
        content = clause["content"]
        level = clause["level"]

        # 조항 번호가 없으면 인덱스 기반 대체
        clause_id = clause_num or f"섹션_{idx+1}"

        # 1. 메타데이터 추출 (조항별로 한 번 수행)
        if use_llm_metadata:
            print(f"    [{idx+1}/{len(clauses)}] 분석: {clause_id} {title[:20]}...")
            meta = extract_metadata(content, clause_id, title, embed_model)
        else:
            print(f"    [{idx+1}/{len(clauses)}] 저장: {clause_id} {title[:20]}...")
            meta = _default_metadata()

        # 2. 긴 조항 재분할 (벡터 검색 품질 향상)
        # v8.7: content가 500자 이상이면 분할, 각 청크에 title 포함

        # content가 500자를 넘으면 여러 파트로 분할
        if len(content) > 500:
            content_parts = _split_recursive(content, chunk_size=500, chunk_overlap=50)
        else:
            content_parts = [content]

        # 각 content 파트마다 title을 붙여서 청크 생성
        for p_idx, content_part in enumerate(content_parts):
            # 각 청크 = title + content_part
            chunk_text = f"{title}\n\n{content_part}"
            
            # 최종 메타데이터
            full_meta = {
                "doc_id": doc_id,
                "doc_title": doc_title,
                "clause_id": clause_id,
                "title": title,
                "clause_level": level,
                "main_section": clause_id.split('.')[0] if '.' in str(clause_id) else clause_id,
                "page": clause.get("page", 1),
                "chunk_part": p_idx + 1 if len(content_parts) > 1 else None,
                **meta
            }

            chunks.append({
                "text": chunk_text.strip(),
                "index": global_idx,
                "metadata": full_meta
            })
            global_idx += 1

    return chunks


# ═══════════════════════════════════════════════════════════════════════════
# 통합 파이프라인
# ═══════════════════════════════════════════════════════════════════════════

def process_document(
    file_path: str,
    content: bytes = None,
    doc_id: str = None,
    max_clause_level: int = None,
    use_llm_metadata: bool = False,
    embed_model=None
) -> Dict:
    """
    PDF 문서 처리 메인 함수

    Args:
        file_path: PDF 파일 경로
        content: PDF 바이너리 (없으면 파일에서 읽음)
        doc_id: 문서 ID (없으면 파일명에서 추출)
        max_clause_level: 최대 조항 레벨 (None이면 모든 레벨 포함)
        use_llm_metadata: LLM 메타데이터 추출 여부
        embed_model: 임베딩 모델 (intent_embedding용)

    Returns:
        Dict: 처리 결과
            - success: 성공 여부
            - chunks: 청크 리스트
            - errors: 에러 메시지
    """
    print(f"\n{'='*60}")
    print(f"문서 처리: {Path(file_path).name}")
    print(f"{'='*60}")

    try:
        # 파일 읽기
        if content is None:
            with open(file_path, 'rb') as f:
                content = f.read()

        # 문서 ID 및 버전 결정
        if doc_id is None:
            # ID 추출: EQ-SOP-00004 형식 (대문자-대문자-숫자)
            id_match = re.search(r'[A-Z]+-[A-Z]+-\d+', file_path)
            doc_id = id_match.group() if id_match else Path(file_path).stem
            
        # 1차 버전 추출: 파일명 패턴 (v1.0, _v2.0 등)
        version = None
        version_match = re.search(r'[vV_-]?(\d+\.\d+)', Path(file_path).name)
        if version_match:
            version = version_match.group(1)
            print(f"     [추출] 파일명에서 버전 감지: {version}")

        doc_title = Path(file_path).stem

        # 1. 문서 변환 (PDF → 마크다운, 텍스트는 직접 사용)
        file_ext = Path(file_path).suffix.lower()
        if file_ext in ['.md', '.markdown', '.txt']:
            print(f"\n[1/4] 텍스트 시스템 감지 ({file_ext}): 변환 생략 및 직접 로드")
            if isinstance(content, bytes):
                markdown = content.decode('utf-8')
            else:
                markdown = content
        elif file_ext in ['.docx', '.doc']:
            print(f"\n[1/4] DOCX → 마크다운 변환")
            markdown = docx_to_markdown(content)
        else:
            print("\n[1/4] PDF → 마크다운 변환")
            markdown = pdf_to_markdown(content)

        # 2. 조항 파싱
        print("\n[2/4] 조항 파싱")
        clauses = parse_clauses(markdown, max_level=max_clause_level)

        if not clauses:
            if file_ext in ['.docx', '.doc']:
                # DOCX는 Word 헤딩 스타일로 재시도
                print("    숫자 조항 없음 → Word 헤딩 스타일로 재파싱")
                clauses = parse_docx_clauses(content)

            if not clauses:
                return {
                    "success": False,
                    "chunks": [],
                    "errors": ["조항을 찾을 수 없습니다."]
                }
            
        # 2.5 지능형 버전 추출 (파일명에 없을 경우 본문 분석)
        if not version and use_llm_metadata:
            print("     [추출] 파일명에 버전 정보가 없어 본문 분석을 시작합니다...")
            try:
                from backend.agent import get_openai_client
                client = get_openai_client()
                # 앞뒤 2000자 분석
                sample_text = markdown[:2000] + "\n...\n" + markdown[-2000:]
                prompt = f"""다음 GMP 문서 내용(개정 이력 포함)을 보고 이 문서의 '현재 버전'을 찾아내세요.
                - 형식은 1.0, 2.0 등 숫자.숫자 여야 합니다.
                - 가장 최근(마지막) 개정이력을 기준으로 하세요.
                - 버전 번호만 답변하세요 (예: 2.0). 찾지 못하면 '1.0'이라고 답하세요.
                
                [문서 내용]
                {sample_text}
                """
                res = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}]
                )
                extracted_v = res.choices[0].message.content.strip()
                v_match_llm = re.search(r'(\d+\.\d+)', extracted_v)
                if v_match_llm:
                    version = v_match_llm.group(1)
                    print(f"     [추출] 본문 분석을 통해 버전 감지: {version}")
            except Exception as ve:
                print(f"     [추출] 본문 버전 추출 실패: {ve}")

        # 3. 청크 생성
        print(f"\n[3/4] 청크 생성 {'(LLM 메타데이터 포함)' if use_llm_metadata else ''}")
        chunks = create_chunks(clauses, doc_id, doc_title, use_llm_metadata, embed_model)

        # 4. 완료
        print(f"\n[4/4] 완료!")
        print(f"  - 조항: {len(clauses)}개")
        print(f"  - 청크: {len(chunks)}개")

        return {
            "success": True,
            "chunks": chunks,
            "doc_id": doc_id,
            "doc_title": doc_title,
            "version": version or "1.0",
            "total_clauses": len(clauses),
            "markdown": markdown,  # 원본 마크다운 추가
            "clauses": clauses,  # 조항 구조 추가
            "errors": []
        }

    except Exception as e:
        print(f"\n❌ 오류: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "chunks": [],
            "errors": [str(e)]
        }


# ═══════════════════════════════════════════════════════════════════════════
# 테스트
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("사용법: python document_pipeline.py <PDF파일> [--llm-meta]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    use_llm = "--llm-meta" in sys.argv

    result = process_document(
        file_path=pdf_path,
        use_llm_metadata=use_llm
    )

    if result["success"]:
        print(f"\n🟢 성공!")
    else:
        print(f"\n🔴 실패: {result['errors']}")
