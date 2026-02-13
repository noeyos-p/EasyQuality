"""
OnlyOffice 서비스 - 에디터 설정 JSON 생성 및 콜백 처리

환경변수:
  ONLYOFFICE_SERVER_URL  : OnlyOffice 서버 주소 (브라우저가 접근)
  ONLYOFFICE_SECRET_KEY  : JWT 서명 키
  BACKEND_URL            : FastAPI 백엔드 주소 (OnlyOffice 서버가 콜백 호출)
"""

import os
import time
import jwt
import httpx
from typing import Dict, Any


ONLYOFFICE_SECRET = os.getenv('ONLYOFFICE_SECRET_KEY', 'your-secret-key')
ONLYOFFICE_SERVER_URL = os.getenv('ONLYOFFICE_SERVER_URL', 'http://localhost:8080')
BACKEND_URL = os.getenv('BACKEND_URL', 'http://localhost:8000')


def create_editor_config(
    doc_id: str,
    version: str,
    user_name: str,
    file_url: str,
    mode: str = "view",
) -> Dict[str, Any]:
    """
    OnlyOffice DocsAPI에 전달할 설정 JSON 생성

    Args:
        doc_id    : 문서 ID (예: EQ-SOP-00001)
        version   : 문서 버전 (예: 1.0)
        user_name : 편집자 이름
        file_url  : S3 presigned URL (OnlyOffice가 DOCX를 가져올 URL)

    Returns:
        OnlyOffice DocsAPI 설정 dict (JWT token 포함)
    """
    # 편집 모드일 때만 콜백 URL 설정
    is_edit_mode = (mode == "edit")

    config = {
        "document": {
            "fileType": "docx",
            # 캐시 무효화용 고유 키: 저장 시마다 새 키 필요
            "key": f"{doc_id}_v{version}_{int(time.time())}",
            "title": f"{doc_id}_v{version}.docx",
            "url": file_url,
            "permissions": {
                "comment": is_edit_mode,
                "copy": True,
                "download": True,
                "edit": is_edit_mode,
                "print": True,
                "review": is_edit_mode,
            },
        },
        "documentType": "word",
        "editorConfig": {
            "user": {
                "id": user_name.replace(" ", "_"),
                "name": user_name,
            },
            "lang": "ko",
            "mode": mode,
            "customization": {
                "autosave": is_edit_mode,
                "forcesave": False,
                "zoom": -2,
            },
        },
    }

    # 편집 모드일 때만 콜백 URL 추가
    if is_edit_mode:
        config["editorConfig"]["callbackUrl"] = f"{BACKEND_URL}/onlyoffice/callback"

    # JWT 서명 (무단 접근 방지)
    token = jwt.encode(config, ONLYOFFICE_SECRET, algorithm="HS256")
    config["token"] = token

    return config


async def download_from_onlyoffice(url: str) -> bytes:
    """OnlyOffice가 제공한 URL에서 편집된 DOCX 다운로드"""
    async with httpx.AsyncClient() as client:
        response = await client.get(url, follow_redirects=True, timeout=60.0)
        response.raise_for_status()
        return response.content


def get_onlyoffice_server_url() -> str:
    return ONLYOFFICE_SERVER_URL
