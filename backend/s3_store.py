"""
S3 스토어 - DOCX 파일 업로드/다운로드/presigned URL 생성

S3 키 구조:
  documents/{doc_id}/v{version}/document.docx
  예: documents/EQ-SOP-00001/v2.0/document.docx
"""

import os
import re
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from typing import List, Dict


class S3Store:
    def __init__(self):
        region = os.getenv('AWS_REGION', 'ap-northeast-2')
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=region,
            config=Config(
                signature_version='s3v4',
                s3={'addressing_style': 'virtual'},
            ),
        )
        self.bucket = os.getenv('S3_BUCKET_NAME')

    def _key(self, doc_id: str, version: str) -> str:
        return f"documents/{doc_id}/v{version}/document.docx"

    def _pdf_key(self, doc_id: str, version: str) -> str:
        return f"documents/{doc_id}/v{version}/document.pdf"

    def upload_pdf(self, doc_id: str, version: str, content: bytes) -> str:
        """원본 PDF를 S3에 저장"""
        key = self._pdf_key(doc_id, version)
        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=content,
            ContentType='application/pdf',
        )
        return key

    def get_pdf_presigned_url(self, doc_id: str, version: str, expires: int = 3600) -> str:
        """S3 PDF 파일의 임시 접근 URL 반환"""
        key = self._pdf_key(doc_id, version)
        url = self.s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': self.bucket, 'Key': key},
            ExpiresIn=expires,
        )
        return url

    def pdf_exists(self, doc_id: str, version: str) -> bool:
        """S3에 PDF 파일이 존재하는지 확인"""
        key = self._pdf_key(doc_id, version)
        try:
            self.s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False

    def get_presigned_url(self, doc_id: str, version: str, expires: int = 3600) -> str:
        """S3 파일의 임시 접근 URL 반환"""
        key = self._key(doc_id, version)
        url = self.s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': self.bucket, 'Key': key},
            ExpiresIn=expires,
        )
        return url

    def upload_docx(self, doc_id: str, version: str, content: bytes) -> str:
        """편집된 DOCX를 S3에 저장, 저장된 S3 키 반환"""
        key = self._key(doc_id, version)
        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=content,
            ContentType='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        )
        return key

    def download_docx(self, doc_id: str, version: str) -> bytes:
        """S3에서 DOCX 파일 다운로드"""
        key = self._key(doc_id, version)
        response = self.s3.get_object(Bucket=self.bucket, Key=key)
        return response['Body'].read()

    def list_versions(self, doc_id: str) -> List[str]:
        """특정 문서의 S3 버전 목록 조회 (버전 문자열 리스트)"""
        prefix = f"documents/{doc_id}/"
        result = self.s3.list_objects_v2(
            Bucket=self.bucket,
            Prefix=prefix,
            Delimiter='/',
        )
        versions = []
        for cp in result.get('CommonPrefixes', []):
            # e.g. documents/EQ-SOP-00001/v1.0/
            part = cp['Prefix'].rstrip('/').split('/')[-1]
            if part.startswith('v'):
                versions.append(part[1:])  # 'v' 접두사 제거
        return sorted(versions)

    def object_exists(self, doc_id: str, version: str) -> bool:
        """S3에 해당 문서/버전 파일이 존재하는지 확인"""
        key = self._key(doc_id, version)
        try:
            self.s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False

    def list_docx_documents(self) -> List[Dict]:
        """
        S3의 DOCX 문서 목록 조회.
        반환 필드:
          - doc_id
          - doc_name
          - doc_type ("docx")
          - version (최신 버전)
          - created_at (최신 object 수정시각)
          - latest_version
          - source ("s3")
        """
        docs: Dict[str, Dict] = {}
        continuation_token = None

        while True:
            params = {
                "Bucket": self.bucket,
                "Prefix": "documents/",
            }
            if continuation_token:
                params["ContinuationToken"] = continuation_token

            resp = self.s3.list_objects_v2(**params)
            for obj in resp.get("Contents", []):
                key = obj.get("Key", "")
                # documents/{doc_id}/v{version}/document.docx
                m = re.match(r"^documents/([^/]+)/v([^/]+)/document\.docx$", key)
                if not m:
                    continue

                doc_id = m.group(1)
                version = m.group(2)
                modified = obj.get("LastModified")

                if doc_id not in docs:
                    docs[doc_id] = {
                        "doc_id": doc_id,
                        "doc_name": doc_id,
                        "doc_type": "docx",
                        "version": version,
                        "created_at": modified,
                        "latest_version": version,
                        "source": "s3",
                    }
                    continue

                current = docs[doc_id]
                if modified and (not current.get("created_at") or modified > current["created_at"]):
                    current["created_at"] = modified
                    current["version"] = version
                    current["latest_version"] = version

            if not resp.get("IsTruncated"):
                break
            continuation_token = resp.get("NextContinuationToken")

        return list(docs.values())

    def delete_docx_versions(self, doc_id: str) -> int:
        """특정 문서의 모든 DOCX 버전 객체를 S3에서 삭제하고 삭제 개수를 반환"""
        prefix = f"documents/{doc_id}/"
        keys_to_delete = []
        continuation_token = None

        while True:
            params = {
                "Bucket": self.bucket,
                "Prefix": prefix,
            }
            if continuation_token:
                params["ContinuationToken"] = continuation_token

            resp = self.s3.list_objects_v2(**params)
            for obj in resp.get("Contents", []):
                key = obj.get("Key", "")
                if key.endswith("/document.docx"):
                    keys_to_delete.append({"Key": key})

            if not resp.get("IsTruncated"):
                break
            continuation_token = resp.get("NextContinuationToken")

        deleted_count = 0
        # delete_objects는 최대 1000개까지 한번에 지원
        for i in range(0, len(keys_to_delete), 1000):
            chunk = keys_to_delete[i:i + 1000]
            if not chunk:
                continue
            res = self.s3.delete_objects(
                Bucket=self.bucket,
                Delete={"Objects": chunk, "Quiet": True}
            )
            deleted_count += len(res.get("Deleted", []))

        return deleted_count

    def has_docx(self, doc_id: str) -> bool:
        """문서에 DOCX 객체가 하나라도 존재하는지 확인"""
        prefix = f"documents/{doc_id}/"
        resp = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix, MaxKeys=50)
        for obj in resp.get("Contents", []):
            if obj.get("Key", "").endswith("/document.docx"):
                return True
        return False
