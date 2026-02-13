"""
S3 스토어 - DOCX 파일 업로드/다운로드/presigned URL 생성

S3 키 구조:
  documents/{doc_id}/v{version}/document.docx
  예: documents/EQ-SOP-00001/v2.0/document.docx
"""

import os
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from typing import List


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
