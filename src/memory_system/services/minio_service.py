"""
MinIO Service for Source Workspace Engine
Handles binary storage for source documents
"""

import io
import hashlib
from typing import Optional, Tuple
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from src.memory_system.config import settings


class MinIOService:
    """Service for interacting with MinIO object storage."""

    def __init__(self):
        self.endpoint = settings.minio_endpoint
        self.access_key = settings.minio_access_key
        self.secret_key = settings.minio_secret_key
        self.bucket = settings.minio_bucket
        self.secure = settings.minio_secure

        self.client = boto3.client(
            's3',
            endpoint_url=f"http://{self.endpoint}",
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=Config(signature_version='s3v4'),
            region_name='auto'
        )
        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self) -> None:
        """Create bucket if it doesn't exist."""
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == '404':
                try:
                    self.client.create_bucket(Bucket=self.bucket)
                except Exception:
                    pass

    def _generate_object_key(self, source_id: str, filename: str) -> str:
        """Generate unique object key for a source."""
        return f"sources/{source_id}/{filename}"

    def compute_hash(self, content: bytes) -> str:
        """Compute SHA256 hash of content."""
        return hashlib.sha256(content).hexdigest()

    async def upload_source(
        self,
        source_id: str,
        filename: str,
        content: bytes,
        content_type: str = "application/octet-stream"
    ) -> Tuple[str, str]:
        """
        Upload source file to MinIO.

        Returns:
            Tuple of (canonical_uri, file_hash)
        """
        object_key = self._generate_object_key(source_id, filename)
        file_hash = self.compute_hash(content)

        try:
            self.client.put_object(
                Bucket=self.bucket,
                Key=object_key,
                Body=content,
                ContentType=content_type
            )
        except Exception as e:
            raise Exception(f"Failed to upload to MinIO: {str(e)}")

        canonical_uri = f"minio://{self.bucket}/{object_key}"
        return canonical_uri, file_hash

    async def download_source(self, source_id: str, filename: str) -> bytes:
        """Download source file from MinIO."""
        object_key = self._generate_object_key(source_id, filename)

        try:
            response = self.client.get_object(Bucket=self.bucket, Key=object_key)
            return response['Body'].read()
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == '404':
                raise FileNotFoundError(f"Source not found: {source_id}/{filename}")
            raise Exception(f"Failed to download from MinIO: {str(e)}")

    async def delete_source(self, source_id: str, filename: str) -> bool:
        """Delete source file from MinIO."""
        object_key = self._generate_object_key(source_id, filename)

        try:
            self.client.delete_object(Bucket=self.bucket, Key=object_key)
            return True
        except Exception:
            return False

    async def get_presigned_url(
        self,
        source_id: str,
        filename: str,
        expires: int = 3600
    ) -> str:
        """Generate presigned URL for downloading."""
        object_key = self._generate_object_key(source_id, filename)

        try:
            url = self.client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket, 'Key': object_key},
                ExpiresIn=expires
            )
            return url
        except Exception as e:
            raise Exception(f"Failed to generate presigned URL: {str(e)}")

    async def exists(self, source_id: str, filename: str) -> bool:
        """Check if source exists in MinIO."""
        object_key = self._generate_object_key(source_id, filename)

        try:
            self.client.head_object(Bucket=self.bucket, Key=object_key)
            return True
        except ClientError:
            return False

    async def get_metadata(self, source_id: str, filename: str) -> dict:
        """Get metadata of a stored object."""
        object_key = self._generate_object_key(source_id, filename)

        try:
            response = self.client.head_object(Bucket=self.bucket, Key=object_key)
            return {
                'content_type': response.get('ContentType'),
                'content_length': response.get('ContentLength'),
                'last_modified': response.get('LastModified'),
                'etag': response.get('ETag')
            }
        except ClientError:
            return {}


minio_service = MinIOService()