# utils/s3_utils.py
import os
import boto3
from botocore.exceptions import ClientError
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
SECRET_KEY = os.getenv("S3_SECRET_KEY")
REGION = os.getenv("S3_REGION")
BUCKET = os.getenv("S3_BUCKET")


class S3Client:
    def __init__(self, bucket_name: str = None, region_name: str = None):
        self.bucket_name = bucket_name or BUCKET

        # Explicitly use credentials from .env if provided
        session_kwargs = {}
        if ACCESS_KEY and SECRET_KEY:
            session_kwargs["aws_access_key_id"] = ACCESS_KEY
            session_kwargs["aws_secret_access_key"] = SECRET_KEY
        if region_name or REGION:
            session_kwargs["region_name"] = region_name or REGION

        # Initialize boto3 session safely
        session = boto3.session.Session(**session_kwargs)
        self.s3 = session.resource('s3')
        self.client = session.client('s3')

    def download_folder(self, s3_prefix: str, local_dir: str, bucket_name: Optional[str] = None):
        """
        Download all objects under s3_prefix (folder) to local_dir preserving subpaths.
        s3_prefix example: 'dataset/handwritten-yolo/test'
        """
        bucket = (bucket_name or self.bucket_name)
        if not bucket:
            raise ValueError("bucket_name must be provided either in constructor or call")

        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

        paginator = self.client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket, Prefix=s3_prefix):
            for obj in page.get('Contents', []):
                key = obj['Key']
                if key.endswith('/'):
                    continue
                relative_path = key[len(s3_prefix):].lstrip('/')
                target_path = local_dir / relative_path
                target_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    self.client.download_file(bucket, key, str(target_path))
                except ClientError as e:
                    print(f"Failed to download {key}: {e}")

    def upload_folder(self, local_dir: str, s3_prefix: str, bucket_name: Optional[str] = None):
        """
        Upload all files under local_dir to s3_prefix/. Keeps relative paths.
        """
        bucket = (bucket_name or self.bucket_name)
        if not bucket:
            raise ValueError("bucket_name must be provided either in constructor or call")

        local_dir = Path(local_dir)
        for local_file in local_dir.rglob('*'):
            if local_file.is_file():
                relative_path = local_file.relative_to(local_dir)
                s3_key = f"{s3_prefix.rstrip('/')}/{relative_path.as_posix()}"
                try:
                    self.client.upload_file(str(local_file), bucket, s3_key)
                except ClientError as e:
                    print(f"Failed to upload {local_file}: {e}")

    def upload_file(self, local_path: str, s3_key: str, bucket_name: Optional[str] = None):
        bucket = (bucket_name or self.bucket_name)
        if not bucket:
            raise ValueError("bucket_name must be provided either in constructor or call")
        try:
            self.client.upload_file(local_path, bucket, s3_key)
        except ClientError as e:
            print(f"Failed to upload {local_path} to s3://{bucket}/{s3_key}: {e}")
