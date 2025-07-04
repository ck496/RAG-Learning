#!/usr/bin/env python3
"""
Utility script for uploading all files in a directory to an S3 bucket.

Usage:
    python upload_utils.py --path /synthetic_dataset --bucket my-bucket-name
"""
import logging
import argparse
import os
from pathlib import Path
import boto3

logger = logging.getLogger(__name__)


def upload_directory(path: str, bucket_name: str) -> None:
    """
    Upload all files under `path` to the specified S3 bucket.
    """
    root = Path(path)
    s3_client = boto3.client('s3')

    if not root.exists() or not root.is_dir():
        logger.error("Provided path '%s' is not a valid directory.", path)
        return

    for root, dirs, files in os.walk(path):
        for file in files:
            file_to_upload = os.path.join(root, file)
            logger.info(f"uploading file {file_to_upload} to {bucket_name}")
            s3_client.upload_file(file_to_upload, bucket_name, file)


def main():
    # Configure logging when run as a script
    logging.basicConfig(
        format='[%(asctime)s] %(levelname)s %(name)s - %(message)s',
        level=logging.INFO
    )
    parser = argparse.ArgumentParser(
        description='Upload a local directory to an S3 bucket'
    )
    parser.add_argument(
        '--path', required=True, help='Local directory to upload'
    )
    parser.add_argument(
        '--bucket', required=True, help='Target S3 bucket name'
    )
    args = parser.parse_args()

    upload_directory(args.path, args.bucket)


if __name__ == '__main__':
    main()
