"""
This script is used to create a KB, injest data and test responses

Usage:
    python create_kb.py
"""


import os
import sys
import time
import boto3
import logging
import pprint
import json
import time
from pathlib import Path

# autopep8: off
# Ensure aws dir is on PYTHONPATH so utils can be imported
AWS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AWS_DIR))

logging.basicConfig(format='[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

from utils.knowledge_base import BedrockKnowledgeBase
from utils.upload_source_data import upload_directory
# autopep8: on


def main():
    logger.debug("Create KB start")
    # 0. Setup Logging, path, timer and pretty print

    current_time = time.time()
    # Format the timestamp as a string
    timestamp_str = time.strftime(
        "%Y%m%d%H%M%S", time.localtime(current_time))[-7:]
    suffix = f"{timestamp_str}"

    # 1. Setup Boto3 Clients
    logger.info(f"Boto3 version: ${boto3.__version__}")

    sts_client = boto3.client('sts')
    session = boto3.session.Session()
    region = session.region_name
    account_id = sts_client.get_caller_identity()["Account"]

    logger.info(f"AWS Region: ${region}. AccountID: ${account_id}")

    # 2. Setup resource name and conifgs
    knowledge_base_name = f"bedrock-lab1-knowledge-base-{suffix}"
    knowledge_base_description = "Multi data source knowledge base."

    # 3. Setup your data sources
    data_bucket_name = f'bedrock-kb-{suffix}-1'
    data_sources = [{"type": "S3", "bucket_name": data_bucket_name}]

    # # 4. Create KB
    knowledge_base = BedrockKnowledgeBase(
        kb_name=f'{knowledge_base_name}',
        kb_description=knowledge_base_description,
        data_sources=data_sources,
        chunking_strategy="FIXED_SIZE",
        suffix=f'{suffix}-f'

    )

    # 6. Upload local data from /synthetic_dataset into data-source bucket
    dataset_dir = AWS_DIR / 'synthetic_dataset'
    if not dataset_dir.exists():
        logger.error("Dataset directory not found: %s", dataset_dir)
        return

    upload_directory(str(dataset_dir), data_bucket_name)

    # 7. Start injestion job for each data source
    #   - KB will fetch the documents in the data source,
    #     pre-process it to extract text, chunk it,
    #     create embeddings of each chunk and then write it to the OSS

    time.sleep(30)
    # sync knowledge base
    knowledge_base.start_ingestion_job()
    # keep the kb_id for invocation later in the invoke request
    kb_id = knowledge_base.get_knowledge_base_id()
    logger.info(f"Save this kb_id for testing: ${kb_id}")
    return kb_id


if __name__ == '__main__':
    main()
