#!/usr/bin/env python3
"""
Utility script for testing a Knowledge Base with retrieve and retrieve_and_generate functions.

Usage:
    python test_kb.py --kb_id KB_ID --query "Your question here"
    python test_kb.py --kb_id KB_ID --query-file path/to/query.txt
"""
import argparse
import os
import sys
import time
import boto3
import logging
import pprint
import json
import time
from pathlib import Path

logging.basicConfig(
    format='[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

foundation_model = "anthropic.claude-3-haiku-20240307-v1:0"
session = boto3.session.Session()
region = session.region_name


def test_knowledge_base(kb_id: str, query: str) -> str:
    """
    Test the knowledge base using the retrieve and generate API. 
    With this API, Bedrock takes care of retrieving the necessary references from the knowledge base 
    and generating the final answer using the specified foundation model from Bedrock.
    """
    bedrock_agent_runtime_client = boto3.client('bedrock-agent-runtime')

    response = bedrock_agent_runtime_client.retrieve_and_generate(
        input={
            "text": query
        },
        retrieveAndGenerateConfiguration={
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {
                'knowledgeBaseId': kb_id,
                "modelArn": "arn:aws:bedrock:{}::foundation-model/{}".format(region, foundation_model),
                "retrievalConfiguration": {
                    "vectorSearchConfiguration": {
                        "numberOfResults": 5
                    }
                }
            }
        }
    )

    result = response['output']['text']
    return result


def main():
    # configure logging only once when run as a script
    logging.basicConfig(
        format='[%(asctime)s] %(levelname)s %(name)s - %(message)s',
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description='Test RAG query responses against a Bedrock knowledge base.'
    )
    # Allow either --query or --query-file not both
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--query',
        help='Query string to send to the Knowledge Base'
    )
    group.add_argument(
        '--query-file',
        type=Path,
        help='Path to a text file containing the query'
    )
    parser.add_argument(
        '--kb_id',
        required=True,
        help='ID of the Knowledge Base to test'
    )
    args = parser.parse_args()

    # load query from file or directly
    if args.query_file:
        try:
            query = args.query_file.read_text(encoding='utf-8')
        except Exception as e:
            logger.error("Failed to read query file '%s': %s",
                         args.query_file, e)
            sys.exit(1)
    else:
        query = args.query

    logger.info("Sending query to KB %s", args.kb_id)
    result = test_knowledge_base(args.kb_id, query)
    print(f"\nResult from KB:\n{result}\n")


if __name__ == "__main__":
    main()
