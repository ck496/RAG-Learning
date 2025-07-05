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


def test_kb_retrieve_generate(kb_id: str, query: str) -> str:
    """
    Test the knowledge base using the retrieve and generate API and get a response based on relavent data in KB.
    Here Bedrock takes care of:
        1) retrieving the necessary references from the knowledge base 
        2) generating the final answer with addintional context using the specified foundation model .
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


def test_kb_retrive_sources(kb_id: str, query: str, numberOfResults: int):
    """
    Test the KB using the retrieve API to get a list of all the data source 
    chunks that were found in the vector store matiching a query 
    """
    bedrock_agent_runtime_client = boto3.client('bedrock-agent-runtime')
    response = bedrock_agent_runtime_client.retrieve(
        knowledgeBaseId=kb_id,
        nextToken='string',
        retrievalConfiguration={
            "vectorSearchConfiguration": {
                "numberOfResults": 5,
            }
        },
        retrievalQuery={
            "text": query
        }
    )

    response_print(response)


def response_print(retrieve_resp):
    """
    Helper function for test_kb_retrive_sources to print out the various chunks 
    that were retrived for a given query
    - structure 'retrievalResults': list of contents. Each list has content, location, score, metadata
    """
    #
    for num, chunk in enumerate(retrieve_resp['retrievalResults'], 1):
        print(f'Chunk {num}: ', chunk['content']['text'], end='\n'*2)
        print(f'Chunk {num} Location: ', chunk['location'], end='\n'*2)
        print(f'Chunk {num} Score: ', chunk['score'], end='\n'*2)
        print(f'Chunk {num} Metadata: ', chunk['metadata'], end='\n'*2)


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
    parser.add_argument(
        "--chunks",
        type=int,
        default=5,               # ← default value if flag is omitted
        help="Number of Chunks to get from Retrive API(default: 5)"
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

    print("\nSending query to KB's Retrive and Generate API %s .....", args.kb_id)
    result_ret_gen = test_kb_retrieve_generate(args.kb_id, query)
    print(
        f"-Test Result : \n\t{result_ret_gen}")

    print(
        "\n\nSending query to KB's Retrive API to get matching chunks %s .....", args.kb_id)
    print(
        f"-Test Result:")
    test_kb_retrive_sources(args.kb_id, query, args.chunks)


if __name__ == "__main__":
    main()
