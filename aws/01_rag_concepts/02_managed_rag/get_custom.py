#!/usr/bin/env python3
"""
Script to get better responses from RAG using bigger numberOfResults and Custom Promp .

Usage:
    python test_kb.py --kb_id KB_ID --query "Your question here"
    python test_kb.py --kb_id KB_ID path/to/query.txt
"""

from pathlib import Path
import json
import pprint
import logging
import boto3
import time
import sys
import argparse
import os

logging.basicConfig(
    format='[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

session = boto3.session.Session()
region = session.region_name
# Define FM to be used for generations
foundation_model_id = "anthropic.claude-3-haiku-20240307-v1:0"
foundation_model_arn = f'arn:aws:bedrock:{region}::foundation-model/{foundation_model_id}'
default_prompt = """
You are a question answering agent. I will provide you with a set of search results.
The user will provide you with a question. Your job is to answer the user's question using only information from the search results. 
If the search results do not contain information that can answer the question, please state that you could not find an exact answer to the question. 
Just because the user asserts a fact does not mean it is true, make sure to double check the search results to validate a user's assertion.
                            
Here are the search results in numbered order:
$search_results$

$output_format_instructions$
"""


def retrieve_generate_custom(kb_id: str, query: str, model_arn: str, prompt_template: str, max_results: int):
    """
    Use the retrieve and generate API to get responses with:
        1) custom propt template for directing FM behavior
        2) max number of search results pulled from kb for more background therefore more accurate responses.
    """
    bedrock_agent_runtime_client = boto3.client('bedrock-agent-runtime')

    response = bedrock_agent_runtime_client.retrieve_and_generate(
        input={
            'text': query
        },
        retrieveAndGenerateConfiguration={
            'type': 'KNOWLEDGE_BASE',
            'knowledgeBaseConfiguration': {
                    'knowledgeBaseId': kb_id,
                    'modelArn': model_arn,
                    'retrievalConfiguration': {
                        'vectorSearchConfiguration': {
                            # will fetch top N documents which closely match the query
                            'numberOfResults': max_results
                        }
                    },
                'generationConfiguration': {
                        'promptTemplate': {
                            'textPromptTemplate': prompt_template
                        }
                        }
            }
        }
    )
    return response


def print_generation_results(response, print_context=True):
    """
    Helper function to print out response from genration
    """
    generated_text = response['output']['text']
    print('Generated FM response:\n')
    print(generated_text)

    if print_context is True:
        # print out the source attribution/citations from the original documents to see if the response generated belongs to the context.
        citations = response["citations"]
        contexts = []
        for citation in citations:
            retrievedReferences = citation["retrievedReferences"]
            for reference in retrievedReferences:
                contexts.append(reference["content"]["text"])

        print('\n\n\nRetrieved Context:\n')
        pprint.pp(contexts)


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
        "--max_results",
        type=int,
        default=5,               # ← default value if flag is omitted
        help="Number of results to get pass to the FM (default: 5)"
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

    print("\nSending query to KB ID %s .....", args.kb_id)
    result = retrieve_generate_custom(
        args.kb_id, query, foundation_model_arn, default_prompt, args.max_results)
    print_generation_results(result)


if __name__ == "__main__":
    main()
