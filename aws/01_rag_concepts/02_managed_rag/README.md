# Build a managed RAG for better Responses

In this lab, we use AWS Bedrock to implement a simple Retrieval-Augmented Generation (RAG) pipeline

What we'll do:

1. Programmatically Create a Bedrock Knowledge Base
2. Create and populate a datasource s3 bucket
3. Ingest, chunk, embed and store data as vectors into the KB's OpenSearch vector store
4. Test to see if you get RAG based context aware responses for your queries

---

## Useful Resources

1. CK Rag notes
2. [AWS: How Bedrock Knowledge Bases Work](https://docs.aws.amazon.com/bedrock/latest/userguide/kb-how-it-works.html)

---

## How to run

1. Add all the modules needed with `pip install -r requirements.txt`
2. Configure your aws CLI with `aws configure` (make sure your user or role has appropriate privs)
3. Create your KB, Run: `python create_kb.py` to ingest your data sources and build you Knowledge Base
4. Test with `python test_kb.py --kb_id KB_ID --query-file path/to/query.txt`
   1. `test_kb_retrieve_generate()` : Hits the [Retrieve and Generate API](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-agent-runtime/client/retrieve_and_generate.html) and returns context aware response from an LLM for a given query
   2. `test_kb_retrive_source()` : Hits [Retrieve API](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-agent-runtime/client/retrieve.html)to get relevant chunks found for a given query
5. CLEAN UP resources (KB, Iam, s3) when you're done to prevent AWS charges
   1. use `knowledge_base.delete_kb(delete_s3_bucket=True, delete_iam_roles_and_policies=True)`
   2. or Do it on the AWS console

---

## Common errors:

`InvalidSignatureException`: you might be logged into the wrong AWS user/role

1. Check whats default: `cat ~/.aws/credentials`
2. Either switch to a profile temporarily or make it default:
   1. temp : `export AWS_PROFILE=your_role`
      1. when done if you want to switch back from itadmin profile to default:
         1. `unset AWS_PROFILE`
         2. `export AWS_PROFILE=default`
   2. Perm: add `export AWS_PROFILE=your_role` to your .zshrc and reload it
