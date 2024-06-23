import asyncio
import os
import chromadb
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, ServiceContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.evaluation import RelevancyEvaluator, generate_question_context_pairs, FaithfulnessEvaluator, BatchEvalRunner

from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor, KeywordExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.instructor import InstructorEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from llama_index.llms.ollama import Ollama
from llama_index.core import Settings

import questions as q
from src.evaluation import DocuEval

# The OPENAI should be in the env variables
print("OPENAI_API_KEY:", os.environ['OPENAI_API_KEY'])
# Load environment variables
load_dotenv()

# init chroma database
chroma_client = chromadb.HttpClient()
chroma_collection = chroma_client.get_or_create_collection(name="DocuRAG")

# init llm
Settings.llm = Ollama(model="llama2", request_timeout=60.0)

# Initialize the document reader
filename_fn = lambda filename: {"file_name": filename}
documents = SimpleDirectoryReader("data/pdf", file_metadata=filename_fn).load_data()

# Testing if the parsing of the docs was a success
print("Documents count: " + str(len(documents)))

# Initialize the embedding model
# openAI Embeddings
# embed_model = OpenAIEmbedding(model='text-embedding-3-large', embed_batch_size=100)

# HuggingFace Embedding
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# Instructor Embedding
# embed_model = InstructorEmbedding(model_name="hkunlp/instructor-base")

# ollama embeddings:
#Settings.embed_model = OllamaEmbedding(
#    model_name="llama2",
#    base_url="http://localhost:11434",
#    ollama_additional_kwargs={"mirostat": 0},
#)

# Define metadata extractors
transformations = [
    SentenceSplitter(),
    # TitleExtractor(nodes=5),
    # KeywordExtractor(keywords=10),
    # OpenAIEmbedding(model='text-embedding-3-large', embed_batch_size=100)
]

# Create ingestion pipeline
pipeline = IngestionPipeline(transformations=transformations)

nodes = pipeline.run(
    documents=documents,
    show_progress=True
)

print(nodes[0].metadata)

# Connect to the ElasticSearchStore(Since there are still some issues with the Hydac in terms
# of access to their Elastic SearchStore, we have to set it up on-premise in a docker container)
# vector_store = ElasticsearchStore(es_url="http://localhost:9200", index_name="RAG-Testing")
print("Connecting to chroma vector store...")
vector_store = ChromaVectorStore(chroma_collection= chroma_collection)

print("Creating storage context")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
print("Creating vector store index...")
#index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, use_async=True)
index = VectorStoreIndex(nodes, storage_context=storage_context)
print("creating query engine")
# Query for testing
query_engine = index.as_query_engine()


print("Starting eval")
#query = "What are the restrictions of the file manager?"
#query = "What is the smallest possible measurement rate that can be set for recordings using the HMG 4000 device, depending on the number of active measurement channels?"
query = "What are the Mechanical connection for the Pressure transmitter HDA 4400?"

evaluator = DocuEval(index, query_engine)

eval_results = evaluator.run_evaluation(q.get_questions()[:100])



print("Total Relevancy Score:" + str(eval_results["relevancy"]))

print("Total Faithfulness Score:" + str(eval_results["faithfulness"]))