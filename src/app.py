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

# init chroma database
chroma_client = chromadb.HttpClient()
chroma_collection = chroma_client.get_or_create_collection(name="DocuRAG")

# init llm
Settings.llm = Ollama(model="llama2", request_timeout=60.0)

# initialize the document reader
filename_fn = lambda filename: {"file_name": filename}
documents = SimpleDirectoryReader("data/pdf", file_metadata=filename_fn).load_data()

# testing if the parsing of the docs was a success
print("Documents count: " + str(len(documents)))

# huggingFace Embedding
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# define metadata extractors
transformations = [
    SentenceSplitter(),
    # hyper parameters for tuning
     TitleExtractor(nodes=5),
     KeywordExtractor(keywords=10),
     OpenAIEmbedding(model='text-embedding-3-large', embed_batch_size=100)
]

# create ingestion pipeline
pipeline = IngestionPipeline(transformations=transformations)

nodes = pipeline.run(
    documents=documents,
    show_progress=True
)

print(nodes[0].metadata)

# connect to the Chroma db
print("Connecting to chroma vector store...")
vector_store = ChromaVectorStore(chroma_collection= chroma_collection)

print("Creating storage context")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
print("Creating vector store index...")
index = VectorStoreIndex(nodes, storage_context=storage_context)
print("creating query engine")

# query for testing
query_engine = index.as_query_engine()


print("Starting eval")
query = "What are the Mechanical connection for the Pressure transmitter HDA 4400?"

evaluator = DocuEval(index, query_engine)

eval_results = evaluator.run_evaluation(q.get_questions()[:100])


# printing out the results
print("Total Relevancy Score:" + str(eval_results["relevancy"]))
print("Total Faithfulness Score:" + str(eval_results["faithfulness"]))
