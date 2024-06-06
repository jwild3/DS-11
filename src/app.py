import os
from dotenv import load_dotenv
from llama_index import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores import ElasticsearchVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor, KeywordExtractor
from llama_index.core.ingestion import IngestionPipeline

# Load environment variables
load_dotenv()

# Initialize the document reader
filename_fn = lambda filename: {"file_name": filename}
documents = SimpleDirectoryReader("../data/pdf", file_metadata=filename_fn).load_data()

# Testing if the parsing of the docs was a success
print(documents[0])
len(documents)

# Initialize the embedding model
embed_model = OpenAIEmbedding(model='text-embedding-3-large', embed_batch_size=100)

# Define metadata extractors
transformations = [
    SentenceSplitter(),
    #TitleExtractor(nodes=5),
    #KeywordExtractor(keywords=10),
    #OpenAIEmbedding(model='text-embedding-3-large', embed_batch_size=100)
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
vector_store = ElasticsearchStore(
    es_url="http://localhost:9200",  
    index_name="RAG-Testing",

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context

# Query for testing
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
print(response)
