import os
import logging.config

from neo4j import GraphDatabase

from dotenv import load_dotenv
load_dotenv()

# Connect to Neo4j database
uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
driver = GraphDatabase.driver(uri, auth=(username, password))


# 1. Initialize the Embedder
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings

embedder = OpenAIEmbeddings(model="text-embedding-ada-002")


# 2. Initialize the HybridRetriever
from neo4j_graphrag.retrievers import HybridRetriever
from neo4j_graphrag.llm import OpenAILLM

retriever = HybridRetriever(
    driver=driver,
    vector_index_name="moviePlots",
    fulltext_index_name="plotFulltext",
    embedder=embedder,
    return_properties=["title", "plot"],
)


# 3. Using the Retriever
from neo4j_graphrag.generation import GraphRAG

llm = OpenAILLM(model_name="gpt-4o", model_params={"temperature": 0})
rag = GraphRAG(retriever=retriever, llm=llm)
query_text = "What is the name of the movie set in 1375 in Imperial China?"
response = rag.search(query_text=query_text, retriever_config={"top_k": 5})
print(response.answer)

driver.close()
