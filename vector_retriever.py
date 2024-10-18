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

# 2. Initialize the VectorRetriever
from neo4j_graphrag.retrievers import VectorRetriever

# Build the retriever
retriever = VectorRetriever(
    driver,
    index_name="moviePlots",
    embedder=embedder,
    return_properties=["title", "plot"],
)

# 3. Using the Retriever
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.llm import OpenAILLM

llm = OpenAILLM(model_name="gpt-4o", model_params={"temperature": 0})
rag = GraphRAG(retriever=retriever, llm=llm)
query_text = "Give me 3 films where a hero goes on a journey"
response = rag.search(query_text=query_text, retriever_config={"top_k": 5})
print(response.answer)

driver.close()
