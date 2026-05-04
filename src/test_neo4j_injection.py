import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

# 1. Load credentials
load_dotenv()
URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")

# 2. The Extracted Data (We are using a snippet of your actual output!)
repo_data = {
  "file_name": "tog_utils.py",
  "dependencies": ["freebase_func", "sentence_transformers", "openai"],
  "functions": [
    {
      "function_name": "retrieve_top_docs",
      "description": "Retrieves the top-n most relevant documents for a given query.",
      "arguments": ["query", "docs", "model", "width"]
    },
    {
      "function_name": "compute_bm25_similarity",
      "description": "Computes BM25 similarity between a query and relations.",
      "arguments": ["query", "corpus", "width"]
    }
  ]
}

# 3. The Cypher Injection Logic
def inject_to_graph(tx, data):
    # A. Create the main File Node
    # Add this inside inject_to_graph(), after the File MERGE
    tx.run(
        """
        MERGE (r:Repository {name: $repo_name})
        MERGE (f:File {name: $file_name, repository: $repo_name})
        MERGE (f)-[:BELONGS_TO]->(r)
        """,
        repo_name="ToG-main",
        file_name=data["file_name"]
    )

    # B. Create Dependency Nodes and link them to the File
    for dep in data["dependencies"]:
        tx.run(
            """
            MERGE (d:Dependency {name: $dep_name})
            WITH d
            MATCH (f:File {name: $file_name})
            MERGE (f)-[:IMPORTS]->(d)
            """,
            dep_name=dep, file_name=data["file_name"]
        )

    # C. Create Function Nodes and link them to the File
    for func in data["functions"]:
        tx.run(
            """
            MERGE (fn:Function {name: $func_name})
            SET fn.description = $desc, fn.arguments = $args
            WITH fn
            MATCH (f:File {name: $file_name})
            MERGE (f)-[:DEFINES]->(fn)
            """,
            func_name=func["function_name"],
            desc=func["description"],
            args=", ".join(func["arguments"]),  # Convert list to string for storage
            file_name=data["file_name"]
        )

# 4. Connect and Execute
print("Connecting to Neo4j...")
driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

with driver.session() as session:
    session.execute_write(inject_to_graph, repo_data)

driver.close()
print("Success! Nodes and Edges injected into the Knowledge Graph.")