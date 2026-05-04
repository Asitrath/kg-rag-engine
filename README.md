# KG-RAG Engine (Knowledge Graph Retrieval-Augmented Generation)

This repository contains the prototype architecture for an automated Knowledge Graph-based RAG system. It was built to bridge the gap between theoretical academic papers and their practical code implementations.

## Architecture Pipeline
1. **Extraction:** Uses Gemini Flash models to parse unstructured academic texts (.mmd) and raw Python code into structured JSON via Pydantic schemas.
2. **Construction:** Injects the extracted Entities (Papers, Authors, Methodologies, Repositories, Files, Functions) into a local **Neo4j** graph database.
3. **Retrieval:** Executes Cypher queries to traverse the graph, linking conceptual queries to specific code implementations.
4. **Augmentation & Generation:** Feeds the retrieved graph context into an LLM to generate hallucination-resistant, context-aware answers.

## Setup Instructions
1. Clone the repository.
2. Create a virtual environment: `python -m venv venv`
3. Install dependencies: `pip install -r requirements.txt`
4. Create a `.env` file in the root directory with the following credentials:
   ```text
   GOOGLE_API_KEY="your_api_key"
   NEO4J_URI="neo4j://localhost:7687"
   NEO4J_USERNAME="neo4j"
   NEO4J_PASSWORD="your_password"

5. Run the injection scripts in src/ to populate your local database.
6. Execute src/rag_engine.py to query the graph.

### **Step 4: Initialize and Commit**
Now, open your terminal (make sure you are in the `kg-rag-engine` folder) and run these commands one by one to officially create the local repository:

```bash
git init
git add .
git commit -m "Initial commit: Core KG-RAG architecture, injection pipelines, and retrieval engine"