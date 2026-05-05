import os
import glob
import time
from dotenv import load_dotenv
from llama_index.llms.google_genai import GoogleGenAI
from pydantic import BaseModel, Field
from typing import List
from neo4j import GraphDatabase

# ==========================================
# 1. SETUP & CREDENTIALS
# ==========================================
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# ==========================================
# 2. DEFINE THE DATA SCHEMA (Pydantic)
# ==========================================
class PaperExtraction(BaseModel):
    paper_title: str = Field(description="The formal title of the academic paper")
    authors: List[str] = Field(description="A list of the authors' names")
    methodologies: List[str] = Field(description="Core algorithms, architectures, or methods used")
    key_findings: List[str] = Field(description="1-2 brief sentences summarizing the main discoveries")
    datasets: List[str] = Field(description="Names of KG datasets used for evaluation e.g. FB15k-237, WD50K, WebQSP, NELL-995")
    metrics: List[str] = Field(description="Evaluation metrics reported e.g. Hits@1, Hits@10, MRR, F1")
    baselines: List[str] = Field(description="Names of baseline models compared against e.g. ToG, RoG, GNN-RAG")
    kg_structure_assumption: str = Field(description="The KG structural form this paper assumes. Must be exactly one of: 'triple-only', 'hyper-relational', 'temporal-quadruple', 'mixed'")

# ==========================================
# 3. NEO4J INJECTION LOGIC
# ==========================================
def inject_paper_to_graph(tx, paper_obj):
    # Convert the Pydantic object to a standard Python dictionary
    data = paper_obj.model_dump()

    # A. Create the main Paper Node
    tx.run(
        """
        MERGE (p:Paper {title: $title})
        SET p.key_findings = $findings
        """,
        title=data["paper_title"],
        findings=" | ".join(data["key_findings"])
    )

    # B. Create Author Nodes and link them to the Paper
    for author in data["authors"]:
        tx.run(
            """
            MERGE (a:Author {name: $name})
            WITH a
            MATCH (p:Paper {title: $title})
            MERGE (a)-[:WROTE]->(p)
            """,
            name=author, title=data["paper_title"]
        )

    # C. Create Methodology Nodes and link them to the Paper
    for method in data["methodologies"]:
        tx.run(
            """
            MERGE (m:Methodology {name: $name})
            WITH m
            MATCH (p:Paper {title: $title})
            MERGE (p)-[:USES_METHOD]->(m)
            """,
            name=method, title=data["paper_title"]
        )

# ==========================================
# 4. MAIN PIPELINE EXECUTION
# ==========================================
def process_all_papers():
    # Initialize the LLM
    print("Connecting to Gemini API...")
    llm = GoogleGenAI(model="gemini-3-flash-preview")
    structured_llm = llm.as_structured_llm(output_cls=PaperExtraction)

    # Initialize Neo4j
    print("Connecting to Neo4j...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    # Find all .mmd files in the target directory
    current_dir = os.getcwd()
    target_folder = os.path.join(current_dir, "data", "papers", "my_output")
    mmd_files = glob.glob(os.path.join(target_folder, "*.mmd"))

    if not mmd_files:
        print(f"No .mmd files found in {target_folder}")
        return

    print(f"Found {len(mmd_files)} papers to process. Starting pipeline...\n")

    with driver.session() as session:
        for file_path in mmd_files:
            filename = os.path.basename(file_path)
            print(f"Processing: {filename}...")
            
            try:
                # Read the markdown file
                with open(file_path, "r", encoding="utf-8") as file:
                    raw_markdown = file.read()
                
                # To save API costs and time, we only feed the first 8000 characters 
                # (plenty of context for title, authors, and abstract)
                text_to_process = raw_markdown[:8000]

                # Run LLM Extraction
                prompt = f"""
                You are a data engineer building a Knowledge Graph. 
                Read the following academic text and extract the required entities exactly as specified.
                TEXT:
                {text_to_process}
                """
                
                response = structured_llm.complete(prompt)
                extracted_data = response.raw # This is the Pydantic object

                # Inject into Graph
                session.execute_write(inject_paper_to_graph, extracted_data)
                print(f"  -> Successfully injected: {extracted_data.paper_title}")

                print("  -> Pausing for 15 seconds to respect API rate limits...")
                time.sleep(15)

            except Exception as e:
                print(f"  -> ERROR processing {filename}: {e}")

    driver.close()
    print("\n✅ Pipeline complete! All papers are now in the Knowledge Graph.")

if __name__ == "__main__":
    process_all_papers()