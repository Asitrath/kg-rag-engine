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
# 2. THE REPO-TO-PAPER MAPPING
# ==========================================
# We map your folder names to unique keywords found in the Paper titles in Neo4j
REPO_MAPPING = {
    "GNN-RAG-main": "Gnn-Rag",
    "STARE-master": "Message Passing for Hyper-Relational", # StarE title
    "TempAgent-master": "Time-aware ReAct", # TempAgent title
    "Temp-R1-main": "Temp-R1",
    "TimeR4-main": "TimeR4",
    "ToG-main": "Think-on-Graph"
}

# ==========================================
# 3. DEFINE CODE SCHEMA (Pydantic)
# ==========================================
class FunctionNode(BaseModel):
    function_name: str = Field(description="The exact name of the function")
    description: str = Field(description="A 1-2 sentence summary of what the function does")
    arguments: List[str] = Field(description="List of arguments the function takes")

class FileNode(BaseModel):
    file_name: str = Field(description="The name of this python file")
    dependencies: List[str] = Field(description="Libraries imported at the top")
    functions: List[FunctionNode] = Field(description="List of functions defined in this file")

# ==========================================
# 4. NEO4J CYPHER INJECTIONS
# ==========================================
def link_repo_to_paper(tx, repo_name, paper_keyword):
    """Creates a Repo node and links it to the Paper node using a keyword search"""
    tx.run(
        """
        MATCH (p:Paper)
        WHERE toLower(p.title) CONTAINS toLower($keyword)
        MERGE (r:Repository {name: $repo_name})
        MERGE (r)-[:IMPLEMENTS]->(p)
        """,
        keyword=paper_keyword, repo_name=repo_name
    )

def inject_code_data(tx, repo_name, file_data):
    """Injects the File, Dependencies, and Functions, linking them to the Repo"""
    data = file_data.model_dump()
    
    # Create File and link to Repo
    tx.run(
        """
        MERGE (r:Repository {name: $repo_name})
        MERGE (f:File {name: $file_name, repository: $repo_name})
        MERGE (f)-[:BELONGS_TO]->(r)
        """,
        repo_name=repo_name, file_name=data["file_name"]
    )

    # Link Dependencies
    for dep in data["dependencies"]:
        tx.run(
            """
            MERGE (d:Dependency {name: $dep_name})
            WITH d
            MATCH (f:File {name: $file_name, repository: $repo_name})
            MERGE (f)-[:IMPORTS]->(d)
            """,
            dep_name=dep, file_name=data["file_name"], repo_name=repo_name
        )

    # Link Functions
    for func in data["functions"]:
        tx.run(
            """
            MERGE (fn:Function {name: $func_name})
            SET fn.description = $desc, fn.arguments = $args
            WITH fn
            MATCH (f:File {name: $file_name, repository: $repo_name})
            MERGE (f)-[:DEFINES]->(fn)
            """,
            func_name=func["function_name"],
            desc=func["description"],
            args=func["arguments"],    # native list, no .join()
            file_name=data["file_name"],
            repo_name=repo_name
        )

    # Link Classes
    for cls in data["classes_defined"]:
        tx.run(
            """
            MERGE (c:Class {name: $cls_name})
            WITH c
            MATCH (f:File {name: $file_name, repository: $repo_name})
            MERGE (f)-[:DEFINES_CLASS]->(c)
            """,
            cls_name=cls, file_name=data["file_name"], repo_name=repo_name
        )

    # Link Data Structures
    for ds in data["data_structures"]:
        tx.run(
            """
            MERGE (s:DataStructure {name: $ds_name})
            WITH s
            MATCH (f:File {name: $file_name, repository: $repo_name})
            MERGE (f)-[:USES_STRUCTURE]->(s)
            """,
            ds_name=ds, file_name=data["file_name"], repo_name=repo_name
        )

    # Link Graph Libraries
    for lib in data["graph_libraries"]:
        tx.run(
            """
            MERGE (gl:GraphLibrary {name: $lib_name})
            WITH gl
            MATCH (f:File {name: $file_name, repository: $repo_name})
            MERGE (f)-[:USES_GRAPH_LIB]->(gl)
            """,
            lib_name=lib, file_name=data["file_name"], repo_name=repo_name
        )

# ==========================================
# 5. MAIN EXECUTION LOOP
# ==========================================
def run_bridge_pipeline():
    print("Connecting to APIs and Database...")
    llm = GoogleGenAI(model="gemini-3-flash-preview")
    structured_llm = llm.as_structured_llm(output_cls=FileNode)
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    base_repo_path = os.path.join(os.getcwd(), "data", "repos")
    
    # SAFETY LIMIT: Only process 1 file per repo to avoid hitting API Rate Limits on Free Tier
    MAX_FILES_PER_REPO = 1 

    with driver.session() as session:
        for repo_folder, paper_keyword in REPO_MAPPING.items():
            repo_path = os.path.join(base_repo_path, repo_folder)
            
            if not os.path.exists(repo_path):
                print(f"⚠️ Skipping {repo_folder} - Folder not found.")
                continue
                
            print(f"\n🔗 Linking Repository: [{repo_folder}] to Paper keyword: [{paper_keyword}]")
            session.execute_write(link_repo_to_paper, repo_folder, paper_keyword)
            
            # Find Python files in the repo
            py_files = glob.glob(os.path.join(repo_path, "**", "*.py"), recursive=True)
            files_processed = 0
            
            for file_path in py_files:
                if files_processed >= MAX_FILES_PER_REPO:
                    break # Move to the next repository
                
                # Skip massive files or empty init files to save tokens
                if os.path.getsize(file_path) > 20000 or file_path.endswith("__init__.py"):
                    continue
                    
                filename = os.path.basename(file_path)
                print(f"  -> Parsing Code: {filename}...")
                
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        raw_code = f.read()
                        
                    prompt = f"Extract the structural architecture of this Python code:\n```python\n{raw_code}\n```"
                    response = structured_llm.complete(prompt)
                    
                    session.execute_write(inject_code_data, repo_folder, response.raw)
                    print(f"  -> Successfully injected code graph for {filename}")
                    
                    files_processed += 1
                    
                    print("  -> Pausing for 15 seconds for API limits...")
                    time.sleep(15)
                    
                except Exception as e:
                    print(f"  -> ❌ ERROR processing {filename}: {e}")

    driver.close()
    print("\n✅ Bridging Complete! Repositories, Code, and Papers are now connected.")

if __name__ == "__main__":
    run_bridge_pipeline()