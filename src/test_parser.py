import os
from dotenv import load_dotenv
from llama_index.core import Document
from llama_index.llms.google_genai import GoogleGenAI
from pydantic import BaseModel, Field
from typing import List

# 1. Load the secret API key from the .env file
load_dotenv()

# 2. Define the Knowledge Graph Schema (The Blueprint)
# This forces Gemini to return structured JSON instead of a conversational paragraph.
class PaperExtraction(BaseModel):
    paper_title: str = Field(description="The formal title of the academic paper")
    authors: List[str] = Field(description="A list of the authors' names")
    methodologies: List[str] = Field(description="Core algorithms, architectures, or methods used (e.g., GNN, LLM, Temporal Reasoning)")
    key_findings: List[str] = Field(description="1-2 brief sentences summarizing the main discoveries")
    datasets: List[str] = Field(description="Names of KG datasets used for evaluation e.g. FB15k-237, WD50K, WebQSP, NELL-995")
    metrics: List[str] = Field(description="Evaluation metrics reported e.g. Hits@1, Hits@10, MRR, F1")
    baselines: List[str] = Field(description="Names of baseline models compared against e.g. ToG, RoG, GNN-RAG")
    kg_structure_assumption: str = Field(description="The KG structural form this paper assumes. Must be exactly one of: 'triple-only', 'hyper-relational', 'temporal-quadruple', 'mixed'")
    aliases: List[str] = Field(description="Short-form names, acronyms, or codebase names this paper is commonly referred to by e.g. 'ToG' for Think-on-Graph, 'RoG' for Reasoning on Graphs, 'StarE' for Message Passing for Hyper-Relational Knowledge Graphs")
    
# 3. Load the Markdown File
current_dir = os.getcwd()
file_path = os.path.join(current_dir, "data", "papers", "triples", "RoG.mmd")

print(f"Loading: {file_path}")
with open(file_path, "r", encoding="utf-8") as file:
    raw_markdown = file.read()

# For this initial prototype, we will only feed the first 5000 characters 
# (usually the Abstract and Introduction) to save time and API costs.
text_to_process = raw_markdown[:5000]

# 4. Initialize the Gemini API
print("Connecting to Gemini API...")
llm = GoogleGenAI(model="models/gemini-3-flash-preview")

# 5. Execute the Extraction
print("Extracting Graph Nodes... (This takes a few seconds)")
# We lock the LLM to our Pydantic schema
structured_llm = llm.as_structured_llm(output_cls=PaperExtraction)

prompt = f"""
You are a brilliant data engineer building a Knowledge Graph. 
Read the following academic text and extract the required entities exactly as specified.

TEXT:
{text_to_process}
"""

response = structured_llm.complete(prompt)

# 6. Print the results
print("\n--- Extracted Knowledge Graph Data ---")
# This prints the result as perfectly formatted JSON!
print(response.raw.model_dump_json(indent=2))