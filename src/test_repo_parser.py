import os
from dotenv import load_dotenv
# 1. NEW IMPORT: Using the updated Google GenAI SDK integration
from llama_index.llms.google_genai import GoogleGenAI
from pydantic import BaseModel, Field
from typing import List

# Load the secret API key
load_dotenv()

# Define the Knowledge Graph Schema for CODE
class FunctionNode(BaseModel):
    function_name: str = Field(description="The exact name of the function")
    description: str = Field(description="A 1-2 sentence summary of what the function does")
    arguments: List[str] = Field(description="List of arguments the function takes")

class FileNode(BaseModel):
    file_name: str = Field(description="The name of this python file")
    dependencies: List[str] = Field(description="Libraries or internal modules imported at the top (e.g., torch, networkx)")
    classes_defined: List[str] = Field(description="Names of any classes defined in this file")
    functions: List[FunctionNode] = Field(description="List of all functions defined in this file")

# Load the Python File
current_dir = os.getcwd()
file_path = os.path.join(current_dir, "data", "repos", "ToG-main", "ToG", "utils.py") 

print(f"Loading Code: {file_path}")
with open(file_path, "r", encoding="utf-8") as file:
    raw_code = file.read()

print("Connecting to Gemini API...")
# 2. NEW INITIALIZATION: Using GoogleGenAI and a stable API model string
llm = GoogleGenAI(model="gemini-3-flash-preview")

print("Extracting Code Architecture... (This takes a few seconds)")
structured_llm = llm.as_structured_llm(output_cls=FileNode)

prompt = f"""
You are a senior software engineer building a Knowledge Graph of a complex codebase.
Read the following Python code and extract its structural architecture exactly as specified.

CODE:
```python
{raw_code}
"""

response = structured_llm.complete(prompt)

print("\n--- Extracted Code Graph Data ---")
print(response.raw.model_dump_json(indent=2))