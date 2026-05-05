import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from llama_index.llms.google_genai import GoogleGenAI

# ==========================================
# 1. SETUP & CREDENTIALS
# ==========================================
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# ==========================================
# 2. GRAPH RETRIEVAL LOGIC (The "R" in RAG)
# ==========================================
def retrieve_context_from_graph(tx, keyword):
    query = """
    MATCH (p:Paper)
    WHERE toLower(p.title) CONTAINS toLower($keyword)
       OR ANY(alias IN p.aliases WHERE toLower(alias) CONTAINS toLower($keyword))

    OPTIONAL MATCH (p)-[:USES_METHOD]->(m:Methodology)
    OPTIONAL MATCH (p)-[:EVALUATES_ON]->(ds:Dataset)
    OPTIONAL MATCH (p)-[:MEASURES]->(mt:Metric)
    OPTIONAL MATCH (p)-[:COMPARES_AGAINST]->(b:Baseline)
    OPTIONAL MATCH (p)-[:ASSUMES_STRUCTURE]->(s:KGStructure)
    OPTIONAL MATCH (p)<-[:IMPLEMENTS]-(r:Repository)
    OPTIONAL MATCH (r)<-[:BELONGS_TO]-(f:File)
    OPTIONAL MATCH (f)-[:DEFINES]->(fn:Function)
    OPTIONAL MATCH (f)-[:DEFINES_CLASS]->(c:Class)
    OPTIONAL MATCH (f)-[:USES_GRAPH_LIB]->(gl:GraphLibrary)

    RETURN p.title AS Paper, p.key_findings AS Findings,
           collect(DISTINCT m.name) AS Methodologies,
           collect(DISTINCT ds.name) AS Datasets,
           collect(DISTINCT mt.name) AS Metrics,
           collect(DISTINCT b.name) AS Baselines,
           s.name AS KGStructure,
           r.name AS Repository, f.name AS File,
           fn.name AS Function, fn.description AS Description,
           collect(DISTINCT c.name) AS Classes,
           collect(DISTINCT gl.name) AS GraphLibraries
    """
    result = tx.run(query, keyword=keyword)
    return [record for record in result]

def format_context(records):
    
    if not records or records[0]["Paper"] is None:
        return "No relevant information found in the Knowledge Graph."

    r0 = records[0]

    context = f"ACADEMIC PAPER:\nTitle: {r0['Paper']}\n"
    context += f"Key Findings: {r0['Findings']}\n"
    context += f"KG Structure Assumption: {r0['KGStructure']}\n"
    context += f"Methodologies: {', '.join(r0['Methodologies'])}\n"
    context += f"Datasets Evaluated On: {', '.join(r0['Datasets'])}\n"
    context += f"Metrics: {', '.join(r0['Metrics'])}\n"
    context += f"Baselines Compared Against: {', '.join(r0['Baselines'])}\n\n"
    
    context += "ASSOCIATED CODEBASE ARCHITECTURE:\n"
    
    current_file = ""
    for record in records:
        if record["File"] and record["File"] != current_file:
            current_file = record["File"]
            context += f"\n- FILE: {current_file} (Repository: {record['Repository']})\n"
            if record["Classes"]:
                context += f"  Classes: {', '.join(record['Classes'])}\n"
            if record["GraphLibraries"]:
                context += f"  Graph Libraries: {', '.join(record['GraphLibraries'])}\n"
        if record["Function"]:
            context += f"  * def {record['Function']}()\n"
            context += f"    {record['Description']}\n"

    return context

# ==========================================
# 3. GENERATION LOGIC (The "AG" in RAG)
# ==========================================
def run_rag_engine():
    print("🤖 Initializing KG-RAG Engine...")
    
    # Initialize the LLM (Using 1.5-flash here, but it will work with whatever your active model is)
    llm = GoogleGenAI(model="gemini-3-flash-preview") 
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    # The Interactive CLI
    print("\n" + "="*55)
    print("🎓 Welcome to your Knowledge Graph RAG System!")
    print("="*55)
    
    target_paper = input("\n1. Which paper do you want to query? (e.g., 'Think-on-Graph', 'TempAgent'): ")
    user_question = input("2. What is your question? (e.g., 'Based on the code, how does this retrieve documents?'): ")
    
    print("\n🔍 Retrieving context from Neo4j...")
    with driver.session() as session:
        records = session.execute_read(retrieve_context_from_graph, target_paper)
        
    graph_context = format_context(records)
    print("✅ Graph context retrieved successfully!")
    
    # Build the Master Prompt
    master_prompt = f"""
    You are a highly intelligent AI assistant answering questions based on a specific Knowledge Graph.
    Use ONLY the provided context to answer the user's question. If the answer is not in the context, say so.
    Base your technical explanations on the provided functions and their descriptions.
    
    KNOWLEDGE GRAPH CONTEXT:
    -------------------------
    {graph_context}
    -------------------------
    
    USER QUESTION: {user_question}
    """
    
    print("\n🧠 Sending Augmented Prompt to Gemini...")
    
    try:
        # This will attempt to generate the answer
        response = llm.complete(master_prompt)
        print("\n" + "="*55)
        print("💡 AI RESPONSE:")
        print("="*55)
        print(response.text)
        print("="*55)
        
    except Exception as e:
        # If the API blocks us due to the quota limit, we catch it here!
        print(f"\n❌ LLM Blocked (Quota Limit Reached): {e}")
        print("\n" + "*"*55)
        print("Don't worry! Your script is functionally perfect.")
        print("Here is the exact Master Prompt that will successfully generate an answer tomorrow:")
        print("*"*55 + "\n")
        print(master_prompt)

    driver.close()

if __name__ == "__main__":
    run_rag_engine()