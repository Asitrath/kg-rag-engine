import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

# 1. Load credentials
load_dotenv()
URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")

def search_graph_for_paper(tx, paper_keyword):
    """
    Using OPTIONAL MATCH ensures we still retrieve the Paper and File
    even if the AI didn't find any specific 'Functions' inside that file.
    """
    query = """
    MATCH (p:Paper)
    WHERE toLower(p.title) CONTAINS toLower($keyword)
    OPTIONAL MATCH (p)<-[:IMPLEMENTS]-(r:Repository)
    OPTIONAL MATCH (r)<-[:BELONGS_TO]-(f:File)
    OPTIONAL MATCH (f)-[:DEFINES]->(fn:Function)
    RETURN p.title AS Paper, r.name AS Repository, f.name AS File, fn.name AS Function, fn.description AS Description
    """
    
    result = tx.run(query, keyword=paper_keyword)
    return [record for record in result]

def run_search():
    print("Connecting to Local Neo4j Knowledge Graph...\n")
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

    search_term = "Think-on-Graph"
    print(f"🔍 Searching Graph for code related to: '{search_term}'...\n")

    with driver.session() as session:
        records = session.execute_read(search_graph_for_paper, search_term)

        if not records or records[0]["Paper"] is None:
            print("No paper found matching that keyword.")
        else:
            print(f"✅ Found {len(records)} connected paths!\n")
            print("-" * 60)
            
            for record in records:
                print(f"📄 PAPER: {record['Paper']}")
                
                if record['Repository']:
                    print(f"   📦 REPO: {record['Repository']}")
                if record['File']:
                    print(f"      📁 FILE: {record['File']}")
                if record['Function']:
                    print(f"         ⚙️  def {record['Function']}()")
                    print(f"             {record['Description']}")
                elif record['File']:
                    print(f"         (No standard functions found/extracted in this file)")
                print("-" * 60)

    driver.close()

if __name__ == "__main__":
    run_search()