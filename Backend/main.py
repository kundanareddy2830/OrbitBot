# ==============================================================================
# FastAPI Backend: Hybrid Search (Local Host Version)
# This script wraps your successful test logic into a FastAPI API.
# ==============================================================================

# --- Step 1: Imports ---
import os
import asyncio
import re
from typing import List

# FastAPI specific imports
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware # Essential for frontend access

# LangChain and other core components
from langchain_community.vectorstores import Chroma
from langchain_community.graphs import Neo4jGraph
from langchain_together import ChatTogether
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings

# Pydantic for API request/response models
from pydantic import BaseModel, Field

# For loading .env file
from dotenv import load_dotenv

print("--- Step 1: Imports Complete ---")

# --- Step 2: Configuration and Connections ---
print("\n--- Step 2: Configuring Connections ---")

# Load environment variables from .env file
load_dotenv()

# API Keys and Credentials from .env
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Local path for ChromaDB (relative to your project folder)
# MAKE SURE your copied ChromaDB folder matches this path/name!
CHROMA_PERSIST_DIR = "OrbitBot/Backend/chroma_db" 
CHROMA_COLLECTION_NAME = "mosdac_knowledge_unified"

# Load HuggingFace API key from .env
HF_API_KEY = os.getenv("HF_API_KEY")

print("✅ Environment and Paths Configured.")

# --- Step 3: Initialize LLM, KG, VectorDB ---
print("\n--- Step 3: Initializing Models ---")

llm = ChatTogether(
    together_api_key=TOGETHER_API_KEY,
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    temperature=0.1,
    max_tokens=2048
)
print("✅ LLM Initialized.")

# Use HuggingFace Inference API for embeddings (no local model, low memory)
embedding_model = HuggingFaceInferenceAPIEmbeddings(
    api_key=HF_API_KEY,
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print("✅ Embedding Model Loaded.")

try:
    graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
    print("✅ Connected to Neo4j KG.")
except Exception as e:
    print(f"❌ Failed to connect to Neo4j: {e}. KG might not be fully functional.")
    graph = None # Set to None to handle gracefully in query functions

try:
    vector_store = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embedding_model
    )
    retriever = vector_store.as_retriever(search_kwargs={'k': 3})
    print(f"✅ Vector DB Loaded: {vector_store._collection.count()} documents.")
except Exception as e:
    print(f"❌ Failed to load ChromaDB: {e}. VectorDB might not be fully functional.")
    vector_store = None # Set to None to handle gracefully in query functions

print("--- Models and Retrievers Initialized ---")

# --- Step 4: Define KG and RAG Query Functions ---
print("\n--- Step 4: Defining Query Functions ---")

# Known clean entity names in your KG (based on your KG creation script's output sample)
key_kg_entities = [
    "MOSDAC", "Kalpana-1", "INSAT-3D", "INSAT-3DR", "Oceansat-2", "SARAL-AltiKa",
    "OCM", "LISS-IV", "ISRO", "NRSC", "Space Applications Centre"
]

async def query_knowledge_graph_async(question: str):
    print("🧠 Querying Knowledge Graph...")
    if graph is None: # Check if graph connection was successful at init
        return "KG is not connected or initialized."

    found_entities = []
    for entity in key_kg_entities:
        if re.search(r'\b' + re.escape(entity) + r'\b', question, re.IGNORECASE):
            found_entities.append(entity)
    
    if not found_entities:
        return "KG: No specific, high-confidence entities found for this query."

    results = []
    for entity in found_entities:
        cypher = f"""
        MATCH (n)
        WHERE toLower(n.name) = toLower('{entity}') OR toLower(n.description) CONTAINS toLower('{entity}')
        RETURN n.name AS name, n.description AS description, labels(n) AS labels
        LIMIT 1
        """
        try:
            query_result = await asyncio.to_thread(graph.query, cypher)
            if query_result:
                for record in query_result:
                    results.append(
                        f"KG Fact: Name='{record.get('name')}', Description='{record.get('description')}'"
                    )
            else:
                results.append(f"KG: No direct fact found for '{entity}'.")
        except Exception as e:
            results.append(f"KG Error for '{entity}': Query execution failed.")
    return "\n".join(results)

async def query_vector_db_async(question: str):
    print("📚 Querying Vector DB...")
    if vector_store is None: # Check if vector_store was successful at init
        return "VectorDB is not loaded or initialized."

    try:
        docs = await asyncio.to_thread(retriever.get_relevant_documents, question)
        return "\n".join([doc.page_content for doc in docs]) if docs else "No documents found."
    except Exception as e:
        return f"VectorDB Error: Data retrieval failed."

print("--- Step 4: Query Functions Defined ---")

# --- Step 5: Initialize FastAPI App and Define Endpoint ---
print("\n--- Step 5: Initializing FastAPI App ---")

app = FastAPI(
    title="MOSDAC Knowledge Navigator API",
    description="Hybrid RAG API for MOSDAC portal.",
    version="1.0.0"
)

# Add CORS middleware to allow requests from your React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Pydantic model for request body
class QueryRequest(BaseModel):
    question: str

# Pydantic model for response body (plain text answer)
class QueryResponse(BaseModel):
    question: str
    answer: str
    kg_context: str
    vector_db_context: str
    
@app.post("/hybrid-search", response_model=QueryResponse)
async def hybrid_search_endpoint(request: QueryRequest):
    user_question = request.question
    
    # Run KG and VectorDB queries in parallel
    kg_task = query_knowledge_graph_async(user_question)
    rag_task = query_vector_db_async(user_question)
    kg_results, rag_results = await asyncio.gather(kg_task, rag_task)

    # Construct the prompt for the LLM
    prompt_for_llm = f"""
You are an expert assistant for ISRO's MOSDAC portal.

Use the following information to answer the user's question clearly and concisely.

--- KG FACTS ---
{kg_results}

--- DOCUMENTS ---
{rag_results}

If the 'KG FACTS' section contains 'KG Error' or 'No relevant entities found' or 'KG is not connected', disregard it and answer solely using 'DOCUMENTS'.
If the questyion is about what are you or who are you then answer that you are OrbitBot a smart ai assistent for ISRO's MOSDAC(Meteorological and Oceanographic Satellite Data Archival Center) portal.which helps to clarify the user's question regarding MOSDAC portal.
If both 'KG FACTS' and 'DOCUMENTS' are weak or indicate no results, provide a helpful fallback answer based on general knowledge about MOSDAC, clarifying that specific information wasn't found.
Ensure your answer directly addresses the USER QUESTION and avoids making up information and also provide a url which is relavent ot give mosdacs url for every question

USER QUESTION: {user_question}

ANSWER:
"""
    
    final_answer_content = "An error occurred while generating the answer."
    try:
        response_from_llm = await llm.ainvoke(prompt_for_llm)
        final_answer_content = response_from_llm.content
    except Exception as e:
        print(f"❌ Failed to generate final answer from LLM: {e}")
        final_answer_content = f"Sorry, I encountered an error while processing your request: {e}"

    return QueryResponse(
        question=user_question,
        answer=final_answer_content,
        kg_context=kg_results,
        vector_db_context=rag_results
    )

@app.get("/")
async def root():
    return {"message": "MOSDAC Knowledge Navigator API is running! Access /docs for Swagger UI."}

@app.head("/")
async def root_head():
    """HEAD method for root endpoint - returns same headers as GET but no body"""
    return {"message": "MOSDAC Knowledge Navigator API is running! Access /docs for Swagger UI."}

@app.head("/hybrid-search")
async def hybrid_search_head():
    """HEAD method for hybrid-search endpoint - returns headers only"""
    return {
        "endpoint": "/hybrid-search",
        "method": "POST",
        "description": "Hybrid RAG search endpoint for MOSDAC queries",
        "content_type": "application/json",
        "request_schema": {
            "question": "string (required)"
        },
        "response_schema": {
            "question": "string",
            "answer": "string", 
            "kg_context": "string",
            "vector_db_context": "string"
        }
    }

print("--- Step 5: FastAPI App Initialized ---")

# --- Step 6: Instructions to Run Locally ---
print("\n\n--- TO RUN YOUR FASTAPI APPLICATION LOCALLY ---")
print("1. Save this code as `main.py` in your project folder.")
print("2. Ensure `requirements.txt` and `.env` are in the same folder.")
print("3. Ensure your `chroma_db_mosdac` folder is copied into this project folder.")
print("4. Open your terminal in the project folder.")
print("5. Create & activate a Python virtual environment (recommended):")
print("   `python -m venv venv`")
print("   `source venv/bin/activate` (macOS/Linux) OR `.\\venv\\Scripts\\activate` (Windows)")
print("6. Install dependencies:")
print("   `pip install -r requirements.txt`")
print("7. Run the server:")
print("   `uvicorn main:app --reload`")
print("\nYour API will then be accessible at `http://127.0.0.1:8000`")
print("Open `http://127.0.0.1:8000/docs` in your browser to test the API via Swagger UI.")
