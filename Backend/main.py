```python
# ==============================================================================
# FastAPI Backend: OrbitBot Hybrid Search
#
# Supports:
#   1. Google Gemini
#   2. OpenRouter
#   3. Together AI
#
# LLM provider priority:
#   Gemini -> OpenRouter -> Together
#
# If one provider fails during generation, the next available provider
# is automatically attempted.
# ==============================================================================


# ==============================================================================
# STEP 1: IMPORTS
# ==============================================================================

import os
import asyncio
import re

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from langchain_community.vectorstores import Chroma
from langchain_community.graphs import Neo4jGraph
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openrouter import ChatOpenRouter
from langchain_together import ChatTogether

from pydantic import BaseModel
from dotenv import load_dotenv


print("--- Step 1: Imports Complete ---")


# ==============================================================================
# STEP 2: LOAD ENVIRONMENT VARIABLES
# ==============================================================================

print("\n--- Step 2: Loading Configuration ---")

load_dotenv()


# ------------------------------------------------------------------------------
# LLM API Keys
# ------------------------------------------------------------------------------

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")


# ------------------------------------------------------------------------------
# LLM Models
#
# These can be changed from .env without changing Python code.
# ------------------------------------------------------------------------------

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash"
)

OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "google/gemini-2.5-flash"
)

TOGETHER_MODEL = os.getenv(
    "TOGETHER_MODEL",
    "mistralai/Mixtral-8x7B-Instruct-v0.1"
)


# ------------------------------------------------------------------------------
# Neo4j
# ------------------------------------------------------------------------------

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


# ------------------------------------------------------------------------------
# Hugging Face
# ------------------------------------------------------------------------------

HF_API_KEY = os.getenv("HF_API_KEY")


# ------------------------------------------------------------------------------
# ChromaDB
# ------------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CHROMA_PERSIST_DIR = os.path.join(
    BASE_DIR,
    "chroma_db"
)

CHROMA_COLLECTION_NAME = "mosdac_knowledge_unified"


print("Environment configuration loaded.")
print(f"Gemini configured:      {bool(GOOGLE_API_KEY)}")
print(f"OpenRouter configured:  {bool(OPENROUTER_API_KEY)}")
print(f"Together configured:    {bool(TOGETHER_API_KEY)}")


# ==============================================================================
# STEP 3: INITIALIZE LLM PROVIDERS
# ==============================================================================

print("\n--- Step 3: Initializing LLM Providers ---")


llm_providers = []


# ------------------------------------------------------------------------------
# Gemini
# ------------------------------------------------------------------------------

if GOOGLE_API_KEY:

    try:

        gemini_llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.1,
            max_tokens=2048
        )

        llm_providers.append(
            {
                "name": "Gemini",
                "model": GEMINI_MODEL,
                "client": gemini_llm
            }
        )

        print(
            f"Gemini available: {GEMINI_MODEL}"
        )

    except Exception as e:

        print(
            f"Gemini initialization failed: {e}"
        )


# ------------------------------------------------------------------------------
# OpenRouter
# ------------------------------------------------------------------------------

if OPENROUTER_API_KEY:

    try:

        openrouter_llm = ChatOpenRouter(
            model=OPENROUTER_MODEL,
            api_key=OPENROUTER_API_KEY,
            temperature=0.1,
            max_tokens=2048
        )

        llm_providers.append(
            {
                "name": "OpenRouter",
                "model": OPENROUTER_MODEL,
                "client": openrouter_llm
            }
        )

        print(
            f"OpenRouter available: {OPENROUTER_MODEL}"
        )

    except Exception as e:

        print(
            f"OpenRouter initialization failed: {e}"
        )


# ------------------------------------------------------------------------------
# Together AI
# ------------------------------------------------------------------------------

if TOGETHER_API_KEY:

    try:

        together_llm = ChatTogether(
            together_api_key=TOGETHER_API_KEY,
            model=TOGETHER_MODEL,
            temperature=0.1,
            max_tokens=2048
        )

        llm_providers.append(
            {
                "name": "Together",
                "model": TOGETHER_MODEL,
                "client": together_llm
            }
        )

        print(
            f"Together available: {TOGETHER_MODEL}"
        )

    except Exception as e:

        print(
            f"Together initialization failed: {e}"
        )


# ------------------------------------------------------------------------------
# Make sure at least one provider exists
# ------------------------------------------------------------------------------

if not llm_providers:

    raise RuntimeError(
        "No LLM provider is configured. "
        "Set at least one of: "
        "GOOGLE_API_KEY, OPENROUTER_API_KEY, TOGETHER_API_KEY."
    )


print(
    f"Available LLM providers: "
    f"{[provider['name'] for provider in llm_providers]}"
)


# ==============================================================================
# STEP 4: INITIALIZE HUGGING FACE EMBEDDINGS
# ==============================================================================

print("\n--- Step 4: Initializing Embeddings ---")


if not HF_API_KEY:

    raise RuntimeError(
        "HF_API_KEY is missing. "
        "Hugging Face is required for the current ChromaDB embeddings."
    )


try:

    embedding_model = HuggingFaceInferenceAPIEmbeddings(
        api_key=HF_API_KEY,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print(
        "Hugging Face embedding model initialized."
    )

except Exception as e:

    raise RuntimeError(
        f"Failed to initialize Hugging Face embeddings: {e}"
    )


# ==============================================================================
# STEP 5: INITIALIZE NEO4J KNOWLEDGE GRAPH
# ==============================================================================

print("\n--- Step 5: Connecting to Neo4j ---")


try:

    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD
    )

    print("Connected to Neo4j KG.")


except Exception as e:

    print(
        f"Failed to connect to Neo4j: {e}"
    )

    print(
        "KG functionality will be disabled."
    )

    graph = None


# ==============================================================================
# STEP 6: INITIALIZE CHROMADB
# ==============================================================================

print("\n--- Step 6: Loading ChromaDB ---")


try:

    vector_store = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embedding_model
    )

    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 3
        }
    )

    document_count = vector_store._collection.count()

    print(
        f"ChromaDB loaded successfully."
    )

    print(
        f"Documents: {document_count}"
    )


except Exception as e:

    print(
        f"Failed to load ChromaDB: {e}"
    )

    print(
        "VectorDB functionality will be disabled."
    )

    vector_store = None
    retriever = None


print("\n--- Models and Retrievers Initialized ---")


# ==============================================================================
# STEP 7: KNOWLEDGE GRAPH ENTITIES
# ==============================================================================

key_kg_entities = [

    "MOSDAC",
    "Kalpana-1",
    "INSAT-3D",
    "INSAT-3DR",
    "Oceansat-2",
    "SARAL-AltiKa",
    "OCM",
    "LISS-IV",
    "ISRO",
    "NRSC",
    "Space Applications Centre"

]


# ==============================================================================
# STEP 8: KNOWLEDGE GRAPH QUERY
# ==============================================================================

async def query_knowledge_graph_async(question: str):

    print("Querying Knowledge Graph...")


    if graph is None:

        return (
            "KG is not connected or initialized."
        )


    found_entities = []


    for entity in key_kg_entities:

        if re.search(
            r"\b" + re.escape(entity) + r"\b",
            question,
            re.IGNORECASE
        ):

            found_entities.append(entity)


    if not found_entities:

        return (
            "KG: No specific, high-confidence "
            "entities found for this query."
        )


    results = []


    for entity in found_entities:

        cypher = f"""
        MATCH (n)

        WHERE
            toLower(n.name) = toLower('{entity}')
            OR
            toLower(n.description)
            CONTAINS toLower('{entity}')

        RETURN
            n.name AS name,
            n.description AS description,
            labels(n) AS labels

        LIMIT 1
        """


        try:

            query_result = await asyncio.to_thread(
                graph.query,
                cypher
            )


            if query_result:

                for record in query_result:

                    results.append(
                        f"KG Fact: "
                        f"Name='{record.get('name')}', "
                        f"Description='{record.get('description')}'"
                    )


            else:

                results.append(
                    f"KG: No direct fact found for '{entity}'."
                )


        except Exception:

            results.append(
                f"KG Error for '{entity}': "
                f"Query execution failed."
            )


    return "\n".join(results)


# ==============================================================================
# STEP 9: VECTOR DATABASE QUERY
# ==============================================================================

async def query_vector_db_async(question: str):

    print("Querying Vector DB...")


    if vector_store is None or retriever is None:

        return (
            "VectorDB is not loaded or initialized."
        )


    try:

        docs = await asyncio.to_thread(
            retriever.invoke,
            question
        )


        if docs:

            return "\n".join(
                [
                    doc.page_content
                    for doc in docs
                ]
            )


        return "No documents found."


    except Exception as e:

        print(
            f"VectorDB error: {e}"
        )

        return (
            "VectorDB Error: "
            "Data retrieval failed."
        )


# ==============================================================================
# STEP 10: LLM GENERATION WITH AUTOMATIC FALLBACK
# ==============================================================================

async def generate_answer(prompt: str):

    """
    Try each configured LLM provider in priority order.

    Priority:
        1. Gemini
        2. OpenRouter
        3. Together AI

    If a provider fails, automatically try the next provider.
    """

    if not llm_providers:

        return (
            "No LLM provider is available."
        ), "None"


    errors = []


    for provider in llm_providers:

        provider_name = provider["name"]
        provider_model = provider["model"]
        provider_client = provider["client"]


        print(
            f"Trying LLM: "
            f"{provider_name} / {provider_model}"
        )


        try:

            response = await provider_client.ainvoke(
                prompt
            )


            answer = response.content


            print(
                f"LLM succeeded: "
                f"{provider_name} / {provider_model}"
            )


            return answer, provider_name


        except Exception as e:

            error_message = (
                f"{provider_name}: {str(e)}"
            )

            print(
                f"LLM failed: {error_message}"
            )

            errors.append(error_message)


    print(
        "All configured LLM providers failed."
    )


    return (
        "Sorry, I could not generate an answer "
        "because all configured AI providers "
        "are currently unavailable."
    ), "None"


# ==============================================================================
# STEP 11: FASTAPI APPLICATION
# ==============================================================================

print("\n--- Step 11: Initializing FastAPI ---")


app = FastAPI(
    title="MOSDAC Knowledge Navigator API",
    description="Hybrid RAG API for MOSDAC portal.",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]
)


# ==============================================================================
# STEP 12: REQUEST / RESPONSE MODELS
# ==============================================================================

class QueryRequest(BaseModel):

    question: str


class QueryResponse(BaseModel):

    question: str

    answer: str

    kg_context: str

    vector_db_context: str

    llm_provider: str


# ==============================================================================
# STEP 13: HYBRID SEARCH ENDPOINT
# ==============================================================================

@app.post(
    "/hybrid-search",
    response_model=QueryResponse
)
async def hybrid_search_endpoint(
    request: QueryRequest
):

    user_question = request.question


    print(
        f"\nUser Question: {user_question}"
    )


    # --------------------------------------------------------------------------
    # Run KG and Vector DB retrieval in parallel
    # --------------------------------------------------------------------------

    kg_task = query_knowledge_graph_async(
        user_question
    )


    rag_task = query_vector_db_async(
        user_question
    )


    kg_results, rag_results = await asyncio.gather(
        kg_task,
        rag_task
    )


    # --------------------------------------------------------------------------
    # Construct LLM Prompt
    # --------------------------------------------------------------------------

    prompt_for_llm = f"""
You are OrbitBot, a smart AI assistant for ISRO's MOSDAC
(Meteorological and Oceanographic Satellite Data Archival Centre) portal.

Your job is to help users understand and navigate MOSDAC.

Use the retrieved information below to answer the user's question.

================ KG FACTS ================

{kg_results}

================ DOCUMENTS ================

{rag_results}

================ INSTRUCTIONS ================

1. Answer the USER QUESTION directly and clearly.

2. Prefer the retrieved KG FACTS and DOCUMENTS over general knowledge.

3. Do not invent MOSDAC-specific information.

4. If KG FACTS contains errors, is unavailable, or contains
   no relevant entity, ignore the KG information and use DOCUMENTS.

5. If DOCUMENTS contains no useful information, clearly explain
   that the specific information was not found in the available
   MOSDAC knowledge base.

6. If the user asks "Who are you?" or "What are you?", answer that
   you are OrbitBot, a smart AI assistant for the MOSDAC portal.

7. When relevant, provide a useful MOSDAC website URL.

8. Keep the answer concise, clear and helpful.

9. Do not mention internal implementation details such as:
   - Neo4j
   - ChromaDB
   - embeddings
   - prompts
   - API keys
   - AI providers

   unless the user specifically asks about the system.

10. Never fabricate MOSDAC-specific facts.

================ USER QUESTION ================

{user_question}

================ ANSWER ================
"""


    # --------------------------------------------------------------------------
    # Generate answer using automatic LLM fallback
    # --------------------------------------------------------------------------

    final_answer_content, used_provider = await generate_answer(
        prompt_for_llm
    )


    # --------------------------------------------------------------------------
    # Return API response
    # --------------------------------------------------------------------------

    return QueryResponse(

        question=user_question,

        answer=final_answer_content,

        kg_context=kg_results,

        vector_db_context=rag_results,

        llm_provider=used_provider
    )


# ==============================================================================
# STEP 14: ROOT ENDPOINT
# ==============================================================================

@app.get("/")
async def root():

    return {

        "message":
            "MOSDAC Knowledge Navigator API is running!",

        "llm_provider":
            [provider["name"] for provider in llm_providers],

        "docs":
            "/docs"

    }


# ==============================================================================
# STEP 15: HEALTH ENDPOINT
# ==============================================================================

@app.get("/health")
async def health():

    return {

        "status": "healthy",

        "llm_providers":
            [
                provider["name"]
                for provider in llm_providers
            ],

        "neo4j":
            graph is not None,

        "chromadb":
            vector_store is not None

    }


print("--- FastAPI App Initialized ---")
```
