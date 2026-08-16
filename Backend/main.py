# ==============================================================================
# FastAPI Backend: OrbitBot Hybrid Search
#
# LLM Priority:
#   1. Google Gemini
#   2. OpenRouter
#   3. Together AI
#
# Retrieval:
#   1. ChromaDB + Hugging Face embeddings
#   2. Neo4j Knowledge Graph
#
# If Neo4j fails, VectorDB still works.
# If VectorDB fails, KG/general response can still work.
# If an LLM fails, the next configured LLM is attempted.
# ==============================================================================


# ==============================================================================
# STEP 1: IMPORTS
# ==============================================================================

import os
import asyncio
import re
import traceback

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from langchain_community.vectorstores import Chroma
from langchain_community.graphs import Neo4jGraph

from huggingface_hub import InferenceClient

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openrouter import ChatOpenRouter
from langchain_together import ChatTogether

from pydantic import BaseModel
from dotenv import load_dotenv


print("====================================================")
print("STEP 1: IMPORTS")
print("====================================================")
print("Imports completed successfully.")


# ==============================================================================
# STEP 2: LOAD ENVIRONMENT VARIABLES
# ==============================================================================

print("\n====================================================")
print("STEP 2: CONFIGURATION")
print("====================================================")

load_dotenv()


# ------------------------------------------------------------------------------
# LLM API KEYS
# ------------------------------------------------------------------------------

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")


# ------------------------------------------------------------------------------
# LLM MODELS
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
# NEO4J
# ------------------------------------------------------------------------------

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


# ------------------------------------------------------------------------------
# HUGGING FACE
# ------------------------------------------------------------------------------

HF_API_KEY = os.getenv("HF_API_KEY")


# ------------------------------------------------------------------------------
# CHROMADB
# ------------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CHROMA_PERSIST_DIR = os.path.join(
    BASE_DIR,
    "chroma_db"
)

CHROMA_COLLECTION_NAME = "mosdac_knowledge_unified"


print("Environment configuration loaded.")

print(
    f"Gemini configured:     {bool(GOOGLE_API_KEY)}"
)

print(
    f"OpenRouter configured: {bool(OPENROUTER_API_KEY)}"
)

print(
    f"Together configured:   {bool(TOGETHER_API_KEY)}"
)

print(
    f"HuggingFace configured: {bool(HF_API_KEY)}"
)

print(
    f"ChromaDB path: {CHROMA_PERSIST_DIR}"
)


# ==============================================================================
# STEP 3: INITIALIZE LLM PROVIDERS
# ==============================================================================

print("\n====================================================")
print("STEP 3: LLM PROVIDERS")
print("====================================================")


llm_providers = []


# ------------------------------------------------------------------------------
# GEMINI
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
# OPENROUTER
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
# TOGETHER AI
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


if not llm_providers:

    raise RuntimeError(
        "No LLM provider is configured. "
        "Configure at least one of "
        "GOOGLE_API_KEY, OPENROUTER_API_KEY, "
        "or TOGETHER_API_KEY."
    )


print(
    "Available LLM providers:",
    [provider["name"] for provider in llm_providers]
)


# ==============================================================================
# STEP 4: INITIALIZE HUGGING FACE API EMBEDDINGS
# ==============================================================================

print("\n====================================================")
print("STEP 4: HUGGING FACE EMBEDDINGS")
print("====================================================")


# ------------------------------------------------------------------------------
# LangChain-compatible Hugging Face API embedding wrapper
# ------------------------------------------------------------------------------

class HuggingFaceAPIEmbeddings:

    def __init__(
        self,
        api_key,
        model_name
    ):

        self.client = InferenceClient(
            provider="hf-inference",
            api_key=api_key
        )

        self.model_name = model_name


    def embed_documents(self, texts):

        embeddings = []

        for text in texts:

            result = self.client.feature_extraction(
                text,
                model=self.model_name
            )

            embeddings.append(
                result.tolist()
                if hasattr(result, "tolist")
                else result
            )

        return embeddings


    def embed_query(self, text):

        result = self.client.feature_extraction(
            text,
            model=self.model_name
        )

        return (
            result.tolist()
            if hasattr(result, "tolist")
            else result
        )


if not HF_API_KEY:

    print(
        "WARNING: HF_API_KEY is missing."
    )

    embedding_model = None

else:

    try:

        embedding_model = HuggingFaceAPIEmbeddings(
            api_key=HF_API_KEY,
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        print(
            "Hugging Face embedding client initialized."
        )

        print(
            "Hugging Face provider: hf-inference"
        )

        print(
            "Embedding model: "
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    except Exception as e:

        print(
            "Hugging Face embedding initialization failed:"
        )

        print(
            f"{type(e).__name__}: {e}"
        )

        embedding_model = None


# ==============================================================================
# STEP 5: CONNECT TO NEO4J
# ==============================================================================

print("\n====================================================")
print("STEP 5: NEO4J")
print("====================================================")


graph = None


if not NEO4J_URI:

    print(
        "NEO4J_URI is missing."
    )

else:

    try:

        graph = Neo4jGraph(
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD
        )

        print(
            "Neo4j connected successfully."
        )

    except Exception as e:

        print(
            "Neo4j connection failed."
        )

        print(
            f"{type(e).__name__}: {e}"
        )

        print(
            "KG functionality will be disabled."
        )

        graph = None


# ==============================================================================
# STEP 6: LOAD CHROMADB
# ==============================================================================

print("\n====================================================")
print("STEP 6: CHROMADB")
print("====================================================")


vector_store = None
retriever = None
document_count = 0


if embedding_model is None:

    print(
        "ChromaDB cannot initialize because "
        "the embedding model is unavailable."
    )

else:

    try:

        print(
            f"Opening ChromaDB at: {CHROMA_PERSIST_DIR}"
        )

        vector_store = Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embedding_model
        )

        document_count = (
            vector_store._collection.count()
        )

        print(
            "ChromaDB loaded successfully."
        )

        print(
            f"Documents: {document_count}"
        )

        retriever = vector_store.as_retriever(
            search_kwargs={
                "k": 3
            }
        )

        print(
            "ChromaDB retriever initialized."
        )

    except Exception as e:

        print(
            "ChromaDB initialization failed."
        )

        print(
            f"Error type: {type(e).__name__}"
        )

        print(
            f"Error message: {e}"
        )

        vector_store = None
        retriever = None


print("\n====================================================")
print("INITIALIZATION SUMMARY")
print("====================================================")

print(
    f"LLM providers: "
    f"{[p['name'] for p in llm_providers]}"
)

print(
    f"Neo4j: {graph is not None}"
)

print(
    f"ChromaDB: {vector_store is not None}"
)

print(
    f"Documents: {document_count}"
)


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

    print("\n[KG] Query started.")

    if graph is None:

        print(
            "[KG] Neo4j unavailable."
        )

        return (
            "KG unavailable."
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

        print(
            "[KG] No high-confidence entities detected."
        )

        return (
            "KG: No specific high-confidence "
            "entities found."
        )


    results = []


    for entity in found_entities:

        cypher = """
        MATCH (n)
        WHERE
            toLower(n.name) = toLower($entity)
            OR
            toLower(coalesce(n.description, ''))
            CONTAINS toLower($entity)

        RETURN
            n.name AS name,
            n.description AS description,
            labels(n) AS labels

        LIMIT 1
        """


        try:

            query_result = await asyncio.to_thread(
                graph.query,
                cypher,
                params={
                    "entity": entity
                }
            )


            if query_result:

                for record in query_result:

                    results.append(
                        "KG Fact: "
                        f"Name='{record.get('name')}', "
                        f"Description='{record.get('description')}'"
                    )

            else:

                results.append(
                    f"KG: No direct fact found for '{entity}'."
                )


        except Exception as e:

            print(
                f"[KG] Error for {entity}: "
                f"{type(e).__name__}: {e}"
            )

            results.append(
                f"KG Error for '{entity}'."
            )


    return "\n".join(results)


# ==============================================================================
# STEP 9: VECTOR DATABASE QUERY
# ==============================================================================

async def query_vector_db_async(question: str):

    print("\n[VECTOR] Query started.")

    if vector_store is None:

        print(
            "[VECTOR] Vector store is unavailable."
        )

        return (
            "VectorDB unavailable."
        )


    if retriever is None:

        print(
            "[VECTOR] Retriever is unavailable."
        )

        return (
            "VectorDB retriever unavailable."
        )


    if not question.strip():

        print(
            "[VECTOR] Empty question."
        )

        return (
            "VectorDB: Empty question."
        )


    try:

        print(
            f"[VECTOR] Searching for: {question}"
        )

        docs = await asyncio.to_thread(
            retriever.invoke,
            question
        )


        if not docs:

            print(
                "[VECTOR] Search completed. "
                "No documents returned."
            )

            return "No documents found."


        print(
            f"[VECTOR] Search successful. "
            f"Retrieved {len(docs)} documents."
        )


        contexts = []


        for index, doc in enumerate(docs):

            content = getattr(
                doc,
                "page_content",
                ""
            )

            if content:

                contexts.append(
                    content
                )

                print(
                    f"[VECTOR] Document {index + 1}: "
                    f"{len(content)} characters"
                )


        if not contexts:

            print(
                "[VECTOR] Documents returned but "
                "contained no page content."
            )

            return (
                "Documents were retrieved but "
                "contained no usable content."
            )


        return "\n\n".join(contexts)


    except Exception as e:

        print("\n")
        print("====================================================")
        print("VECTOR DATABASE RETRIEVAL ERROR")
        print("====================================================")

        print(
            f"Error type: {type(e).__name__}"
        )

        print(
            f"Error message: {str(e)}"
        )

        print("----------------------------------------------------")

        traceback.print_exc()

        print("====================================================")


        return (
            "VectorDB Error: "
            f"{type(e).__name__}: {str(e)}"
        )


# ==============================================================================
# STEP 10: LLM GENERATION WITH FALLBACK
# ==============================================================================

async def generate_answer(prompt: str):

    if not llm_providers:

        return (
            "No LLM provider is available.",
            "None"
        )


    print("\n====================================================")
    print("LLM GENERATION")
    print("====================================================")


    errors = []


    for provider in llm_providers:

        provider_name = provider["name"]
        provider_model = provider["model"]
        provider_client = provider["client"]


        print(
            f"Trying: {provider_name} / {provider_model}"
        )


        try:

            response = await provider_client.ainvoke(
                prompt
            )


            answer = response.content


            if not answer:

                raise RuntimeError(
                    "LLM returned an empty response."
                )


            print(
                f"LLM succeeded: "
                f"{provider_name}"
            )


            return (
                answer,
                provider_name
            )


        except Exception as e:

            error_message = (
                f"{provider_name}: "
                f"{type(e).__name__}: {e}"
            )

            print(
                f"LLM failed: {error_message}"
            )

            errors.append(
                error_message
            )


    print(
        "All configured LLM providers failed."
    )


    return (
        "Sorry, I could not generate an answer "
        "because all configured AI providers "
        "are currently unavailable.",
        "None"
    )


# ==============================================================================
# STEP 11: FASTAPI APPLICATION
# ==============================================================================

print("\n====================================================")
print("STEP 11: FASTAPI")
print("====================================================")


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

    user_question = request.question.strip()


    print("\n")
    print("====================================================")
    print("NEW HYBRID SEARCH REQUEST")
    print("====================================================")

    print(
        f"Question: {user_question}"
    )


    if not user_question:

        return QueryResponse(

            question=user_question,

            answer="Please provide a question.",

            kg_context="",

            vector_db_context="",

            llm_provider="None"

        )


    # --------------------------------------------------------------------------
    # RUN KG + VECTOR SEARCH IN PARALLEL
    # --------------------------------------------------------------------------

    kg_task = query_knowledge_graph_async(
        user_question
    )

    vector_task = query_vector_db_async(
        user_question
    )


    kg_results, vector_results = await asyncio.gather(
        kg_task,
        vector_task
    )


    print(
        "\nRetrieval completed."
    )

    print(
        f"KG result length: {len(kg_results)}"
    )

    print(
        f"Vector result length: {len(vector_results)}"
    )


    # --------------------------------------------------------------------------
    # CONSTRUCT LLM PROMPT
    # --------------------------------------------------------------------------

    prompt_for_llm = f"""
You are OrbitBot, a smart AI assistant for ISRO's MOSDAC
(Meteorological and Oceanographic Satellite Data Archival Centre) portal.

Your job is to help users understand and navigate MOSDAC.

Use the retrieved information below to answer the user's question.

================ KG FACTS ================

{kg_results}

================ DOCUMENTS ================

{vector_results}

================ INSTRUCTIONS ================

1. Answer the USER QUESTION directly and clearly.

2. Prefer the retrieved KG FACTS and DOCUMENTS over general knowledge.

3. Never invent MOSDAC-specific information.

4. If KG is unavailable or contains no useful information,
   simply rely on the DOCUMENTS.

5. If the DOCUMENTS say that no useful information was found,
   clearly state that the specific information was not found
   in the available MOSDAC knowledge base.

6. If the user asks "Who are you?" or "What are you?",
   answer that you are OrbitBot, a smart AI assistant for
   the MOSDAC portal.

7. When relevant, provide the useful official MOSDAC website URL.

8. Keep the response concise, clear and helpful.

9. Do not mention internal implementation details such as
   Neo4j, ChromaDB, embeddings, prompts, API keys, or
   AI providers unless the user specifically asks about them.

10. Never fabricate MOSDAC-specific facts.

================ USER QUESTION ================

{user_question}

================ ANSWER ================
"""


    # --------------------------------------------------------------------------
    # GENERATE ANSWER
    # --------------------------------------------------------------------------

    final_answer_content, used_provider = await generate_answer(
        prompt_for_llm
    )


    # --------------------------------------------------------------------------
    # RETURN RESPONSE
    # --------------------------------------------------------------------------

    return QueryResponse(

        question=user_question,

        answer=final_answer_content,

        kg_context=kg_results,

        vector_db_context=vector_results,

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

        "llm_providers":
            [
                provider["name"]
                for provider in llm_providers
            ],

        "docs":
            "/docs"

    }


# ==============================================================================
# STEP 15: HEALTH ENDPOINT
# ==============================================================================

@app.get("/health")
async def health():

    return {

        "status":
            "healthy",

        "llm_providers":
            [
                provider["name"]
                for provider in llm_providers
            ],

        "neo4j":
            graph is not None,

        "chromadb":
            vector_store is not None,

        "documents":
            document_count

    }


# ==============================================================================
# STARTUP COMPLETE
# ==============================================================================

print("\n====================================================")
print("FASTAPI APPLICATION INITIALIZED")
print("====================================================")
