# OrbitBot: MOSDAC Knowledge Navigator

OrbitBot is a full-stack AI-powered assistant for ISRO's MOSDAC (Meteorological and Oceanographic Satellite Data Archival Center) portal it is a part of  Bharatiya Antariksh Hackathon 2025 conducted by @ISRO . It leverages advanced Retrieval-Augmented Generation (RAG) techniques, combining a knowledge graph, vector database, and large language models to provide accurate, context-rich answers about MOSDAC's data and services.

---

## 🌐Live Demo

- **Frontend:** [https://orbit-bot.vercel.app/](https://orbit-bot.vercel.app/)
- **Backend API:** [https://orbitbot.onrender.com/](https://orbitbot.onrender.com/)

---

## 🚀 Features

- **Hybrid RAG Search:** Combines knowledge graph and vector search for robust answers.
- **Modern UI:** Built with React, Vite, TypeScript, and Tailwind CSS.
- **FastAPI Backend:** Python backend with LangChain, ChromaDB, Neo4j, and HuggingFace integration.
- **Seamless Deployment:** Frontend on Vercel, backend on Render.
- **API Documentation:** Swagger UI available at `/docs` on the backend.

---

## 🗂️ Project Structure

```
ISRO/
  OrbitBot/
    Backend/      # FastAPI backend, vector DB, KG, LLM logic
    UI/           # Vite + React frontend
```

---

## 🏗️ Architecture & Workflow

![Workflow](UI/public/workflow)

---
## 🛰️ Data Pipeline & Extraction

This project uses a robust data pipeline to crawl, parse, and extract structured and unstructured content from the MOSDAC portal:

**Technologies Used:**
- **Python** (Google Colab environment)
- **crawl4ai** for deep web crawling and link discovery
- **Playwright** for headless browser automation (dynamic content rendering)
- **BeautifulSoup (bs4)** for HTML parsing and data extraction
- **pandas, lxml, requests** for data manipulation and HTTP requests
- **Google Drive** for persistent storage of extracted data

**Workflow Summary:**
- Deep crawl of the MOSDAC portal to discover all relevant URLs
- Filtering and deduplication of discovered links
- Parsing HTML content to extract FAQs, tables, and other structured data
- Saving extracted data to Google Drive for further processing and use in the AI assistant

## 🗃️ Data Unification & Corpus Creation

After extraction, all collected data (Markdown, JSON FAQs, tables, links, PDF text) is consolidated into a single, unified JSON corpus for downstream AI processing.

**Technologies Used:**
- **Python** (Google Colab environment)
- **PyPDF2** for PDF text extraction
- **json, os, re, typing** for file I/O, regex, and data processing

**Workflow Summary:**
- Read and process unstructured Markdown files (web page content)
- Read and process structured JSON files (FAQs, tables, links)
- Extract text from PDF files
- Normalize and combine all data into a consistent document format
- Assign source URLs and content type details for traceability
- Save the unified corpus as a single JSON file for use in the AI assistant

## 🧠 Vector Database Creation

After unification, the corpus is embedded and indexed for semantic search using a vector database.

**Technologies Used:**
- **Python** (Google Colab)
- **ChromaDB** for vector storage and retrieval
- **LangChain** for vector DB, embeddings, and document chunking
- **Sentence Transformers** (HuggingFace, `all-MiniLM-L6-v2`) for text embeddings
- **langdetect** for language filtering
- **json, os, shutil, re** for file and text processing

**Workflow Summary:**
- Load the unified corpus from Google Drive
- Clean and filter text (remove noise, non-English, etc.)
- Split documents into chunks for embedding
- Generate embeddings using Sentence Transformers
- Store embeddings in ChromaDB for semantic search
- Verify the database with a sample semantic search
- Copy the resulting ChromaDB back to Google Drive for persistence

---

## 🕸️ Knowledge Graph Creation

A knowledge graph is built from the unified corpus to enable relationship-based and entity-centric queries.

**Technologies Used:**
- **Python** (Google Colab)
- **Neo4j** (via `py2neo`) for graph database storage
- **spaCy** (with `en_core_web_lg`) for NLP and entity extraction
- **langdetect** for language filtering
- **tqdm** for progress bars
- **pandas** for data manipulation
- **json, os, re** for file and text processing

**Workflow Summary:**
- Load the unified corpus from Google Drive
- Clean and filter text for NER
- Use spaCy (with custom entity rules) to extract entities (satellites, sensors, organizations, etc.)
- Filter and deduplicate entities
- Populate Neo4j with clean nodes (entities) and their sources
- Scan corpus for sentences containing multiple entities to extract relationship triplets (subject, predicate, object)
- Populate Neo4j with relationships between entities


## 🖥️ Frontend (UI)

The frontend is a modern, responsive web application built with React, Vite, TypeScript, and Tailwind CSS. It provides an intuitive chat interface for users to interact with the AI assistant, view responses, and access quick actions. The UI is optimized for both desktop and mobile devices, ensuring a seamless experience across platforms.

**Key Features:**
- Conversational chat interface
- Quick action buttons for common queries
- Dark mode support
- Responsive design for all screen sizes
- Easy integration with the backend API

### Tech Stack

- React + Vite + TypeScript
- Tailwind CSS

### Local Development

```sh
cd OrbitBot/UI
npm install
npm run dev
```
Visit [http://localhost:5173](http://localhost:5173) (or as shown in your terminal).

### API Endpoint Configuration

- The frontend communicates with the backend at `https://orbitbot.onrender.com`.
- For local development, you can set the API base URL using a `.env` file:
  ```
  VITE_API_BASE_URL=http://127.0.0.1:8000
  ```
  In your code, use:
  ```js
  const apiUrl = import.meta.env.VITE_API_BASE_URL;
  ```

### Deployment

- Deployed on [Vercel](https://vercel.com/).
- To deploy your own, connect your repo to Vercel, set the project root to `UI`, and set environment variables as needed.

---

## 🧠 Backend (API)

The backend is a scalable FastAPI application that powers the AI assistant. It orchestrates hybrid search using both a knowledge graph and a vector database, leverages large language models for answer generation, and exposes secure, well-documented API endpoints for the frontend.

**Key Features:**
- Hybrid RAG (Retrieval-Augmented Generation) search
- Integration with ChromaDB (vector DB) and Neo4j (knowledge graph)
- Uses HuggingFace and TogetherAI for embeddings and LLMs
- RESTful API with automatic Swagger documentation
- CORS-enabled for frontend integration
  
### Tech Stack

- FastAPI (Python 3.11)
- LangChain, ChromaDB, Neo4j, HuggingFace
- CORS enabled for frontend integration

### Setup & Run Locally

1. **Install Python 3.11** (see `runtime.txt`)
2. **Install dependencies:**
   ```sh
   cd OrbitBot/Backend
   python -m venv venv
   source venv/bin/activate  # or .\venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```
3. **Configure environment variables:**  
   Create a `.env` file in `Backend/` with:
   ```
   TOGETHER_API_KEY=your_together_api_key
   HF_API_KEY=your_huggingface_api_key
   NEO4J_URI=bolt://your_neo4j_host:7687
   NEO4J_USERNAME=your_neo4j_username
   NEO4J_PASSWORD=your_neo4j_password
   ```
4. **Ensure ChromaDB data is present:**  
   Place your ChromaDB folder at `OrbitBot/Backend/chroma_db/`.
5. **Run the server:**
   ```sh
   uvicorn main:app --reload
   ```
   Access API docs at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### Deployment

- Deployed on [Render](https://render.com/).
- For your own deployment, set environment variables in the Render dashboard.

---

## 📦 Backend Requirements

See `OrbitBot/Backend/requirements.txt` for all dependencies.

---

## 📝 API Usage

- **POST `/hybrid-search`**  
  Request:
  ```json
  { "question": "What is MOSDAC?" }
  ```
  Response:
  ```json
  {
    "question": "...",
    "answer": "...",
    "kg_context": "...",
    "vector_db_context": "..."
  }
  ```

- **GET `/`**  
  Health check endpoint.



---

## 🛡️ Security

- **Never commit secrets:**  
  Ensure `.env` is in `.gitignore` and never pushed to the repository.
- **Regenerate API keys** if accidentally exposed.

---


---

## 📄 License

MIT License

---

## 📢 Acknowledgements

- ISRO MOSDAC: [https://mosdac.gov.in/](https://mosdac.gov.in/)
- HACK2SKILL:[https://hack2skill.com/](https://hack2skill.com/)
- ISRO:[https://www.isro.gov.in/](https://www.isro.gov.in/)


---

## 👥 Team

* **Mohammad Arif**
* **Tamma Kundana Reddy**
---

## 
