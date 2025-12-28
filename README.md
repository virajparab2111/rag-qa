# **📄 RAG Policy QA System**

A **Retrieval-Augmented Generation (RAG)** based question-answering system for policy documents (PDFs & text files).  
The system supports **dense retrieval + reranking**, **source citations**, **answer confidence scoring**, and **JSON-constrained outputs**, powered by **ChromaDB** and **Phi-3 Mini (Ollama)**.

## **🚀 Features**

- 📚 Multi-document ingestion (PDF, TXT)
- 🔍 Semantic search using Sentence Transformers
- 🔁 Cross-encoder reranking for high-precision retrieval
- 🧠 LLM-based answering using **Phi-3 Mini**
- 📌 Source-level citations per answer bullet
- 📊 Confidence score for each answer
- 📦 Fully local (no external APIs required)

## **🛠️ Setup Instructions**

### **1️⃣ Install Python dependencies**

````bash
pip install -r requirements.txt
````

### **2️⃣ Install Ollama**

Download and install Ollama from:

👉 <https://ollama.com/download>

Verify installation:

````bash
ollama --version
````

### **3️⃣ Pull the Phi-3 Mini model**


````bash
ollama pull phi3:mini
````
This downloads the LLM used for answering questions.

### **4️⃣ Run the RAG pipeline**

````bash
python rag.py
````

The system will:

- Load documents from pdf_data/ and data/
- Build a ChromaDB vector store (if not already present)
- Perform retrieval, reranking, and answer generation

## **🧱 Project Architecture**

{content: 
rag-policy-qa/

│

├── src/

│ ├── data_loader.py # Loads PDFs & text files with metadata

│ ├── embedding.py # Chunking & embedding pipeline

│ ├── vectorstore.py # ChromaDB + cross-encoder reranking

│ ├── search.py # RAG orchestration (retrieve → prompt → LLM)

│ ├── prompt.py # Strict JSON-based RAG prompt template

│ ├── utils.py # Context building & evaluation helpers

│

├── pdf_data/ # Policy PDFs (tracked intentionally)

├── data/ # Text-based policy documents

│

├── rag.py # Main entry point

├── requirements.txt # Python dependencies

├── README.md # Project documentation

└── .gitignore
}

## **🔍 Retrieval & Answer Flow**

- **Document Loading**
  - PDFs and text files are loaded with source metadata
- **Chunking & Embedding**
  - Documents are split into overlapping chunks
  - Embeddings generated using all-MiniLM-L6-v2
- **Vector Storage**
  - Chunks stored in **ChromaDB**
- **Dense Retrieval**
  - Top-K chunks retrieved via vector similarity
- **Cross-Encoder Reranking**
  - Retrieved chunks reranked using ms-marco-MiniLM
- **LLM Answering**
  - Context injected into a **strict JSON prompt**
  - Phi-3 Mini generates:
    - Bullet-point answers
    - Source citations
    - Confidence score

## **📤 Output Format (Guaranteed)**

```json
{

"answer": \[

{

"point": "Refund is granted if the train is cancelled by the railways.",

"sources": \["CancellationRulesforIRCTCTrain.pdf"\]

}

\],

"confidence": 0.95

}
```

## **⚠️ Notes**

- Vector databases (chroma_store/, chroma_db/) are **not committed**
- All inference runs **locally**
- No internet or paid APIs required after setup

## **📌 Future Improvements**

- Web UI for document upload & querying
- Dockerized deployment
- Advanced evaluation metrics (precision/recall)
- Support for multi-language policies

## **📄 License**

This project is for educational and research purposes.
