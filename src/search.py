from langchain_community.chat_models import ChatOllama
from src.vectorstore import ChromaVectorStore
from src.data_loader import load_all_documents
from src.utils import build_context, render_rag_prompt


class RAGSearch:
    def __init__(
        self,
        persist_dir: str = "chroma_store",
        collection_name: str = "policy_docs",
        embedding_model: str = "all-MiniLM-L6-v2",
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        llm_model: str = "phi3:mini",
    ):
        # -----------------------------
        # Vector store (Chroma + rerank)
        # -----------------------------
        self.vectorstore = ChromaVectorStore(
            persist_dir=persist_dir,
            collection_name=collection_name,
            embedding_model=embedding_model,
            rerank_model=rerank_model,
        )

        # Build store if empty
        if self.vectorstore.collection.count() == 0:
            print("[INFO] No existing Chroma collection found. Building new store...")
            docs = load_all_documents(
                r"C:\Users\Asus_owner\PycharmProjects\rag\pdf_data"
            )
            self.vectorstore.build_from_documents(docs)
        else:
            print(
                f"[INFO] Found {self.vectorstore.collection.count()} documents in ChromaDB. Using existing collection.")

        # -----------------------------
        # LLM (Ollama – Phi-3 Mini)
        # -----------------------------
        self.llm = ChatOllama(
            model=llm_model,
            temperature=0.0  # deterministic for policy QA
        )

        print(f"[INFO] Ollama LLM initialized: {llm_model}")

    def search(self, query: str, retrieve_k: int = 20, final_k: int = 5) -> str:
        # Step 1: Retrieve + rerank
        results = self.vectorstore.query_with_rerank(
            query_text=query,
            retrieve_k=retrieve_k,
            final_k=final_k
        )

        # Step 2: Build citation-aware context
        context = build_context(results)

        # Step 3: Render constrained prompt
        prompt = render_rag_prompt(
            context=context,
            question=query
        )

        # Step 4: Invoke LLM
        response = self.llm.invoke(prompt)

        # Phi-3 returns clean JSON well with strict prompting
        return response.content


# --------------------------------------------------
# Example usage
# --------------------------------------------------
if __name__ == "__main__":
    rag = RAGSearch()

    query = "Do you support crypto payments?"
    result = rag.search(query, retrieve_k=20, final_k=3)

    print("RAG JSON Output:")
    print(result)
