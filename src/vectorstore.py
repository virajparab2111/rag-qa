import os
from typing import List, Any
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, CrossEncoder
from src.embedding import EmbeddingPipeline


class ChromaVectorStore:
    def __init__(
        self,
        persist_dir: str = "chroma_store",
        collection_name: str = "documents",
        embedding_model: str = "all-MiniLM-L6-v2",
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        os.makedirs(self.persist_dir, exist_ok=True)

        # Embedding model (query-time)
        self.embedding_model = embedding_model
        self.embedder = SentenceTransformer(embedding_model)

        # Cross-encoder reranker
        self.reranker = CrossEncoder(rerank_model)

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.client = chromadb.Client(
            Settings(
                is_persistent = True,
                persist_directory=self.persist_dir,
                anonymized_telemetry=False,
            )
        )

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )

        print(f"[INFO] Embedding model: {embedding_model}")
        print(f"[INFO] Reranker model: {rerank_model}")
        print(f"[INFO] ChromaDB collection: {collection_name}")

    # --------------------------------------------------
    # Build vector store
    # --------------------------------------------------
    def build_from_documents(self, documents: List[Any]):
        if self.collection.count() > 0:
            print("[INFO] Collection already populated — skipping rebuild.")
            return

        print(f"[INFO] Building vector store from {len(documents)} documents...")

        emb_pipe = EmbeddingPipeline(
            model_name=self.embedding_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_chunks(chunks)

        ids = [f"chunk_{i}" for i in range(len(chunks))]
        documents_text = [chunk.page_content for chunk in chunks]

        # Keep metadata minimal but expandable
        metadatas = [
            {
                "text": chunk.page_content,
                **chunk.metadata
            }
            for chunk in chunks
        ]

        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=documents_text,
            metadatas=metadatas,
        )

        print("[INFO] Vector store built (auto-persisted by Chroma).")

    # --------------------------------------------------
    # Basic search (no rerank)
    # --------------------------------------------------
    def query(self, query_text: str, top_k: int = 5):
        print(f"[INFO] Querying (no rerank): '{query_text}'")

        query_emb = self.embedder.encode([query_text]).tolist()

        results = self.collection.query(
            query_embeddings=query_emb,
            n_results=top_k,
            include=["distances", "documents", "metadatas"],
        )

        if not results["documents"]:
            return []

        return [
            {
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            }
            for i in range(len(results["documents"][0]))
        ]

    def query_with_rerank(
        self,
        query_text: str,
        retrieve_k: int = 20,
        final_k: int = 5,
    ):
        """
        Step 1: Dense retrieval
        Step 2: Cross-encoder reranking
        Step 3: Return final_k results
        """
        print(f"[INFO] Querying with rerank: '{query_text}'")

        query_emb = self.embedder.encode([query_text]).tolist()

        results = self.collection.query(
            query_embeddings=query_emb,
            n_results=retrieve_k,
            include=["documents", "metadatas"],
        )

        if not results["documents"]:
            return []

        docs = results["documents"][0]
        metas = results["metadatas"][0]

        # Prepare (query, doc) pairs
        pairs = [(query_text, doc) for doc in docs]

        # Cross-encoder scoring
        scores = self.reranker.predict(pairs)

        # Sort by rerank score
        reranked = sorted(
            zip(docs, metas, scores),
            key=lambda x: x[2],
            reverse=True,
        )

        return [
            {
                "document": doc,
                "metadata": meta,
                "rerank_score": float(score),
            }
            for doc, meta, score in reranked[:final_k]
        ]


# --------------------------------------------------
# Example usage
# --------------------------------------------------
if __name__ == "__main__":
    from src.data_loader import load_all_documents

    docs = load_all_documents(
        r"C:\Users\Asus_owner\PycharmProjects\rag\pdf_data"
    )

    store = ChromaVectorStore(
        persist_dir="chroma_store",
        collection_name="faq_docs",
    )

    store.build_from_documents(docs)

    results = store.query_with_rerank(
        "What is the Cancellation charges for sleeper class?",
        retrieve_k=15,
        final_k=3,
    )
    for r in results:
        print(r)
