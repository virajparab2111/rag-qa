from datetime import datetime
from src.data_loader import load_all_documents
from src.vectorstore import ChromaVectorStore
from src.search import RAGSearch

# --------------------------------------------------
# Questions to test
# --------------------------------------------------
questions = [
    ("What happens if a train is cancelled by the railways?", "Answerable"),
    ("What is the clerkage charge for RAC or waitlisted ticket cancellation?", "Answerable"),
    ("What happens if AC facility is not provided during the journey?", "Partially Answerable"),
    ("Can I reschedule my ticket without cancellation charges?", "Answerable"),
]

# --------------------------------------------------
# Load documents & initialize vector store
# --------------------------------------------------
DATA_DIR = r"C:\Users\Asus_owner\PycharmProjects\rag\pdf_data"

documents = load_all_documents(DATA_DIR)

vectorstore = ChromaVectorStore(
    persist_dir="chroma_store",
    collection_name="faq_docs"
)

if vectorstore.collection.count() == 0:
    vectorstore.build_from_documents(documents)

# Initialize RAG pipeline
rag = RAGSearch()

# --------------------------------------------------
# Output file setup
# --------------------------------------------------
OUTPUT_FILE = "output_file.txt"

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("RAG Evaluation Output\n")
    f.write("=" * 80 + "\n")
    f.write(f"Run time: {datetime.now()}\n\n")

# --------------------------------------------------
# Run questions
# --------------------------------------------------
for q, category in questions:
    print("\n-----------------------------")
    print(f"Q ({category}): {q}")
    print("Answer:")

    result = rag.search(
        query=q,
        retrieve_k=30,
        final_k=7
    )

    print(result)

    # Write to output.txt
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write("-" * 80 + "\n")
        f.write(f"Question: {q}\n")
        f.write(f"Expected: {category}\n")
        f.write("Answer JSON:\n")
        f.write(result.strip() + "\n\n")

print(f"\n[INFO] Results saved to {OUTPUT_FILE}")
