from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
    JSONLoader
)
from langchain_community.document_loaders.excel import UnstructuredExcelLoader


def _enrich_metadata(docs, file_path: Path):
    """Attach standardized metadata to each LangChain Document."""
    enriched = []
    for i, doc in enumerate(docs):
        doc.metadata = doc.metadata or {}

        doc.metadata.update({
            "doc_id": f"{file_path.stem}_{i}",
            "source_doc": file_path.name,
            "file_path": str(file_path),
            "file_type": file_path.suffix.lower().replace(".", "")
        })

        enriched.append(doc)
    return enriched


def load_all_documents(data_dir: str) -> List[Any]:
    """
    Load all supported files and attach standardized metadata.
    Supported: PDF, TXT, CSV, Excel, Word, JSON
    """
    data_path = Path(data_dir).resolve()
    print(f"[DEBUG] Data path: {data_path}")

    documents = []

    loaders = [
        ("*.pdf", PyPDFLoader),
        ("*.txt", TextLoader),
        ("*.csv", CSVLoader),
        ("*.xlsx", UnstructuredExcelLoader),
        ("*.docx", Docx2txtLoader),
        ("*.json", JSONLoader),
    ]

    for pattern, loader_cls in loaders:
        files = list(data_path.glob(f"**/{pattern}"))
        print(f"[DEBUG] Found {len(files)} {pattern} files")

        for file_path in files:
            print(f"[DEBUG] Loading {file_path}")
            try:
                loader = loader_cls(str(file_path))
                loaded_docs = loader.load()

                enriched_docs = _enrich_metadata(loaded_docs, file_path)
                documents.extend(enriched_docs)

                print(f"[DEBUG] Loaded {len(enriched_docs)} docs from {file_path}")

            except Exception as e:
                print(f"[ERROR] Failed to load {file_path}: {e}")

    print(f"[DEBUG] Total loaded documents: {len(documents)}")
    return documents


# --------------------------------------------------
# Example usage
# --------------------------------------------------
if __name__ == "__main__":
    docs = load_all_documents(
        r"C:\Users\Asus_owner\PycharmProjects\rag\pdf_data"
    )
    print(f"Loaded {len(docs)} documents.")
    if docs:
        print("Example document content:")
        print(docs[0].page_content[:500])
        print("Example metadata:")
        print(docs[0].metadata)