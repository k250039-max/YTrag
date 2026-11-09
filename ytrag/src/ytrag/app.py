from ytrag.data_loader import load_all_documents
from ytrag.embedding import EmbeddingPipeline
from ytrag.vectorstore import VectorStore
from ytrag.retriever import RAGRetriever
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "notebook" / "pdf"


def main():
    docs = load_all_documents(str(DATA_DIR))
    if not docs:
        print("No documents found in the 'data' folder.")
        return

    pipeline = EmbeddingPipeline()
    chunks = pipeline.chunk_documents(docs)
    embeddings = pipeline.embed_documents(chunks)

    store = VectorStore()
    store.add_document(chunks, embeddings)

    retriever = RAGRetriever(store, pipeline)

    while True:
        query = input("\nAsk a question (or 'exit'): ").strip()
        if query.lower() in ("exit", "quit", "q"):
            break
        if not query:
            continue
        results = retriever.retrieve(query, top_k=5)
        for r in results:
            print(f"\n[{r['rank']}] score={r['similarity_score']:.3f} | {r['metadata'].get('source', '')}")
            print(r["content"][:500])


if __name__ == "__main__":
    main()