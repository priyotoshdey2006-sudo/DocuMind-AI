
from collections import Counter
from app.retrieval import HybridRetriever

retriever = HybridRetriever()

print("\n========== DUPLICATE INSPECTION ==========")

for source in [
    "Queue Question Answers.pdf",
    "Stack Question Answers.pdf",
]:
    chunks = [
        chunk for chunk in retriever.chunks
        if chunk["metadata"]["source"] == source
    ]

    print(f"\nDocument: {source}")
    print(f"Total chunks: {len(chunks)}")

    texts = [chunk["text"].strip() for chunk in chunks]
    counts = Counter(texts)

    duplicates = {
        text: count
        for text, count in counts.items()
        if count > 1
    }

    print(f"Unique text chunks: {len(counts)}")
    print(f"Repeated text chunks: {len(duplicates)}")

    for text, count in duplicates.items():
        print(f"\nRepeated {count} times:")
        print(text[:200])
