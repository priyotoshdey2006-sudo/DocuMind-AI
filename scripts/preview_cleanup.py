

import json
from pathlib import Path
from collections import defaultdict

CHUNKS_PATH = Path("data/chunks.json")

with CHUNKS_PATH.open("r", encoding="utf-8") as file:
    chunks = json.load(file)

seen_content = set()
id_content = defaultdict(set)
unique_chunks = []

for chunk in chunks:
    metadata = chunk["metadata"]
    chunk_id = chunk["id"]
    text = " ".join(chunk["text"].split())

    id_content[chunk_id].add(text)

    key = (metadata["source"], metadata["page"], text)

    if key not in seen_content:
        seen_content.add(key)
        unique_chunks.append(chunk)

conflicting_ids = {
    chunk_id: contents
    for chunk_id, contents in id_content.items()
    if len(contents) > 1
}

print("\n========== SAFETY CHECK ==========")
print(f"Original chunks: {len(chunks)}")
print(f"Unique chunks: {len(unique_chunks)}")
print(f"Duplicates: {len(chunks) - len(unique_chunks)}")
print(f"Conflicting IDs: {len(conflicting_ids)}")

if conflicting_ids:
    print("\nWARNING: Some IDs contain different text.")
    for chunk_id in conflicting_ids:
        print(chunk_id)
    print("Do not perform cleanup until these are investigated.")
else:
    print("\nNo conflicting chunk IDs detected.")

print("\nPreview only. No files were modified.")
