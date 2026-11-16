
import json
from pathlib import Path

CHUNKS_PATH = Path("data/chunks.json")

with CHUNKS_PATH.open("r", encoding="utf-8") as file:
    chunks = json.load(file)

seen = set()
unique_chunks = []

for chunk in chunks:
    metadata = chunk["metadata"]

    key = (
        metadata["source"],
        metadata["page"],
        " ".join(chunk["text"].split())
    )

    if key not in seen:
        seen.add(key)
        unique_chunks.append(chunk)

removed = len(chunks) - len(unique_chunks)

# Save cleaned data to a temporary file first.
temp_path = CHUNKS_PATH.with_suffix(".clean.tmp")

temp_path.write_text(
    json.dumps(unique_chunks, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

# Replace the original only after writing succeeds.
temp_path.replace(CHUNKS_PATH)

print("\n========== CLEANUP COMPLETE ==========")
print(f"Original chunks : {len(chunks)}")
print(f"Remaining chunks: {len(unique_chunks)}")
print(f"Removed chunks  : {removed}")

print("\nDocument-wise totals:")

for source in sorted({c["metadata"]["source"] for c in unique_chunks}):
    count = sum(
        1 for c in unique_chunks
        if c["metadata"]["source"] == source
    )
    print(f"{source}: {count}")

print("\nCleaned chunks.json successfully.")
