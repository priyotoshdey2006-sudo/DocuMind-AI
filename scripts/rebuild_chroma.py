
import shutil
from pathlib import Path

import chromadb

DB_PATH = Path("data/chroma_db")
COLLECTION_NAME = "documind_ollama"

if not DB_PATH.exists():
    raise FileNotFoundError("ChromaDB directory does not exist.")

client = chromadb.PersistentClient(path=str(DB_PATH))

existing = client.get_collection(COLLECTION_NAME)

print("Old ChromaDB chunks:", existing.count())

# Delete only the target collection.
client.delete_collection(COLLECTION_NAME)

print("Old collection deleted successfully.")

# Create a fresh database directory.
shutil.rmtree(DB_PATH)

DB_PATH.mkdir(parents=True, exist_ok=True)

print("Old ChromaDB directory replaced.")
print("Ready to rebuild from cleaned chunks.json.")
