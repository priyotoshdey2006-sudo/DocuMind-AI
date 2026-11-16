


import json
import re
import math
import urllib.request
import urllib.error
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi


class HybridRetriever:
    def __init__(
        self,
        chunks_path: str = "data/chunks.json",
        db_path: str = "data/chroma_db_clean",
        collection_name: str = "documind_ollama",
        embedding_model: str = "nomic-embed-text",
        ollama_url: str = "http://localhost:11434",
    ):
        self.chunks_path = Path(chunks_path)
        self.embedding_model = embedding_model
        self.ollama_url = ollama_url.rstrip("/")

        self.chunks = json.loads(
            self.chunks_path.read_text(encoding="utf-8")
        )

        if not self.chunks:
            raise ValueError(f"No chunks found in {chunks_path}")

        self.client = chromadb.PersistentClient(path=db_path)

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        self._rebuild_bm25()
        self._index_missing_chunks()

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def _get_embedding(
        self,
        text: str,
        is_query: bool = False,
    ) -> list[float]:

        prefix = "search_query: " if is_query else "search_document: "

        payload = {
            "model": self.embedding_model,
            "prompt": prefix + text,
        }

        request = urllib.request.Request(
            url=f"{self.ollama_url}/api/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                result = json.loads(response.read().decode("utf-8"))

            embedding = result.get("embedding")

            if not embedding:
                raise RuntimeError("Ollama returned an empty embedding.")

            magnitude = math.sqrt(
                sum(value * value for value in embedding)
            )

            if magnitude == 0:
                raise RuntimeError("Ollama returned a zero embedding.")

            return [value / magnitude for value in embedding]

        except urllib.error.URLError as error:
            raise RuntimeError(
                "Could not connect to Ollama. Check that Ollama is running."
            ) from error

    def _rebuild_bm25(self):
        self.texts = [chunk["text"] for chunk in self.chunks]

        self.tokenized = [
            self._tokenize(text) for text in self.texts
        ]

        self.bm25 = BM25Okapi(self.tokenized)

    def _index_missing_chunks(self):
        stored_ids = set(self.collection.get()["ids"])

        missing_chunks = [
            chunk for chunk in self.chunks
            if chunk["id"] not in stored_ids
        ]

        if not missing_chunks:
            print(
                f"ChromaDB already contains {self.collection.count()} chunks."
            )
            return

        print(
            f"Generating Ollama embeddings for "
            f"{len(missing_chunks)} chunks..."
        )

        embeddings = []

        for index, chunk in enumerate(missing_chunks, start=1):
            embeddings.append(self._get_embedding(chunk["text"]))

            if index % 10 == 0 or index == len(missing_chunks):
                print(f"Embedded {index}/{len(missing_chunks)} chunks")

        self.collection.add(
            ids=[chunk["id"] for chunk in missing_chunks],
            documents=[chunk["text"] for chunk in missing_chunks],
            metadatas=[
                {
                    **chunk["metadata"],
                    "chunk_id": chunk["id"],
                }
                for chunk in missing_chunks
            ],
            embeddings=embeddings,
        )

        print(f"Successfully indexed {len(missing_chunks)} chunks.")

    def add_chunks(self, new_chunks: list[dict]) -> int:
        if not new_chunks:
            return 0

        known_ids = {chunk["id"] for chunk in self.chunks}

        unique_chunks = [
            chunk for chunk in new_chunks
            if chunk["id"] not in known_ids
        ]

        if not unique_chunks:
            return 0

        embeddings = [
            self._get_embedding(chunk["text"])
            for chunk in unique_chunks
        ]

        self.collection.add(
            ids=[chunk["id"] for chunk in unique_chunks],
            documents=[chunk["text"] for chunk in unique_chunks],
            metadatas=[
                {
                    **chunk["metadata"],
                    "chunk_id": chunk["id"],
                }
                for chunk in unique_chunks
            ],
            embeddings=embeddings,
        )

        self.chunks.extend(unique_chunks)

        self.chunks_path.parent.mkdir(parents=True, exist_ok=True)

        temp_path = self.chunks_path.with_suffix(".tmp")

        temp_path.write_text(
            json.dumps(
                self.chunks,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        temp_path.replace(self.chunks_path)

        self._rebuild_bm25()

        return len(unique_chunks)

    def list_documents(self) -> list[dict]:
        """Return available PDF filenames and their chunk counts."""

        counts = {}

        for chunk in self.chunks:
            source = chunk["metadata"]["source"]
            counts[source] = counts.get(source, 0) + 1

        return [
            {"filename": filename, "chunks": count}
            for filename, count in sorted(counts.items())
        ]

    def search_bm25(
        self,
        query: str,
        k: int = 5,
        source: str | None = None,
    ) -> list[dict]:

        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        candidates = [
            index
            for index, chunk in enumerate(self.chunks)
            if source is None or chunk["metadata"]["source"] == source
        ]

        ranked = sorted(
            candidates,
            key=lambda index: scores[index],
            reverse=True,
        )[:k]

        return [
            {
                **self.chunks[index],
                "score": float(scores[index]),
            }
            for index in ranked
        ]

    def search_dense(
        self,
        query: str,
        k: int = 5,
        source: str | None = None,
    ) -> list[dict]:

        collection_size = self.collection.count()

        if collection_size == 0:
            return []

        query_embedding = self._get_embedding(
            query,
            is_query=True,
        )

        query_args = {
            "query_embeddings": [query_embedding],
            "n_results": min(k, collection_size),
            "include": ["documents", "metadatas", "distances"],
        }

        if source is not None:
            query_args["where"] = {"source": source}

        results = self.collection.query(**query_args)

        matches = []

        if not results["documents"] or not results["documents"][0]:
            return []

        for text, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            matches.append({
                "id": metadata["chunk_id"],
                "text": text,
                "metadata": {
                    "source": metadata["source"],
                    "page": metadata["page"],
                },
                "score": 1.0 - float(distance),
            })

        return matches

    def search_hybrid(
        self,
        query: str,
        k: int = 5,
        pool: int = 20,
        source: str | None = None,
    ) -> list[dict]:
        """Combine BM25 and dense results using RRF."""

        bm25_results = self.search_bm25(
            query, pool, source=source
        )

        dense_results = self.search_dense(
            query, pool, source=source
        )

        by_id = {}
        rrf_scores = {}

        for results in (bm25_results, dense_results):
            for rank, result in enumerate(results, start=1):
                chunk_id = result["id"]

                by_id[chunk_id] = result

                rrf_scores[chunk_id] = (
                    rrf_scores.get(chunk_id, 0.0)
                    + 1.0 / (60 + rank)
                )

        ranked_ids = sorted(
            rrf_scores,
            key=rrf_scores.get,
            reverse=True,
        )[:k]

        return [
            {
                **by_id[chunk_id],
                "score": rrf_scores[chunk_id],
            }
            for chunk_id in ranked_ids
        ]

    def search_reranked(
        self,
        query: str,
        k: int = 5,
        pool: int = 20,
        source: str | None = None,
    ) -> list[dict]:
        """Compatibility method with document-level filtering."""

        return self.search_hybrid(
            query=query,
            k=k,
            pool=pool,
            source=source,
        )
