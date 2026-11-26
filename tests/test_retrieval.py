
import pytest

from app.retrieval import HybridRetriever


def make_chunk(chunk_id, text, source, page=1):
    return {
        "id": chunk_id,
        "text": text,
        "metadata": {
            "source": source,
            "page": page,
        },
    }


class FakeCollection:
    def __init__(self, results=None, count=2):
        self.results = results or {}
        self.size = count
        self.last_query = None

    def count(self):
        return self.size

    def query(self, **kwargs):
        self.last_query = kwargs
        return self.results


@pytest.fixture
def retriever():
    instance = HybridRetriever.__new__(HybridRetriever)

    instance.chunks = [
        make_chunk("1", "Python programming language", "python.pdf"),
        make_chunk("2", "Java object oriented programming", "java.pdf"),
        make_chunk("3", "Python machine learning", "python.pdf"),
    ]

    instance.collection = FakeCollection()
    instance._rebuild_bm25()

    return instance


def test_tokenize_converts_text_to_lowercase():
    result = HybridRetriever._tokenize("Python PROGRAMMING 123")

    assert result == ["python", "programming", "123"]


def test_empty_bm25_index_returns_empty_results():
    instance = HybridRetriever.__new__(HybridRetriever)
    instance.chunks = []
    instance._rebuild_bm25()

    assert instance.search_bm25("Python") == []


def test_bm25_returns_relevant_results(retriever):
    results = retriever.search_bm25("Python")

    assert results
    assert all("score" in item for item in results)
    assert all("id" in item for item in results)


def test_bm25_filters_by_selected_document(retriever):
    results = retriever.search_bm25(
        "programming",
        source="java.pdf",
    )

    assert results
    assert all(
        item["metadata"]["source"] == "java.pdf"
        for item in results
    )


def test_list_documents_returns_correct_counts(retriever):
    documents = retriever.list_documents()

    assert documents == [
        {"filename": "java.pdf", "chunks": 1},
        {"filename": "python.pdf", "chunks": 2},
    ]


def test_dense_search_returns_empty_for_empty_collection(retriever):
    retriever.collection = FakeCollection(count=0)

    results = retriever.search_dense("Python")

    assert results == []


def test_dense_search_filters_by_selected_document(retriever):
    retriever._get_embedding = lambda text, is_query=False: [1.0, 0.0]

    retriever.collection = FakeCollection(
        count=2,
        results={
            "documents": [["Java programming"]],
            "metadatas": [[
                {
                    "chunk_id": "2",
                    "source": "java.pdf",
                    "page": 1,
                }
            ]],
            "distances": [[0.1]],
        },
    )

    results = retriever.search_dense(
        "Java",
        source="java.pdf",
    )

    assert len(results) == 1
    assert results[0]["id"] == "2"
    assert results[0]["metadata"]["source"] == "java.pdf"
    assert retriever.collection.last_query["where"] == {
        "source": "java.pdf"
    }


def test_dense_search_handles_no_matches(retriever):
    retriever._get_embedding = lambda text, is_query=False: [1.0, 0.0]

    retriever.collection = FakeCollection(
        count=2,
        results={
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        },
    )

    assert retriever.search_dense("unknown") == []


def test_hybrid_search_combines_bm25_and_dense(retriever, monkeypatch):
    bm25_results = [
        make_chunk("1", "Python programming", "python.pdf")
    ]

    dense_results = [
        make_chunk("2", "Python machine learning", "python.pdf")
    ]

    monkeypatch.setattr(
        retriever,
        "search_bm25",
        lambda query, k, source=None: bm25_results,
    )

    monkeypatch.setattr(
        retriever,
        "search_dense",
        lambda query, k, source=None: dense_results,
    )

    results = retriever.search_hybrid("Python", k=5)

    result_ids = {item["id"] for item in results}

    assert result_ids == {"1", "2"}
    assert all(item["score"] > 0 for item in results)


def test_hybrid_search_respects_result_limit(retriever, monkeypatch):
    results = [
        make_chunk(str(i), f"Python topic {i}", "python.pdf")
        for i in range(1, 4)
    ]

    monkeypatch.setattr(
        retriever,
        "search_bm25",
        lambda query, k, source=None: results,
    )

    monkeypatch.setattr(
        retriever,
        "search_dense",
        lambda query, k, source=None: [],
    )

    output = retriever.search_hybrid("Python", k=2)

    assert len(output) == 2
