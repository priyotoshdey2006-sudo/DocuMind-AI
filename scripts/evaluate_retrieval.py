
from app.retrieval import HybridRetriever

K = 5

TEST_CASES = [
    {
        "question": "What is a stack?",
        "source": "Stack Question Answers.pdf",
        "relevant_pages": [1],
    },
    {
        "question": "What is the FIFO principle?",
        "source": "Queue Question Answers.pdf",
        "relevant_pages": [1],
    },
]


def precision_at_k(results, relevant_pages, k):
    if k <= 0:
        return 0.0

    relevant_count = sum(
        1 for result in results[:k]
        if result["metadata"]["page"] in relevant_pages
    )
    return relevant_count / k


def recall_at_k(results, relevant_pages, k):
    if not relevant_pages:
        return 0.0

    retrieved_pages = {
        result["metadata"]["page"]
        for result in results[:k]
        if result["metadata"]["page"] in relevant_pages
    }

    return len(retrieved_pages) / len(set(relevant_pages))


def reciprocal_rank(results, relevant_pages):
    for rank, result in enumerate(results, start=1):
        if result["metadata"]["page"] in relevant_pages:
            return 1 / rank
    return 0.0


def evaluate_method(retriever, method, test_case):
    search_function = getattr(retriever, method)

    results = search_function(
        query=test_case["question"],
        k=K,
        source=test_case["source"],
    )

    relevant_pages = test_case["relevant_pages"]

    return {
        "precision": precision_at_k(results, relevant_pages, K),
        "recall": recall_at_k(results, relevant_pages, K),
        "mrr": reciprocal_rank(results, relevant_pages),
        "results": results,
    }


def main():
    retriever = HybridRetriever()

    methods = [
        "search_bm25",
        "search_dense",
        "search_hybrid",
    ]

    totals = {
        method: {"precision": 0, "recall": 0, "mrr": 0}
        for method in methods
    }

    print("\n========== RETRIEVAL EVALUATION ==========")

    for test_case in TEST_CASES:
        print(f"\nQuestion: {test_case['question']}")
        print(f"Source: {test_case['source']}")

        for method in methods:
            result = evaluate_method(retriever, method, test_case)

            for metric in ("precision", "recall", "mrr"):
                totals[method][metric] += result[metric]

            print(f"\n{method.upper()}")
            print(f"Precision@{K}: {result['precision']:.3f}")
            print(f"Recall@{K}:    {result['recall']:.3f}")
            print(f"MRR:           {result['mrr']:.3f}")

            for item in result["results"]:
                print(
                    f"  Page {item['metadata']['page']} | "
                    f"Score: {item['score']:.4f}"
                )

    count = len(TEST_CASES)

    print("\n========== AVERAGE COMPARISON ==========")
    print(
        f"{'Method':<18}"
        f"{'Precision@5':<16}"
        f"{'Recall@5':<14}"
        f"{'MRR':<10}"
    )

    for method in methods:
        values = totals[method]

        print(
            f"{method:<18}"
            f"{values['precision'] / count:<16.3f}"
            f"{values['recall'] / count:<14.3f}"
            f"{values['mrr'] / count:<10.3f}"
        )


if __name__ == "__main__":
    main()
