"""Tests for Historical Resolution Retriever."""

import pytest
from src.retriever import HistoricalResolutionRetriever


@pytest.fixture(scope="module")
def retriever():
    return HistoricalResolutionRetriever()


def test_retriever_initialization(retriever):
    assert len(retriever.kb_items) >= 500
    assert retriever.tfidf_matrix.shape[0] == len(retriever.kb_items)


def test_retriever_query(retriever):
    results = retriever.retrieve("Where is my package delivery?", top_k=3)
    assert len(results) == 3
    for r in results:
        assert "customer_text" in r
        assert "response_text" in r
        assert 0.0 <= r["similarity_score"] <= 1.0


def test_retriever_empty_query(retriever):
    results = retriever.retrieve("")
    assert results == []
