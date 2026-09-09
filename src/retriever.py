"""Retriever module for Grounded Customer Support Resolution Search.

Uses TF-IDF vectorization with cosine similarity to retrieve historically
verified AmazonHelp resolutions for incoming customer queries.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import KB_DATA_PATH, TOP_K_RESOLUTIONS, TFIDF_MAX_FEATURES
from src.data_loader import load_knowledge_base


class HistoricalResolutionRetriever:
    """Retrieves top-k historically proven support resolutions for a given customer query."""

    def __init__(self, kb_path: Optional[str] = None):
        self.kb_items = load_knowledge_base()
        self.corpus = [item["customer_text"] for item in self.kb_items]
        self.vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)

    def retrieve(
        self, query: str, top_k: int = TOP_K_RESOLUTIONS
    ) -> List[Dict[str, any]]:
        """Retrieve top-k most similar historical customer queries and their responses.

        Args:
            query: The incoming customer message.
            top_k: Number of historical resolutions to retrieve.

        Returns:
            List of dicts containing customer_text, response_text, similarity_score.
        """
        if not query or not query.strip():
            return []

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []

        for idx in top_indices:
            score = float(similarities[idx])
            item = self.kb_items[idx]
            results.append({
                "tweet_id": item.get("tweet_id", ""),
                "customer_text": item["customer_text"],
                "response_text": item["response_text"],
                "similarity_score": round(score, 4),
            })

        return results
