"""
Hybrid RAG Retriever module with Dense and Keyword Search,
similarity scoring, and confidence-based relevance gating.
"""

import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class RetrievalResult:
    """Encapsulates a single retrieved document chunk with relevance scores."""
    def __init__(
        self,
        id: str,
        category: str,
        question: str,
        answer: str,
        score: float,
        text: str,
        keywords: Optional[List[str]] = None
    ):
        self.id = id
        self.category = category
        self.question = question
        self.answer = answer
        self.score = round(float(score), 4)
        self.text = text
        self.keywords = keywords or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "question": self.question,
            "answer": self.answer,
            "score": self.score,
            "text": self.text,
            "keywords": self.keywords,
        }

    def __repr__(self) -> str:
        return f"<RetrievalResult id='{self.id}' score={self.score} question='{self.question[:40]}...'>"


class HybridRetriever:
    """
    Combines dense semantic vector retrieval with keyword/BM25 retrieval,
    with automatic fallbacks and similarity score thresholding.
    """

    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        dense_model_name: str = "all-MiniLM-L6-v2",
        use_dense: bool = True,
        relevance_threshold: float = 0.45,
        hybrid_weight: float = 0.65  # Weight on dense (0.65 dense, 0.35 keyword)
    ):
        self.chunks = chunks
        self.relevance_threshold = relevance_threshold
        self.hybrid_weight = hybrid_weight
        self.dense_model_name = dense_model_name
        self.dense_model = None
        self.dense_embeddings = None
        self.dense_backend = "tfidf"  # 'sentence-transformers' or 'tfidf'

        self.corpus_texts = [c["text"] for c in chunks]
        self._init_keyword_retriever()

        if use_dense:
            self._init_dense_retriever()
        else:
            self._init_tfidf_dense_fallback()

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer for BM25 and keyword matching."""
        clean = re.sub(r"[^\w\s-]", " ", text.lower())
        return [w for w in clean.split() if len(w) > 1]

    def _init_keyword_retriever(self):
        """Initializes BM25 and TF-IDF for keyword matching."""
        tokenized_corpus = [self._tokenize(text) for text in self.corpus_texts]
        if BM25_AVAILABLE and len(tokenized_corpus) > 0:
            self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            self.bm25 = None

        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words="english"
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.corpus_texts)

    def _init_dense_retriever(self):
        """Attempts to load SentenceTransformer; falls back to TF-IDF embeddings on failure."""
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                # Load model
                self.dense_model = SentenceTransformer(self.dense_model_name)
                self.dense_embeddings = self.dense_model.encode(
                    self.corpus_texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                self.dense_backend = "sentence-transformers"
                return
            except Exception as e:
                # Fallback gracefully
                self._init_tfidf_dense_fallback()
        else:
            self._init_tfidf_dense_fallback()

    def _init_tfidf_dense_fallback(self):
        """Initializes normalized TF-IDF vector space as the vector retrieval engine."""
        self.dense_backend = "tfidf"
        self.dense_embeddings = self.tfidf_matrix

    def _compute_dense_scores(self, query: str) -> np.ndarray:
        """Computes dense cosine similarity scores."""
        if self.dense_backend == "sentence-transformers" and self.dense_model is not None:
            query_vec = self.dense_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            # Dot product of normalized vectors equals cosine similarity
            sims = np.dot(self.dense_embeddings, query_vec.T).flatten()
            return np.clip(sims, 0.0, 1.0)
        else:
            query_vec = self.tfidf_vectorizer.transform([query])
            sims = cosine_similarity(self.tfidf_matrix, query_vec).flatten()
            return np.clip(sims, 0.0, 1.0)

    def _compute_keyword_scores(self, query: str) -> np.ndarray:
        """Computes normalized keyword/BM25 scores."""
        if self.bm25 is not None:
            tokens = self._tokenize(query)
            if not tokens:
                return np.zeros(len(self.chunks))
            raw_scores = np.array(self.bm25.get_scores(tokens))
            max_score = np.max(raw_scores)
            if max_score > 0:
                return raw_scores / max_score
            return np.zeros(len(self.chunks))
        else:
            query_vec = self.tfidf_vectorizer.transform([query])
            sims = cosine_similarity(self.tfidf_matrix, query_vec).flatten()
            return np.clip(sims, 0.0, 1.0)

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        threshold: Optional[float] = None
    ) -> Tuple[List[RetrievalResult], bool]:
        """
        Retrieves the top_k most relevant chunks for the given query.

        Returns:
            results: List of RetrievalResult objects sorted by score descending.
            is_relevant: Boolean indicating whether the best score meets the relevance threshold.
        """
        if not query or not query.strip():
            return [], False

        thresh = threshold if threshold is not None else self.relevance_threshold

        dense_scores = self._compute_dense_scores(query)
        keyword_scores = self._compute_keyword_scores(query)

        # Hybrid linear combination
        hybrid_scores = (
            self.hybrid_weight * dense_scores +
            (1.0 - self.hybrid_weight) * keyword_scores
        )

        ranked_indices = np.argsort(hybrid_scores)[::-1]

        results: List[RetrievalResult] = []
        for idx in ranked_indices[:top_k]:
            score = float(hybrid_scores[idx])
            chunk = self.chunks[idx]
            results.append(
                RetrievalResult(
                    id=chunk["id"],
                    category=chunk["category"],
                    question=chunk["question"],
                    answer=chunk["answer"],
                    score=score,
                    text=chunk["text"],
                    keywords=chunk.get("keywords", [])
                )
            )

        best_score = results[0].score if results else 0.0
        is_relevant = best_score >= thresh

        return results, is_relevant
