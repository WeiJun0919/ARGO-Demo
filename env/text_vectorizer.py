"""
Text vectorizer module for converting text data to numerical features.

Supports multiple vectorization methods:
- TF-IDF: Classic bag-of-words approach
- Count: Simple word count vectorization
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


class TextVectorizer:
    """Text vectorizer wrapper that converts text to numerical features."""

    def __init__(
        self,
        method="tfidf",
        max_features=1000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        random_state=42,
    ):
        """
        Initialize text vectorizer.

        Args:
            method: Vectorization method - 'tfidf', 'count', or 'embed' (placeholder)
            max_features: Maximum number of features to extract
            ngram_range: N-gram range for text features
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            random_state: Random seed
        """
        self.method = method
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.random_state = random_state
        self.vectorizer = None
        self.is_fitted = False

    def fit(self, texts):
        """
        Fit the vectorizer on text data.

        Args:
            texts: List or array of text strings
        """
        texts = self._ensure_list(texts)

        if self.method == "tfidf":
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                max_df=self.max_df,
                stop_words="english",
            )
        elif self.method == "count":
            self.vectorizer = CountVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                max_df=self.max_df,
                stop_words="english",
            )
        else:
            raise ValueError(f"Unknown vectorization method: {self.method}")

        self.vectorizer.fit(texts)
        self.is_fitted = True
        return self

    def transform(self, texts):
        """
        Transform text to numerical features.

        Args:
            texts: List or array of text strings

        Returns:
            numpy array of shape (n_samples, n_features)
        """
        if not self.is_fitted:
            raise RuntimeError("Vectorizer must be fitted before transform")

        texts = self._ensure_list(texts)
        return self.vectorizer.transform(texts).toarray()

    def fit_transform(self, texts):
        """
        Fit and transform in one step.

        Args:
            texts: List or array of text strings

        Returns:
            numpy array of shape (n_samples, n_features)
        """
        self.fit(texts)
        return self.transform(texts)

    def _ensure_list(self, texts):
        """Ensure input is a list of strings."""
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        elif hasattr(texts, 'tolist'):
            texts = texts.tolist()
        return [str(t) if t is not None else "" for t in texts]

    @property
    def n_features(self):
        """Return the number of features after vectorization."""
        if self.vectorizer is None:
            return 0
        return len(self.vectorizer.get_feature_names_out())

    def get_feature_names(self):
        """Return feature names."""
        if self.vectorizer is None:
            return []
        return self.vectorizer.get_feature_names_out().tolist()


def create_text_vectorizer(method="tfidf", **kwargs):
    """
    Factory function to create a text vectorizer.

    Args:
        method: Vectorization method
        **kwargs: Additional arguments passed to TextVectorizer

    Returns:
        TextVectorizer instance
    """
    return TextVectorizer(method=method, **kwargs)
