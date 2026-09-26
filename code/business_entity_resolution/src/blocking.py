import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

class TfidfBlocker:
    """
    Character n-gram TF-IDF blocking over normalized business names.
    Country is used as a soft partition: candidates are preferred from
    the same country, but cross-country candidates are retained as a
    fallback so the country field remains an open set.
    """
    def __init__(self, top_k=30, min_score=0.05):
        self.top_k = top_k
        self.min_score = min_score
        self.vectorizer = None
        self.matrix = None
        self.s2 = None

    def fit(self, source2):
        self.s2 = source2.reset_index(drop=True)
        corpus = self.s2["name_norm"].fillna("").tolist()
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            min_df=1,
            sublinear_tf=True,
            max_features=200_000,
        )
        self.matrix = self.vectorizer.fit_transform(corpus)
        return self

    def generate(self, source1):
        if self.vectorizer is None:
            raise RuntimeError("Call fit(source2) first.")

        queries = self.vectorizer.transform(source1["name_norm"].fillna("").tolist())
        n_neighbors = min(self.top_k, self.matrix.shape[0])
        distances, indices = NearestNeighbors(
            n_neighbors=n_neighbors, metric="cosine", algorithm="brute"
        ).fit(self.matrix).kneighbors(queries)

        rows = []
        for i in range(len(source1)):
            s1 = source1.iloc[i]
            same_country = str(s1["country_norm"])
            for d, j in zip(distances[i], indices[i]):
                score = 1.0 - float(d)
                s2 = self.s2.iloc[int(j)]
                if score < self.min_score:
                    continue
                # Same-country candidates are retained; cross-country candidates
                # are also retained when the name signal is sufficiently strong.
                if s2["country_norm"] != same_country and score < 0.35:
                    continue
                rows.append({
                    "source1_entity_id": s1["entity_id"],
                    "candidate_entity_id": s2["entity_id"],
                    "block_name_score": score,
                })
        return rows
