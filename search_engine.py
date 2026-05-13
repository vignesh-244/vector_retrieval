import math
import os
import re
from collections import Counter


class SearchEngine:
    def __init__(self, documents_dir="documents"):
        self.documents_dir = documents_dir
        self.documents = []
        self.vocabulary = []
        self.idf_values = {}
        self.document_vectors = {}

    def build_index(self):
        self.documents = self._load_documents()
        self.vocabulary = self._build_vocabulary(self.documents)
        self.idf_values = self._compute_idf(self.documents, self.vocabulary)
        self.document_vectors = self._build_document_vectors()

    def search(self, query, limit=3):
        if not query.strip():
            return []

        if not self.documents:
            self.build_index()

        query_vector = self._build_query_vector(query)
        results = []

        for document in self.documents:
            score = self._cosine_similarity(
                query_vector,
                self.document_vectors.get(document["filename"], {}),
            )
            if score > 0:
                results.append(
                    {
                        "document": document["filename"],
                        "score": round(score, 4),
                        "snippet": self._make_snippet(document["content"]),
                    }
                )

        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:limit]

    def _load_documents(self):
        documents = []

        if not os.path.isdir(self.documents_dir):
            return documents

        for filename in sorted(os.listdir(self.documents_dir)):
            path = os.path.join(self.documents_dir, filename)
            if not os.path.isfile(path) or not filename.endswith(".txt"):
                continue

            with open(path, "r", encoding="utf-8") as file:
                content = file.read()

            documents.append(
                {
                    "filename": filename,
                    "content": content,
                    "tokens": self._tokenize(content),
                }
            )

        return documents

    def _tokenize(self, text):
        text = text.lower()
        return re.findall(r"[a-z0-9]+", text)

    def _build_vocabulary(self, documents):
        words = set()
        for document in documents:
            words.update(document["tokens"])
        return sorted(words)

    def _compute_idf(self, documents, vocabulary):
        total_documents = len(documents)
        idf_values = {}

        for word in vocabulary:
            matching_documents = 0
            for document in documents:
                if word in document["tokens"]:
                    matching_documents += 1

            idf_values[word] = math.log((total_documents + 1) / (matching_documents + 1)) + 1

        return idf_values

    def _build_document_vectors(self):
        vectors = {}
        for document in self.documents:
            vectors[document["filename"]] = self._build_tfidf_vector(document["tokens"])
        return vectors

    def _build_query_vector(self, query):
        tokens = self._tokenize(query)
        return self._build_tfidf_vector(tokens)

    def _build_tfidf_vector(self, tokens):
        if not tokens:
            return {}

        token_counts = Counter(tokens)
        total_tokens = len(tokens)
        vector = {}

        for word, count in token_counts.items():
            if word not in self.idf_values:
                continue

            tf = count / total_tokens
            vector[word] = tf * self.idf_values[word]

        return vector

    def _cosine_similarity(self, first_vector, second_vector):
        if not first_vector or not second_vector:
            return 0.0

        dot_product = 0.0
        for word, value in first_vector.items():
            dot_product += value * second_vector.get(word, 0.0)

        first_magnitude = math.sqrt(sum(value * value for value in first_vector.values()))
        second_magnitude = math.sqrt(sum(value * value for value in second_vector.values()))

        if first_magnitude == 0 or second_magnitude == 0:
            return 0.0

        return dot_product / (first_magnitude * second_magnitude)

    def _make_snippet(self, content):
        lines = [line.strip() for line in content.splitlines() if line.strip()]

        if len(lines) >= 2:
            snippet = " ".join(lines[:2])
        elif lines:
            snippet = lines[0]
        else:
            snippet = content.strip()

        return snippet[:200]
