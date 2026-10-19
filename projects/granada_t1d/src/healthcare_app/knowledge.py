"""Versioned, local lexical vector retrieval of sourced educational summaries."""
import hashlib
import json
from pathlib import Path
import numpy as np
from collections import Counter
import re
from urllib.parse import urlparse
from sklearn.feature_extraction.text import TfidfVectorizer


class KnowledgeBase:
    def __init__(self, path=None, semantic_model=None):
        raw = Path(path or Path(__file__).with_name("knowledge.json")).read_bytes()
        self.fingerprint = hashlib.sha256(raw).hexdigest()
        self.documents = json.loads(raw)
        if not self.documents or len({d["id"] for d in self.documents}) != len(self.documents):
            raise ValueError("Knowledge base is empty or has duplicate evidence IDs.")
        trusted_hosts = {"medlineplus.gov", "diabetesjournals.org", "doi.org", "www.niddk.nih.gov"}
        if any(urlparse(d.get("url", "")).scheme != "https" or urlparse(d.get("url", "")).hostname not in trusted_hosts or not d.get("content") or not d.get("reviewed") for d in self.documents):
            raise ValueError("Knowledge base requires versioned, reviewed sources from approved medical publishers.")
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=512)
        corpus = [f"{d['title']} {d['category']} {' '.join(d['tests'])} {d['content']}" for d in self.documents]
        self.matrix = self.vectorizer.fit_transform(corpus).toarray().astype(np.float32)
        self.char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=4096)
        self.char_matrix = self.char_vectorizer.fit_transform(corpus)
        self.tokens = [Counter(self._tokenize(text)) for text in corpus]
        self.lengths = np.asarray([sum(tokens.values()) for tokens in self.tokens])
        self.avg_length = float(self.lengths.mean())
        self.document_frequency = Counter(token for tokens in self.tokens for token in tokens)
        self.retrieval_method = "Hybrid word/character TF-IDF + BM25; applicability filtered; no learned semantic embeddings"
        self.encoder = None
        if semantic_model:
            if semantic_model != "BAAI/bge-base-en-v1.5":
                raise ValueError("Only the reviewed BGE model is supported. No arbitrary model code is loaded.")
            from sentence_transformers import SentenceTransformer
            import torch
            torch.set_num_threads(2)
            revision = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
            self.encoder = SentenceTransformer(semantic_model, device="cpu", local_files_only=True,
                                              revision=revision, trust_remote_code=False, model_kwargs={"use_safetensors": True})
            self.dense_matrix = self.encoder.encode(corpus, normalize_embeddings=True, show_progress_bar=False)
            self.retrieval_method = "Offline BGE-base-en-v1.5 embeddings + BM25 + word/character TF-IDF; applicability filtered"
            self.fingerprint = hashlib.sha256(raw + semantic_model.encode() + revision.encode()).hexdigest()
        self.by_id = {d["id"]: d for d in self.documents}

    def retrieve(self, query, top_k=3, min_score=0.12):
        if not isinstance(query, str) or not query.strip() or len(query) > 2000:
            return []
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        lexical = self.matrix @ self.vectorizer.transform([query]).toarray()[0]
        query_tokens = self._tokenize(query)
        dense = None
        if self.encoder is not None:
            embedding = self.encoder.encode(["Represent this sentence for searching relevant passages: " + query], normalize_embeddings=True, show_progress_bar=False)[0]
            dense = np.asarray(self.dense_matrix @ embedding)
        if not any(token in self.document_frequency for token in query_tokens) and (dense is None or dense.max() < 0.72):
            return []
        bm25 = np.zeros(len(self.documents))
        for token in set(query_tokens):
            df = self.document_frequency.get(token, 0)
            if not df:
                continue
            idf = np.log(1 + (len(self.documents) - df + 0.5) / (df + 0.5))
            frequency = np.asarray([tokens.get(token, 0) for tokens in self.tokens])
            bm25 += idf * frequency * 2.5 / (frequency + 1.5 * (0.25 + 0.75 * self.lengths / self.avg_length))
        bm25 /= (bm25.max() or 1)
        character = (self.char_matrix @ self.char_vectorizer.transform([query]).T).toarray().ravel()
        scores = 0.65 * lexical + 0.25 * bm25 + 0.10 * character
        if dense is not None:
            scores = 0.80 * scores + 0.20 * np.clip((dense - 0.4) / 0.6, 0, 1)
        ranked = np.argsort(-scores, kind="stable")[:min(top_k, len(scores))]
        return [{**self.documents[int(i)], "score": round(float(scores[i]), 4), "rank": n + 1} for n, i in enumerate(ranked) if scores[i] >= min_score]

    def _tokenize(self, text):
        stop = self.vectorizer.get_stop_words()
        return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1 and t not in stop]

    def for_event(self, event):
        if not event["recognized_name"]:
            return []
        query = event["name"]
        if event["kind"] == "medication":
            query += " medicine medication"
        return [d for d in self.retrieve(query, top_k=len(self.documents), min_score=0.01)
                if event["name"] in d.get("events", []) or event["kind"] == "medication" and d["id"] == "NLM-MEDICINES"][:3]

    def for_test(self, test):
        # Vector retrieval followed by explicit curated applicability. No patient similarity.
        candidates = {d["id"] for d in self.documents if test in d["tests"]}
        return [d for d in self.retrieve(test, top_k=len(self.documents), min_score=0.01) if d["id"] in candidates]
