"""
Embedded Chroma vector store with two collections: one for text embeddings
(transcript/OCR/PDF text, sentence-transformers space) and one for image
embeddings (CLIP space). Kept separate because the two embedding spaces are
not directly comparable — cross-modal matching happens by embedding the
*query* into both spaces and querying each collection independently (see
src/retrieval/query_engine.py), not by mixing vectors from different models
in one index.
"""

from __future__ import annotations

import chromadb

import config

TEXT_COLLECTION = "text_embeddings"
IMAGE_COLLECTION = "image_embeddings"
EVENT_COLLECTION = "event_embeddings"


class VectorStore:
    def __init__(self, persist_dir=None):
        self._client = chromadb.PersistentClient(path=str(persist_dir or config.CHROMA_DIR))
        self.text = self._client.get_or_create_collection(TEXT_COLLECTION, metadata={"hnsw:space": "cosine"})
        self.image = self._client.get_or_create_collection(IMAGE_COLLECTION, metadata={"hnsw:space": "cosine"})
        self.events = self._client.get_or_create_collection(EVENT_COLLECTION, metadata={"hnsw:space": "cosine"})

    @staticmethod
    def _clean_metadata(metadata: dict) -> dict:
        return {k: v for k, v in metadata.items() if v is not None}

    def add_text(self, unit_id: str, vector: list[float], metadata: dict, document: str = "") -> None:
        self.text.upsert(ids=[unit_id], embeddings=[vector], metadatas=[self._clean_metadata(metadata)],
                          documents=[document])

    def add_image(self, unit_id: str, vector: list[float], metadata: dict, document: str = "") -> None:
        self.image.upsert(ids=[unit_id], embeddings=[vector], metadatas=[self._clean_metadata(metadata)],
                           documents=[document])

    def add_event(self, event_id: str, vector: list[float], metadata: dict, document: str = "") -> None:
        self.events.upsert(ids=[event_id], embeddings=[vector], metadatas=[self._clean_metadata(metadata)],
                            documents=[document])

    def query_text(self, vector: list[float], top_k: int) -> list[tuple[str, float]]:
        return self._query(self.text, vector, top_k)

    def query_image(self, vector: list[float], top_k: int) -> list[tuple[str, float]]:
        return self._query(self.image, vector, top_k)

    def query_events(self, vector: list[float], top_k: int) -> list[tuple[str, float]]:
        return self._query(self.events, vector, top_k)

    @staticmethod
    def _query(collection, vector: list[float], top_k: int) -> list[tuple[str, float]]:
        if collection.count() == 0:
            return []
        result = collection.query(query_embeddings=[vector], n_results=min(top_k, collection.count()))
        ids = result["ids"][0]
        distances = result["distances"][0]
        # cosine distance -> similarity in [0,1]
        return [(uid, max(0.0, 1.0 - dist)) for uid, dist in zip(ids, distances)]

    def reset(self) -> None:
        for name in (TEXT_COLLECTION, IMAGE_COLLECTION, EVENT_COLLECTION):
            self._client.delete_collection(name)
        self.text = self._client.get_or_create_collection(TEXT_COLLECTION, metadata={"hnsw:space": "cosine"})
        self.image = self._client.get_or_create_collection(IMAGE_COLLECTION, metadata={"hnsw:space": "cosine"})
        self.events = self._client.get_or_create_collection(EVENT_COLLECTION, metadata={"hnsw:space": "cosine"})
