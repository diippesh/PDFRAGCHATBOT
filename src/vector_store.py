"""
Vector store layer.

Wraps FAISS + HuggingFace embeddings behind a small interface so the rest of
the app never touches LangChain's vector store API directly. Supports both
ephemeral (in-memory, per-session) and persisted (on-disk) indices.
"""

from __future__ import annotations

import os
import shutil
from functools import lru_cache

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

from src.config import Settings
from src.exceptions import VectorStoreNotReadyError
from src.logging_config import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _load_embeddings(model_name: str, device: str) -> HuggingFaceEmbeddings:
    """Embedding models are expensive to load — cache one per process."""
    logger.info("Loading embedding model '%s' on %s", model_name, device)
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def _load_reranker(model_name: str, device: str) -> CrossEncoder:
    """Load the cross-encoder only when a query is actually made."""
    logger.info("Loading reranker model '%s' on %s", model_name, device)
    return CrossEncoder(model_name, device=device)


class RerankingRetriever:
    """Retrieve candidates with FAISS, then order them with a cross-encoder."""

    def __init__(
        self,
        retriever: VectorStoreRetriever,
        model_name: str,
        device: str,
        top_k: int,
    ):
        self._retriever = retriever
        self._model_name = model_name
        self._device = device
        self._top_k = top_k

    def invoke(self, query: str) -> list[Document]:
        candidates = self._retriever.invoke(query)
        if len(candidates) <= 1:
            return candidates

        model = _load_reranker(self._model_name, self._device)
        scores = model.predict([(query, doc.page_content) for doc in candidates])
        ranked = sorted(
            zip(scores, candidates, strict=True),
            key=lambda item: float(item[0]),
            reverse=True,
        )
        return [doc for _, doc in ranked[: self._top_k]]


class VectorStoreManager:
    """Builds, persists, and queries a FAISS vector store."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._embeddings = _load_embeddings(settings.embedding_model, settings.embedding_device)
        self._store: FAISS | None = None

    @property
    def is_ready(self) -> bool:
        return self._store is not None

    def build(self, chunks: list[Document]) -> None:
        """Build a fresh in-memory index from document chunks."""
        logger.info("Building FAISS index from %d chunk(s)", len(chunks))
        self._store = FAISS.from_documents(chunks, self._embeddings)

    def add(self, chunks: list[Document]) -> None:
        """Add chunks to an existing index, or build one if none exists."""
        if self._store is None:
            self.build(chunks)
        else:
            self._store.add_documents(chunks)

    def persist(self, session_id: str) -> str:
        """Save the current index to disk, namespaced by session/collection id."""
        if self._store is None:
            raise VectorStoreNotReadyError("Cannot persist an index that hasn't been built.")
        path = os.path.join(self._settings.vector_store_dir, session_id)
        os.makedirs(path, exist_ok=True)
        self._store.save_local(path)
        logger.info("Persisted vector store to %s", path)
        return path

    def load(self, session_id: str) -> bool:
        """Load a previously persisted index. Returns False if none exists."""
        path = os.path.join(self._settings.vector_store_dir, session_id)
        if not os.path.isdir(path):
            return False
        self._store = FAISS.load_local(path, self._embeddings, allow_dangerous_deserialization=True)
        logger.info("Loaded vector store from %s", path)
        return True

    def delete(self, session_id: str) -> None:
        path = os.path.join(self._settings.vector_store_dir, session_id)
        if os.path.isdir(path):
            shutil.rmtree(path)
            logger.info("Deleted persisted vector store at %s", path)

    def as_retriever(self) -> VectorStoreRetriever | RerankingRetriever:
        if self._store is None:
            raise VectorStoreNotReadyError(
                "No documents have been indexed yet. Upload and process a document first."
            )
        base_retriever = self._store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": (
                    self._settings.reranker_candidate_k
                    if self._settings.enable_reranking
                    else self._settings.retriever_top_k
                )
            },
        )
        if not self._settings.enable_reranking:
            return base_retriever
        return RerankingRetriever(
            base_retriever,
            self._settings.reranker_model,
            self._settings.embedding_device,
            self._settings.retriever_top_k,
        )
