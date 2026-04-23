from __future__ import annotations

import re
from pathlib import Path

from langchain_core.documents import Document

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "about",
    "can",
    "do",
    "for",
    "i",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "the",
    "to",
    "what",
    "you",
    "your",
}


class LocalMarkdownRetriever:
    def __init__(self, knowledge_path: str | Path | None = None) -> None:
        default_path = Path(__file__).resolve().parent.parent / "data" / "knowledge_base.md"
        self.knowledge_path = Path(knowledge_path or default_path)
        self.documents = self._load_documents(self.knowledge_path)

    def retrieve(self, query: str, top_k: int = 2) -> list[Document]:
        query_tokens = self._tokenize(query)
        scored: list[tuple[int, Document]] = []
        for document in self.documents:
            score = self._score_document(query_tokens, query.lower(), document)
            if score > 0:
                scored.append((score, document))

        if not scored:
            return [self.documents[0]]

        scored.sort(key=lambda item: item[0], reverse=True)
        return [document for _, document in scored[:top_k]]

    def _load_documents(self, path: Path) -> list[Document]:
        lines = path.read_text(encoding="utf-8").splitlines()
        current_title = "AutoStream Overview"
        current_lines: list[str] = []
        documents: list[Document] = []

        def flush() -> None:
            if not current_lines:
                return
            content = " ".join(line.strip() for line in current_lines if line.strip())
            content = re.sub(r"\s+", " ", content).strip()
            documents.append(
                Document(
                    page_content=content,
                    metadata={"title": current_title},
                )
            )

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("# "):
                continue
            if line.startswith("## "):
                flush()
                current_title = line[3:].strip()
                current_lines = []
                continue
            current_lines.append(line.lstrip("- ").strip())

        flush()
        return documents

    def _score_document(
        self,
        query_tokens: set[str],
        query_lower: str,
        document: Document,
    ) -> int:
        title = str(document.metadata.get("title", "")).lower()
        content = document.page_content.lower()
        document_tokens = self._tokenize(f"{title} {content}")
        overlap = len(query_tokens & document_tokens)

        if "pricing" in query_lower or "price" in query_lower or "plan" in query_lower:
            if "plan" in title:
                overlap += 4

        if "refund" in query_lower or "support" in query_lower or "policy" in query_lower:
            if "policies" in title:
                overlap += 5

        if "feature" in query_lower or "autostream" in query_lower:
            if "overview" in title:
                overlap += 2

        if "pro" in query_lower and "pro" in title:
            overlap += 4

        if "basic" in query_lower and "basic" in title:
            overlap += 4

        return overlap

    def _tokenize(self, text: str) -> set[str]:
        tokens = re.findall(r"[A-Za-z0-9']+", text.lower())
        return {token for token in tokens if token not in STOPWORDS}


def serialize_documents(documents: list[Document]) -> list[dict[str, str]]:
    return [
        {
            "title": str(document.metadata.get("title", "")),
            "content": document.page_content,
        }
        for document in documents
    ]

