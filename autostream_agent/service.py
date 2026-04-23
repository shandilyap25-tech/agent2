from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .graph import AutoStreamConversationGraph
from .llm import ResponseComposer
from .rag import LocalMarkdownRetriever


@dataclass(slots=True)
class ChatResult:
    session_id: str
    reply: str
    intent: str
    lead_info: dict[str, str]
    missing_fields: list[str]
    lead_captured: bool
    retrieved_docs: list[dict[str, str]]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class AutoStreamAgent:
    def __init__(self, knowledge_path: str | Path | None = None) -> None:
        retriever = LocalMarkdownRetriever(knowledge_path=knowledge_path)
        composer = ResponseComposer()
        self.graph = AutoStreamConversationGraph(retriever, composer)

    def chat(self, session_id: str, message: str) -> ChatResult:
        state = self.graph.invoke(session_id=session_id, user_text=message)
        return ChatResult(
            session_id=session_id,
            reply=self.graph.last_ai_message(state),
            intent=state.get("intent", "casual_greeting"),
            lead_info=dict(state.get("lead_info", {})),
            missing_fields=list(state.get("missing_fields", [])),
            lead_captured=bool(state.get("lead_captured", False)),
            retrieved_docs=list(state.get("retrieved_docs", [])),
        )

