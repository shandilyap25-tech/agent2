from __future__ import annotations

from typing import Annotated

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Literal, TypedDict

from .intents import (
    INTENT_HIGH,
    contains_product_keywords,
    detect_intent,
    extract_lead_details,
    get_missing_lead_fields,
)
from .llm import ResponseComposer
from .rag import LocalMarkdownRetriever, serialize_documents
from .tools import mock_lead_capture


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    intent: str
    lead_info: dict[str, str]
    missing_fields: list[str]
    collecting_lead: bool
    ready_for_tool: bool
    lead_captured: bool
    needs_retrieval: bool
    needs_lead_workflow: bool
    retrieved_docs: list[dict[str, str]]
    tool_output: str | None
    response: str


class AutoStreamConversationGraph:
    def __init__(
        self,
        retriever: LocalMarkdownRetriever,
        composer: ResponseComposer,
    ) -> None:
        self.retriever = retriever
        self.composer = composer
        self.graph = self._build_graph()

    def invoke(self, session_id: str, user_text: str) -> AgentState:
        config = {"configurable": {"thread_id": session_id}}
        return self.graph.invoke(
            {"messages": [HumanMessage(content=user_text)]},
            config=config,
        )

    @staticmethod
    def last_ai_message(state: AgentState) -> str:
        for message in reversed(state.get("messages", [])):
            if isinstance(message, AIMessage):
                return str(message.content)
        return ""

    def _build_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("analyze_intent", self._analyze_intent)
        builder.add_node("retrieve_knowledge", self._retrieve_knowledge)
        builder.add_node("lead_workflow", self._lead_workflow)
        builder.add_node("capture_lead", self._capture_lead)
        builder.add_node("respond", self._respond)

        builder.add_edge(START, "analyze_intent")
        builder.add_conditional_edges(
            "analyze_intent",
            self._route_after_analyze,
            {
                "retrieve_knowledge": "retrieve_knowledge",
                "lead_workflow": "lead_workflow",
                "respond": "respond",
            },
        )
        builder.add_conditional_edges(
            "retrieve_knowledge",
            self._route_after_retrieve,
            {
                "lead_workflow": "lead_workflow",
                "respond": "respond",
            },
        )
        builder.add_conditional_edges(
            "lead_workflow",
            self._route_after_lead_workflow,
            {
                "capture_lead": "capture_lead",
                "respond": "respond",
            },
        )
        builder.add_edge("capture_lead", "respond")
        builder.add_edge("respond", END)

        return builder.compile(checkpointer=MemorySaver())

    def _analyze_intent(self, state: AgentState) -> AgentState:
        user_text = self._last_human_message(state)
        existing_lead = dict(state.get("lead_info", {}))
        collecting_lead = bool(state.get("collecting_lead", False)) and not bool(
            state.get("lead_captured", False)
        )

        lead_info = extract_lead_details(user_text, existing_lead)
        intent = detect_intent(user_text, collecting_lead=collecting_lead)

        if intent == INTENT_HIGH:
            collecting_lead = True

        missing_fields = get_missing_lead_fields(lead_info) if collecting_lead else []
        ready_for_tool = collecting_lead and not missing_fields
        needs_retrieval = self._needs_retrieval(user_text, intent)
        needs_lead_workflow = collecting_lead and not bool(state.get("lead_captured", False))

        return {
            "intent": intent,
            "lead_info": lead_info,
            "missing_fields": missing_fields,
            "collecting_lead": collecting_lead,
            "ready_for_tool": ready_for_tool,
            "needs_retrieval": needs_retrieval,
            "needs_lead_workflow": needs_lead_workflow,
            "retrieved_docs": [],
            "tool_output": None,
            "response": "",
        }

    def _retrieve_knowledge(self, state: AgentState) -> AgentState:
        user_text = self._last_human_message(state)
        documents = self.retriever.retrieve(user_text)
        return {"retrieved_docs": serialize_documents(documents)}

    def _lead_workflow(self, state: AgentState) -> AgentState:
        lead_info = dict(state.get("lead_info", {}))
        missing_fields = get_missing_lead_fields(lead_info)
        return {
            "missing_fields": missing_fields,
            "ready_for_tool": not missing_fields,
        }

    def _capture_lead(self, state: AgentState) -> AgentState:
        lead_info = dict(state.get("lead_info", {}))
        tool_output = mock_lead_capture(
            lead_info["name"],
            lead_info["email"],
            lead_info["platform"],
        )
        return {
            "tool_output": tool_output,
            "lead_captured": True,
            "collecting_lead": False,
            "needs_lead_workflow": False,
            "ready_for_tool": False,
            "missing_fields": [],
        }

    def _respond(self, state: AgentState) -> AgentState:
        user_text = self._last_human_message(state)
        response = self.composer.compose(
            user_text=user_text,
            intent=state.get("intent", "casual_greeting"),
            retrieved_docs=state.get("retrieved_docs", []),
            lead_info=state.get("lead_info", {}),
            missing_fields=state.get("missing_fields", []),
            tool_output=state.get("tool_output"),
        )
        return {
            "response": response,
            "messages": [AIMessage(content=response)],
        }

    def _route_after_analyze(
        self, state: AgentState
    ) -> Literal["retrieve_knowledge", "lead_workflow", "respond"]:
        if state.get("needs_retrieval"):
            return "retrieve_knowledge"
        if state.get("needs_lead_workflow"):
            return "lead_workflow"
        return "respond"

    def _route_after_retrieve(
        self, state: AgentState
    ) -> Literal["lead_workflow", "respond"]:
        if state.get("needs_lead_workflow"):
            return "lead_workflow"
        return "respond"

    def _route_after_lead_workflow(
        self, state: AgentState
    ) -> Literal["capture_lead", "respond"]:
        if state.get("ready_for_tool"):
            return "capture_lead"
        return "respond"

    def _needs_retrieval(self, user_text: str, intent: str) -> bool:
        if contains_product_keywords(user_text):
            return True
        return intent != INTENT_HIGH and user_text.strip().endswith("?")

    @staticmethod
    def _last_human_message(state: AgentState) -> str:
        for message in reversed(state.get("messages", [])):
            if isinstance(message, HumanMessage):
                return str(message.content)
        return ""

