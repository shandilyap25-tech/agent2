from __future__ import annotations

import os
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


class ResponseComposer:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or os.getenv("AUTOSTREAM_MODEL", "gpt-4o-mini")
        self.client = self._build_client()

    def compose(
        self,
        *,
        user_text: str,
        intent: str,
        retrieved_docs: list[dict[str, str]],
        lead_info: dict[str, str],
        missing_fields: list[str],
        tool_output: str | None,
    ) -> str:
        if self.client is not None:
            try:
                return self._compose_with_openai(
                    user_text=user_text,
                    intent=intent,
                    retrieved_docs=retrieved_docs,
                    lead_info=lead_info,
                    missing_fields=missing_fields,
                    tool_output=tool_output,
                )
            except Exception:
                pass

        return self._compose_offline(
            user_text=user_text,
            intent=intent,
            retrieved_docs=retrieved_docs,
            lead_info=lead_info,
            missing_fields=missing_fields,
            tool_output=tool_output,
        )

    def _build_client(self) -> ChatOpenAI | None:
        if not os.getenv("OPENAI_API_KEY"):
            return None
        return ChatOpenAI(model=self.model_name, temperature=0.2)

    def _compose_with_openai(
        self,
        *,
        user_text: str,
        intent: str,
        retrieved_docs: list[dict[str, str]],
        lead_info: dict[str, str],
        missing_fields: list[str],
        tool_output: str | None,
    ) -> str:
        context = "\n".join(
            f"- {doc['title']}: {doc['content']}" for doc in retrieved_docs
        )
        system_prompt = (
            "You are AutoStream's sales assistant. Use only the provided context. "
            "Be concise, accurate, and natural. If lead capture succeeded, confirm it. "
            "If details are still missing, ask only for the missing fields."
        )
        user_prompt = (
            f"Intent: {intent}\n"
            f"Latest user message: {user_text}\n"
            f"Retrieved context:\n{context or '- None'}\n"
            f"Lead info: {lead_info}\n"
            f"Missing fields: {missing_fields}\n"
            f"Tool output: {tool_output or 'None'}"
        )
        response = self.client.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
        return str(response.content).strip()

    def _compose_offline(
        self,
        *,
        user_text: str,
        intent: str,
        retrieved_docs: list[dict[str, str]],
        lead_info: dict[str, str],
        missing_fields: list[str],
        tool_output: str | None,
    ) -> str:
        parts: list[str] = []

        context_summary = self._summarize_docs(retrieved_docs)
        if context_summary:
            parts.append(context_summary)

        if tool_output:
            name = lead_info.get("name", "there")
            email = lead_info.get("email", "")
            platform = lead_info.get("platform", "")
            parts.append(
                f"Thanks {name}. I've captured your lead successfully with {email} for your {platform} workflow. "
                "Our team can follow up with next steps."
            )
            return " ".join(parts).strip()

        if missing_fields:
            if intent == "high_intent_lead" and not parts:
                parts.append("Happy to help you get started with AutoStream.")
            parts.append(self._lead_prompt(missing_fields))
            return " ".join(parts).strip()

        if intent == "casual_greeting":
            return (
                "Hi! I can help with AutoStream pricing, features, refund policy, or getting you started on a plan."
            )

        if parts:
            return " ".join(parts).strip()

        return (
            "I can help with AutoStream pricing, features, refund policy, or capturing your details if you'd like to sign up."
        )

    def _summarize_docs(self, retrieved_docs: list[dict[str, str]]) -> str:
        if not retrieved_docs:
            return ""
        return " ".join(doc["content"] for doc in retrieved_docs[:2]).strip()

    def _lead_prompt(self, missing_fields: list[str]) -> str:
        if missing_fields == ["name"]:
            return "What name should I use for the lead?"
        if missing_fields == ["email"]:
            return "What is the best email address to use?"
        if missing_fields == ["platform"]:
            return "Which creator platform do you use, such as YouTube or Instagram?"

        field_labels = {
            "name": "name",
            "email": "email address",
            "platform": "creator platform",
        }
        requested = ", ".join(field_labels[field] for field in missing_fields[:-1])
        if len(missing_fields) > 1:
            requested += f" and {field_labels[missing_fields[-1]]}"
        return f"To capture your lead, please share your {requested}."

