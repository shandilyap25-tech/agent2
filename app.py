from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from autostream_agent.service import AutoStreamAgent

app = FastAPI(title="AutoStream Lead Agent", version="1.0.0")
agent = AutoStreamAgent()


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Stable session ID for conversation memory.")
    message: str = Field(..., description="Latest user message.")


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    intent: str
    lead_info: dict[str, str]
    missing_fields: list[str]
    lead_captured: bool
    retrieved_docs: list[dict[str, str]]


@app.get("/")
def read_root() -> dict[str, str]:
    return {
        "name": "AutoStream Lead Agent",
        "status": "ok",
        "docs": "/docs",
    }


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    result = agent.chat(session_id=request.session_id, message=request.message)
    return ChatResponse(**result.to_dict())


def main() -> None:
    session_id = "cli-session"

    print("AutoStream conversational agent")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Session ended.")
            break

        result = agent.chat(session_id=session_id, message=user_input)
        print(f"Agent: {result.reply}")
        print(
            f"[intent={result.intent} missing={result.missing_fields or '-'} lead_captured={result.lead_captured}]"
        )
        print()


if __name__ == "__main__":
    main()
