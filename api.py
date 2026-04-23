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


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    result = agent.chat(session_id=request.session_id, message=request.message)
    return ChatResponse(**result.to_dict())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

