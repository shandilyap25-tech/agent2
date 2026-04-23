from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
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


HTML_UI = """
<!DOCTYPE html>
<html>
<head>
    <title>AutoStream Agent</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: 'Inter', sans-serif; background: #0f172a; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }
        #chat-container { background: #1e293b; width: 100%; max-width: 450px; height: 80vh; border-radius: 16px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); display: flex; flex-direction: column; overflow: hidden; border: 1px solid #334155; }
        #header { background: #2563eb; color: white; padding: 18px; text-align: center; font-weight: 600; font-size: 18px; letter-spacing: 0.5px; }
        #messages { flex: 1; padding: 20px; overflow-y: auto; display: flex; flex-direction: column; gap: 12px; }
        .message { max-width: 85%; padding: 12px 16px; border-radius: 12px; font-size: 15px; line-height: 1.5; color: #f8fafc; }
        .user-message { background: #3b82f6; align-self: flex-end; border-bottom-right-radius: 4px; }
        .agent-message { background: #334155; align-self: flex-start; border-bottom-left-radius: 4px; border: 1px solid #475569; }
        #input-container { display: flex; padding: 15px; border-top: 1px solid #334155; background: #1e293b; gap: 10px; }
        #user-input { flex: 1; padding: 12px; border: 1px solid #475569; border-radius: 8px; outline: none; background: #0f172a; color: white; font-size: 15px; }
        #user-input::placeholder { color: #94a3b8; }
        #send-btn { background: #2563eb; color: white; border: none; padding: 12px 20px; border-radius: 8px; cursor: pointer; font-weight: 600; transition: background 0.2s; }
        #send-btn:hover { background: #1d4ed8; }
        .metadata { font-size: 11px; color: #94a3b8; margin-top: 6px; font-family: monospace; }
        .highlight { color: #10b981; font-weight: bold; }
    </style>
</head>
<body>
    <div id="chat-container">
        <div id="header">AutoStream AI Agent</div>
        <div id="messages">
            <div class="message agent-message">Hi! I can help you with AutoStream pricing, features, or sign you up for a plan.</div>
        </div>
        <div id="input-container">
            <input type="text" id="user-input" placeholder="Type your message here..." onkeypress="handleKeyPress(event)" autocomplete="off" />
            <button id="send-btn" onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script>
        const sessionId = "demo-" + Math.random().toString(36).substring(7);
        const messagesDiv = document.getElementById("messages");
        const inputField = document.getElementById("user-input");

        function addMessage(text, sender, meta="") {
            const msgDiv = document.createElement("div");
            msgDiv.className = "message " + (sender === "user" ? "user-message" : "agent-message");
            msgDiv.innerText = text;
            if(meta) {
                const metaDiv = document.createElement("div");
                metaDiv.className = "metadata";
                metaDiv.innerHTML = meta;
                msgDiv.appendChild(metaDiv);
            }
            messagesDiv.appendChild(msgDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        async function sendMessage() {
            const text = inputField.value.trim();
            if (!text) return;
            
            addMessage(text, "user");
            inputField.value = "";
            inputField.disabled = true;

            try {
                const response = await fetch("/chat", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ session_id: sessionId, message: text })
                });
                const data = await response.json();
                
                let metaText = `intent: ${data.intent}`;
                if(data.lead_captured) metaText += " | <span class='highlight'>[LEAD CAPTURED]</span>";
                else if(data.missing_fields && data.missing_fields.length > 0) metaText += ` | missing: ${data.missing_fields.join(", ")}`;
                
                addMessage(data.reply, "agent", metaText);
            } catch (err) {
                addMessage("Error connecting to server.", "agent");
            }
            
            inputField.disabled = false;
            inputField.focus();
        }

        function handleKeyPress(e) {
            if (e.key === "Enter") sendMessage();
        }
    </script>
</body>
</html>
"""

@app.get("/")
def read_root() -> HTMLResponse:
    return HTMLResponse(content=HTML_UI)


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
