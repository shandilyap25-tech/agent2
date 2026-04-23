from __future__ import annotations

from autostream_agent.service import AutoStreamAgent


def main() -> None:
    agent = AutoStreamAgent()
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

