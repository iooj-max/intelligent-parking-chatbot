#!/usr/bin/env python3
"""
CLI interface for parking chatbot.

Provides a simple REPL (Read-Eval-Print Loop) for interacting with the
LangGraph-based parking chatbot. Users can:
- Ask questions about parking facilities
- Make parking reservations
- Exit with 'exit', 'quit', or 'bye'

Usage:
    python -m src.main
"""

import logging

from langchain_core.messages import AIMessage, HumanMessage

from src.chatbot.graph import graph
from src.chatbot.state import ChatbotState

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def print_message(role: str, content: str):
    """
    Pretty print message with role indicator.

    Args:
        role: Message role (e.g., "User", "Assistant")
        content: Message content
    """
    separator = "=" * 60
    print(f"\n{separator}")
    print(f"{role.upper()}:")
    print(f"{content}")
    print(f"{separator}\n")


def main():
    """Main chatbot REPL loop."""
    # Print welcome message
    print("\n" + "=" * 60)
    print("🅿️  PARKING CHATBOT")
    print("=" * 60)
    print("\nWelcome! I can help you with:")
    print("  • Parking facility information")
    print("  • Real-time availability and pricing")
    print("  • Making parking reservations")
    print("\nType 'exit', 'quit', or 'bye' to quit")
    print("=" * 60 + "\n")

    # Initialize conversation state
    state: ChatbotState = {
        "messages": [],
        "mode": "info",
        "intent": None,
        "context": None,
        "reservation": {
            "completed_fields": [],
            "validation_errors": {},
        },
        "error": None,
        "iteration_count": 0,
    }

    # Main conversation loop
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()

            # Skip empty input
            if not user_input:
                continue

            # Check for exit commands
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("\nGoodbye! Thank you for using the Parking Chatbot. 👋\n")
                break

            # Add user message to state
            state["messages"].append(HumanMessage(content=user_input))

            # Invoke graph
            logger.info(f"Processing user input: {user_input[:50]}...")
            result = graph.invoke(state)

            # Extract AI response (last message in result)
            if result["messages"]:
                # Find last AI message
                last_ai_message = None
                for msg in reversed(result["messages"]):
                    if isinstance(msg, AIMessage):
                        last_ai_message = msg
                        break

                if last_ai_message:
                    print_message("Assistant", last_ai_message.content)
                else:
                    print("\n(No response generated)\n")

            # Update state for next iteration
            state = result

            # Show error if any (for debugging)
            if result.get("error"):
                logger.warning(f"Error in conversation: {result['error']}")
                # Error messages are already handled in nodes, so we don't print again

        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Goodbye! 👋\n")
            break

        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}", exc_info=True)
            print(f"\n❌ Error: {str(e)}\n")
            print("Please try again or type 'exit' to quit.\n")


if __name__ == "__main__":
    main()
