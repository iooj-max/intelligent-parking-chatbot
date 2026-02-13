#!/usr/bin/env python3
"""
Manual multilingual testing script for agentic chatbot.

Tests:
1. English queries
2. Russian queries
3. Spanish queries
4. French queries
5. Tool calling behavior
6. Reservation switching
"""

import os
from langchain_core.messages import HumanMessage
from src.chatbot.graph import graph
from src.chatbot.state import ChatbotState


def test_query(query: str, language: str):
    """Test a single query and print result."""
    print(f"\n{'='*60}")
    print(f"Language: {language}")
    print(f"Query: {query}")
    print(f"{'='*60}")

    initial_state: ChatbotState = {
        "messages": [HumanMessage(content=query)],
        "mode": "info",
        "intent": None,
        "context": None,
        "reservation": {"completed_fields": [], "validation_errors": {}},
        "error": None,
        "iteration_count": 0,
    }

    try:
        result = graph.invoke(initial_state)

        # Print result
        print(f"\nMode: {result.get('mode')}")
        print(f"Iteration count: {result.get('iteration_count')}")

        # Print messages
        for msg in result.get("messages", []):
            if hasattr(msg, "content"):
                print(f"\n{msg.__class__.__name__}: {msg.content[:200]}...")

        # Check reservation state
        if result.get("mode") == "reservation":
            print(f"\n✅ RESERVATION MODE ACTIVATED")
            print(f"Parking ID: {result['reservation'].get('parking_id')}")
            print(f"Completed fields: {result['reservation'].get('completed_fields')}")

    except Exception as e:
        print(f"\n❌ Error: {e}")


def main():
    """Run multilingual tests."""
    print("🤖 Multilingual Agentic Chatbot Test")
    print("=" * 60)

    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set. Set it in .env file.")
        return

    # Test cases
    test_cases = [
        # English
        ("What are your hours?", "English"),
        ("Is downtown parking available?", "English"),
        ("I want to book parking", "English"),

        # Russian
        ("Какие у вас часы работы?", "Russian"),
        ("Есть ли места на парковке?", "Russian"),
        ("Забронировать парковку", "Russian"),

        # Spanish
        ("¿Cuáles son sus horarios?", "Spanish"),
        ("¿Cuánto cuesta?", "Spanish"),
        ("Quiero reservar estacionamiento", "Spanish"),

        # French
        ("Quels sont vos horaires?", "French"),
        ("Je veux réserver un parking", "French"),

        # Edge cases
        ("I don't want to book anything", "English - Negation"),
        ("Compare prices for downtown and airport", "English - Multi-tool"),
    ]

    for query, language in test_cases:
        test_query(query, language)
        print("\nPress Enter to continue...")
        input()

    print("\n✅ All tests completed!")


if __name__ == "__main__":
    main()
