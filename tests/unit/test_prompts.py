"""Tests for prompt template construction."""

from __future__ import annotations

from parking_agent import prompts


def test_scope_guardrail_prompt_contains_expected_variables() -> None:
    """Scope prompt exposes all runtime placeholders used by graph nodes."""
    prompt = prompts.scope_guardrail_prompt()
    variables = set(prompt.input_variables)
    assert {"user_input", "summary", "conversation"}.issubset(variables)


def test_reservation_confirmation_prompt_mentions_admin_waiting_notice() -> None:
    """Reservation confirmation prompt includes admin review requirement text."""
    prompt = prompts.reservation_confirmation_prompt()
    text = "\n".join(part.prompt.template for part in prompt.messages)
    assert "administrator" in text
    assert "confirmation can take some time" in text


def test_info_react_system_prompt_lists_supported_capabilities() -> None:
    """Info agent system prompt includes reservation and parking capabilities."""
    system_text = prompts.info_react_system_prompt()
    assert "Parking Information Agent" in system_text
    assert "Booking a parking reservation" in system_text
