"""Agent orchestration: facility validation and info ReAct agent."""

from __future__ import annotations

import logging
from typing import Any, cast

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphRecursionError
from langchain_openai import ChatOpenAI

from parking_agent.clients import build_postgres_uri
from parking_agent.prompts import info_react_system_prompt, recursion_limit_fallback_prompt
from parking_agent.utils import message_content_to_text
from src.config import settings


def _safe_tool_error_message() -> str:
    return "The requested data is currently unavailable."


_RECURSION_LIMIT_MESSAGE = (
    "The requested data could not be found. Please try again with a different query."
)


def _fallback_message_in_user_language(user_message: str) -> str:
    """Generate recursion-limit fallback message in the user's language."""
    if not (user_message or str(user_message).strip()):
        return _RECURSION_LIMIT_MESSAGE
    try:
        llm = ChatOpenAI(model=settings.parking_agent_model)
        prompt = recursion_limit_fallback_prompt()
        response = llm.invoke(
            prompt.format_messages(user_message=(user_message or "")[:500])
        )
        text = (response.content or "").strip() if response else ""
        return text or _RECURSION_LIMIT_MESSAGE
    except Exception:
        return _RECURSION_LIMIT_MESSAGE


def _extract_clarifying_question_from_tool_call(tool_call: Any) -> str | None:
    """Extract question from ask_clarifying_question tool call."""
    if not isinstance(tool_call, dict):
        if hasattr(tool_call, "model_dump"):
            tool_call = tool_call.model_dump()
        else:
            return None
    if str(tool_call.get("name", "")).strip() != "ask_clarifying_question":
        return None
    args = tool_call.get("args") or {}
    if not isinstance(args, dict):
        return None
    q = args.get("question")
    if q is None:
        return None
    text = (q if isinstance(q, str) else str(q)).strip()
    return text if text else None


def _extract_final_agent_text(agent_output: Any) -> str:
    """Extract the final text or clarifying question from agent output."""
    if not isinstance(agent_output, dict):
        return ""
    messages = agent_output.get("messages")
    if not isinstance(messages, list):
        return ""

    for message in reversed(messages):
        if isinstance(message, AIMessage):
            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls:
                for tc in tool_calls:
                    q = _extract_clarifying_question_from_tool_call(tc)
                    if q:
                        return q
                continue
            text = message_content_to_text(message.content)
            if text:
                return text
        if isinstance(message, BaseMessage):
            text = message_content_to_text(message.content)
            if text:
                return text
        if isinstance(message, dict):
            role = str(message.get("role", "")).strip().lower()
            if role not in {"assistant", "ai"}:
                continue
            tool_calls = message.get("tool_calls") or []
            for tc in tool_calls:
                q = _extract_clarifying_question_from_tool_call(tc)
                if q:
                    return q
            text = message_content_to_text(message.get("content", ""))
            if text:
                return text
    return ""


def run_info_react_agent(
    user_input: Any,
    conversation_summary: Any,
    *,
    config: RunnableConfig | None = None,
) -> str:
    """Run a unified ReAct agent with SQL and static retrieval tools."""
    try:
        from langchain_community.agent_toolkits import (  # pyright: ignore[reportMissingImports]
            SQLDatabaseToolkit,
        )
        from langchain_community.utilities import SQLDatabase  # pyright: ignore[reportMissingImports]
        from langchain.agents import create_agent

        from parking_agent.tools import (
            ask_clarifying_question,
            retrieve_static_parking_info,
        )
    except Exception:
        logging.getLogger(__name__).exception("Failed to initialize info ReAct agent")
        return _safe_tool_error_message()

    model_name = settings.parking_agent_model
    prompt_input = message_content_to_text(user_input)
    summary = message_content_to_text(conversation_summary)
    if summary:
        prompt_input = (
            f"Latest user request:\n{prompt_input}\n\n"
            f"Conversation summary (context):\n{summary}"
        )

    db = None
    try:
        db = SQLDatabase.from_uri(
            build_postgres_uri(),
            include_tables=[
                "parking_facilities",
                "space_availability",
                "pricing_rules",
                "special_hours",
                "working_hours",
            ],
        )
        model = ChatOpenAI(model=model_name, max_retries=0)
        toolkit = SQLDatabaseToolkit(db=db, llm=model)
        info_tools = [
            ask_clarifying_question,
            retrieve_static_parking_info,
            *toolkit.get_tools(),
        ]
        info_agent = create_agent(
            model=model,
            tools=info_tools,
            system_prompt=info_react_system_prompt(),
            name="parking_info_react_agent",
        )

        invoke_config = dict(config or {})
        tags = list(cast(list, invoke_config.get("tags", [])))
        tags.append("info_react_agent")
        invoke_config["tags"] = tags
        invoke_config.setdefault("run_name", "info_react_agent_invoke")

        try:
            agent_output = info_agent.invoke(
                {"messages": [{"role": "user", "content": prompt_input}]},
                config=cast(RunnableConfig, invoke_config),
            )
            final_text = _extract_final_agent_text(agent_output)
            return final_text or _safe_tool_error_message()
        except GraphRecursionError:
            return _fallback_message_in_user_language(prompt_input)
    except Exception:
        logging.getLogger(__name__).exception("run_info_react_agent failed")
        return _safe_tool_error_message()
    finally:
        if db is not None and hasattr(db, "_engine"):
            db._engine.dispose()
