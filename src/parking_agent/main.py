"""Telegram bot entrypoint for the parking agent."""

from __future__ import annotations

import asyncio
import logging
import os
import warnings
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from typing import Any, cast
from uuid import uuid4

from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import Command
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update  # pyright: ignore[reportMissingImports]
from telegram.ext import Application  # pyright: ignore[reportMissingImports]
from telegram.ext import CallbackQueryHandler  # pyright: ignore[reportMissingImports]
from telegram.ext import CommandHandler  # pyright: ignore[reportMissingImports]
from telegram.ext import ContextTypes  # pyright: ignore[reportMissingImports]
from telegram.ext import MessageHandler  # pyright: ignore[reportMissingImports]
from telegram.ext import filters  # pyright: ignore[reportMissingImports]

from parking_agent.graph import (
    ExecutionState,
    RoutingState,
    build_execution_graph,
    build_routing_graph,
)
from parking_agent.chat_history_store import ChatHistoryStore
from parking_agent.message_reducer import RECENT_MESSAGES_TO_KEEP
from parking_agent.mcp_reservation_status import (
    append_reservation_status,
    get_latest_reservation_status,
    reservation_is_pending,
)
from parking_agent.prompts import reservation_already_pending_prompt
from parking_agent.schemas import ReservationData
from parking_agent.utils import message_content_to_text
from src.config import settings

_FALLBACK_REPLY = "The requested data is currently unavailable."


def _load_env() -> None:
    dotenv_path = find_dotenv(usecwd=True)
    if not dotenv_path:
        repo_root_env = Path(__file__).resolve().parents[2] / ".env"
        if repo_root_env.exists():
            dotenv_path = str(repo_root_env)
    if dotenv_path:
        load_dotenv(dotenv_path=dotenv_path, override=False)


def _configure_warnings() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"^Pydantic serializer warnings:.*",
        category=UserWarning,
        module=r"pydantic\.main",
    )


def _chat_id_from_update(update: Update) -> str:
    chat = update.effective_chat
    if chat is not None:
        return str(chat.id)
    user = update.effective_user
    if user is not None:
        return f"user-{user.id}"
    return f"session-{uuid4().hex}"


def _thread_id_for_intent(chat_id: str, intent: str) -> str:
    thread_suffix = "reservation" if intent == "reservation" else "info"
    return f"tg:{chat_id}:{thread_suffix}"


def _conversation_id_for_chat(chat_id: str) -> str:
    return f"tg:{chat_id}"


def _latest_assistant_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = message_content_to_text(message.content)
            if text:
                return text
        if isinstance(message, dict):
            role = str(message.get("role", "")).strip().lower()
            if role in {"assistant", "ai"}:
                text = message_content_to_text(message.get("content", ""))
                if text:
                    return text
        if isinstance(message, BaseMessage):
            text = message_content_to_text(message.content)
            if text:
                return text
    return ""


def _format_recent_messages(messages: list[BaseMessage], max_messages: int = 6) -> str:
    recent = messages[-max_messages:]
    lines = []
    for message in recent:
        role = getattr(message, "type", message.__class__.__name__.lower())
        text = message_content_to_text(getattr(message, "content", ""))
        lines.append(f"{role}: {text}")
    return "\n".join(lines)


def _blocked_reservation_response(shared_messages: list[BaseMessage], user_input: str) -> str:
    llm = ChatOpenAI(model=settings.parking_agent_model)
    prompt = reservation_already_pending_prompt()
    conversation = _format_recent_messages(shared_messages, max_messages=RECENT_MESSAGES_TO_KEEP)
    response = llm.invoke(
        prompt.format_messages(
            conversation=conversation,
            user_input=user_input,
        )
    )
    return message_content_to_text(response.content) or _FALLBACK_REPLY


def _invoke_routing_graph_for_text(
    routing_graph_app: Any,
    shared_messages: list[BaseMessage],
    shared_summary: str,
    user_input: str,
    thread_id: str,
    conversation_id: str,
) -> tuple[str | None, str | None, str]:
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id, "conversation_id": conversation_id},
        "recursion_limit": 20,
    }
    state_input = cast(
        RoutingState,
        {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *shared_messages,
                HumanMessage(content=user_input),
            ],
            "conversation_summary": shared_summary,
        },
    )
    result = routing_graph_app.invoke(state_input, config=config)
    messages = cast(list[Any], result.get("messages", []))
    scope_value = result.get("scope_decision")
    scope_decision = str(scope_value) if isinstance(scope_value, str) else None
    intent_value = result.get("intent")
    intent = str(intent_value) if isinstance(intent_value, str) else None
    response_text = _latest_assistant_text(messages) if scope_decision == "out_of_scope" else ""
    return scope_decision, intent, response_text


def _invoke_execution_graph_for_text(
    execution_graph_app: Any,
    shared_messages: list[BaseMessage],
    shared_summary: str,
    user_input: str,
    thread_id: str,
    conversation_id: str,
    intent: str,
) -> tuple[str, bool, dict[str, Any] | None, str | None, bool]:
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id, "conversation_id": conversation_id},
        "recursion_limit": 20,
    }
    state_input: ExecutionState = cast(
        ExecutionState,
        {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *shared_messages,
                HumanMessage(content=user_input),
            ],
            "intent": intent,
            "conversation_summary": shared_summary,
        },
    )
    result = execution_graph_app.invoke(state_input, config=config)
    messages = cast(list[Any], result.get("messages", []))
    interrupted = bool(result.get("__interrupt__"))
    awaiting_user_confirmation = bool(result.get("awaiting_user_confirmation"))
    reservation: dict[str, Any] | None = None
    updated_summary_value = result.get("conversation_summary")
    updated_summary = (
        str(updated_summary_value)
        if isinstance(updated_summary_value, str)
        else None
    )
    if interrupted:
        response_text = ""
        reservation = cast(dict[str, Any], result.get("reservation") or {})
    else:
        response_text = _latest_assistant_text(messages) or _FALLBACK_REPLY
    return (
        response_text,
        interrupted,
        reservation,
        updated_summary,
        awaiting_user_confirmation,
    )


async def _persist_chat_turn(
    history_store: ChatHistoryStore,
    chat_id: str,
    user_text: str,
    assistant_text: str,
    updated_summary: str | None = None,
) -> None:
    await asyncio.to_thread(history_store.append_user_message, chat_id, user_text)
    if assistant_text.strip():
        await asyncio.to_thread(history_store.append_ai_message, chat_id, assistant_text)
    if isinstance(updated_summary, str):
        await asyncio.to_thread(history_store.set_summary, chat_id, updated_summary)


def _format_admin_reservation_message(reservation: dict[str, Any]) -> str:
    """Format reservation details for admin review message in English."""
    lines = ["New parking reservation request – please review", ""]
    fields = [
        ("customer_name", "Customer"),
        ("facility", "Facility"),
        ("date", "Date"),
        ("start_time", "Start time"),
        ("duration_hours", "Duration (hours)"),
        ("vehicle_plate", "Vehicle plate"),
    ]
    for key, label in fields:
        value = reservation.get(key)
        if value is not None and str(value).strip():
            lines.append(f"{label}: {value}")
    lines.append("")
    lines.append("Please review and approve or reject using the buttons below.")
    return "\n".join(lines)


def _chat_id_from_reservation_thread_id(thread_id: str) -> str | None:
    if not thread_id.startswith("tg:") or not thread_id.endswith(":reservation"):
        return None
    parts = thread_id.split(":")
    if len(parts) < 3:
        return None
    return ":".join(parts[1:-1])


def _resume_reservation_thread_with_admin_decision(
    graph_app: Any, thread_id: str, conversation_id: str, admin_decision: str
) -> str:
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id, "conversation_id": conversation_id},
    }
    result = graph_app.invoke(Command(resume=admin_decision), config=config)
    messages = cast(list[Any], result.get("messages", []))
    return _latest_assistant_text(messages) or _FALLBACK_REPLY


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    admin_chat_id = settings.telegram_admin_chat_id.strip()
    chat_id = _chat_id_from_update(update)
    is_admin = bool(admin_chat_id) and chat_id == admin_chat_id
    if is_admin:
        text = (
            "Admin panel. You will receive parking reservation requests for approval. "
            "Use Approve/Reject buttons to process them."
        )
    else:
        text = "Hello! I can help with parking facility information and making parking reservations."
    await update.message.reply_text(text)


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    user_text = (update.message.text or "").strip()
    if not user_text:
        await update.message.reply_text("Please send a non-empty text message.")
        return

    admin_chat_id = settings.telegram_admin_chat_id.strip()
    chat_id = _chat_id_from_update(update)
    if admin_chat_id and chat_id == admin_chat_id:
        await update.message.reply_text("Please wait for a user request.")
        return

    routing_graph_app = context.application.bot_data.get("routing_graph_app")
    execution_graph_app = context.application.bot_data.get("execution_graph_app")
    history_store = context.application.bot_data.get("chat_history_store")
    if routing_graph_app is None or execution_graph_app is None or history_store is None:
        await update.message.reply_text(_FALLBACK_REPLY)
        return

    try:
        conversation_id = _conversation_id_for_chat(chat_id)
        info_thread_id = _thread_id_for_intent(chat_id=chat_id, intent="info")
        reservation_thread_id = _thread_id_for_intent(chat_id=chat_id, intent="reservation")
        chat_thread_modes = cast(
            dict[str, str], context.application.bot_data.setdefault("chat_thread_modes", {})
        )
        pending_threads = cast(
            dict[str, str],
            context.application.bot_data.setdefault("pending_reservation_threads", {}),
        )
        if await reservation_is_pending(reservation_thread_id):
            pending_threads[chat_id] = reservation_thread_id
        else:
            pending_threads.pop(chat_id, None)
        has_pending_reservation = chat_id in pending_threads
        shared_messages = await asyncio.to_thread(
            history_store.get_recent_messages,
            chat_id,
            RECENT_MESSAGES_TO_KEEP,
        )
        shared_summary = await asyncio.to_thread(history_store.get_summary, chat_id)
        routing_thread_id = info_thread_id

        scope_decision, intent, out_of_scope_text = (
            await asyncio.to_thread(
                _invoke_routing_graph_for_text,
                routing_graph_app,
                shared_messages,
                shared_summary,
                user_text,
                routing_thread_id,
                conversation_id,
            )
        )
        if scope_decision == "out_of_scope":
            await _persist_chat_turn(
                history_store,
                chat_id,
                user_text,
                out_of_scope_text or _FALLBACK_REPLY,
            )
            await update.message.reply_text(out_of_scope_text or _FALLBACK_REPLY)
            return

        allowed_intents = {"info_retrieval", "reservation"}
        if intent not in allowed_intents:
            logging.getLogger(__name__).warning(
                "Invalid intent '%s' received from routing graph for chat_id=%s.",
                intent,
                chat_id,
            )
            await _persist_chat_turn(
                history_store,
                chat_id,
                user_text,
                _FALLBACK_REPLY,
            )
            await update.message.reply_text(_FALLBACK_REPLY)
            return

        response_text = _FALLBACK_REPLY
        should_create_pending_file = False
        reservation: dict[str, Any] | None = None
        updated_summary: str | None = None
        awaiting_user_confirmation = False

        if intent == "reservation":
            if has_pending_reservation:
                response_text = await asyncio.to_thread(
                    _blocked_reservation_response,
                    shared_messages,
                    user_text,
                )
                chat_thread_modes[chat_id] = "info"
            else:
                (
                    response_text,
                    should_create_pending_file,
                    reservation,
                    updated_summary,
                    awaiting_user_confirmation,
                ) = (
                    await asyncio.to_thread(
                        _invoke_execution_graph_for_text,
                        execution_graph_app,
                        shared_messages,
                        shared_summary,
                        user_text,
                        reservation_thread_id,
                        conversation_id,
                        "reservation",
                    )
                )
                chat_thread_modes[chat_id] = "reservation"
        else:
            (
                response_text,
                should_create_pending_file,
                reservation,
                updated_summary,
                awaiting_user_confirmation,
            ) = (
                await asyncio.to_thread(
                    _invoke_execution_graph_for_text,
                    execution_graph_app,
                    shared_messages,
                    shared_summary,
                    user_text,
                    info_thread_id,
                    conversation_id,
                    "info_retrieval",
                )
            )
            chat_thread_modes[chat_id] = "info"

        await _persist_chat_turn(
            history_store,
            chat_id,
            user_text,
            response_text,
            updated_summary=updated_summary,
        )

        if should_create_pending_file:
            if chat_id not in pending_threads:
                pending_threads[chat_id] = reservation_thread_id
                await append_reservation_status(
                    thread_id=reservation_thread_id,
                    status="pending",
                    reservation=cast(ReservationData, reservation or {}),
                )
            chat_thread_modes[chat_id] = "info"
            admin_chat_id = settings.telegram_admin_chat_id.strip()
            if admin_chat_id and reservation:
                admin_text = _format_admin_reservation_message(reservation)
                keyboard = InlineKeyboardMarkup.from_row(
                    [
                        InlineKeyboardButton(
                            "Approve",
                            callback_data=f"approve:{reservation_thread_id}",
                        ),
                        InlineKeyboardButton(
                            "Reject",
                            callback_data=f"reject:{reservation_thread_id}",
                        ),
                    ]
                )
                try:
                    await context.application.bot.send_message(
                        chat_id=int(admin_chat_id),
                        text=admin_text,
                        reply_markup=keyboard, # approve / reject buttons
                    )
                except Exception:
                    logging.getLogger(__name__).exception(
                        "Failed to send admin notification for reservation"
                    )
        if response_text:
            sent_message = await update.message.reply_text(response_text)
            if intent == "reservation" and awaiting_user_confirmation:
                confirmation_message_ids = cast(
                    dict[str, int],
                    context.application.bot_data.setdefault(
                        "reservation_confirmation_message_ids", {}
                    ),
                )
                confirmation_message_ids[reservation_thread_id] = sent_message.message_id
    except Exception:
        logging.getLogger(__name__).exception("Telegram message handling failed")
        await update.message.reply_text(_FALLBACK_REPLY)


async def handle_admin_callback(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle Approve/Reject button clicks from admin. Resume graph and notify user."""
    if update.callback_query is None:
        return
    callback = update.callback_query
    admin_chat_id = settings.telegram_admin_chat_id.strip()
    if not admin_chat_id:
        await callback.answer("Admin notification is not configured.")
        return
    if str(callback.from_user.id) != admin_chat_id:
        await callback.answer("Only the administrator can use these buttons.")
        return
    data = callback.data or ""
    if not data.startswith("approve:") and not data.startswith("reject:"):
        await callback.answer("Invalid callback data.")
        return
    action, _, thread_id = data.partition(":")
    if not thread_id or not thread_id.startswith("tg:") or not thread_id.endswith(":reservation"):
        await callback.answer("Invalid thread ID.")
        return
    latest_status = await get_latest_reservation_status(thread_id)
    if latest_status is None:
        await callback.answer("Reservation already processed or not found.")
        return
    if latest_status != "pending":
        await callback.answer("Reservation already processed.")
        return
    new_status = "approved" if action == "approve" else "rejected"
    await callback.answer(f"{new_status.capitalize()}")
    callback_message = callback.message
    admin_decision_text = new_status.capitalize()
    try:
        await callback.edit_message_reply_markup(reply_markup=None)
    except Exception:
        pass
    if callback_message is not None:
        try:
            await context.application.bot.send_message(
                chat_id=int(admin_chat_id),
                text=admin_decision_text,
                reply_to_message_id=callback_message.message_id,
            )
        except Exception:
            logging.getLogger(__name__).warning(
                "Failed to send admin decision history reply for thread %s.",
                thread_id,
            )

    chat_id = _chat_id_from_reservation_thread_id(thread_id)
    if chat_id is None:
        return
    execution_graph_app = context.application.bot_data.get("execution_graph_app")
    history_store = context.application.bot_data.get("chat_history_store")
    if execution_graph_app is None or history_store is None:
        return
    try:
        conversation_id = _conversation_id_for_chat(chat_id)
        response_text = await asyncio.to_thread(
            _resume_reservation_thread_with_admin_decision,
            execution_graph_app,
            thread_id,
            conversation_id,
            new_status,
        )
        confirmation_message_ids = cast(
            dict[str, int],
            context.application.bot_data.setdefault(
                "reservation_confirmation_message_ids", {}
            ),
        )
        reply_to_message_id = confirmation_message_ids.get(thread_id)
        try:
            await context.application.bot.send_message(
                chat_id=int(chat_id),
                text=response_text,
                reply_to_message_id=reply_to_message_id,
            )
        except Exception:
            if reply_to_message_id is None:
                raise
            logging.getLogger(__name__).warning(
                "Failed to send admin decision as reply for thread %s. Sending plain message.",
                thread_id,
            )
            await context.application.bot.send_message(
                chat_id=int(chat_id),
                text=response_text,
            )
        await asyncio.to_thread(history_store.append_ai_message, chat_id, response_text)
        chat_thread_modes = cast(
            dict[str, str],
            context.application.bot_data.setdefault("chat_thread_modes", {}),
        )
        pending_threads = cast(
            dict[str, str],
            context.application.bot_data.setdefault("pending_reservation_threads", {}),
        )
        chat_thread_modes[chat_id] = "info"
        pending_threads.pop(chat_id, None)
        confirmation_message_ids.pop(thread_id, None)
        await append_reservation_status(
            thread_id=thread_id,
            status=new_status,
        )
    except Exception:
        logging.getLogger(__name__).exception(
            "Failed to process admin decision for thread %s", thread_id
        )


async def handle_unsupported_message(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    _ = context
    if update.message is None:
        return
    await update.message.reply_text("Please send a text message.")


def _start_health_server() -> None:
    """Start a minimal HTTP server for Cloud Run health checks (listens on PORT)."""
    port = int(os.environ.get("PORT", "8080"))

    class HealthHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logging.getLogger(__name__).info("Health server listening on port %d", port)


def run_bot() -> None:
    _load_env()
    _configure_warnings()
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    token = settings.telegram_bot_token.strip()
    if not token:
        raise ValueError("TELEGRAM_BOT_TOKEN is required to run the Telegram bot.")

    _start_health_server()

    routing_graph_app = build_routing_graph(checkpointer=InMemorySaver())
    execution_graph_app = build_execution_graph(checkpointer=InMemorySaver())
    application = Application.builder().token(token).build()
    application.bot_data["routing_graph_app"] = routing_graph_app
    application.bot_data["execution_graph_app"] = execution_graph_app
    application.bot_data["chat_history_store"] = ChatHistoryStore()
    application.bot_data.setdefault("chat_thread_modes", {})
    application.bot_data.setdefault("pending_reservation_threads", {})
    application.bot_data.setdefault("reservation_confirmation_message_ids", {})

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    application.add_handler(CallbackQueryHandler(handle_admin_callback))
    application.add_handler(
        MessageHandler(~filters.TEXT & ~filters.COMMAND, handle_unsupported_message)
    )
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    run_bot()
