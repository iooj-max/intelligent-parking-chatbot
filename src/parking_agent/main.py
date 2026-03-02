"""Telegram bot entrypoint for the parking agent."""

from __future__ import annotations

import asyncio
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update  # pyright: ignore[reportMissingImports]
from telegram.ext import Application  # pyright: ignore[reportMissingImports]
from telegram.ext import CallbackQueryHandler  # pyright: ignore[reportMissingImports]
from telegram.ext import CommandHandler  # pyright: ignore[reportMissingImports]
from telegram.ext import ContextTypes  # pyright: ignore[reportMissingImports]
from telegram.ext import MessageHandler  # pyright: ignore[reportMissingImports]
from telegram.ext import filters  # pyright: ignore[reportMissingImports]

from parking_agent.graph import GraphState, build_graph
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


def _invoke_graph_for_text(
    graph_app: Any, user_input: str, thread_id: str, conversation_id: str
) -> tuple[str, bool, bool, dict[str, Any] | None]:
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id, "conversation_id": conversation_id},
        "recursion_limit": 20,
    }
    result = graph_app.invoke(
        cast(GraphState, {"messages": [HumanMessage(content=user_input)]}),
        config=config,
    )
    messages = cast(list[Any], result.get("messages", []))
    should_create_pending_file = bool(result.get("should_create_pending_file"))
    interrupted = bool(result.get("__interrupt__"))
    if interrupted and should_create_pending_file:
        response_text = ""
    else:
        response_text = _latest_assistant_text(messages) or _FALLBACK_REPLY
    handoff_to_reservation = bool(result.get("handoff_to_reservation"))
    reservation: dict[str, Any] | None = None
    if should_create_pending_file:
        reservation = cast(dict[str, Any], result.get("reservation") or {})
    return response_text, handoff_to_reservation, should_create_pending_file, reservation


def _status_file_path(status_dir: Path, thread_id: str) -> Path:
    return status_dir / f"{thread_id}.txt"


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


def _reservation_is_pending(status_dir: Path, thread_id: str) -> bool:
    status_path = _status_file_path(status_dir, thread_id)
    if not status_path.exists():
        return False
    return status_path.read_text(encoding="utf-8").strip().lower() == "pending"


def _transfer_state_between_threads(
    graph_app: Any,
    source_thread_id: str,
    target_thread_id: str,
    conversation_id: str,
) -> None:
    source_config: RunnableConfig = {
        "configurable": {
            "thread_id": source_thread_id,
            "conversation_id": conversation_id,
        },
        "recursion_limit": 20,
    }
    target_config: RunnableConfig = {
        "configurable": {
            "thread_id": target_thread_id,
            "conversation_id": conversation_id,
        },
        "recursion_limit": 20,
    }
    source_snapshot = graph_app.get_state(source_config)
    source_values = cast(dict[str, Any], getattr(source_snapshot, "values", {}) or {})
    if not source_values:
        return

    transfer_payload: dict[str, Any] = {}
    if "messages" in source_values:
        transfer_payload["messages"] = source_values["messages"]
    if "conversation_summary" in source_values:
        transfer_payload["conversation_summary"] = source_values["conversation_summary"]
    if "reservation" in source_values:
        transfer_payload["reservation"] = source_values["reservation"]
    if "reservation_validation" in source_values:
        transfer_payload["reservation_validation"] = source_values["reservation_validation"]
    if "missing_field" in source_values:
        transfer_payload["missing_field"] = source_values["missing_field"]
    if "missing_field_reason" in source_values:
        transfer_payload["missing_field_reason"] = source_values["missing_field_reason"]
    if "scope_decision" in source_values:
        transfer_payload["scope_decision"] = source_values["scope_decision"]
    if "scope_reasoning" in source_values:
        transfer_payload["scope_reasoning"] = source_values["scope_reasoning"]
    if "intent" in source_values:
        transfer_payload["intent"] = source_values["intent"]
    if "intent_reasoning" in source_values:
        transfer_payload["intent_reasoning"] = source_values["intent_reasoning"]
    if "handoff_to_reservation" in source_values:
        transfer_payload["handoff_to_reservation"] = source_values["handoff_to_reservation"]
    if "awaiting_user_confirmation" in source_values:
        transfer_payload["awaiting_user_confirmation"] = source_values["awaiting_user_confirmation"]
    if "admin_decision" in source_values:
        transfer_payload["admin_decision"] = source_values["admin_decision"]
    if "should_create_pending_file" in source_values:
        transfer_payload["should_create_pending_file"] = source_values["should_create_pending_file"]

    if transfer_payload:
        graph_app.update_state(target_config, transfer_payload)


def _process_user_text(
    graph_app: Any, user_input: str, chat_id: str, current_mode: str
) -> tuple[str, bool, bool, dict[str, Any] | None]:
    conversation_id = _conversation_id_for_chat(chat_id)
    thread_id = _thread_id_for_intent(chat_id=chat_id, intent=current_mode)
    return _invoke_graph_for_text(graph_app, user_input, thread_id, conversation_id)


def _resume_reservation_thread_with_admin_decision(
    graph_app: Any, thread_id: str, conversation_id: str, admin_decision: str
) -> str:
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id, "conversation_id": conversation_id},
        "recursion_limit": 20,
    }
    result = graph_app.invoke(Command(resume=admin_decision), config=config)
    messages = cast(list[Any], result.get("messages", []))
    return _latest_assistant_text(messages) or _FALLBACK_REPLY


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    await update.message.reply_text(
        "I can help with parking facility information and making parking reservations."
    )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    user_text = (update.message.text or "").strip()
    if not user_text:
        await update.message.reply_text("Please send a non-empty text message.")
        return

    graph_app = context.application.bot_data.get("graph_app")
    if graph_app is None:
        await update.message.reply_text(_FALLBACK_REPLY)
        return

    try:
        chat_id = _chat_id_from_update(update)
        conversation_id = _conversation_id_for_chat(chat_id)
        status_dir = cast(Path, context.application.bot_data["reservation_status_dir"])
        chat_thread_modes = cast(
            dict[str, str], context.application.bot_data.setdefault("chat_thread_modes", {})
        )
        pending_threads = cast(
            dict[str, str],
            context.application.bot_data.setdefault("pending_reservation_threads", {}),
        )
        reservation_thread_id = _thread_id_for_intent(chat_id=chat_id, intent="reservation")
        if _reservation_is_pending(status_dir, reservation_thread_id):
            pending_threads[chat_id] = reservation_thread_id
        current_mode = chat_thread_modes.get(chat_id, "info")
        if chat_id in pending_threads:
            current_mode = "info"
            chat_thread_modes[chat_id] = "info"
        response_text, handoff_to_reservation, should_create_pending_file, reservation = (
            await asyncio.to_thread(
                _process_user_text, graph_app, user_text, chat_id, current_mode
            )
        )

        if should_create_pending_file and current_mode == "reservation":
            if chat_id not in pending_threads:
                pending_threads[chat_id] = reservation_thread_id
                _status_file_path(status_dir, reservation_thread_id).write_text(
                    "pending", encoding="utf-8"
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
                        reply_markup=keyboard,
                    )
                except Exception:
                    logging.getLogger(__name__).exception(
                        "Failed to send admin notification for reservation"
                    )
        elif handoff_to_reservation and current_mode == "info" and chat_id not in pending_threads:
            info_thread_id = _thread_id_for_intent(chat_id=chat_id, intent="info")
            await asyncio.to_thread(
                _transfer_state_between_threads,
                graph_app,
                info_thread_id,
                reservation_thread_id,
                conversation_id,
            )
            chat_thread_modes[chat_id] = "reservation"
        if response_text:
            await update.message.reply_text(response_text)
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
    status_dir = cast(Path, context.application.bot_data.get("reservation_status_dir"))
    if status_dir is None:
        await callback.answer("Status directory not available.")
        return
    status_path = _status_file_path(status_dir, thread_id)
    if not status_path.exists():
        await callback.answer("Reservation already processed or not found.")
        return
    status_value = status_path.read_text(encoding="utf-8").strip().lower()
    if status_value != "pending":
        await callback.answer("Reservation already processed.")
        return
    new_status = "approved" if action == "approve" else "rejected"
    await callback.answer(f"{new_status.capitalize()}")
    try:
        await callback.edit_message_reply_markup(reply_markup=None)
    except Exception:
        pass

    chat_id = _chat_id_from_reservation_thread_id(thread_id)
    if chat_id is None:
        return
    graph_app = context.application.bot_data.get("graph_app")
    if graph_app is None:
        return
    try:
        conversation_id = _conversation_id_for_chat(chat_id)
        response_text = await asyncio.to_thread(
            _resume_reservation_thread_with_admin_decision,
            graph_app,
            thread_id,
            conversation_id,
            new_status,
        )
        await context.application.bot.send_message(
            chat_id=int(chat_id), text=response_text
        )
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
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        processed_path = status_path.parent / f"{status_path.stem}_processed_{timestamp}{status_path.suffix}"
        status_path.rename(processed_path)
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

    graph_app = build_graph(checkpointer=InMemorySaver())
    reservation_status_dir = Path.cwd() / "runtime" / "reservation_status"
    reservation_status_dir.mkdir(parents=True, exist_ok=True)
    application = Application.builder().token(token).build()
    application.bot_data["graph_app"] = graph_app
    application.bot_data["reservation_status_dir"] = reservation_status_dir
    application.bot_data.setdefault("chat_thread_modes", {})
    application.bot_data.setdefault("pending_reservation_threads", {})

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    application.add_handler(CallbackQueryHandler(handle_admin_callback))
    application.add_handler(
        MessageHandler(~filters.TEXT & ~filters.COMMAND, handle_unsupported_message)
    )
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    run_bot()
