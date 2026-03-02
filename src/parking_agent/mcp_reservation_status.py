"""MCP-backed reservation status storage."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, cast

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import CallToolResult

from parking_agent.schemas import RESERVATION_FIELD_ORDER, ReservationData
from src.config import settings

ReservationStatus = Literal["pending", "approved", "rejected"]
_ALLOWED_STATUSES: set[str] = {"pending", "approved", "rejected"}


def _status_file_path(thread_id: str) -> str:
    status_dir = settings.mcp_reservation_status_dir.rstrip("/")
    return f"{status_dir}/{thread_id}.txt"


def _normalize_status(status: str) -> ReservationStatus:
    normalized = str(status or "").strip().lower()
    if normalized not in _ALLOWED_STATUSES:
        raise ValueError("Reservation status must be one of pending/approved/rejected.")
    return cast(ReservationStatus, normalized)


def _extract_text_content(result: CallToolResult) -> str:
    parts: list[str] = []
    for item in result.content:
        text = getattr(item, "text", None)
        if isinstance(text, str):
            parts.append(text)
            continue
        if isinstance(item, dict):
            dict_text = item.get("text")
            if isinstance(dict_text, str):
                parts.append(dict_text)
                continue
        if item is not None:
            parts.append(str(item))
    return "\n".join(parts).strip()


async def _call_filesystem_tool(name: str, arguments: dict[str, Any]) -> CallToolResult:
    timeout_seconds = float(settings.mcp_filesystem_timeout_seconds)
    async with streamablehttp_client(
        settings.mcp_filesystem_url,
        timeout=timeout_seconds,
    ) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(name=name, arguments=arguments)
    if result.isError:
        raise RuntimeError(_extract_text_content(result) or f"MCP tool call failed: {name}")
    return result


async def _read_status_file(path: str) -> str:
    try:
        result = await _call_filesystem_tool("read_text_file", {"path": path})
    except RuntimeError as exc:
        message = str(exc).lower()
        if "enoent" in message or "no such file" in message:
            return ""
        raise
    return _extract_text_content(result)


def _build_fields_segment(reservation: ReservationData) -> str:
    values = [str(reservation.get(field, "")).strip() for field in RESERVATION_FIELD_ORDER]
    return "|".join(values)


def _parse_last_line(file_content: str) -> str | None:
    lines = [line.strip() for line in file_content.splitlines() if line.strip()]
    if not lines:
        return None
    return lines[-1]


def _extract_fields_segment(log_line: str) -> str:
    parts = log_line.rsplit(" | ", 2)
    if len(parts) != 3:
        raise ValueError("Invalid reservation status log entry format.")
    return parts[0].strip()


def _extract_status(log_line: str) -> ReservationStatus | None:
    parts = log_line.rsplit(" | ", 2)
    if len(parts) != 3:
        return None
    status = str(parts[1]).strip().lower()
    if status not in _ALLOWED_STATUSES:
        return None
    return cast(ReservationStatus, status)


def _current_timestamp_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


async def append_reservation_status(
    thread_id: str,
    status: str,
    reservation: ReservationData | None = None,
) -> None:
    normalized_status = _normalize_status(status)
    file_path = _status_file_path(thread_id)
    await _call_filesystem_tool(
        "create_directory",
        {"path": settings.mcp_reservation_status_dir},
    )
    existing_content = await _read_status_file(file_path)

    if reservation is not None:
        fields_segment = _build_fields_segment(reservation)
    else:
        latest_line = _parse_last_line(existing_content)
        if latest_line is None:
            raise ValueError("Cannot append status without existing reservation fields.")
        fields_segment = _extract_fields_segment(latest_line)

    new_entry = f"{fields_segment} | {normalized_status} | {_current_timestamp_iso()}"
    if existing_content and not existing_content.endswith("\n"):
        existing_content = f"{existing_content}\n"
    updated_content = f"{existing_content}{new_entry}\n"
    await _call_filesystem_tool(
        "write_file",
        {"path": file_path, "content": updated_content},
    )


async def get_latest_reservation_status(thread_id: str) -> ReservationStatus | None:
    file_path = _status_file_path(thread_id)
    content = await _read_status_file(file_path)
    latest_line = _parse_last_line(content)
    if latest_line is None:
        return None
    return _extract_status(latest_line)


async def reservation_is_pending(thread_id: str) -> bool:
    return (await get_latest_reservation_status(thread_id)) == "pending"
