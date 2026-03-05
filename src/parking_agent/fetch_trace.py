"""CLI utility to fetch a LangSmith trace by trace_id via SDK."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from dotenv import load_dotenv
from langsmith import Client
from langsmith.utils import LangSmithAuthError

def _serialize(value: Any) -> Any:
    """Convert SDK objects into JSON-serializable values."""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialize(v) for v in value]
    return value


def _run_to_dict(run: Any) -> dict[str, Any]:
    """Extract stable run fields for export."""
    return {
        "id": str(getattr(run, "id", "")),
        "trace_id": str(getattr(run, "trace_id", "")),
        "name": getattr(run, "name", None),
        "run_type": getattr(run, "run_type", None),
        "status": getattr(run, "status", None),
        "error": getattr(run, "error", None),
        "start_time": _serialize(getattr(run, "start_time", None)),
        "end_time": _serialize(getattr(run, "end_time", None)),
        "inputs": _serialize(getattr(run, "inputs", None)),
        "outputs": _serialize(getattr(run, "outputs", None)),
        "metadata": _serialize(getattr(run, "metadata", None)),
        "tags": _serialize(getattr(run, "tags", None)),
        "parent_run_id": _serialize(getattr(run, "parent_run_id", None)),
        "child_run_ids": _serialize(getattr(run, "child_run_ids", None)),
        "events": _serialize(getattr(run, "events", None)),
        "extra": _serialize(getattr(run, "extra", None)),
    }


def _list_trace_runs(client: Client, trace_id: str) -> list[Any]:
    """Load all runs for trace_id using SDK cursor pagination."""
    # Client.list_runs handles cursor-based pagination internally.
    # Passing unsupported page controls (e.g. offset) can lead to repeated pages.
    return list(
        client.list_runs(
            trace_id=trace_id,
        )
    )


def fetch_trace(trace_id: str) -> dict[str, Any]:
    """Fetch all runs for a trace_id using LangSmith SDK."""
    UUID(trace_id)

    client = Client()
    runs = []

    # If trace_id is also a run_id of a root run, include it directly.
    try:
        root = client.read_run(trace_id)
        runs.append(root)
    except Exception:
        pass

    # Query full trace by trace_id.
    queried_runs = _list_trace_runs(client, trace_id)

    existing_ids = {str(getattr(run, "id", "")) for run in runs}
    for run in queried_runs:
        run_id = str(getattr(run, "id", ""))
        if run_id and run_id not in existing_ids:
            runs.append(run)

    runs_sorted = sorted(
        runs,
        key=lambda run: (
            getattr(run, "start_time", datetime.max),
            str(getattr(run, "id", "")),
        ),
    )

    return {
        "trace_id": trace_id,
        "run_count": len(runs_sorted),
        "runs": [_run_to_dict(run) for run in runs_sorted],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch LangSmith trace data by trace_id and print JSON."
    )
    parser.add_argument(
        "--trace-id",
        required=True,
        help="Trace UUID from LangSmith.",
    )
    parser.add_argument(
        "--output",
        help="Optional path to save JSON output (default: traces/<trace_id>.json).",
    )
    return parser


def main() -> int:
    load_dotenv()
    parser = _build_parser()
    args = parser.parse_args()

    try:
        data = fetch_trace(args.trace_id)
    except ValueError:
        print("Invalid trace_id format: expected UUID.", file=sys.stderr)
        return 2
    except LangSmithAuthError as error:
        print(
            f"LangSmith authentication failed: {error}. "
            "Check LANGSMITH_API_KEY and LANGSMITH_ENDPOINT.",
            file=sys.stderr,
        )
        return 3
    except Exception as error:  # noqa: BLE001
        print(f"Failed to fetch trace: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    payload = json.dumps(data, ensure_ascii=False, indent=2)
    output_path = (
        Path(args.output) if args.output else Path("traces") / f"{args.trace_id}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(payload + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
