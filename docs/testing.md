# Testing Guide

This document describes what is tested, how test coverage is organized, and how to run tests locally.

## Folder Structure

Tests are organized by type:

| Folder | Purpose |
|--------|---------|
| `tests/unit/` | Fast isolated unit tests; no external services |
| `tests/integration/` | Multi-component orchestration tests (mocked dependencies) |
| `tests/system/` | System-level load and behavior smoke tests |
| `tests/llm/` | Tests that call real LLM (OpenAI); require API key and network |

Shared `conftest.py` remains in `tests/` root.

## Scope

The test suite covers:

- Unit tests for core helpers, schemas, prompt builders, retrieval, and adapters.
- Integration tests for orchestration steps between routing, execution, and admin approval flow.
- System tests for end-to-end helper-level orchestration behavior.
- Load smoke tests (pytest-based repeated-call tests) for:
  - chatbot interactive dialogue mode,
  - administrator confirmation flow,
  - MCP reservation status recording/storage,
  - orchestration step chaining.

## Final Coverage Matrix by Module

| Module | Total tests | Covered behaviors | Test files |
|---|---:|---|---|
| `src/parking_agent/__init__.py` | 2 | package exports, package metadata/docstring | `tests/unit/test_package_exports.py` |
| `src/parking_agent/agent_runners.py` | 4 | clarifying tool-call extraction, final AI text extraction | `tests/unit/test_agent_runners.py` |
| `src/parking_agent/chat_history_store.py` | 2 | message persistence/retrieval, summary persistence and path safety | `tests/unit/test_chat_history_store.py` |
| `src/parking_agent/clients.py` | 2 | postgres URI and weaviate client construction | `tests/unit/test_clients.py` |
| `src/parking_agent/eval/__init__.py` | 2 | package docstring and export shape | `tests/unit/test_package_exports.py` |
| `src/parking_agent/eval/performance_eval.py` | 3 | percentile math, AI text extraction, latency summary aggregation | `tests/unit/test_eval_performance_eval.py` |
| `src/parking_agent/eval/retrieval_eval.py` | 3 | dataset loading/validation, aggregate retrieval metrics | `tests/unit/test_eval_retrieval_eval.py` |
| `src/parking_agent/facility_validation.py` | 4 | facility validation response structure and matching behavior | `tests/llm/test_facility_validation.py` |
| `src/parking_agent/fetch_trace.py` | 2 | serialization of SDK-like objects, run merge/dedup/sort logic | `tests/unit/test_fetch_trace.py` |
| `src/parking_agent/graph.py` | 4 | reservation merge logic, date/time validators | `tests/unit/test_graph.py` |
| `src/parking_agent/main.py` | 9 | chat ID derivation, admin formatting, orchestration helper calls, admin resume flow, repeated invocation behavior | `tests/unit/test_main.py`, `tests/integration/test_system_integration_orchestration.py`, `tests/system/test_system_load.py` |
| `src/parking_agent/mcp_reservation_status.py` | 5 | status normalization/parsing, append/get latest status, repeated MCP write path | `tests/unit/test_mcp_reservation_status.py`, `tests/system/test_system_load.py` |
| `src/parking_agent/message_reducer.py` | 2 | trimming message history | `tests/unit/test_message_reducer.py` |
| `src/parking_agent/prompts.py` | 4 | prompt input variables, reservation/admin constraints in prompt text, scope classification coverage set | `tests/unit/test_prompts.py`, `tests/llm/test_scope_guardrail.py` |
| `src/parking_agent/retrieval.py` | 3 | deduplication, normalized document output | `tests/unit/test_retrieval.py` |
| `src/parking_agent/schemas.py` | 3 | schema parsing/validation for scope and reservation extraction | `tests/unit/test_schemas.py` |
| `src/parking_agent/tools.py` | 4 | facility match derivation, unresolved extraction, tool wrappers | `tests/unit/test_tools.py` |
| `src/parking_agent/utils/__init__.py` | 2 | utility package exports and metadata/docstring | `tests/unit/test_package_exports.py` |
| `src/parking_agent/utils/messages.py` | 4 | text extraction from string/list/non-text message content | `tests/unit/test_utils_messages.py` |

## Final Coverage Matrix by Test Type

| Test type | Covered modules/components | Test files |
|---|---|---|
| Unit | prompts, schemas, retrieval, tools, clients, message reducer, chat history storage, trace export, eval helpers, package exports | `tests/unit/` |
| Integration | orchestration transitions (routing -> execution -> admin resume) | `tests/integration/` |
| System | helper-level end-to-end runtime behavior, load smoke tests | `tests/system/` |
| LLM | scope guardrail classification, facility validation with DB+LLM pathway | `tests/llm/` |

## Required System Testing Scenarios

### 1) Chatbot in interactive dialogue mode

Validated by:

- `test_load_interactive_dialogue_mode_smoke`

What is checked:

- repeated execution-path invocations return stable responses,
- no interruption flag for info flow,
- response extraction remains consistent under load-smoke repetition.

### 2) Administrator confirmation functionality

Validated by:

- `test_orchestration_reservation_path_with_admin_resume`
- `test_load_admin_confirmation_resume_smoke`

What is checked:

- reservation path can be resumed with admin decision,
- admin-resume operation remains stable under repeated calls.

### 3) MCP server recording and storage process

Validated by:

- `test_append_reservation_status_writes_combined_log`
- `test_get_latest_reservation_status_reads_last_line`
- `test_load_mcp_recording_storage_smoke`

What is checked:

- correct status normalization and append semantics,
- latest status parsing from storage log,
- repeated write/read behavior in MCP storage path via mocked I/O boundary.

### 4) Integration of all orchestration steps

Validated by:

- `test_orchestration_reservation_path_with_admin_resume`
- `test_orchestration_out_of_scope_short_circuit`
- `test_load_orchestration_step_chain_smoke`

What is checked:

- out-of-scope short-circuit handling,
- in-scope reservation orchestration progression,
- admin interruption/resume path and final user-facing response extraction.

## How to Run Tests

Do not run all commands blindly in production environments. Use a virtualenv and `.env` configured for local development.

### Full suite

```bash
.venv/bin/pytest -v
```

### By test type

System tests:

```bash
.venv/bin/pytest -v -m system
```

Integration tests:

```bash
.venv/bin/pytest -v -m integration
```

Load smoke tests:

```bash
.venv/bin/pytest -v -m load
```

Unit-focused run (exclude integration/system/load):

```bash
.venv/bin/pytest -v -m "not integration and not system and not load"
```

Exclude LLM tests (no API key or network required):

```bash
.venv/bin/pytest -v -m "not llm"
```

### By folder

```bash
.venv/bin/pytest -v tests/unit/
.venv/bin/pytest -v tests/integration/
.venv/bin/pytest -v tests/system/
.venv/bin/pytest -v tests/llm/
```

### By component/module

```bash
.venv/bin/pytest -v tests/integration/test_system_integration_orchestration.py
.venv/bin/pytest -v tests/system/test_system_load.py
.venv/bin/pytest -v tests/unit/test_mcp_reservation_status.py
.venv/bin/pytest -v tests/unit/test_chat_history_store.py
.venv/bin/pytest -v tests/unit/test_eval_retrieval_eval.py
.venv/bin/pytest -v tests/unit/test_eval_performance_eval.py
```

## Manual Validation Steps (Local)

1. Ensure environment and dependencies are ready:

```bash
source .venv/bin/activate
pip install -e .
```

2. Validate test collection and markers:

```bash
.venv/bin/pytest --collect-only -q
.venv/bin/pytest --markers
```

Expected:

- `system`, `integration`, `load` markers are listed.
- New test files are discoverable in collection output.

3. Run orchestrated system/integration coverage only (no LLM):

```bash
.venv/bin/pytest -v tests/integration/ tests/system/
```

Expected:

- orchestration, admin resume, and MCP load-smoke tests are executed,
- no real DB/Weaviate/MCP network dependency is required for load-smoke tests due to mocks.

4. Run key module-focused checks:

```bash
.venv/bin/pytest -v tests/unit/test_chat_history_store.py tests/unit/test_mcp_reservation_status.py
```

Expected:

- local runtime file persistence behavior and MCP status helper behavior are validated.

## Notes

- `tests/llm/` contains tests that call real OpenAI LLM; they require valid `OPENAI_API_KEY` in `.env` and network access.
- Load tests in this repository are smoke-level repeated-call checks implemented in pytest (not external load tooling).
