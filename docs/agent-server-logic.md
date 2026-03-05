# Agent and Server Logic

## Server Entry Logic

The server entry point is `src/parking_agent/main.py`.

Key responsibilities:

- Accept Telegram user messages and admin callbacks.
- Build thread identifiers:
  - Info thread: `tg:{chat_id}:info`
  - Reservation thread: `tg:{chat_id}:reservation`
- Build conversation identifier: `tg:{chat_id}`.
- Invoke routing graph first, then execution graph.
- Persist user/assistant turns and updated summary.
- Manage pending reservation approval status.

## Routing Graph Logic

`build_routing_graph()` in `src/parking_agent/graph.py`.

Flow:

1. `scope_guardrail`
   - LLM classifies input as in-scope or out-of-scope.
2. Conditional route:
   - Out of scope -> `out_of_scope_response` -> `final_response_guardrail`.
   - In scope -> `intent_router`.
3. `intent_router`
   - LLM returns intent: `info_retrieval` or `reservation`.
4. `final_response_guardrail`
   - Validates generated response and can rewrite unsafe response.

## Execution Graph Logic

`build_execution_graph()` in `src/parking_agent/graph.py`.

Entry routing:

- `info_retrieval` -> `info_agent_llm` -> `final_response_guardrail`.
- `reservation` -> reservation sub-flow.

Reservation sub-flow:

1. `reservation_extract`.
2. `reservation_check` (missing/invalid fields).
3. Conditional route:
   - Missing/invalid -> `reservation_ask`.
   - Complete -> `reservation_confirm`.
   - Awaiting confirmation -> `reservation_confirmation_decision`.
4. Confirmation decision route:
   - Confirmed -> `reservation_wait_admin_decision` (interrupt/resume).
   - Cancelled -> `reservation_cancelled_response`.
   - Modified -> loop back to `reservation_extract`.
5. All paths pass through `final_response_guardrail`.
6. `update_conversation_summary` runs before graph end.

## Parallel Thread Model

The system uses separate LangGraph thread IDs per chat intent:

- Info thread keeps information-retrieval state.
- Reservation thread keeps reservation state.

This isolates intent-specific state while still sharing conversation-level context from persisted chat history/summary.

## Admin Approval Logic

- On confirmed reservation, execution interrupts and waits for admin input.
- Admin receives Approve/Reject buttons in Telegram.
- Callback resumes the reservation thread via `Command(resume=...)`.
- Final user message is generated based on admin decision.
