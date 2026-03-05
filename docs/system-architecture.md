# System Architecture

## Overview

The system is a parking-domain chatbot built as a LangGraph workflow with Telegram as the user channel.

Core runtime layers:

1. Interface layer: Telegram bot receives user/admin messages.
2. Orchestration layer: LangGraph routing and execution graphs.
3. LLM layer: OpenAI models for guardrails, routing, extraction, and response generation.
4. Data layer:
   - PostgreSQL for dynamic parking data.
   - Weaviate for static knowledge retrieval.
5. Observability layer: LangSmith tracing.

## Main Components

- `src/parking_agent/main.py`
  - Telegram handlers.
  - Graph invocation orchestration.
  - Thread/conversation ID management.
  - Admin approval callback handling.

- `src/parking_agent/graph.py`
  - `build_routing_graph()` for scope and intent routing.
  - `build_execution_graph()` for info and reservation execution flows.
  - Reservation validation and confirmation sub-flow.

- `src/parking_agent/tools.py`
  - Tool entry points used by the information retrieval agent.

- `src/parking_agent/facility_validation.py`
  - Facility matching and validation against available facilities.

- `src/data/`
  - Data loading pipeline for PostgreSQL and Weaviate.

## Data and Storage Boundaries

- PostgreSQL (dynamic/transactional data):
  - `parking_facilities`
  - `working_hours`
  - `special_hours`
  - `pricing_rules`
  - `space_availability`

- Weaviate (static semantic knowledge):
  - `booking_process.md`
  - `faq.md`
  - `features.md`
  - `general_info.md`
  - `location.md`
  - `policies.md`

- Runtime files (`runtime/`):
  - Reservation status artifacts.
  - Chat history and summary artifacts.

## High-Level Request Flow

1. User sends a Telegram message.
2. Routing graph performs scope validation and intent routing.
3. Execution graph handles one of two intents:
   - Information retrieval.
   - Reservation workflow.
4. Final response guardrail validates assistant output.
5. Conversation summary is updated for future turns.
6. For reservations requiring approval, an admin decision resumes interrupted execution.
