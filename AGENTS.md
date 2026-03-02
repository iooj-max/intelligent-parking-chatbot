# Repository Guidelines

## Project Structure & Technical stack

- Programming Language: Python3.
- Agent and workflow logic must be implemented using LangChain and LangGraph.
- Tracing, evaluation, and observability must use LangSmith.
- Relational data persistence must use PostgreSQL.
- Vector search and embeddings storage must use Weaviate.
- Do not add alternative frameworks, databases, or storage backends unless explicitly instructed.
- Debugging and inspection must be performed using LangSmith Studio.
- Required infrastructure services are provided locally via Docker.
- All code must be written in English.
- Never use any other language in code (identifiers, comments, strings intended for users/logs, error messages).

## Project Context & Scope

- This project implements an intelligent, RAG-based parking assistant chatbot.
- The agent is designed to interact with users about parking-related topics only.
- The primary responsibilities of the agent are:
  - Answering user questions using data stored in the system databases.
  - Assisting users with parking-related actions such as reservations.
- The agent must explicitly detect and reject out-of-scope requests.
  - Out-of-scope handling must be performed through LLM-based classification, not keyword matching.
  - When a request is out of scope, the agent must respond briefly that the request is out of scope and optionally suggest a relevant parking-related direction.
  - No database access or tool invocation must occur for out-of-scope requests.
- The agent uses LangGraph as the central orchestration mechanism.
  - All decision-making, routing, and branching must be expressed as graph logic.
- The system is designed to support multi-step interactions.
  - When required information for an action (e.g. reservation) is missing, the agent must ask the user for the necessary details instead of making assumptions.
- The agent is not a generic conversational assistant.
  - It must not attempt to answer questions outside the parking domain.
  - It must prioritize correctness, control, and predictable behavior over conversational breadth.

## General Rules

- All bot-related functionality must be implemented using LangChain, LangGraph, and LangSmith.
- Orchestration and control flow must be implemented exclusively with LangGraph.
  - Do not use keyword-based or string-matching logic to drive behavior.
  - Never implement conditional logic by checking whether user input contains specific words or phrases.
  - Use LangGraph conditional edges for all branching and decision-making.
- Natural language responses must never be hardcoded as literal phrases.
  - Do not store predefined response texts such as exact sentences or quotes.
  - Instructions to the model must describe what to communicate, not what exact words to say.
- Agent state must be stored deliberately and kept minimal.
  - Persist only information that is necessary for future reasoning or task continuation.
  - Avoid unbounded or accumulative state growth.
- The agent must follow the ReAct pattern.
  - The LLM decides autonomously whether a tool invocation is required.
  - Tool usage is driven by model reasoning and feedback loops, not by external rules.
  - This pattern must be used to handle complex and multi-step scenarios.
- LangGraph guardrails must be used to enforce safety, validation, and execution constraints.
  - Guardrails must handle input validation, output validation, and execution boundaries.
  - Do not implement ad-hoc safety checks outside the LangGraph flow.
  - Guardrails must be part of the graph logic, not external wrappers.
- LangGraph SQL agents must be used for all interactions with relational databases, except for facility validation where a simple direct db call can be implemented. 
  - Do not construct SQL queries manually or via string concatenation.
  - Do not implement custom database logic outside the SQL agent abstraction.
  - The SQL agent is the single entry point for querying and reasoning over PostgreSQL data.
- Retrieval and RAG logic must be implemented via LangChain retrievers and LangGraph nodes.
  - Do not access Weaviate directly outside agent tools or graph nodes.
  - Retrieval must be part of the graph execution flow.
- Tool usage must be model-driven.
  - Do not manually route or force tool invocation based on external conditions.
  - Tools must be invoked only through the ReAct loop.
- Prefer structured outputs over free-form text.
  - Use schemas or typed outputs where applicable.
  - Do not parse structured data from raw natural language responses.
- Before starting work or planning changes, familiarize yourself with the best-practice sources listed in this document.

## API Usage & Verification Rules

- Do not rely on assumed or remembered APIs.
- Before using a library API, verify it against official documentation or the official repository source.
- Prefer reading the actual source code or reference docs over examples from blogs or tutorials.
- Assume APIs correspond to the versions specified in project dependency files.
- If versions are not specified, assume the latest stable release.
- If an API is ambiguous or recently changed, clarify before implementation.

## Coding Style & Naming Conventions

- Follow Python PEP 8 for code style and formatting.
- Follow PEP 257 for docstrings.

## Build, Test, and Development Commands

- None at this time.

## Testing Guidelines

- Do not run tests unless explicitly instructed.
- Do not modify existing tests unless explicitly instructed.
- If tests fail, do not change test code to make them pass.
- Tests are treated as a source of truth for expected behavior.

## Commit & Pull Request Guidelines

- All development work must be done on the `develop` branch.
- Do not make direct changes to the `main` branch.
- Changes may be merged into `main` only via a pull request.
- Do not create commits or pull requests unless explicitly instructed.

## Security & Configuration Tips

- Do not commit real secrets.
- Do not hardcode credentials, tokens, connection strings, or API keys.
- Configuration must be provided via environment variables or configuration files excluded from version control.

## Agent-Specific Instructions

- Before writing a library specific code, use context7 MCP tools.
- After writing code use linters to check if there are any issues.

## Best-practice sources for development

Use these sources as the primary references for architecture patterns, APIs, and production guidance. Prefer official documentation and official organization repositories. Treat blogs, gists, and forums as secondary sources and always verify against official materials.

### Tier 1 — Official documentation (highest trust)

- LangChain OSS Python Docs  
  - https://docs.langchain.com/oss/python/
- LangGraph Docs  
  - https://docs.langchain.com/oss/python/langgraph/overview  
  - https://langchain-ai.github.io/langgraph/guides/  
  - https://langchain-ai.github.io/langgraph/reference/
- LangChain Academy  
  - https://academy.langchain.com/

### Tier 2 — Official GitHub repositories (source of truth for implementation)

- https://github.com/langchain-ai/langgraph
- https://github.com/langchain-ai/langchain
- https://github.com/langchain-ai/langchain-academy

### Tier 3 — Community and semi-official sources

- Use only when:
  - The pattern is not covered in Tier 1–2, or
  - An end-to-end reference application is needed for adaptation.
- Verification rule:
  - Any API usage must be validated against official documentation or official repositories.

### Domain allowlist for web search tools

- docs.langchain.com
- academy.langchain.com
- langchain-ai.github.io
- github.com/langchain-ai

### Exclusions and low-trust sources

- Medium articles, personal blogs, gists, and Reddit must not be treated as best-practice references.
- If used at all, they may serve only as idea generators and must be verified against Tier 1–2 sources.

## TASKS CONSTRAINTS

### Scope Validation & Intent Routing

- Use LLM-based guardrails implemented inside LangGraph.
- Do not use keyword-based or rule-based matching.
- Do NOT suggest other things the user can ask.
- The output must NOT include suggestions or alternatives when out of scope.
- Route via LangGraph conditional edges only.
- If user asks bot about what bot can do or it's functionality mention that bot can help with parking facility information and making parking reservations.
- Supported intents:
  - Retrieve information from databases
  - Start or continue a reservation workflow
- For out-of-scope cases, respond with a short message stating that the request is out of scope.
- For out-of-scope cases, do NOT access databases or invoke tools.
- The Parking Agent scope includes any parking-related questions about:
  - parking facilities
  - policies (rules)
  - allowed vehicle types
  - pricing
  - availability
  - hours
  - location
  - amenities
  - general parking information
- The reservation intent includes booking, reserving, or starting a reservation.

### Reservation Information & Validation

- The user must be asked for one piece of information at a time.
- A reservation requires the following information from the user:
  - **Customer name**: Any non-empty string
  - **Parking facility**: Must be in the list of available parking facilities
  - **Date**: Must be today or a future date in YYYY-MM-DD format
  - **Start time**: Must be a valid time in HH:MM format (24-hour)
  - **Duration**: Must be a positive integer between 1 and 168 (maximum 1 week)


## Data section

There are two databases available for the agent.

### Dynamic data (postgres)

- parking_facilities;
- working_hours;
- special_hours;
- pricing_rules;
- space_availability;

### Static data (weaviate)

- booking_process.md (How to Book Airport Parking)
- faq.md (Frequently Asked Questions)
- features.md (Parking Features & Amenities)
- general_info.md (Overview, Contact Information, etc)
- location.md (Location & Access)
- policies.md (Parking Policies & Rules)


## Manual test guidance required

After making code changes, always provide:
- A short list of manual validation steps I can run locally.
- Concrete command examples (copy/paste ready) to exercise the changed behavior (e.g., CLI commands, curl requests, SQL queries, minimal repro scripts).
- If applicable, example inputs + expected outputs, and which files/lines should change or which logs/messages I should see.

## Debugging behavior

- If verification is needed, ask me to run the suggested commands and share the output.
- Do not iterate with workaround-style changes “until it passes”; instead, wait for my test results and then propose the next targeted change.

## If there was/is a Plan (plan-mode)

- Strictly follow the agreed plan.
- Do not implement any solution, workaround, refactor, dependency change, or behavioral change that was not in the plan.
- If new information (errors, constraints, unexpected behavior) requires deviating from the plan:
  - Stop before making further code changes.
  - Propose an updated plan with alternatives and trade-offs.
  - Wait for explicit approval before proceeding.
- When multiple options exist, present them and ask which one to execute; do not choose unilaterally.