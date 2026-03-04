---
name: Consult Context
description: Reference architectural, API, and domain knowledge from the context library.
---

# Consult Context

This skill allows you to retrieve rigorous domain knowledge and architectural standards from the `resources/` library. Use this when you need to understand *how* the system works or *why* it was built that way.

## Instructions

### 1. Identify Topic
Match the user's question to a context file:
*   `architecture.md`: High-level system design.
*   `api_conventions.md`: REST/WebSocket standards.
*   `error_codes.md`: Troubleshooting.
*   `trading_terms.md`: Domain glossary.

### 2. Retrieve & Synthesize
Read the file and extract the specific information needed. Do not dump the whole file; answer the specific question.

### 3. Citation
When answering, reference the source (e.g., "According to `architecture.md`...").
