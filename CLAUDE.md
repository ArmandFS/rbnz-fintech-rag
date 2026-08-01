# Project Overview

You are an experienced software engineer. Build a modular fintech RAG application focused on Reserve Bank of New Zealand (RBNZ) financial and regulatory documents.

The intended build order is:

1. Core Python RAG pipeline
2. FastAPI wrapper for the RAG service
3. Node/Express backend API
4. React/TypeScript frontend

## Repository Shape

The actual git repository root is:

```text
/Users/armandsurbakti/fintech-rag-project/fintech-rag
```

The RAG code lives in:

```text
rag/
```

---

## Coding Style

- Never create duplicate components.
- Prefer reusable functions.
- Use descriptive variable names.
- Keep functions under ~50 lines when practical
- Avoid uncessary abstractions.


---


## Communicate Clearly

- State assumptions explicitly.

- Explain important tradeoffs briefly when relevant.

- Ask questions only when ambiguity materially affects correctness.

- If multiple valid approaches exist, summarize the options briefly before proceeding.

- Do not hide uncertainty or missing information.


---

## When implementing features

Always:

- Make the smallest possible change.
- Preserve backwards compatibility.
- Do not remove working functionality.
- Ask before deleting code.

---

## Testing

Whenever possible

- run tests
- fix linting
- ensure build passes

---

## Git

Never commit.

Never push.

Only suggest commit messages.

