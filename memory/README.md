# Agent Memory

Shared knowledge base for all coding agents working on this repo (Claude Code, Codex, Cursor, Copilot, etc.).

## How Agents Should Use This Folder

### On session start — read these files first
1. **`architecture.md`** — project structure, pipelines, key file paths
2. **`conventions.md`** — coding patterns, model roster, prompt rules
3. **`lessons.md`** — hard-won debugging insights and gotchas

These three files are the "always-relevant" context. Read them before making changes.

### During a session
- **`changelog.md`** — rolling log of recent sessions. Skim for context on what changed recently, but don't treat old entries as current truth. Architecture/conventions/lessons files are the source of truth.

### After a session — update what you learned
- **Stable facts** (project structure changed, new pipeline added, model swapped) → update `architecture.md` or `conventions.md`
- **Debugging insight or gotcha** confirmed across multiple instances → add to `lessons.md`
- **Session summary** (what was built/fixed/tested) → append to `changelog.md`

### Rules
1. **Topical, not chronological.** Architecture, conventions, and lessons are organized by topic. Don't append timestamped entries to them — edit the relevant section.
2. **Keep it concise.** These files are injected into agent context windows. Every line costs tokens. If a fact is no longer true, delete it.
3. **Changelog gets trimmed.** Keep only the last ~5 sessions. Older entries should be deleted — their durable lessons belong in `lessons.md`, not in the log.
4. **Don't duplicate.** If a fact is in `architecture.md`, don't also put it in `lessons.md`. Each file has a clear scope.
5. **Verify before writing.** Don't add speculative conclusions from a single file read. Confirm against project code/docs first.

## Legacy Files (root directory)
- **`MEMORY.md`** and **`codingagentmemory.md`** in the repo root are **legacy** files. They are kept for reference but are no longer the active memory. All new memory updates should go into this `memory/` folder.

## File Overview

```
memory/
├── README.md          # This file — usage instructions
├── architecture.md    # Project structure, pipelines, key files
├── conventions.md     # Coding patterns, model roster, prompt rules
├── lessons.md         # Debugging insights, gotchas
└── changelog.md       # Rolling session log (last ~5 sessions)
```
