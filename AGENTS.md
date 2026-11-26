# AGENTS.md

This file is not a map of the current implementation; it describes how an agent should work in this repo. The work here is to produce a notebook that carries a reader from raw data to a fund decision.

## Reader

- Path: Make every change clarify the reader's path from raw data to a fund decision.
- First read: Apply a cold first-read check; re-express code that makes the reader hold several states or rules in mind.

## Code

- Reuse: Before owning custom code, ask what the operation must do, how the present stack solves it, and whether the data can be reshaped to use that primitive; prefer the provided mechanism already present in the stack.
- Dependency: Add a package only when it removes a whole concern with less surface than the code it replaces.
- Sprawl: Assume it accumulates quietly; merge duplicates, reject forwarding layers, and leave touched code simpler even when the diff grows.
- Concern: Give each kind of knowledge one named home; extract only when one sentence without "and" describes it.
- Locality: Keep implementation-detail caveats at the detail itself; do not promote them to repo-wide rules or point at details elsewhere in the code, which go stale uncaught.

## Change

- Bounds: Do not change the notebook's presentation or namespace unless asked; do not change staging or commit state unless explicitly asked; read-only inspection is fine.
- Ground: Read before changing; do not infer responsibility from names; establish tool mechanisms from documentation or source.
- Verify: Check what changed means rather than whether checks pass; confirm outputs semantically, reproduce failures where they occur; prefer that evidence over clean checks, assumed equivalence, or line, file, and diff counts; clean checks prove correctness, not clarity or equivalence.
- IDE: Treat editor diagnostics as authoritative for the reader's environment; reproduce them with the same interpreter and a fresh language-server cache before changing code or dismissing them.

## Collaboration

- Correction: Restate goal and constraints after correction, but restating is not permission to continue; after two rejections of the same approach, ask one precise question.
- Friction: When a correction, rejection, surprising result, workaround, or unclear rule shows that this file did not settle the decision, surface it in chat as it happens: name the trigger, classify it as `rule` if an existing rule governed the decision or `gap` if none did, name the smallest useful update.
- Reporting: Report decisions and why; numbers are evidence, not the message.
- Completion: Before the final response, walk every bullet of `AGENTS.md` in order. If only a compacted version is in context, load the file from disk first. Formulate one reason per bullet that reports how it was satisfied, how it was violated, or why it was not applicable. Keep `agent/proof.py` reason IDs in sync with these labels. Feed those reasons and any structured open friction as JSON on stdin to `python3 agent/proof.py`; use `--example` for the input shape and never hand-write `proof.json`, the check block, or the receipt. Append the printed block with horizontal rule to the final response.
