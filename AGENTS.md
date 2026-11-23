# AGENTS.md

The deliverable is a notebook a reader can follow from raw data to a fund decision. Sprawl is the failure mode: units that do not earn their place — helpers that only forward, functions that only restate their code, mechanics mixed into choices, state left in the reader's way. It accumulates quietly, so avoiding new sprawl is not enough; it has to be reduced on sight.

- Organize by concern, not by caller. Each kind of knowledge has one home; consolidate it when it appears in several places. A new file needs a distinct responsibility. A unit whose only job is to sequence existing calls or relay between them is a layer, not a concern.
- Name a piece for what it is, independent of where it is used today. Extract a general operation only when it has a name that stands on its own; a single caller is fine when the concept is real, but not when the only available name references that caller. Put it with the concern it implements, not in a miscellaneous bucket.
- Before giving code its own function, say what it does in one short sentence. If the sentence needs "and", the function is doing too much — find a different boundary.
- The notebook tells the story: state each choice and call one named action for each step, with the values that matter visible. Every step is named, whether or not it involved a choice; only its mechanics stay in source modules, however short they are.
- Reduce sprawl in the code you touch. Merge near-duplicates, fold thin layers into their natural homes, and remove what no longer earns its place. Leave the area simpler than you found it, even when that means a larger change; the reader sees the result, not the size of the diff.
- When a problem involves a tool, framework, or runtime, establish the mechanism from its documentation or source before proposing a fix. Prefer the provided mechanism to custom machinery.
- Verify by reading the result as a newcomer, reproduce failures in the environment where they occur, and treat generated outputs as part of the contract — confirm them after execution. Clean checks prove correctness, not clarity or equivalence.
- When corrected, restate the goal and constraints before acting again. If an approach is rejected twice, stop and ask one precise question instead of producing another design. Report what you decided and why; numbers from checks are evidence, not the message.
