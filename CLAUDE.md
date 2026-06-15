# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Scope

The active code lives in `src/` (Common Lisp). The `C++/` directory is an abandoned earlier attempt — ignore it. Design context lives in `notes/Overview.txt` and `notes/Notes.pdf` (the latter is unfiltered, roughly chronological scratch notes).

This is a research/personal project released to the public domain. It began as only the **input component** of a three-part design (Input → Processing → Output). `Plan.md` (repo root) lays out a phased build toward continual, Hebbian-style learning that adds the Processing and Output components; **Phases 0–7 are implemented** — data-structure foundations, output generation, the association bridge, spreading-activation inference, reward-modulated reinforcement with decay/pruning, an interactive teaching loop (`main`), and persistence. The input→processing→output design is functionally whole and survives restarts: you can teach the running system (input, response) pairs and it reproduces or generalizes them on later inputs, across sessions. **Phase 7** adds a pure-Hebbian **concept graph** (`concepts.lisp`) that generalizes to novel inputs: trained on a starter knowledge base, never-taught "horses"/"birds" are recognized as walking on their legs while legless animals (snakes/fish), vehicles, furniture, and nonsense are excluded. It is auto-populated slot-free from `learn`, persists with the network, and uses an adaptive per-category threshold. `train-from-file` imports a training set (`src/training-set.txt`, 105 facts); `export-kb`/`import-kb` save and reload the whole knowledge base. A separate **attention head** (`attention.lisp`), built as Hebbian fast weights, adds copy/binding — the original FIRST GOAL `say X → X` for a *novel* X (taught only `say dog/cat/house`, it answers `say car → car`). See `Plan.md` §3.5–§3.6. The original design notes are in `notes/`.

## Running the code

There is no real build system. `src/Makefile` only provides `make clean` to remove Lisp fasl artifacts (`*.fas`, `*.fasl`, `*.lib`, `*.lx64fsl`).

To run, load and start from a Common Lisp REPL **with the working directory set to `src/`**
(so relative paths — the training file, the saved knowledge base — resolve there). Load the
whole system with one call:

```lisp
(load "load.lisp")   ; loads every component in dependency order; works on all impls
(main)               ; NOT (ai::main) — see note below
```

`load.lisp` also defines `(load-system)`, which reloads everything after an edit. Under the
hood it loads each component by pathname; each file `(provide …)`s, so the internal
`(require …)` forms are no-ops. (CLISP alone can also `(load "ai.lisp")` directly — its
`require` searches the current directory — but SBCL/CCL/ECL do not, which is why `load.lisp`
loads by pathname.)

`main` is an interactive teaching loop (Phase 5). Each turn it reads an **input** sentence, prints what it would answer (or "I don't know"), then reads a **teacher** line — the correct response, or a confirm word (`yes`/`y`/`right`/`correct`/`ok`) to accept a correct guess — and learns from it. It exits on `quit.`/`exit.` (or end-of-input) and prints the input neuron tree. It loads saved memory from `*save-file*` (default `ai-network.sexp` in the working directory) on entry and saves it on exit, so learning persists across runs.

**Note on `(main)` vs `(ai::main)`:** `ai.lisp` has no `defpackage`/`in-package` (unlike the component files), so its only function, `main`, is interned into whatever package is current at load time — normally `common-lisp-user`. There is no `ai` package; call `(main)`, not `(ai::main)`.

**Input format:** all input is lowercased (`string-downcase` in `getword`), and every sentence must end with a terminator — `.`, `!`, or `?` — which is read as its own token. This applies to the exit commands too: type `quit.` or `exit.` (a bare `quit` with no terminator just makes the reader block waiting for more input).

**Tests:** a small smoke-test suite lives in `src/tests.lisp` (115 assertions). It loads the components by pathname — so it stays loadable on CLISP, SBCL, CCL, and ECL, though `make test` now runs SBCL (Blake's primary) — and checks the data-structure foundations, the Output component, the association bridge, inference, the reinforcement/decay dynamics, the interactive teaching loop, persistence (a save→reset→reload round-trip), concept-graph generalization (auto-populated, adaptive, persisted), the starter knowledge base (`train-from-file` on `training-set.txt`, then `export-kb`/`import-kb`), the string interface, and the attention copy head (`say X → X` for novel X). Tests that touch files bind `*save-file*` / use throwaway paths and delete them. Run it with `make test` (SBCL; non-zero exit on failure) or interactively, from `src/`, with `(load "tests.lisp")`.

For a manual end-to-end check of the teaching loop (lines alternate input / teacher; note the `(main)` line and that input is piped to the REPL, not passed via `clisp -x`, which would hijack `*standard-input*`). This teaches "say hi" → "hi there", then re-asks and confirms the correct guess:

```sh
cd src
printf '(load "load.lisp")\n(main)\nsay hi.\nhi there.\nsay hi.\nyes.\nquit.\n(ext:quit)\n' | clisp -q
```

## Architecture

The system converts input sentences into a network of neuron structures, maps them to output structures, and generates responses. Nine source files (`data-structures`, `line-input`, `input`, `output`, `processing`, `persist`, `concepts`, `attention` are each their own package; `ai.lisp` is the package-less top level):

- **`data-structures.lisp`** (package `data-structures`) — defines `neuron`, `named-neuron` (neuron + `name` slot for word neurons), and `dendrite` (an edge to another neuron, carrying a `weight`, a `kind` — `:sequence` / `:extender` / `:association` — and a `from` back-pointer to its source, set on `:association` edges so decay can prune them). Holds the global `*dictionary*` hash table (word string → named-neuron), the `*next-neuron-id*` counter, and the `*output-roots*` / `*associations*` registries (empty for now — reserved for the Processing/Output components per `Plan.md`). `reset` clears all of them. `dump-dictionary` recursively prints the network.

- **`line-input.lisp`** (package `line-input`) — tokenizes stdin into words terminated by `.`, `!`, or `?`. `create-line` returns a list of `named-neuron`s for one sentence, interning new words into `*dictionary*` via `intern-word`. Also exports `intern-words` (intern a list of word strings without reading stdin — used by inference) and the `add` macro (push onto a list variable).

- **`input.lisp`** (package `input`) — the input algorithm, extracted from `ai.lisp` (Phase 3) so both `main` and `processing` can call it without a circular dependency. `build-structure` (described below) turns a sentence into the neuron network and returns the `active` meaning neurons; `connect`, `connects-to`, and `find-connecting-neuron` are its helpers.

- **`ai.lisp`** — the package-less top-level entry point: the interactive teaching-loop `main` (read input → `respond` → read teacher line → `learn`) plus `train-from-file` (bulk-learn a training set of `input => answer` lines) and the `require`/`use-package` forms that pull the whole system together. (The input algorithm now lives in `input.lisp`.)

- **`output.lisp`** (package `output`) — the start of the **Output** component (Phase 1 of `Plan.md`), the mirror of the input network. `build-output-structure` builds (or reuses, keyed by text in `*responses*`) a divergent extender chain `root → w1 → w2 → …` for a response and registers the root in `*output-roots*`; `produce-output` walks the chain and returns the emitted words; `output-sentence` joins them. Output nodes are fresh neurons, never the input word neurons in `*dictionary*`, so the two networks never share an extender chain.

- **`processing.lisp`** (package `processing`) — the start of the **Processing** component (Phase 2 of `Plan.md`), the bridge between the two networks. `associate(input-endings, output-root)` wires every input "meaning" neuron (the `active` set returned by `build-structure`) to an output root with a weighted `:association` dendrite — created at `*assoc-initial-weight*` or strengthened by `*assoc-strengthen*` (clamped to `*assoc-max-weight*`) and registered in `*associations*`. These cross-component edges are the only ones later phases reinforce and decay; structural `:sequence`/`:extender` edges are permanent. `respond(input-words)` does inference (Phase 3): it builds the input, spreads each meaning neuron's association `weight`s into the output roots' `current-value`, and returns the highest-scoring root's response (winner-take-all above `threshold`) — or `nil` ("I don't know") if nothing is activated. `learn(input-words, correct-words)` is one continual-learning turn (Phase 4): it reinforces the correct pathway, weakens a wrong guess, runs a global `decay-associations` step (shrinking unused weights, pruning those below `*assoc-prune-threshold*`, GC'ing orphaned roots), and homeostatically adapts the correct root's threshold — the rule `Δw = η·pre·post·r − λ·w` from `Plan.md` §3.4.

- **`persist.lisp`** (package `persist`) — persistence (Phase 6). `save-network`/`load-network` serialise the whole graph to one readable s-expression (neurons written flat and keyed by `id`, every reference as an id, rebuilt in two passes via an id→neuron map so shared structure and the cyclic `from` back-pointers survive) and reload it; `main` loads `*save-file*` on entry and saves on exit, so learning persists across restarts. `export-kb`/`import-kb` save and reload the entire knowledge base (input network, outputs, associations, **and** the concept graph) to/from a chosen file.

- **`concepts.lisp`** (package `concepts`) — the Phase 7 **concept graph**, where generalization lives. `relate(subject, predicate, answer)` Hebbianly strengthens an edge between a subject concept (a shared `*dictionary*` neuron) and a `(predicate:answer)` state neuron in `*concept-graph*`; `category-strength`/`recognizes-p` answer "does X belong to this category?" by degree-normalized spreading activation. Similar subjects share state neighbours, so a novel word generalizes in while dissimilar ones (and nonsense) are excluded — no slots, no rules. Edges live in `*concept-graph*` (not in `axon` slots) so the input network is untouched. The graph is **auto-populated from `learn`** (`note-relationship` relates each input word to its complement-frame + answer; the slot-free `infer-p`/`infer-strength` query tries every word-as-subject decomposition), the membership threshold is **adaptive** (relative to each category's members, via `member-baseline` × `*concept-fraction*`), and it **persists** with the rest of the network. See `Plan.md` §3.5 / Phase 7.

- **`attention.lisp`** (package `attention`) — a transformer-like **attention head realized as Hebbian fast weights**: each token/position gets a deterministic vector, a sentence is bound as `M = Σ roleᵢ ⊗ tokenᵢ` (outer products), and a role-query retrieves a token (`r·M`, then decode) — attention with keys = roles, values = token vectors, no backprop. This gives the **copy/binding** capability the associative net and concept graph lack — the project's original FIRST GOAL, `say X → X` for a *novel* X. `learn` calls `note-copy` to learn (in `*copy-cues*`, persisted) that a cue word triggers "copy the next word"; once a cue passes `*copy-threshold*`, `respond` fires `copy-response` (an induction head) before falling back to associative spreading. `attention-experiment.lisp` is a standalone proof-of-concept.

### How `build-structure` works

This is the non-obvious part — read `notes/Overview.txt` alongside the code. For each word in the input sentence, the system maintains an `active` list of "frontier" neurons representing the running partial sentence and its subset variants. For each new word neuron:

1. The word's **extender neuron** (`neuron-extender`, lazily created via `get-extender`) represents "this word in sequence" and seeds the next frontier.
2. For each prior frontier neuron `pn`, the system also adds (a) `pn`'s extender (representing the prior partial-sentence advancing past the new word — the "missing word" combination) and (b) either an existing neuron that both `pn` and the new word already connect to, or a freshly created joining neuron (`connect pn (new-next-neuron neuron)`).

The result: a single sentence builds neurons not only for the full sequence but for every subset preserving order, so the network is less brittle to exact wording. `find-connecting-neuron` is what makes repeated training reuse existing nodes instead of duplicating structure. `build-structure` returns this final `active` frontier — the candidate "meaning" neurons that the Processing component consumes (`associate` wires them to an output root during teaching; `respond` fires them during inference).

Each `neuron` has an `axon` (list of outgoing `dendrite`s) and at most one designated `extender` neuron, which is also present in the axon list. `connects-to` and `find-connecting-neuron` walk the axon to detect existing edges — and skip `:association` edges, so the cross-component links added by `processing` never interfere with input structure-building.

**Activation slots:** `neuron`'s `current-value` and `threshold` (both `single-float`) and `dendrite`'s `weight` belong to the Processing side, not structure-building. `associate` sets association `weight`s; `respond` spreads them into roots' `current-value` and fires the one whose value exceeds its `threshold`; `learn` homeostatically adapts `threshold` (it drifts toward a small fraction of a root's activation). The input algorithm never touches these. (`*next-neuron-id*` remains debug-only — see below.)

## Conventions

- `data-structures`, `line-input`, `input`, `output`, `processing`, and `persist` are each their own package and use `(provide ...)` / `(require ...)` so they can be loaded individually (in dependency order). `ai.lisp` is the top-level entry point: it `require`s and `use-package`s all six but has no `defpackage`/`in-package`/`provide` of its own, so its symbols (`main` and its helpers) land in the current package (see the run note above).
- New exported symbols must be added to the `:export` list of the relevant `defpackage`.
- Package names and exported symbols are written as case-sensitive strings (e.g. `(defpackage "data-structures" …)`, `"NEURON-AXON"`), so the exact casing matters — export strings are uppercase, package names lowercase. Because the package names are lowercase, you can't reach their symbols with the reader's `pkg:sym` syntax (the reader upcases `pkg` to `DATA-STRUCTURES`, which doesn't exist) — use string designators (`(use-package "data-structures")`) or `(find-symbol "RESET" "data-structures")`. `ai.lisp` does exactly this to `shadowing-import` `reset` before `use-package`, because SBCL's `CL-USER` inherits `SB-PROFILE:RESET` and would otherwise raise a name conflict.
- `*next-neuron-id*` / a neuron's `id` are not part of the algorithm (only `dump-neuron`'s debug output uses the id for display) — but `persist` reuses each neuron's `id` as its serialization key, so ids must stay unique.
