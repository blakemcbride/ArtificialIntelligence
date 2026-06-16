# Plan.md

A design document and build plan for turning the existing **input component** into a
complete, continually-learning question→answer system. Written for the project at
its current state (only `src/` Input component built; see `CLAUDE.md` and
`notes/Overview.txt`).

This file has two parts:

1. **The vision** — the problems being solved and the mechanism Blake described.
2. **The build plan** — a concrete, phased path from today's code to a working system.

---

## Part 1 — The Vision

### The two goals (and why)

These are the two design goals the whole project exists to pursue. Everything in the
build plan below is in service of them.

**Goal 1 — Continual learning, with no separate, nonvolatile pre-training phase.**
An LLM's knowledge is frozen at training time. Training and inference are separate
phases: you build the model against a fixed dataset, then deploy it read-only. New
experience during use changes nothing. We want a system with **no separate training
phase** — it learns *online*, one interaction at a time, the way a person picks
things up during a conversation. Learning *is* using it.
*Why:* so the system gets **continually smarter** over time, rather than carrying a
static body of knowledge that changes only when someone retrains it.

**Goal 2 — Hebbian learning rather than backpropagation.**
LLMs adjust their weights with backpropagation: a global error signal is propagated
backward through every layer to compute a gradient for every weight. We instead use
local rules where a connection changes based only on the activity of the two neurons
it joins, plus a coarse "that worked / that didn't" signal — strengthen what succeeds,
let what goes unused slowly fade. There are **two reasons**, and neither is a claim that
Hebbian learning is the better learning algorithm:

- **(a) Biological plausibility.** Backpropagation is, as a pure optimizer, definitely
  *better* than Hebbian learning — but the brain almost certainly is **not** using it
  (there is no global backward pass and no exact gradient transport in neural tissue). It
  far more likely uses something local, like Hebbian learning. The goal is a brain-like
  mechanism, not the strongest optimizer.
- **(b) Simplicity and computational cost.** Hebbian learning is **far, far** simpler and
  far less computationally expensive than backpropagation — a handful of local weight
  updates per fact, with no backward pass over the entire network.

### The mechanism Blake described (the heart of the design)

Processing one sentence with the existing input component ends with a set of *active*
neurons, each representing one possible meaning of the sentence (one order-preserving
subset of its words — see Part 1 of the architecture below). The intended learning
loop is:

1. Start **blank** — empty `*dictionary*`, no associations.
2. Present an **input** sentence. The system builds its structure → a set of
   candidate "meaning" neurons. With nothing learned yet, it has **no idea how to
   respond**.
3. A teacher provides the **response** sentence.
4. The system **maps out the response just as it did the input, only in reverse** —
   it builds a structure for the response, but one arranged to *generate* words rather
   than *absorb* them.
5. **Every candidate ending input neuron is wired to the output series** for that
   response. The meaning(s) of the input now point at the words of the answer.
6. As more input/output pairs arrive, the connections behind **successful** responses
   **strengthen**, and connections that go **unused slowly weaken** (their weight
   diminishes a small amount each time). Over many interactions the useful pathways
   consolidate and the noise decays away. That consolidation *is* the learning.

This is, in modern terms, a **reward-modulated Hebbian associative memory** with
**synaptic decay** — see "The learning rule" below. It needs no epochs, no replay
buffer, and no backward pass.

---

## Part 2 — Where we are today (the starting point)

The whole plan builds on the existing input component, so the relevant facts (all
verified against `src/`):

- **`build-structure` (`src/ai.lisp`)** walks a sentence's word-neurons and maintains
  an `active` frontier. For each new word it (a) seeds the next frontier with the
  word's *extender*, and (b) for every prior frontier neuron `pn`, adds `pn`'s
  extender (the "skip this word" / missing-word path) and a joining neuron shared by
  `pn` and the new word (reused via `find-connecting-neuron`, created otherwise). The
  result is a network covering the full sentence **and every order-preserving subset**,
  so retrieval is tolerant of inexact wording.
- **The candidate "meaning" neurons we need are exactly the final value of `active`.**
  ⚠️ **But `build-structure` currently returns `nil`** — its body is just the outer
  `dolist`. The first task in Phase 0 is to make it return `active`.
- **`find-connecting-neuron` is the retrieval key.** Because identical or overlapping
  inputs land on the *same* shared neurons, an association attached to a neuron is
  automatically recalled the next time an overlapping sentence is presented. Partial
  overlap → partial activation → similarity-based (content-addressable) recall. This is
  what makes the associative scheme generalize at all.
- **Two neuron slots are already reserved for activation but unused:**
  `current-value` and `threshold` (`src/data-structures.lisp`). They are the natural
  home for spreading activation and firing. (They are `fixnum` today and will need to
  become floats — see Phase 0.)
- **`dendrite` already carries a `weight` (single-float, currently always `0.0`).**
  This is where association strengths live.
- Conventions to preserve: one package per file, `(provide …)`/`(require …)`, add new
  exported symbols to the package `:export` list, package names are lowercase strings
  and exported symbols are uppercase strings.

What is **missing** and must be built: the **Output** component (generation), the
**Processing/association** bridge (the middle), the **activation engine** (inference),
the **learning rule** (reinforce + decay), an **interactive teaching loop**, and
**persistence** (so learning survives across runs — essential for "continual" to mean
anything).

---

## Part 3 — Design of the missing pieces

### 3.1 Output component — "the reverse of the input"

The input is **convergent**: many words fan *in* to a meaning neuron. The output is the
mirror image — **divergent**: one idea neuron fans *out* into a word sequence.

Realize it with the primitive already in the code — the **extender chain** — but
traversed *forward* to generate:

```
[output root] --extender--> [w1] --extender--> [w2] --extender--> [w3] --extender--> nil
                            "hi"               "there"            "."
```

- The **root** is a plain `neuron`. Firing it triggers generation.
- Each subsequent node is a `named-neuron` whose `name` is the word to emit.
- **Producing the response** = start at the root, follow `neuron-extender` until `nil`,
  emitting each `named-neuron-name`. (Input *reads* a chain of word-neurons; output
  *writes* one. Same structure, opposite direction — exactly "in reverse.")

Identical responses should share one chain. Phase 1 keys them by the full response
string in a `*responses*` registry (root ← response string); a later phase can give
output the same structural reuse the input enjoys.

### 3.2 Processing component — the association bridge

After step 4, we hold a set of input **meaning** neurons `I = {i₁ … iₖ}` (the returned
`active`) and one output **root** neuron `O`. Wiring them is the core operation:

> For each iⱼ ∈ I, add an **association dendrite** iⱼ → O.

Association dendrites live in the same `axon` list as structural ones but are **tagged
differently** (Phase 0 adds a `kind` to `dendrite`: `:sequence`, `:extender`,
`:association`). The distinction matters because **structural edges are permanent
memory and must not decay**, while **association weights are what reinforcement and
decay act on.**

### 3.3 Activation engine — inference (responding)

Discrete time steps, one event per step (matching the existing "one word per interval"
model):

1. Present the input sentence; build/refresh its structure; collect the returned
   `active` meaning neurons and **fire** them (set them active).
2. **Spread activation** across each fired neuron's `:association` dendrites, adding
   `weight` into the target root's `current-value`.
3. **Select**: among output roots whose `current-value ≥ threshold`, take the winner
   (**winner-take-all**; optional lateral inhibition between competing roots).
4. If a winner fires → **generate** its sentence via 3.1. If none clears threshold →
   the system **"has no idea how to respond"** → ask the teacher (this *is* the
   blank-slate state from the vision, and the trigger for learning).
5. Reset `current-value`s for the next turn.

Because partially-overlapping inputs share meaning neurons, a never-before-seen
sentence that overlaps a learned one will still push activation toward the right root —
graceful, similarity-based generalization.

### 3.4 The learning rule — reward-modulated Hebbian + decay

A **three-factor / local** rule. Each association synapse iⱼ→O updates from only:
pre-neuron activity, post-neuron activity, and a global scalar reward (the teacher
signal) — never a back-propagated gradient.

```
Δw(i→O) = η · pre(i) · post(O) · r     − λ · w(i→O)
          └─── Hebbian, gated by ───┘     └─ decay ─┘
               reward r
```

- **Hebbian term** — "fire together, wire together." When input meaning `i` and output
  root `O` are co-active, their link strengthens. `r` is the third factor (a coarse
  neuromodulatory "reward," like dopamine): `r > 0` when the produced answer matched the
  teacher's answer (or when the teacher supplies a fresh correct pair), `r ≤ 0` when it
  was wrong. This is the brain-plausible alternative to backprop.
- **Decay term** — every association weight is multiplied down (or has a small constant
  subtracted) **each learning step**, so links that are never reinforced fade. This is
  Blake's "inactive neurons' strength diminishes a small amount." Decay also doubles as
  **capacity control**: synapses that fall below ε are pruned, and roots/neurons left
  with no associations can be garbage-collected, keeping an always-growing network
  bounded.

Suggested starting constants (all tunable): initial weight on first co-activation
`w₀ ≈ 0.5`; learning rate `η ≈ 0.2`; decay `λ ≈ 0.005`/step; prune threshold
`ε ≈ 0.01`. Thresholds should be **homeostatic** (each output root adapts its
`threshold` toward its typical input drive) so that inputs which fire many vs. few
meaning neurons are compared fairly.

### 3.5 Generalization / abstraction — the "say" goal, via a Hebbian concept graph

The deeper goal (Blake, refining `notes/Overview.txt`): **not** a relay that copies input,
but **concept formation**. From "dogs walk on their legs", "people walk on their legs",
"goats walk on their legs", tease out the *general idea* — the subject is a legged, walking,
animate thing — that includes a novel "horse" but excludes rulers, bottles, and **even
snakes** (animate, but legless). And it must be **pure Hebbian**: no slots, no variables,
no rules — the idea must *be* connection strengths.

**Why the literal associative scheme can't do it (measured, not assumed).** A probe of the
Phases 0–6 system showed its only generalization is *surface* — shared word-subsequences.
It answers "yes" to "do horses walk on their legs" for the *same* reason it answers "yes"
to "do blickets walk on their legs": both share the "walk on their legs" frame, wired to
"yes". Similar subjects converge *only at the output answer* (shared-neuron overlap was 2
for every pair, including dog/snake); there is no intermediate concept neuron that dog/goat/
horse share and snake/blicket lack. So meaning-based inclusion/exclusion is impossible — not
a data-volume problem, a structural one.

**The mechanism (pure Hebbian, emergent — no slots).** A **concept graph**: each concept is
a neuron; words are the shared `*dictionary*` neurons; a *state* is a (predicate, answer)
that an input→output relationship co-activated — `has-legs:yes` (and `has-legs:no` is a
*different* node, so a snake's leglessness is in the data, not inferred). A taught
relationship Hebbianly strengthens the undirected edge subject ↔ state. Two subjects become
similar by **sharing state neighbours** — that shared-neighbour structure *is* the general
idea ("the system learns general ideas through shared neurons" — Blake). Generalization is
**degree-normalized spreading activation**: how strongly a subject's activation reaches a
(predicate:answer) state. A novel word reaches it through the states it shares with the
taught members; a dissimilar word does not.

Two ingredients make exclusion crisp, and both are faithful, not cheats:
- **Answer polarity is part of the state** (`has-legs:yes` ≠ `has-legs:no`), so being *asked*
  about legs doesn't make a snake look legged.
- **Degree normalization** dilutes promiscuous hubs (`is-animal:yes`, shared by everything)
  so discriminating states (`has-legs:yes`) decide — which is what excludes "even snakes".

**It works** (`concepts.lisp`; reproduced standalone in `concept-graph-experiment.lisp`).
Trained on dogs/goats/people walk-on-legs=yes, snake = legless slithering animal, ruler =
legless tool, and a horse taught *only* has-legs=yes + is-animal=yes (never the walk
question):

      do X walk on their legs?      strength   (threshold 0.04)
        dog/goat/person  0.2753     --> yes  (recall)
        horse            0.0668     --> yes  (GENERALIZED — never taught it)
        snake            0.0226     excluded
        ruler            0.0013     excluded
        blicket          0.0000     excluded

Horse generalizes in *because of its properties* (shares has-legs:yes and is-animal:yes with
the exemplars); snake/ruler/blicket are excluded — the generalization-with-exclusion Blake
asked for, from connection strengths and spreading activation alone.

**Now built (see Phase 7 below):** the graph is auto-populated slot-free from `learn` (every
input word related to its complement-frame + answer; queries try every word-as-subject
decomposition); it persists; and the threshold is adaptive (relative to each category's
members). Trained on a 105-fact starter set, horses/birds generalize "walk on their legs"
while snakes/fish/worms/vehicles/furniture/objects are excluded.

**Honest limits / open frontier:** the near-miss margin depends on the corpus having *enough
discriminating traits* — with legs as the only discriminator snakes leaked in by a hair;
adding feet/running separated them (~2×). Robust discrimination on large, noisy corpora and
contrastive (negative-evidence) weighting remain open, as does folding concept-graph
generalization into the interactive `respond` loop. This matches Blake's expectation that
"these shortcomings and the answers to them would evolve with the use of the system."

### 3.6 Attention / binding — the copy capability (Hebbian fast weights)

The concept graph generalizes *categories* but cannot **copy a specific filler**. The
original FIRST GOAL (`notes/Overview.txt`) — `say X → X` for an unseen X — needs *binding*,
and that is exactly what attention provides.

Key fact: **attention is a Hebbian associative memory.** Linear attention,
`out = q · (Σ kᵢ⊗vᵢ)`, is a query against an outer-product ("fast weight") matrix — pure
Hebbian, no backprop. (Softmax attention ≈ a modern Hopfield network; same family.)

`attention.lisp` builds this: each token and each position gets a deterministic unit
vector; a sentence is bound as `M = Σ roleᵢ ⊗ tokenᵢ`; retrieving the token at a role `r` is
`r · M`, decoded to the nearest token — attention with keys = roles, values = token vectors.
What is *learned* (slow, Hebbian, in `*copy-cues*`): that a cue word triggers "copy the next
word" — strengthened whenever a one-word output equals the word right after an input word.
Once a cue clears `*copy-threshold*`, `respond` fires `copy-response` (an induction head)
before falling back to associative spreading.

**It works** — taught only `say dog→dog`, `say cat→cat`, `say house→house`, the system
answers `say car → car` (car never seen), plus `say elephant`, `say xylophone`, even
nonsense `say grobnar`. The filler is routed *by reference*, so it generalizes to any word
— which a lookup, the associative net, and the concept graph all cannot. It copies by
reference, not by rote repetition, so it is not the trivial relay rejected earlier; and it
complements the concept graph (binding vs. categories). Threshold-gated so it never hijacks
ordinary responses. Standalone proof-of-concept in `src/attention-experiment.lisp`.

Open: multi-word copies and other roles/offsets; learning the cue more distributionally;
and giving the input network distributed (vector) token codes throughout, not only in the
attention head.

### 3.7 Conversation memory and reply composition (Future.md items)

Two user-facing additions from `Future.md`, both built on the existing pieces.

**Conversation memory (follow-ups).** A short follow-up leans on the previous turn: after
"do dogs have legs?", "and cats?" becomes "do cats have legs?".  The fragment's content word
replaces the previous sentence's most concept-similar word (`concept-similarity` = shared
concept-graph neighbours), so *which* word to swap is decided by the learned graph, not a
grammar rule.  The previous turn is held in `*last-turn*`; `resolve-followup` rewrites the
fragment, `ask` answers the result via `infer-answer` (the accurate concept-graph path) and
remembers it, and `main` does the same each turn.  A follow-up is recognized by a small
surface cue (a single word, or a leading "and"/"or"/"what about"/"how about").

**Template / fragment composition.** So replies aren't canned.  When a taught multi-word
answer reuses an input word, `note-template` records `input-minus-that-word` (a frame) ->
`output-with-that-word-as-:slot`.  The genuine slot recurs across examples ("what is a dog"
-> "a dog is an animal", ...cat...), so its template accumulates strength while coincidental
reuses (function words) scatter across frames and stay weak.  Once a template clears
`*compose-threshold*`, `compose` fills its slot with the matching input word (copy by
reference), so `respond` (after the copy head, before associative recall) answers "what is a
horse?" with "a horse is an animal" — a sentence never seen verbatim.  `*templates*` persists
and resets with the rest of the system.

Open: pronoun / coreference resolution (only ellipsis is handled today); multi-slot and
multi-word-filler templates; composing from several fragments rather than one template.

### 3.8 Learned operations and distributed concept vectors

Two further additions, aimed at *understanding* (a computed answer) and at *non-brittleness*
(graceful interpolation), within the two constraints (continual + Hebbian, no backprop).

**Learned operations** (`operations.lisp`).  A fact is stored; an *operation* is computed
over the current knowledge, so its answer changes as the system learns ("how many animals
do you know" rises when you teach a new animal).  The design is a small set of general
primitives (`count`, `similar`) plus a **learned** mapping from a question phrasing to
(operation + category slot): teach "how many animals do you know → count animals" once and
it generalizes to any category via the slot.  Membership is found **generically** — the
positive "are/is _ C" frame that the most subjects share (the shared frame accumulates
members; idiosyncratic decompositions don't), so it works for any category, not a fixed
list.  This is *not* a per-question function; the count primitive is reused.  Honest line:
a small primitive substrate is innate (like the brain's), the language that triggers it is
learned.

**Distributed concept vectors** (`vectors.lisp`) — the CMAC / sparse-distributed-memory /
hyperdimensional direction.  Non-brittleness comes from distributed, continuous
representations: a novel thing lands *near* known ones and interpolates instead of falling
off the symbolic cliff.  Each concept is a high-dimensional vector built online by
superposing the random codes of the words it co-occurs with (Hebbian accumulation, IDF-
weighted, mean-centered) — a few vector adds per fact, no backprop.  Similarity becomes
geometry: dog↔cat and red↔blue clusters *emerge* from shared company, and a word from one
fact generalizes by proximity.  **Empirical head-to-head:** the vectors win decisively at
similarity and graceful generalization (used now for follow-up resolution and a `similar`
operation), but they do *not* give crisp counts — category labels don't sit near their
members, and there is no clean membership boundary (similarity decays continuously).  So the
discrete concept graph stays the home of crisp set-operations while the vectors provide
non-brittle similarity: **the two are complementary**, which is the real finding.

*Pushed further.*  Non-brittle membership **recognition** now works (`recognized-member-p`,
k-NN over the vectors with naive number-normalisation): a novel word taught a few traits is
recognized as a member by resemblance — *zebus*, never told "are zebus animals," is
recognized as an animal and rejected as a vehicle — while framed members stay crisp.  What
did **not** pan out (tried and reverted, an honest record): binding answer-polarity into the
codes, and idf² weighting, both failed to fix the one blemish — legless animals (shark,
snake) drift toward birds — and some hurt other pairs.  That drift is **data sparsity** (few
fish facts, a dense bird cluster acting as an attractor), not a representation bug; the cure
is more/cleaner data, not a weighting trick.  A non-brittle *count* also stays out of reach
(no clean boundary in the space → it over-counts), so **counting stays on the crisp concept
graph** while the vectors do similarity and recognition.  Open: richer learned embeddings;
consistent number (singular/plural) in the knowledge base.

**Learning from raw text** (`read-text` / `read-text-file`).  A first step toward
self-supervised learning: each sentence feeds the distributed-vector co-occurrence
(*unsupervised* similarity — read about a new thing and it clusters with its kin, no
teacher), and simple declarative patterns also teach facts — "X is the Y of Z" → a
relational fact ("Kigali is the capital of Rwanda" → answers "what is the capital of
Rwanda"), "X is a Y" → membership, "X is the largest Y" → a superlative fact ("what is the
largest planet" → Jupiter), and "X was a ROLE" → "who was X" (so past-tense history works);
questions are skipped.  Pattern extraction is deliberately light (regular sentences
only); turning arbitrary prose into facts is the open, hard problem (and heavy parsing drifts
back toward the symbolic style).  But similarity from raw text is free and continual, exactly
in the project's spirit.

A ~690-sentence corpus `src/prose.txt` (history, world geography, and science, written as
plain sentences) ships with the system.  `(read-text-file "prose.txt")` grows the KB *from prose*: it teaches new
capitals (Reykjavik → Iceland), people ("who was Ada Lovelace" → a mathematician), and
relational facts ("what is the birthplace of democracy" → Greece) that the curated KB never
contained, and the new entities cluster by similarity (Iceland's nearest neighbours come out
as other European countries).  This is the "feed it a body of prose" demonstration.

---

## Part 4 — The build plan (phased, each phase independently testable)

Each phase lists its **goal**, **changes** (with concrete code touchpoints), and a
**done-when** check. There is no test harness today; Phase 0 adds a minimal one so the
continual-learning behavior can be regression-checked.

### Phase 0 — Foundations & cleanup  ✅ done
- [x] **Return the meaning neurons:** `build-structure` now returns `active`
      (`src/ai.lisp`).
- [x] **Float activation:** `neuron` `current-value` and `threshold` are now
      `single-float` (`src/data-structures.lisp`); `dendrite-weight` was already a float.
- [x] **Tag edges:** `dendrite` has a `kind` slot (`:sequence` | `:extender` |
      `:association`); `connect`/`new-next-neuron` set it.
- [x] **Registries:** `*output-roots*` and `*associations*` globals added and cleared by
      `reset` (empty until later phases populate them).
- [x] **Minimal test harness:** `src/tests.lisp` (16 assertions) plus a `make test`
      target; documented in `CLAUDE.md`.
- **Done when:** ✓ `build-structure` returns a non-nil set; `dump-dictionary` still
      works; the suite is green.
- **Also fixed along the way (ANSI / portability cleanup that the above surfaced):**
  - `neuron`'s `id` slot used `:read-only` with no value — invalid ANSI (odd-length
    slot-option plist). CLISP tolerated it; SBCL/CCL/ECL rejected it. Now `:read-only t`.
  - `ai.lisp`'s `(use-package "data-structures")` collided with `SB-PROFILE:RESET`
    (inherited by SBCL's `CL-USER`); it now `shadowing-import`s `reset` first (via
    `find-symbol`, since the lowercase package name can't be named with `pkg:sym` syntax).
  - The suite loads its components by pathname, so it passes on **CLISP, SBCL, CCL, and
    ECL** (`16 run, 0 failed` on each). A one-call loader (`load.lisp` / `(load-system)`)
    loads everything by pathname on any implementation; `(load "ai.lisp")` works directly
    only under CLISP — see `CLAUDE.md` › "Running the code".

### Phase 1 — Output component (generation)  ✅ done
- [x] `src/output.lisp` (package `output`): `build-output-structure(words)` builds (or
      reuses) a divergent extender chain `root → w1 → w2 → …` and returns its **root**;
      `produce-output(root)` walks `neuron-extender` and returns the emitted word list;
      `output-sentence(root)` joins them. Chains are interned by response text in
      `*responses*` (added to `data-structures`, cleared by `reset`); every root is
      registered in `*output-roots*`. Output nodes are fresh neurons, distinct from the
      input word neurons in `*dictionary*`, so the two networks never share a chain.
- **Done when:** ✓ building a response then `produce-output` reproduces it verbatim —
      e.g. `("hello" "world" ".")` round-trips and an identical response reuses the same
      root. 14 new assertions; the suite is now 30, green on CLISP / SBCL / CCL / ECL.

### Phase 2 — Association bridge + supervised wiring  ✅ done
- [x] `src/processing.lisp` (package `processing`): `associate(input-endings,
      output-root)` creates an `:association` dendrite from each (distinct) ending to the
      root at weight `w₀ = *assoc-initial-weight*` (0.5), or strengthens an existing one
      by `*assoc-strengthen*` (0.2, clamped to `*assoc-max-weight*` = 1.0). New
      associations are registered in `*associations*` for later decay/pruning.
- [x] **Structural walkers now skip `:association` edges** (the Phase 0 deferral, now
      due): `connects-to` and `find-connecting-neuron` (`ai.lisp`) ignore them, so
      re-running `build-structure` once associations exist can't mistake an output root
      for a structural join. Without this, inference (Phase 3) would corrupt the network.
- **Done when:** ✓ after one (input, output) pair, association dendrites exist from every
      input ending to the correct root (demo: "say hi" → 3 endings → 3 associations at
      0.5; re-training → 0.7). 11 new assertions; suite now 41, green on CLISP/SBCL/CCL/ECL.

### Phase 3 — Inference (responding)  ✅ done
- [x] `processing:respond(input-words)`: build the input structure → spread each
      meaning neuron's association weights into the output roots' `current-value` →
      winner-take-all (highest `current-value` strictly above its `threshold`) →
      `produce-output`, or `nil` ("I don't know") when nothing is activated.
- [x] **Refactor (the deliberate decision flagged in Phase 2):** the input algorithm
      (`build-structure`, `connect`, `connects-to`, `find-connecting-neuron`) moved out
      of `ai.lisp` into a new `input` package (`src/input.lisp`), so both `main` and
      `processing` can depend on it with no circular `require`. `ai.lisp` keeps only
      `main` and stays package-less, so the documented `(main)` call is unchanged.
      Word-string interning was factored into `line-input:intern-word`/`intern-words`.
- **Done when:** ✓ teach (in→out), then `in` reproduces `out`; a subset of a taught
      input retrieves its response (demo: teach "say hi"→"hi there"; `respond '("hi")`
      → "hi there"); an unrelated input yields `nil`. 6 new assertions; suite now 47,
      green on CLISP/SBCL/CCL/ECL.

### Phase 4 — Reinforcement & decay (the continual-learning dynamics)  ✅ done
- [x] `processing:learn(input-words, correct-words)` runs one teacher-confirmed turn of
      the §3.4 rule `Δw = η·pre·post·r − λ·w`: it sees what the system would answer, then
      **reinforces** the correct pathway (`associate`, r>0), **weakens** a wrong guess
      (`weaken`, r<0), runs a global **`decay-associations`** step, and homeostatically
      **adapts** the correct root's threshold. Returns the pre-update guess.
- [x] `decay-associations` shrinks every association by `*assoc-decay*` (λ), **prunes**
      those ≤ `*assoc-prune-threshold*` (ε) from their source axon and `*associations*`
      (via the new `dendrite-from` back-pointer), and **GC's** roots left with no
      incoming associations (dropping them from `*output-roots*`/`*responses*`; their
      chains are then reclaimed by the Lisp GC).
- **Done when:** ✓ repeated correct pairs strengthen and clamp at `*assoc-max-weight*`
      (stabilize); an unused association decays below ε and is pruned (root GC'd, response
      forgotten); a long one-off script stays bounded. Demo: teach "say hi"→"hi there",
      re-teach →"hello" (self-corrects), then a one-off "zonk" is forgotten while the
      reinforced pair survives. 13 new assertions; suite now 60, green on CLISP/SBCL/CCL/ECL.

### Phase 5 — Interactive teaching loop  ✅ done
- [x] `main` (`src/ai.lisp`) is now a conversation: each turn reads an **input**
      sentence, prints what it would answer (`respond`, or "I don't know"), reads a
      **teacher** line — the correct response, or a confirm word (`*confirm-words*`:
      yes/y/right/correct/ok) to accept a correct guess — and `learn`s from it. Ends on
      `quit.`/`exit.` or end-of-input (`getword` now returns nil at EOF instead of
      erroring). Helpers: `words-of`, `quit-line-p`, `confirm-p`.
- **Done when:** ✓ a scripted session improves over its turns. Demo: teach "say hi"→
      "hi there" and "the dog barks"→"woof"; re-presenting "say hi" then answers
      "hi there" (confirmed → reinforced); a novel input says "I don't know" and is
      taught. 3 new assertions (driving `main` over a string-stream script); suite now
      63, green on CLISP/SBCL/CCL/ECL.

### Phase 6 — Persistence (makes "continual" real across runs)  ✅ done
- [x] `src/persist.lisp` (package `persist`): `save-network`/`load-network` serialise
      the whole graph (`*dictionary*`, `*output-roots*`, `*responses*`, `*associations*`,
      every neuron's threshold/current-value/extender/axon, every dendrite's
      weight/kind/from, and the id counter) to one readable s-expression and rebuild it.
      Neurons are written flat, keyed by their unique `id`, with all references as ids,
      then rebuilt in two passes (create, then wire) via an id→neuron map — preserving
      shared structure and the cyclic `from` back-pointers. `main` loads `*save-file*` on
      entry and saves on exit.
- **Done when:** ✓ learn in one process, restart, and the learned responses are still
      produced. Demo: session 1 teaches "say hi"→"hi there" and quits (writes
      `demo-mem.kb`); a separate session-2 process loads it and answers "hi there".
      6 new assertions (save→reset→reload round-trip); suite now 69, green on CLISP/SBCL/CCL/ECL.

### Phase 7 — Generalization via a Hebbian concept graph (the "say" goal)  ✅ done
- [x] `src/concepts.lisp` (package `concepts`) — the validated mechanism (§3.5). `relate`
      Hebbianly wires subject ↔ (predicate:answer) state in `*concept-graph*`;
      `category-strength` / `recognizes-p` do degree-normalized spreading activation.
      Reset-aware and loaded into the system. Concepts are neurons (words shared with
      `*dictionary*`); edges live in a dedicated weighted table so the input network's
      traversal/dump/persist stay untouched.
- [x] **Validated:** a novel "horse" generalizes into "walks-on-legs" (0.067) while snake
      (0.023), ruler (0.001), and a nonsense "blicket" (0.000) are excluded — pure Hebbian,
      no slots. 8 assertions; suite now 77, green on SBCL.
- [x] **Auto-populate, slot-free:** `learn` now calls `note-relationship`, relating every
      input word to its complement-frame + answer; the slot-free `infer-strength`/`infer-p`
      query tries every word-as-subject decomposition and takes the strongest. The concept
      graph grows automatically from taught pairs — no explicit `relate` calls needed.
- [x] **Persisted:** `*concepts*` + `*concept-graph*` are saved/reloaded by
      `save-network`/`load-network` (state neurons collected; weighted edges serialized as
      id triples).
- [x] **Adaptive threshold:** membership is judged relative to each category's own taught
      members (`member-baseline` × `*concept-fraction*`, with a small floor) — no hardcoded
      cutoff.
- [x] **End-to-end on a starter KB:** `train-from-file` imports a training set
      (`input => answer` per line; `#`/`;` comments); `src/generalization-test.txt` (105 facts)
      covers several categories. After import, **horses/birds generalize** "walk on their
      legs" (never taught) while **snakes/fish/worms/cars/tables/rocks are excluded** —
      including "even snakes". `export-kb`/`import-kb` round-trip the whole KB. Suite 104.
- [x] **Broad starter KB + startup:** `src/knowledge-base.txt` (~2,100 facts — greetings, world geography, history,
      animals/traits, colors, opposites, categories, numbers, copy and composition
      examples) exercises every capability; `main` auto-learns it (`*starter-kb*`) on a
      first start with no saved memory, so a fresh system answers, copies, and composes out
      of the box. Suite now **129**, green on SBCL.
- **Scale finding:** robustly excluding near-misses (legless animals still share
  animal/move/breathe with walkers) needs *enough discriminating traits* — faithful to
  "exclusion requires learning the distinctive traits". One discriminator (legs only) let
  snakes leak by a hair; adding feet/running separated them cleanly (~2×).
- **Still open:** contrastive (negative-evidence) weighting for sharper near-miss
  discrimination on noisy corpora; folding concept-graph generalization into the
  interactive `respond` loop (today it is queried via `infer-p` / `infer-strength`).

### Phase 7b — Attention / binding head: the copy capability  ✅ done
- [x] `src/attention.lisp` (package `attention`): a transformer-like attention head built
      as **Hebbian fast weights** (outer-product binding + role-query retrieval; no
      backprop). `learn` learns copy cues (`note-copy` → `*copy-cues*`, persisted); once a
      cue clears `*copy-threshold*`, `respond` fires `copy-response` (an induction head)
      before associative spreading. Words/positions get deterministic vector codes.
- **Done when:** ✓ the original FIRST GOAL — taught only `say dog/cat/house`, the system
      answers `say car → car` (car never seen), generalizing to any filler by routing it
      *by reference*. 6 new tests; suite now **115**, green on SBCL. PoC:
      `attention-experiment.lisp`.  See §3.6.

### Phase 7c — Conversation memory + reply composition  ✅ done
- [x] **Conversation memory** (`ai.lisp`): `resolve-followup` / `ask` fold a follow-up into
      the previous turn (`*last-turn*`) by concept-similarity, answering via `infer-answer`.
- [x] **Template composition** (`attention.lisp`): `note-template` / `compose` learn
      `frame → template-with-:slot` and fill the slot by reference, so `respond` composes
      replies never seen verbatim. Both persist with the network.
- **Done when:** ✓ after "do dogs have legs?", "and cats?" → yes (resolved to "do cats have
      legs"); taught "what is a dog/cat", "what is a horse?" → "a horse is an animal". 9 new
      tests; suite now **124**, green on SBCL.  See §3.7 / `Future.md`.

### Phase 7d — Learned operations + distributed concept vectors  ✅ done
- [x] **Learned operations** (`operations.lisp`): general primitives `count` / `similar` +
      a learned question→operation mapping (`*op-templates*`, persisted).  Taught once,
      "how many animals do you know" counts **generically** (any category, by the shared
      membership frame) and rises as the system learns; "what is similar to X" returns
      related concepts.  `respond` runs an operation before copy/compose/recall.
- [x] **Distributed concept vectors** (`vectors.lisp`): online Hebbian co-occurrence vectors
      (`*cooccur*`, persisted), IDF-weighted + mean-centered; `similarity` / `nearest` now
      drive follow-up resolution and the `similar` operation.  Empirically great for
      similarity / graceful generalization, not for crisp counting — complementary to the
      concept graph (see §3.8).
- [x] **Non-brittle membership recognition** (`recognized-member-p`): framed (crisp) or, for
      a word never explicitly told, k-NN resemblance in the vector space — a novel word
      taught a few traits is recognized as a member, non-members rejected.  (Polarity-binding
      and idf² were tried for the legless-animal drift and reverted — that drift is data
      sparsity, not a bug; see §3.8.)
- **Done when:** ✓ "how many <category> do you know" generic; dog↔cat / red↔blue clusters
      emerge; follow-ups resolve by vector similarity; a novel "zebus" is recognized as an
      animal without ever being told.  13 new tests; suite now **142**, green on SBCL.
      See §3.8 / `Future.md`.

### Phase 8 — Generation (the Output component, for real)  ✅ done
The system so far can **retrieve** an answer (`respond` walks one stored output chain) and
**infer** membership (the concept graph), but it cannot **generate** — it can't answer
"tell me about France" or "why is a cat an animal?" by *assembling a new sentence* from
what it knows. A standalone proof of concept (`src/generation-experiment.lisp`) confirmed
the fix and the trade-offs; Phase 8 promotes it into the live system. Generation is two
pieces, both pure online counting (Hebbian, no backprop, continual — same as every prior
phase):

- **Content selection — *what* to say.** Which facts mention the topic. Reuse the
  knowledge already stored: the **concept graph** (`*concept-graph*`) for relations and
  `*cooccur*` for topical relatedness. No new "knowledge base" — generation reads the same
  structures inference already grows.
- **Surface realization — *how* to say it.** Two new, small substrates (`generation.lisp`,
  globals in `data-structures` so `reset`/persist stay uniform):
  - **`*facts*`** — clean `(subject relation object)` triples, captured at the *same* parse
    points that already exist: `extract-fact`'s declarative patterns (prose via `read-text`)
    and a few high-value question frames from `learn` (`what is the R of S` → `(S R A)`,
    `who was S`, `is S a C`). The concept graph keeps doing inference; `*facts*` holds the
    *declarative* form generation needs, so realization is **grounded in a real edge** (no
    "capital of Latvia" hallucinations).
  - **`*transitions*`** (word → next-word counts) + **`*sentence-starts*`** — the
    genuinely missing sequential model, grown from every sentence read/learned. The PoC
    showed `*cooccur*` alone is order-less; this adds the order, for fluent connective
    tissue and a fallback when no structured fact exists.
- **Entry points** (in `respond`, so `main`/`ask` get them for free):
  - `describe(topic)` → "tell me about X" / "describe X" / "what do you know about X":
    gather `*facts*` triples mentioning X, render each through a per-relation frame (the
    slot-fill behind `compose`), and **aggregate several into one short paragraph** —
    discourse the retrieval path can't produce.
  - `why(input)` → "why is X a Y": confirm via the concept graph that X is a Y, then cite
    the strongest **shared states** (the discriminating traits X shares with Y's members)
    as the reason — "a cat is an animal because it has legs, has fur, …".
- **Stays Hebbian / continual:** every substrate is an incrementally-bumped count grown at
  learn/read time; nothing is trained offline; all of it persists (Phase 6) so generation
  improves continuously as the system reads and is taught.
- **Honest limits (why this is a real research step, not a finish line):** the realizer is
  template + bigram, so coherence is short-range and it can only state what was actually
  read (no inference leaps in the prose itself); quality scales with how much it has read.
  Higher-order context (the input network's order-preserving frontier) and richer frames
  are the follow-on work.
- **Done:** ✓ implemented in `src/generation.lisp` (package `generation`), with the three
  stores (`*facts*`, `*transitions*`, `*sentence-starts*`) in `data-structures`, fed by
  `learn` (QA frames) and `read-text` (declaratives), persisted in `persist.lisp`, and
  wired into `respond` (a recognized request is *owned* by generation — it answers or says
  "I don't know", never falling through to brittle recall). Out of the box (starter KB),
  `(respond "tell me about france")` → "France is a country. It is in europe. Its capital is
  paris."; after `(read-text "A cat is a mammal. A mammal is an animal.")`,
  `(respond "why is a cat an animal")` → "A cat is an animal because a cat is a mammal, and
  a mammal is an animal." 9 new tests (describe, why, QA-derived facts, the transition
  model, persistence); suite now **172**, green on SBCL. The `why` explanation is the is-a
  chain only; trait-based "because" and higher-order realization are follow-on work.

### Phase 9 — Learned relation discovery (un-hardcoding "is a")  ✅ done
The prose reader hardcodes the relationship markers ("is a", "is the Y of", …) and assumes
the subject/category sit in fixed positions — the one rule-based seam in an otherwise
non-rule-based design. Phase 9 shows that seam can be **learned**, by Hebbian counting (no
backprop, continual). Three signals, all just counts (`relations.lisp`, stores in
`data-structures`, persisted):

- **Connectors** — a membership connector ("is a") links many subjects to few category
  hubs (high subjects-per-category); a relational connector / verb is ~one-to-one. The gap
  is the signal; "is"/"a" are never special to the code.
- **Function words** — frequent tokens that are *rarely a head* (the/a/is) — so a frequent
  *content* word like "animal" is not mistaken for glue.
- **Heads** — the real subject/category in a span is the word that most often serves as a
  head, so modifiers ("small brown") and relative-clause words ("that barks") fall away.
  **Iterative bootstrapping** (in the PoC) and **online accumulation** (in the live layer)
  let the model teach itself heads it only ever sees inside complex sentences.

It runs **alongside** the hardcoded patterns, not as a hard replacement: a learned
recognizer needs accumulated evidence (`*rel-min-support*`), so the hardcoded path is the
cold-start floor and the learned layer takes over / extends it as more is read. `read-text`
feeds it (`relations:observe`) and emits learned membership facts into the generation store;
`relation-of` / `membership-connectors` query it.

- **Done:** ✓ `relations.lisp` (package `relations`) productizes
  `relation-discovery-experiment.lisp`; wired into `read-text`, persisted, in `system-stats`.
  On a small batch the learned layer correctly stays below threshold (needs data — the whole
  premise); on `prose.txt` (694 sentences) it discovers `is a` (5.4), `was a` (4.6),
  `is an` (4.3), `was an` (3.0) as **membership** and `is in` / `are mountains in` as
  relational, and parses "the bird that sings sweetly is an animal" → bird / animal. 5 new
  tests; suite now **181**, green on SBCL. Full removal of the hardcoded patterns (so the
  learned layer is the *only* path) is the follow-on, once validated at larger scale.

### Phase 10 — Evaluation & tooling
- [ ] Metrics over a held-out teaching script: response accuracy over time, network
      growth, weight distribution, prune counts. Extend `dump-dictionary` to show
      association weights, roots, and thresholds.
- **Done when:** a single command reports accuracy-over-time and size for a script.

---

## Part 5 — A worked example (end to end)

```
(blank system)
> hello.                 ; build input structure → meaning neurons; no associations
< (I don't know)         ; nothing clears threshold
teacher: hi there.       ; teacher supplies the response
                         ; → build output chain root→"hi"→"there"→"."
                         ; → associate every "hello" ending neuron → that root  (Phase 2, r>0)

> hello.                 ; same input reuses the same ending neurons
< hi there.              ; their associations drive the root over threshold → generate (Phase 3)
                         ; correct guess → reinforce those synapses (Phase 4, r>0)

(many turns later, "hello" associations that kept being used are strong;
 a one-off wrong wiring that never recurred has decayed below ε and been pruned.)
```

And the open challenge (Phase 7): after `say dog→dog`, `say cat→cat`, `say house→house`,
the literal scheme has no link for the unseen `say car`; only a *relational*
("emit the word after 'say'") abstraction can answer `car`.

---

## Part 6 — Cross-cutting concerns, risks, open problems

- **Unbounded growth.** Structure-building + continual learning grows the net forever.
  Mitigation: decay-to-prune on associations, GC of orphaned neurons, and possibly caps
  on subset-neuron fan-out per sentence. *Watch this from Phase 4 on.*
- **Combinatorial subset explosion.** A long sentence already spawns many subset
  neurons. May need to bound subset depth or weight subsets by length.
- **Catastrophic interference.** Local Hebbian updates can overwrite older memories.
  Helpers: the representation is naturally **sparse** (few endings per sentence);
  structural reuse keeps shared sub-phrases stable; consider **metaplasticity** (a
  synapse's learning rate slows as it consolidates).
- **Credit assignment without backprop.** The bet of this project is that local
  reward-modulated Hebbian learning is *enough* for one-step question→answer. Multi-step
  reasoning may strain it; the eligibility/decay mechanics are where to extend if so.
- **Threshold calibration.** Different inputs fire different numbers of endings;
  homeostatic per-root thresholds (and/or normalizing by ending count) keep selection
  fair. Needs empirical tuning (Phase 8).
- **Build-vs-retrieve.** Recommendation: **always build** input structure (cheap,
  monotonic, unsupervised, continual), but only **associate/reinforce** when a teacher
  signal is present (supervised). This cleanly separates "perceiving" from "learning to
  answer."
- **Variable binding (Phase 7)** is genuine open research — expect iteration, not a
  one-shot solution.

---

## Part 7 — Design decisions to confirm (with recommendations)

These are real forks; defaults are picked so building can start without blocking.

1. **Teaching protocol (Phase 5).** *Recommend:* alternating lines — system reads an
   input, answers or says "I don't know," then reads the next line as the teacher's
   correct answer, with a short control token to *confirm* a correct guess vs. *replace*
   a wrong one. (Alternative: explicit `in:` / `out:` prefixes.)
2. **Decay cadence.** *Recommend:* decay all association weights once per teaching turn
   (simple, predictable). (Alternative: decay only on touched neighborhoods — cheaper,
   less uniform.)
3. **Reward signal granularity.** *Recommend:* start with `r ∈ {+1 correct, 0 unknown,
   −small wrong}`. (Alternative: graded reward by overlap between guess and answer.)
4. **Output reuse depth.** *Recommend:* Phase 1 keys whole responses by string; defer
   structural output reuse to a later pass.
5. **File layout.** *Recommend:* `output.lisp`, `processing.lisp`, `persist.lisp`,
   `tests.lisp`, each its own package per the existing convention; keep `data-structures`
   as the shared base and `ai.lisp` as the top-level loop.

---

## Part 8 — How this addresses the two original problems

| Problem | How this design answers it |
|---|---|
| **1. LLMs don't learn in use** | There is **no separate training phase**. Every interaction builds structure and updates association weights *at inference time*. Persistence (Phase 6) carries that learning across sessions. Learning *is* using the system. |
| **2. Backprop is implausible** | No gradients, no backward pass. Connections change by a **local, reward-modulated Hebbian rule** (pre × post × reward) plus **decay** — "fire together, wire together; let the unused fade." Winner-take-all ≈ lateral inhibition; pruning ≈ synaptic elimination; adaptive thresholds ≈ homeostatic plasticity. The whole learning story is local and brain-motivated. |

---

*Author: Blake McBride. Public-domain research project. This plan describes the
intended Processing and Output components and the continual-learning loop that connect
to the already-built Input component in `src/`.*
