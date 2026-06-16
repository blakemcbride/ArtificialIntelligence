# A transformer from Hebbian parts — design memo

*What an LLM does that this system didn't, and how far local (Hebbian, no-backprop) learning
can rebuild it. A record of the experiment arc and the one frontier that remains.*

This memo summarizes a series of proof-of-concept experiments (the `src/*-experiment.lisp`
files) and one productized module (`src/induction.lisp`). It is design context, not a
specification — the live system is described in `CLAUDE.md` and `Plan.md`.

---

## 1. The question

The project has two non-negotiable goals (see `Plan.md` Part 1):

1. **Continual learning** — no separate, frozen pre-training phase; learning never stops.
2. **Hebbian, not backpropagation** — local weight changes (biologically plausible, cheap),
   no global backward pass.

The recurring question was: *what would it take to make this work like a modern LLM?* After
setting aside training data/scale, the honest answer was a single architectural thing the
system lacked:

> **A deep, compositional, context-dependent representation** — what a transformer builds by
> stacking self-attention many layers deep. An LLM recomputes every token's meaning from the
> whole context, recursively; this system's "meaning" was shallow (word subsets + bag
> co-occurrence + one attention head) and modular (a separate hand-built mechanism per
> capability).

The catch: depth is normally what **backpropagation** buys. So the real question narrowed to:
**can the transformer's key behaviors be rebuilt with *local* learning instead?**

The arc below answers "yes" for each ingredient in isolation, and isolates the one part that
remains open: composing them at depth with a local rule.

---

## 2. The build-up (each step a PoC)

| Step | File | What it shows | Honest limit |
|---|---|---|---|
| Attention head | `attention.lisp` (live) | One attention head as Hebbian fast weights (outer-product bind, role-query retrieve) — copy/binding | single, shallow |
| Distributed reps | `vectors.lisp` (live) | Meaning as geometry; similar things cluster; graceful generalization | not compositional |
| **Stacking** | `attention-stack-experiment.lisp` | Composing heads stacks: *k* heads follow a relation *k* hops (transitive containment), reaching answers one head can't. A **cleanup nonlinearity** between heads keeps deep stacks sharp. | same matrix each layer |
| **In-context learning** | `induction-head-experiment.lisp` → `induction.lisp` (live) | A two-layer **induction head** (previous-token head → cleanup → induction head) continues a prompt's pattern with **no training** — the interpretability circuit behind in-context learning, in local fast weights. Wired into `respond` as `continue …`. | layer roles **hand-wired** |
| **A head learns its role** | `learned-attention-experiment.lisp` | A single head **learns its attention offset** from data by a local reward rule. Same architecture, different data → different function (period-2 → k=1, the previous-token head; period-3 → k=2). | positional only; discrete |
| **Two learned heads, stacked** | `learned-induction-experiment.lisp` | **Both** heads of the induction circuit are discovered from data by local reward; layer 1's role is rewarded only **through** layer 2's prediction — **credit assignment through depth, locally**. Rediscovers `(o1=1, r2=0)`; 100% held-out incl. novel tokens; one-layer scores 0%. | learns a discrete choice from a small menu |
| **Learned content weights** | `learned-qk-attention-experiment.lisp` | A content-based head whose **continuous query-key weight matrix `M`** (score = `qᵀ M k`) is learned by a local reward-modulated Hebbian rule. Discovers `M ≈ identity` ("attend to the same token, copy its successor") — a **general** operation that does in-context learning on **tokens never seen in training**. | one layer; single head |
| **Composition at depth** | `deep-composition-experiment.lisp` | Two layers trained by **local, layer-wise objectives** (no backprop) **compose** into an induction circuit, and **depth is required**: with a position-local value read, a single trained content layer can only return the matched token (0.37), while the 2-layer circuit returns its successor (1.00, incl. novel tokens). Layer 1 learns the previous-token head from its *own* target; layer 2 a continuous content matrix on top. | greedy per-layer (not one unified rule); 2 layers; toy scale |

All of the above use only local rules — outer-product binding and reward-modulated Hebbian
updates — and **no backpropagation**.

---

## 3. The through-line

Each step relaxed one hand-wired assumption, moving toward a real transformer:

- **hand-wired → learned**: from a fixed induction circuit to heads that discover their role.
- **discrete → continuous**: from picking an offset/config to learning a `dim×dim` weight matrix.
- **positional → content-based**: from "attend k back" to "attend to whatever matches" (`qᵀ M k`).
- **single → stacked**: heads compose into multi-hop and into a two-layer induction circuit,
  with credit flowing through depth.
- **memorized → general**: the learned content head generalizes to unseen tokens (in-context),
  because it learned an *operation*, not facts.

The cleanup step that recurs (decode → re-encode between layers) is the local-learning echo of
**why transformer layers interleave a nonlinearity**: without it, stacked linear maps collapse.

---

## 4. Composition at depth — first PoC, and what still remains

**Depth composition under local learning now has a working PoC**
(`deep-composition-experiment.lisp`): a two-layer induction circuit trained by **local,
layer-wise objectives** (no backprop) — layer 1 learns the previous-token head from its own
target, layer 2 a continuous content matrix on top — that **cooperate**, generalize to novel
tokens, and **require depth** (a single trained content layer fails). Combined with the earlier
steps, every transformer behavior now has a local-learning realization.

What still remains is **depth at scale under one rule**:

> Stack **many** cooperating layers (not just two), **multi-head**, trained by a **single
> unified local rule** (rather than a hand-chosen objective per layer), with content-based
> continuous weights throughout — at a scale where competence, not just mechanism, shows.

The current PoC is *greedy / layer-wise* (each layer given its own local objective) and only two
layers deep. Turning that into many layers trained jointly by one local rule is the genuine
research step. The established backprop-free approaches aim exactly here:

- **Predictive coding** (local error units, Hebbian-like updates that approximate the gradient),
- **Forward-Forward** (Hinton 2022 — a local goodness objective per layer, no backward pass),
- **Equilibrium propagation** (energy-based, local updates).

None match a backprop-trained transformer yet, but they are the route consistent with the two
goals: local, continual, no global gradient.

---

## 5. Honest caveats

- These are **toy-scale** PoCs: small vocabularies, short sequences, hundreds of examples.
- They demonstrate **mechanisms**, not quality — nothing here approaches an LLM's competence.
- The live system still relies on its complementary hand-built components (concept graph,
  generation, relations); the attention/transformer work is exploratory and, except for the
  induction head (`induction.lisp`), lives in `*-experiment.lisp` standalone files.
- The point is **existence**: each transformer behavior *can* be obtained with local learning.
  The remaining question is whether they can be made to **cooperate at depth and scale** that
  way — which is open.

---

## 6. Where it stands, in one line

> From "can a Hebbian net even learn 'is a'?" to a **two-layer attention circuit composed at
> depth under local learning** — the whole transformer skeleton rebuilt from local parts, with
> only *depth at scale under one unified rule* left to solve.

*Files:* `src/attention-stack-experiment.lisp`, `src/induction-head-experiment.lisp`,
`src/induction.lisp`, `src/learned-attention-experiment.lisp`,
`src/learned-induction-experiment.lisp`, `src/learned-qk-attention-experiment.lisp`,
`src/deep-composition-experiment.lisp`.
*See also:* `Plan.md` Phase 10, `CLAUDE.md` (component map), `notes/BlockDiagram.tex`.
