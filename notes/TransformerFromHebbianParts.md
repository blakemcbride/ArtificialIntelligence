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
| **One unified rule at depth** | `forward-forward-experiment.lisp` | **Forward-Forward**: the *same* local goodness objective at **every** layer, **no backprop, no weight transport**. Trains nets to depth 5; deepest representation stays discriminative; all depths beat a linear model on a nonlinear task. | depth doesn't *improve* a toy task (FF finicky); not yet applied to attention |
| **Uniting them (the frontier)** | `ff-attention-experiment.lisp` | Deep self-attention trained **entirely** by Forward-Forward (one local rule, no backprop; closed-form local update `x_q ⊗ w`). **Above chance but not competent** (≈0.30 vs 0.17). Maps *why* it's hard: FF's activity-goodness is a poor objective for context-dependent prediction; the reward rule has a good objective but credit-through-depth is unsolved. | **the open research problem** — backprop-free transformer training is unsolved |
| **Credit through depth, locally** | `predictive-coding-experiment.lisp` | **Predictive coding** (local value + error nodes; settle latents, then a local Hebbian weight update `e ⊗ pre`). **Validated**: its local update reproduces the **backprop gradient** through a 3-weight-layer net (cosine **0.98 / 0.99 / 1.00**, numerically checked), and trained by PC *alone* it solves continuous-XOR (linear ≈ chance, 1-hidden ≈ 0.92, 2-hidden ≈ 0.95). This is the **missing half** FF lacked: a *good* (prediction/compatibility) objective **with** credit through depth — both local, no backward pass. | MLP toy, not yet attention; PC is finicky (needs bias, settled inference, momentum) |

All of the above use only local rules — outer-product binding, reward-modulated Hebbian
updates, and predictive-coding error settling — and **no backpropagation**.

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

**Depth under *one unified* rule also has a PoC** (`forward-forward-experiment.lisp`): Hinton's
**Forward-Forward** — the *same* local objective (raise a "goodness" on correct-label inputs,
lower it on wrong-label inputs) at **every** layer, **no backward pass, no weight transport**.
It trains nets up to depth 5, the deepest layer's representation stays discriminative, and all
depths beat a linear model on a nonlinear task. *Honest finding:* on a task one hidden layer
already solves, added depth doesn't raise accuracy (naive FF is finicky) — the milestone is the
**rule** (one local objective, arbitrary depth, no backprop), not a depth win on a toy task.

So both halves exist in PoC form: cooperating learned layers
(`deep-composition-experiment.lisp`) and a single unified local rule at depth
(`forward-forward-experiment.lisp`). **Putting them together was attempted**
(`ff-attention-experiment.lisp`): a deep self-attention stack trained **entirely** by
Forward-Forward — one local rule, no backprop, each layer updated by a closed-form local
outer product (the within-layer goodness gradient works out to `x_q ⊗ w`).

**Honest result: above chance but not competent** (best ≈0.30 at depth 3 vs 0.17 chance on
in-context induction). And the *reason* sharpens the frontier into a precise open question:

- **FF's objective is the wrong shape for context-dependent prediction.** Its "goodness" is
  *activity magnitude*; but the correct next token varies per prompt, so "how active the
  appended token makes the net" is a poor signal to maximize.
- **The reward rule (`learned-qk`) has the *right* objective** (it learns induction at 1.00)
  — but **credit-through-depth without backprop is unsolved** for many layers.

So each half works alone, and combining them fails for an understandable reason. The open
problem, now sharply stated:

> Find a **local** learning rule that has a **good (compatibility) objective** *and* assigns
> **credit through depth** — so deep attention can be trained without backprop to real
> competence.

This is genuinely unsolved (backprop-free training of transformers is an open research
problem). The established backprop-free approaches are the route toward it:

- **Predictive coding** (local error units, Hebbian-like updates that approximate the gradient),
- **Forward-Forward** (Hinton 2022 — a local goodness objective per layer, no backward pass),
- **Equilibrium propagation** (energy-based, local updates).

None match a backprop-trained transformer yet, but they are the route consistent with the two
goals: local, continual, no global gradient.

**Progress on the frontier — the predictive-coding route, taken and validated**
(`predictive-coding-experiment.lisp`). Of those three, predictive coding directly supplies the
half FF lacked: a *good* objective (predict the target — the same compatibility objective the
reward rule used to reach 1.00) **together with** credit through depth. We implemented it here
and checked it honestly:

- **It really does credit-through-depth, locally.** PC's purely local weight update
  (`e ⊗ pre`, after settling the latents) matches the **true backprop gradient** layer by
  layer through a 3-weight-layer net — cosine **0.98 / 0.99 / 1.00**, numerically verified. No
  global backward pass, no weight transport across the graph.
- **And it trains depth.** By PC alone, continuous-XOR (a checkerboard a linear model cannot
  separate) is solved — linear ≈ chance, 1-hidden ≈ 0.92, 2-hidden ≈ 0.95.

So the open problem's *first half on a generic net* now has a working, validated local solution
in this codebase. What remains — and what makes the transformer case still genuinely open — is
applying this same local credit assignment **to the attention stack** and beating FF's 0.30 on
the in-context induction task. Honest caveats: PC is a toy here (MLP, not attention) and is
finicky in practice (needs biases, enough inference settling, and momentum to train stably).
**The next experiment is `pc-attention`: PC over the 2-layer attention induction circuit.**

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

> From "can a Hebbian net even learn 'is a'?" to every transformer ingredient rebuilt with
> local, no-backprop learning — and a direct attempt to unite them (deep attention trained
> entirely by Forward-Forward) that lands **above chance but not competent**, sharpening the
> one open problem to: *a local rule with a good (compatibility) objective that also assigns
> credit through depth*. The **predictive-coding** route now supplies that half on a generic
> net — its local update matches the backprop gradient through depth (cosine ≈ 1) and trains
> deep nets — so the remaining open step is narrowed to carrying it into the **attention**
> stack. Precisely mapped, and now actively being closed.

*Files:* `src/attention-stack-experiment.lisp`, `src/induction-head-experiment.lisp`,
`src/induction.lisp`, `src/learned-attention-experiment.lisp`,
`src/learned-induction-experiment.lisp`, `src/learned-qk-attention-experiment.lisp`,
`src/deep-composition-experiment.lisp`, `src/forward-forward-experiment.lisp`,
`src/ff-attention-experiment.lisp`, `src/predictive-coding-experiment.lisp`.
*See also:* `Plan.md` Phase 10, `CLAUDE.md` (component map), `notes/BlockDiagram.tex`.
