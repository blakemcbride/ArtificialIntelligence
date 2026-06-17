# OwnSona — turning a RAG memory into a *learning* memory

*An enhancement plan for [OwnSona](https://github.com/blakemcbride/Ownsona): the steps that move
it from a static retrieval store toward "a memory that genuinely learns," distilled from the
ArtificialIntelligence project's experiments. Ordered by **evidence**, not enthusiasm.*

The goal: a memory that makes an LLM *behave as if it learns* after deployment. The honest ceiling
(below) is that this makes the **memory** learn, not the frozen LLM — but that is the achievable and
worthwhile target, and it is where a learning memory provably beats plain RAG.

---

## 1. What OwnSona is today

- **Backend:** PostgreSQL 16 + `pgvector`; embeddings via an OpenAI-compatible API.
- **Retrieval:** vector-similarity (`recall` / `search_memory`), trigram text match (`text_search`),
  near-duplicate clustering (`find_near_duplicates`).
- **Lifecycle:** a static `importance` score, a `last_confirmed_at` timestamp, `confirm` to refresh
  staleness, soft/hard `forget`, a `keep` protection flag.
- **Access:** OAuth 2.1 per-user; MCP server usable from ChatGPT/Claude/Gemini/Grok.
- **Learning:** none. Per its own docs it is "purely a retrieval-augmented context store."

It already has the *right primitives* (`importance`, `last_confirmed_at`, `confirm`,
`find_near_duplicates`, soft-delete) — but they are **static and manual**. The work is to make them
**dynamic and feedback-driven**.

## 2. The evidence (why these enhancements, in this order)

The ArtificialIntelligence project ran a head-to-head between a learning controller and a fair
RAG memory (`src/controller-vs-rag-experiment.lisp`, four arms, three regimes). The result, honestly
reported:

| Regime | LLM-only | RAG (lexical) | Learning policy |
|---|---|---|---|
| **Recall** (exact contexts recur) | chance | **1.00** | **1.00** (tie) |
| **Generalize** (novel combos, same predictive feature) | chance | **0.85** | **0.85** (tie) |
| **Drift** (the right answer changes) | chance | **~0.5** | **~1.0** (win) |

The lesson: a good RAG already ties a learning memory on **recall** and **generalization**. The
learning memory's **one clean win is non-stationarity** — when the correct answer *changes*, its
reward-modulated **decay** un-learns the stale answer while append-only RAG stays anchored to it.
The win is gated by a **feedback signal** (without one, nothing learns). So the roadmap leads with
feedback + salience/decay, then contradiction handling — the proven differentiators — and treats
generalization features (graph/relations) as optional, since they are *not* where the edge lies.

## 3. Roadmap

### Tier 1 — the prerequisite + the proven win

**1.1 A feedback signal.** OwnSona never learns whether a recalled memory *helped*. Add the loop:
- **Explicit:** a new MCP tool `reinforce(memory_ids, delta)` (or `feedback(ids, helpful)`), called
  by the assistant after using recalled context. Have `recall`/`search_memory` return the memory
  IDs so a later call can credit them.
- **Implicit (free signals already present):** re-retrieval (recalled again → +), `confirm` (→ +),
  a *contradicting* new `remember` (the old memory was wrong → −), explicit user correction.

This is the crux; everything else depends on it.

**1.2 Dynamic salience = reinforcement + decay.** Replace the static `importance` with a *learned*
salience that rises on reward/use/confirm and decays with time/disuse, and **rank by it**. This is
the ArtificialIntelligence project's reward-modulated Hebbian rule, `w ← w(1−λ) + η·r`, applied to
memory rows — the exact mechanism that won the drift regime.

Concretely on Postgres/pgvector:
```
ALTER TABLE memories
  ADD COLUMN salience    double precision DEFAULT 1.0,  -- learned weight (seed from importance)
  ADD COLUMN use_count   integer          DEFAULT 0,
  ADD COLUMN reward_sum  double precision DEFAULT 0.0,
  ADD COLUMN last_used_at timestamptz;
```
- On retrieval-and-helped / `confirm`: `salience ← (1−λ)·salience + η·reward`, bump `use_count`,
  set `last_used_at`.  (e.g. η≈0.3, λ≈0.02 — see the project's `*assoc-*`/`*select-*` constants.)
- Decay lazily at read time or in a nightly job: effective salience = `salience · exp(−λ·age_days)`.
- Rank: `score = cosine_sim · g(salience) · recency` instead of cosine alone; one ORDER BY.
- Prune: soft-delete rows whose decayed salience falls below a floor (respect the `keep` flag).

Tier 1 converts OwnSona from "static notes" into "a memory that learns." Highest ROI, evidence-backed.

### Tier 2 — non-stationarity (the eval's actual win)

**2.1 Automatic contradiction detection & supersession.** Plain RAG *appends* — which is exactly
why it lost the drift regime. On a new `remember`, find high-similarity, same-tag/entity memories;
ask **the in-loop LLM** "do these conflict?"; if yes, **supersede** — down-weight or soft-delete the
stale memory (or set a `superseded_by` column). You already have `update`/`forget` and
`find_near_duplicates`; this automates them on conflict. This is the "un-learn the old answer"
behavior that beat append-only RAG.

### Tier 3 — consolidation (fast → slow; complementary learning systems)

**3.1 A consolidation ("sleep") job.** Periodically: cluster near-duplicates (you have the tool),
and use the **LLM to merge** each cluster into one canonical/higher-level memory, extract
entities/relations, and demote the raw episodes once consolidated. This is the episodic→semantic
consolidation that bounds store growth, removes contradictions, and sharpens retrieval. Let the LLM
do the extraction — it is far better than hand-built NLP.

### Tier 4 — optional, and NOT evidence-backed as wins

- **Learned retrieval/selection policy** (the full propose→select→reinforce controller): learn
  *which* memories to surface per query type from feedback. Most of its benefit is already captured
  by salience ranking (1.2); only worth it later.
- **Relation/concept graph (graph-RAG)** for multi-hop: the eval showed generalization is a *tie*
  with good RAG, so this is nice-to-have, not a differentiator. Skip unless multi-hop is a real need.

## 4. The honest ceiling

All of this makes the **memory** learn — salience, forgetting, supersession, consolidation. It does
**not** make the frozen LLM itself learn or reason better (that needs weight changes, which an MCP
memory cannot do). So "make an LLM learn" honestly becomes **"give the LLM a memory that genuinely
learns,"** and Tiers 1–2 deliver the part that provably beats plain RAG. (Full reasoning in
`SystemAnalysis.md`, Parts 3–5.)

## 5. Provenance

Distilled from the ArtificialIntelligence project: the reward-modulated Hebbian rule and
decay/pruning (`src/processing.lisp`), the controller (`src/controller.lisp`), and the head-to-head
(`src/controller-vs-rag-experiment.lisp`). The salience-and-decay rule to add to OwnSona is the same
mechanism, expressed in SQL over memory rows. See `SystemAnalysis.md` for the full analysis chain
(cost, continual learning, bootstrapping from a pretrained model, controller-vs-RAG).
