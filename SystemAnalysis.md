# System analysis: backpropagation vs. Hebbian, and continual learning

*Why this project uses a Hebbian (local, no-backprop) network, which parts of that rationale
hold up, whether a hybrid could buy modern-LLM quality without LLM-scale training, why a trained
network can't keep learning and how to get past it, and — given no budget to pretrain — whether
an existing open model can be bootstrapped into the system. Written to be accurate rather than
encouraging: some of the common motivations are true, one is largely a misconception, and the
distinctions decide what is worth building next.*

The driving goal: **modern-LLM capability that can still learn continually after deployment.**
Setting aside the multi-user / safety / reproducibility reasons for freezing a model (those are
deployment *choices*, addressed in Part 1, Claim 2), Part 3 isolates the genuine *technical*
barrier to learning-after-training and the architecture that gets past it; Part 4 asks whether,
without the time/compute/data to pretrain, an existing open model can be reused as the backbone;
and Part 5 lands on the architecture worth building — this system as a controller that uses the
LLM to *propose*, learns *which proposal to pick*, and calls the LLM again to *execute*.

See also: `notes/TransformerFromHebbianParts.md` (the experiment arc), and the standalone PoCs
`src/predictive-coding-experiment.lisp` / `src/pc-attention-experiment.lisp` referenced below.

---

## Part 1 — the three claims behind "use Hebbian, not backprop"

### Claim 1: "Backprop is *extremely* more computationally expensive than Hebbian." — misleading

Per weight, per example, the gap is a **small constant factor, not orders of magnitude.**
The standard FLOP accounting:

- A **forward** pass costs ~`2·N` FLOPs (`N` = parameters).
- **Backprop's backward** pass adds ~`4·N` (input-gradient + weight-gradient), so one backprop
  **training step** ≈ `6·N` — about **3× a forward pass**.
- A one-shot **Hebbian** update (`Δw = η·pre·post`) is ~1 multiply per weight, so ~`3·N` total
  including the forward — roughly **2× cheaper than backprop per step**. Real, but a constant.

Where backprop's cost *actually* explodes is not the per-update arithmetic but:

1. **Scale** — `params × tokens × passes`. LLMs are expensive because they are huge models over
   ~10¹³ tokens, not because the gradient is costly per weight.
2. **Activation memory** — backprop must store every layer's activations for the backward
   sweep; local rules don't. Often the real practical bottleneck.
3. **Global synchronization** — a full forward-then-backward sweep with weight transport:
   sequential, and hostile to locality, parallelism, and batch-size-1 online updates.

And a crucial catch, demonstrated directly by this project's own experiments: the *cheap*
one-shot Hebbian rule is cheap **because it does less** (no credit assignment through depth). The
local rules that actually **match** backprop's power — predictive coding, equilibrium
propagation — recover much of the cost through **iterative settling**
(`src/predictive-coding-experiment.lisp` used 30–60 inference steps per example, *more* expensive
than backprop). So "local ⇒ cheap" and "local ⇒ as capable as backprop" are in tension.

**Honest version:** backprop is ~2–3× a forward pass per step; the "extreme" cost is scale +
memory + global sweeps. Cheap Hebbian buys the constant factor and the memory/locality win — but
the capable local rules give the constant factor back.

### Claim 2: "Backprop's slowness is why LLMs separate training from use." — mostly false

The train/use split is **primarily not about backprop's cost.** Even with instantaneous
backprop, LLMs would still separate them, because of:

1. **Catastrophic forgetting / the stability–plasticity dilemma** — continually updating weights
   on a live stream destroys old knowledge and destabilizes the model. *This is the big one.*
   Frozen weights guarantee stable, reproducible behavior.
2. **Safety and data curation** — training data is filtered, deduplicated, RLHF-tuned. You don't
   let a deployed model learn from arbitrary user input (poisoning, drift, abuse).
3. **Reproducibility & deployment economics** — one checkpoint, tested and versioned, served
   identically to millions; you cannot have each user's session mutating shared weights.
4. **The optimization regime** — SGD wants shuffled, i.i.d., multi-epoch, large-batch *offline*
   data. Online single-example learning is a different and harder regime regardless of the
   gradient rule.

Cost is *a* contributing factor (you cannot afford to retrain per interaction), but it is not
*the* reason. The deep reasons are forgetting and control.

### Claim 3: "If the Hebbian net works, continual learning follows, since training is cheap." — necessary, not sufficient

The cost argument is valid: cheap local updates make per-step online learning *affordable*,
which backprop's regime discourages. A real enabler, and a legitimate motivation.

But cheapness does not *give* you continual learning. The hard part is **catastrophic
forgetting**, and Hebbian nets suffer it too (naive Hebbian is famously unstable — runaway
weights, interference). This project already fights it with **decay, pruning, and homeostatic
thresholds** — those *are* the stability mechanisms doing the real work, not the cheapness. So
"get the Hebbian net to work" → "continual learning" still requires solving stability, which is
orthogonal to compute cost.

(Note: the system *already* learns continually at its current capability level. The open
question was never "can it learn online" — it is "can it reach LLM *quality* while doing so,"
which is the deep-representation problem mapped in `notes/TransformerFromHebbianParts.md`.)

---

## Part 2 — would a hybrid (backprop in some areas, Hebbian in others) get LLM quality without the expensive training?

Yes — hybrids are real, used in practice, and likely the most realistic route to
"LLM-ish quality without retraining-on-every-interaction." But one tension must stay explicit,
because it bears directly on Goal 1 (no separate, frozen pre-training phase).

The main patterns, ordered best-to-worst fit for this project's goals:

**A. Backprop once, offline, to *meta-learn a local rule* — then run purely local online.**
"Learning to learn": use expensive backprop *once* to discover a local plasticity rule (or
attention structure) that thereafter runs cheaply and continually with no backward pass. The
expensive global optimization is amortized into a reusable local rule. This **preserves the
goal**: at *use* time, learning is local and continual. The cleanest fit.

**B. Frozen backprop-pretrained backbone + Hebbian/online adaptation on top.**
Take strong representations learned once by backprop (offline), freeze them, and do fast local
learning on top for continual adaptation. Gives backprop-quality features + cheap online
updates — pragmatic and powerful. **The catch:** the backbone *was* trained offline with
backprop, so a separate pre-training phase is reintroduced for it — a partial retreat from
Goal 1. Essentially the LLM model with a cheap adaptive cap.

**C. Backprop for the hard credit-assignment core, Hebbian for the broad associative memory.**
Motivated by this project's *own* findings: predictive coding (local) could not escape the joint
credit-through-depth plateau on the attention induction circuit
(`src/pc-attention-experiment.lisp`), while the Hebbian associative / counting machinery is
genuinely good at broad, cheap, stable storage. So: backprop where deep credit assignment is
essential (a *small*, deep core), Hebbian where scale and continual storage matter — keeping
backprop's cost bounded because it runs on a small module.

**Two caveats that apply to every hybrid:**

1. **LLM quality comes specifically from deep representations learned by backprop over massive
   data.** To get that quality, *something* must do that deep learning. Using backprop for it —
   even "in certain areas" — gives those areas an expensive, offline, separated training phase.
   No free lunch: you can move, shrink, or amortize the backprop (option A), but you cannot get
   its *product* without paying for it somewhere.
2. **The data-exposure cost is irreducible.** Even if every weight update were free, absorbing
   LLM-level breadth means *forward-passing* through ~10¹³ tokens; that forward cost dominates
   and does not vanish with a cheaper learning rule. The genuine appeal of the Hebbian approach
   is not avoiding that cost — it is **amortizing it over the system's operational lifetime**
   (learn-as-you-go) instead of a discrete, frozen training run.

**Recommendation.** For "LLM quality + continual + not-retraining-constantly," **option A or C**
is the intellectually honest hybrid that mostly preserves the goals: use backprop *sparingly and
offline/once* (to meta-learn a rule, or to train a small deep core), and keep the large,
continual, online part local. The pure-purist version (no backprop anywhere, fully continual,
LLM quality) is exactly the open problem being mapped here — and the PC-attention result suggests
the blocker is not "local vs. global" at all; it is **joint optimization through depth**.

---

## Part 3 — why a trained network can't keep learning, and the path past it

This is the actual goal: modern-LLM capability *that still learns continually*. Remove the
deployment reasons for freezing a model (multi-user, safety, reproducibility — all *choices*),
and what remains is the genuine technical barrier. None of it is about backprop's speed.

### The real barrier — three intertwined phenomena

1. **Catastrophic forgetting (interference).** Knowledge lives in *shared, overlapping* weights
   tuned to an equilibrium over the whole i.i.d. training distribution. A later gradient step on
   new data moves weights that *also* encode old knowledge — and nothing marks which weights are
   load-bearing for what — so the old degrades. Training never sees this because every concept is
   constantly rehearsed; continual use delivers a *non-stationary stream*, so the recent
   overwrites the past.
2. **Loss of plasticity.** Networks trained a long time progressively *lose the ability to learn
   new things at all* — units saturate, effective rank collapses, gradients shrink (Dohare et
   al., 2024). Continual updating makes a net *ossify*, not just forget.
3. **One-shot integration failure.** Gradient descent integrates a fact by seeing it many times
   across the corpus. A *single* online exposure either fails to stick, or — pushed hard enough
   to stick — overfits and damages neighbors. "Learn this one new fact now" is exactly the regime
   SGD is worst at.

One sentence: **a dense distributed network stores knowledge as a delicate global equilibrium,
and online non-stationary updates perturb that equilibrium faster than they build on it.**
Freezing is simply the cheapest guaranteed defense.

### The correction that matters here

**This barrier is not solved by being Hebbian.** Forgetting, plasticity loss, and one-shot
failure are properties of *shared distributed representations under non-stationary updates* —
they afflict naive Hebbian nets just as much (untamed Hebbian is notoriously unstable: runaway
weights, interference). So "get a Hebbian net working" does not, by itself, get past
"can't learn after training."

Why does *this* system nonetheless learn continually and survive restarts? Not because it is
Hebbian — because of its **architecture**: sparse / discrete stores (concept graph, a separate
output network), **decay and pruning**, **homeostatic thresholds**, and **retrieval-style
memory** instead of one monolithic weight blob. Those are anti-interference mechanisms, and they
are rule-agnostic.

### The goal therefore decomposes into two *orthogonal* problems

- **Continual learning** — *already largely solved* in this architecture, and the general recipe
  (below) is independent of backprop vs. Hebbian.
- **Modern-LLM capability** — deep, compositional representations: the open problem the
  experiment arc maps, whose blocker is *joint credit assignment through depth*
  (`notes/TransformerFromHebbianParts.md`), not the learning rule or its cost.

The trap in the original framing was coupling them ("Hebbian → cheap → continual → therefore the
LLM problem"). They are separate. Cheapness helps continual *affordability*; it does nothing for
capability *or* for interference.

### What actually gets past forgetting (rule-agnostic)

Every working approach is architectural, not a matter of the gradient rule:

1. **Replay / rehearsal** — keep (or *generate*) old examples and interleave them, restoring the
   i.i.d. condition. The most reliable method; *generative replay* (the model dreams its past) is
   the elegant form.
2. **Weight protection** — estimate which weights are load-bearing for old knowledge (Fisher
   information; EWC) and resist moving them. Works until capacity saturates.
3. **Sparsity / modularity** — sparse or localized codes so new learning barely overlaps old, or
   fresh capacity allocated per new knowledge (adapters, experts). This project's concept graph
   and pruning live here.
4. **Complementary learning systems** — a *fast* episodic store for one-shot learning + a *slow*
   consolidated store, with periodic consolidation/replay between them. How brains do it
   (hippocampus ↔ neocortex, replay during sleep); it solves both forgetting *and* one-shot
   integration at once.

### The synthesis most likely to reach the goal

Combine them: **a deep "slow" backbone for capability + a fast, sparse, local, continual memory
for ongoing learning + consolidation/replay that migrates fast-store knowledge into the slow
store without overwriting it.** That is complementary learning systems, and it is the only known
architecture delivering *both* LLM-level capability *and* learn-after-deployment. Its pragmatic
form today is "frozen deep model + retrieval / episodic memory + occasional consolidation" — the
model *learns by writing to memory*, not by perturbing weights. This system is already shaped
like that (it has the memory / retrieval / generation pieces).

The one genuinely open choice left is *how the deep backbone is built*:

- **backprop once, offline** (the hybrid options A / C above) — concedes a pre-training phase for
  the backbone, but is achievable now; or
- **grown locally / continually** — preserves the purist goal, but is the unsolved frontier
  (joint credit assignment through depth).

**Bottom line for the project:** treat continual learning as a *memory-systems* problem that is
mostly cracked, and spend the remaining effort on the *capability / depth* side — because that,
not forgetting and not compute cost, is what stands between this system and modern-LLM
capability.

---

## Part 4 — bootstrapping from a pretrained model (no budget to pretrain)

The practical constraint: there is no time, compute, or data to pretrain a model at modern scale.
So the question is whether an existing **open-weight** model can be reused — perhaps "run through
a conversion program" into this system. Short answer: **no conversion, but yes composition.**

### Why "conversion" can't work

An LLM stores knowledge as **dense float matrices** tuned by backprop (attention, MLP, a learned
embedding table). This system stores knowledge as **discrete neurons, dendrites, a concept graph,
and count-based stores.** There is no decompiler from one to the other:

- You cannot read a transformer's weights and emit concept-graph edges — the knowledge isn't
  *located* anywhere mappable; it is smeared across billions of parameters in a basis nothing
  here shares.
- Distilling the LLM into this architecture would lose almost everything, because the architecture
  cannot *represent* what the LLM knows (the capability gap the experiment arc maps).

So "run it through a converter" is the wrong frame. The right word is **composition**.

### What works: the LLM as the frozen "slow store"

This is Part 3's complementary-learning-systems synthesis, instantiated with a real model:

- **The open-weight LLM = the deep "slow" backbone** — the capability there is no budget to
  train. Frozen. Runs on the GPU (the *right* job for that GPU; the Hebbian workload never was).
- **This system = the fast, continual, local memory layer** around it — the part already built,
  and the part it is genuinely good at.

The whole then **learns continually even though the backbone is frozen** — exactly Part 3's
resolution: don't keep updating the deep weights, keep a fast memory that does. Three wirings:

1. **Retrieval / episodic memory (most achievable).** LLM frozen; this system stores new
   knowledge and is queried at inference; retrieved facts are fed into the LLM's context. The LLM
   "learns after training" by consulting continually-updated memory. (RAG, with this system as the
   memory.)
2. **LLM embeddings as features.** Replace the random / co-occurrence codes with the LLM's learned
   embeddings (run text through it, take hidden states); the Hebbian layers learn on top of
   LLM-quality representations.
3. **Continually fine-tune the open model itself** (LoRA / adapters + replay). Keeps it a
   transformer, not this system — a continual *LLM*, but abandoning the Hebbian substrate and
   still fighting forgetting.

Integration is by IPC, not linking: run the model under **Ollama or llama.cpp** (a local HTTP
server) and have the Lisp call it for generation or embeddings. No retraining, no data, modest
compute — the training is inherited. That directly dissolves the time/compute/data problem.

### Is wiring 1 better than a plain RAG system?

In principle yes; in practice today, probably not — *unless* the contribution is the learning
dynamics. Wiring 1 (frozen LLM + this system as the memory layer) differs from a plain vector-store
RAG in four ways that genuinely matter for a *continually-learning* memory: **generalization**
(the concept graph answers about novel inputs from shared structure, where RAG can only return what
is literally stored near the query), **relational / multi-hop** reasoning (is-a chains, spreading
activation, vs. a vector store with no edges), **salience and forgetting** (Hebbian reinforcement,
decay, pruning, vs. a store that treats every chunk as equally permanent), and **consolidation**
(aggregating inputs into triples and graph structure rather than dumping raw chunks).

Two honest deflations, though:

1. **"Plain" RAG is not the competition.** Mature systems — GraphRAG, entity/relation extraction,
   hierarchical summarization, agentic memory (MemGPT/Letta) — already build knowledge graphs,
   consolidate, dedupe, and do multi-hop. Most of the *structural* advantages over naive RAG are
   standard features of *good* RAG.
2. **The in-loop LLM is the better extractor.** In wiring 1 the frozen LLM is present anyway, and
   it can build the graph, pull relations, resolve contradictions, and summarize far better than
   the toy, brittle `relations.lisp` / `generation.lisp` (which degrade on messy data). Using the
   weaker language engine to structure memory while the stronger one sits idle is the wrong split;
   good RAG uses the LLM to do exactly that structuring.

So as a *retrieval / structuring* layer, this system does not clearly beat good RAG and probably
loses. **Its one genuine, differentiated edge is the learning *dynamics*** — use-based salience,
decay-when-unused, pruning — which neither plain nor graph-RAG has (RAG memory is static curation;
this is a memory that weights by experience and forgets). The honest framing: this is not "a better
text retriever," it is **a learning memory with salience and forgetting**, and it should be judged
on what RAG *cannot* do, not on retrieval accuracy.

> **Measured (Part 5's `controller-vs-rag-experiment.lisp`).** The head-to-head narrowed this
> claim, honestly: on **recall** all learners tie, and on **generalization** a fair lexical RAG
> *ties* the learning policy (~0.85) — so generalization is *not* the edge it was assumed to be.
> The clean win is **non-stationarity**: when the environment changes, the controller's decay
> un-learns the stale answer (~1.0) while append-only episodic RAG stays anchored to the old one
> (~0.5). The differentiator is specifically **salience/forgetting**, not generalization or recall.

Mapped to the fork: for the **product** goal, use good RAG / agentic memory — do not reinvent it.
For the **research** goal, wiring 1 is meaningful only as a vehicle for the dynamics, and the
constructive middle path is to let those dynamics **augment** a RAG store rather than replace it —
Hebbian salience to rank/prune what the index keeps, the concept graph to expand retrieval
multi-hop, generalization as a fallback when nearest-neighbor retrieval misses. In one line:
wiring 1 ≈ "RAG, but with a learning, decaying, generalizing memory instead of a static index" —
worth it only if those dynamics earn their keep, which is tested on the queries RAG fails, not the
ones it already handles.

### The catch — this conflicts with both founding goals

Using their model as the backbone **violates both founding goals**: it *is* a nonvolatile
pre-training phase (theirs), breaking Goal 1; and it *is* backprop-trained, breaking Goal 2. The
memory layer around it can stay Hebbian/local, but the **capability comes from their backprop
pretraining.** So this is a genuine fork:

- **If the goal is the research question** — *can a local, no-backprop system reach LLM
  capability?* — grafting an LLM **sidesteps the question** and defeats the point.
- **If the goal is a product** — *a capable assistant that learns continually after deployment* —
  grafting is the pragmatic win, and the division of labor (LLM = capability, this system =
  continual memory) is principled, not a cop-out.

### A genuine partial import (the one real "conversion")

One bounded artifact *can* be lifted directly: the LLM's **token embedding table** (a `vocab × d`
matrix). Import it (or use a dedicated embedding model) to give `vectors.lisp` real semantic
geometry instead of random/co-occurrence codes — cheap, useful, and the learning stays local. It
still imports a backprop-trained artifact (mild goal tension), but far milder than adopting a
whole backbone.

### Practical notes

- **Prefer a permissively-licensed open model** for a public-domain project: Apache-2.0 (Mistral,
  Qwen2.5) or MIT (Phi), or **OLMo** (fully open — weights, code, *and* data). Avoid the more
  restrictive community licenses if license cleanliness matters.
- A 7–14B model, quantized, runs comfortably on a strong consumer GPU.

**Recommendation:** for the product goal, start with wiring 1 or 2 — a frozen open model on the
GPU, this system as the continual memory/retrieval layer — optionally importing the embedding
table (the partial conversion). For the research goal, keep the backbone home-grown and treat the
LLM, if used at all, only as a disposable scaffold or evaluation oracle.

---

## Part 5 — controller + LLM advisor: propose, learn-to-select, execute

Part 4's wirings make the LLM the brain and this system its memory — which (the RAG comparison
showed) cannot make the *system's behavior* learn. Flipping it — **this system as the controller,
the LLM as a tool it calls** — puts learning where it governs behavior. But a pure flip fails on
one fact: in a controller-plus-tool design, overall intelligence is capped by the *controller's*
decision quality, and this system is too weak to plan from scratch. The resolution is to **split
control into planning (hard — delegate to the LLM) and judging/selecting (learnable — keep it).**

### The loop

1. **Perceive** — input arrives; this system retrieves relevant memory / state (concept graph,
   vectors, facts).
2. **Advise (LLM call 1)** — ask the LLM to *propose* `K` candidate actions / plans, given the
   input plus the injected memory context. Strong planning, borrowed.
3. **Decide (this system)** — score the candidates with a *learned selection policy* and
   choose / edit / veto / combine, conditioned on persistent memory and what past outcomes say
   worked. **This is where the learning lives.**
4. **Execute (LLM call 2)** — the LLM carries out the chosen action and produces the result.
   Strong execution, borrowed.
5. **Reinforce** — the outcome (user feedback, task success, a self-consistency check) updates the
   selection policy and the memory; decay / pruning fade stale policy. The loop that makes it
   learn.

### Why this is the learnable one — and fits machinery already built

**Choosing is far easier to learn than planning.** Generating a good plan is open-ended search
(what this system fails at); *ranking a handful of LLM-proposed options by expected value given
state and history* is a small, bounded, contextual-bandit-shaped problem. And the learning rule
for it **already exists here**: the reward-modulated Hebbian update `Δw = η·pre·post·r`
(`processing.lisp`) *is* "strengthen the choice that led to a good outcome." So Part 5 is not a
rewrite — it points existing parts at a new, well-suited target:

- **selector / value function** → reward-modulated Hebbian associations (`processing.lisp`), with
  the **reward signal coming from the teaching loop** (the user's confirm/correct already in
  `main`), or from task-success / self-consistency checks for autonomous use;
- **context features** ("which situation is this?") → `vectors.lisp` similarity + the concept
  graph (`concepts.lisp`);
- **forgetting of stale policy** → the existing decay / pruning;
- **the LLM** → an external tool reached by IPC (Ollama / llama.cpp HTTP), called twice per step.

### What it learns, and why it beats RAG

Not *reasoning* — the LLM's reasoning stays frozen — but **judgment, policy, and memory**: which
of the LLM's outputs are good for which situations, which approaches keep failing, what to
remember, when to override the obvious suggestion. A durable, parametric, lifetime-spanning policy
— *taste / expertise* over a fixed reasoning engine. RAG cannot do this: it feeds the model notes
but never develops a policy, and an LLM agent's in-session judgment evaporates at the context
boundary. This is *learning how to act, and keeping it* — the strong sense of "really learning,"
now realistically powered. Lineage: the **"LLM proposes, a learned value/verifier disposes"**
pattern (learned rerankers, verifier-guided generation, RLHF reward models, AlphaGo's
policy-proposes / value-selects) — here with a *continually-learning* selector instead of a frozen
one.

### Caveats

- **The selector must meaningfully diverge from rubber-stamping the LLM's first idea** — its value
  is entirely the persistent, experiential, personalized judgment the stateless LLM lacks. If it
  almost always accepts the top proposal, it is decorative and the design has collapsed back into
  an LLM agent.
- **It needs a real outcome signal** to learn from (the teaching loop provides one; autonomous
  tasks need success/consistency checks). No reward → no policy learning.
- **Cost: two+ LLM calls per action** — fine for a personal / research system.
- **The ceiling is still the LLM's reasoning** — this learns *judgment over* a fixed engine, not
  better reasoning. That is the right and honest scope.

**Verdict:** of the three control framings, this is the one to build. It assigns each component its
strength (LLM: plan + execute; this system: select, steer, remember, and *learn the policy*),
turns "the system learns" into a problem the existing reward-modulated Hebbian machinery suits, and
the learning is durable in a way RAG cannot touch — provided the selector earns its place by
diverging from naive acceptance and improving with experience. The eval that proves it: **does the
system get measurably better at a recurring task through experience** (better choices, fewer wasted
calls, higher success) in a way a frozen LLM-agent + RAG does not?

### Status — prototyped, and measured against RAG

This is built: `src/llm.lisp` (call an LLM as a tool — `:mock`, local `:ollama`/llama.cpp, cloud
`:openai`/`:anthropic`; dependency-free, via `curl`) and `src/controller.lisp` (the
propose → learn-to-select → execute loop). The selector is the existing reward-modulated Hebbian
rule over a persistent `*selector*` store; `controller-reward` is the outcome signal. Offline (with
the `:mock` provider) the loop demonstrably learns to prefer the rewarded proposal within a few
rounds and the policy persists across save/reload (covered by the test suite).

The head-to-head-vs-RAG evaluation has now been **run** (`src/controller-vs-rag-experiment.lisp`,
offline, four arms — LLM-only, a fair feedback-using lexical RAG, the real exact-keyed controller,
and a feature-keyed controller variant — across three regimes). **It overturned the going
hypothesis, honestly:**

- **Recall** (exact contexts recur): all three learners tie (~1.00); LLM-only at chance. The
  controller is **not** beneficial above RAG here — recall is recall.
- **Generalize** (novel combinations of the same vocabulary): a decent lexical RAG holds up
  (~0.85) and **ties** the feature controller; the *exact*-keyed controller can't generalize
  (~chance) because it keys on the whole input. So generalization is **not** a clean controller win
  — nearest-neighbor retrieval is competitive.
- **Drift** (the mapping changes mid-stream): the controllers clearly **beat** RAG (~1.00 vs ~0.5),
  because the reward-modulated **decay** un-learns the stale mapping while RAG's append-only
  episodic memory stays anchored to the old, now-wrong answer.

So the controller's genuine edge over RAG is **non-stationarity / forgetting** — exactly the
"salience and forgetting" property predicted in Part 4's RAG comparison — **not** recall and **not**
generalization. Two concrete implications: (a) keep the decay — it is the real differentiator; (b)
to even *match* RAG's generalization, the selector must be **feature-keyed** (the current
whole-input key generalizes at chance). Caveats: lexical RAG and a toy task; an embedding RAG would
generalize better on the static regimes. Still to do: feature-key the selector, wire the controller
into the interactive `main` loop, and repeat the eval against an embedding RAG on a real task.

Backprop is only ~2–3× a forward pass per step — its real expense is scale, activation memory,
and global forward/backward sweeps, not per-weight arithmetic; cheap Hebbian saves the constant
factor by doing less (no deep credit assignment), and the local rules that match backprop's power
pay much of it back via iterative settling. The separation of training from use in LLMs is driven
mainly by catastrophic forgetting, safety/data-curation, and reproducibility — not by backprop's
speed — so a cheaper rule alone does not unlock continual learning; stability (which this project
handles via decay/pruning/homeostasis) is the real prerequisite. A hybrid *can* approach LLM
quality with far less ongoing cost, but only by paying for deep representations somewhere: the
goal-preserving versions use backprop **once, offline** (to meta-learn a local rule, or to train a
small deep core) and keep the broad, continual learning local — accepting that the irreducible
cost is forward-pass exposure to data, now amortized over the system's lifetime rather than spent
in a frozen training run. And the reason trained nets can't keep learning is not cost at all but
**catastrophic forgetting, loss of plasticity, and one-shot integration failure** — interference
in shared distributed weights under a non-stationary stream, which a Hebbian rule shares — so the
goal splits cleanly: continual learning is a *memory-systems* problem this architecture largely
already solves (sparse stores, decay, pruning, retrieval — a fast/slow complementary system),
while the remaining work is *capability / depth*, the deep-representation frontier that neither
forgetting nor compute cost, but joint credit assignment through depth, governs. With no budget to
pretrain, that capability can be *borrowed*: an open model cannot be *converted* into this system
(incompatible representations) but can be *composed* with it — frozen LLM backbone on the GPU for
capability, this system as the continual memory layer around it (optionally importing the model's
embedding table as the one genuine partial conversion) — which dissolves the resource problem but
trades away both founding goals for the backbone, so the choice between *research* (home-grown
backbone) and *product* (borrowed backbone) is the real decision left. And the architecture worth
building flips the usual composition: rather than this system serving as memory for an LLM brain
(which leaves behavior frozen, no better than RAG), make *this system the controller* and the LLM
a tool — it asks the LLM to **propose** options, learns (via the reward-modulated Hebbian rule it
already has, with the teaching loop as the reward) **which proposal to pick**, then calls the LLM
to **execute** — so what becomes durably learnable is *judgment and policy over a fixed reasoning
engine*, the one form of "really learning" that RAG and context-stuffing cannot provide.
