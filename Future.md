# Future — what it would take to feel like a modern LLM (from the user's side)

**Scope.** This is about *user-facing behavior* — what a person experiences using an LLM
like ChatGPT — **not** about copying how LLMs work internally. This project deliberately
does **not** mimic LLM internals: no backpropagation, no next-token-prediction machinery,
no separate training phase. The question here is only: which user-visible capabilities
would the system need, and what would each actually require of this architecture?

For context on what exists today, see `Plan.md` (design) and `CLAUDE.md` (code map). The
system is currently an associative memory + concept graph + attention/copy head with
continual (online) Hebbian learning.

---

## Already in reach — the bones exist

- **Saying "I don't know"** when unsure — done.
- **Q&A over what it has been taught** — done; the only limit is how much it knows.
- **Generalizing an answer to new inputs** — the concept graph (categories, e.g. a
  never-taught *horse* "walks on its legs") and the copy/binding head (fill-in-the-blank,
  e.g. `say X → X` for a novel X) already do this.

## Addable in this style — high value

These would most make it *feel* like an LLM while staying true to the design:

- **Multi-turn conversation memory.** ✅ **Built** (`ask` / `resolve-followup` in `ai.lisp`;
  `Plan.md` §3.7): a follow-up leans on the previous turn (`*last-turn*`) — "and cats?"
  becomes "do cats have legs?" by replacing the most concept-similar word — and is answered
  via the concept graph. Still open: pronoun / coreference resolution ("it" / "that"); only
  ellipsis follow-ups are handled today.
- **Paraphrase tolerance.** A user says the same thing many ways ("do dogs walk?", "can a
  dog walk?", "dogs — do they?") and expects the same understanding. Needs synonym /
  rephrasing generalization on the input side, stronger than today's exact-frame matching.
- **Following simple instructions / operations.** ✅ **Started** (`operations.lisp`;
  `Plan.md` §3.8): *learned operations* over the system's own knowledge — you teach what a
  question MEANS once ("how many animals do you know → count animals") and it generalizes to
  any category and is recomputed as the system learns. `count` and `similar` exist; more
  (compare, exists, list) are additions of the same shape.
- **Distributed representations (non-brittleness).** ✅ **Started** (`vectors.lisp`;
  `Plan.md` §3.8): Hebbian co-occurrence concept vectors (CMAC / hyperdimensional family)
  give similarity as geometry — dog↔cat clusters emerge, novel words interpolate, no cliff.
  Strong for similarity/generalization; crisp counting stays with the discrete graph (no
  clean boundary in the vector space). The two are complementary.
- **Chained reasoning.** "A is a B, B's are C, so A is C." Multi-hop over the knowledge;
  spreading activation is a start.

## The genuinely hard ones — they need more than association

- **Producing original, fluent prose on open topics** (write an email, explain something
  in a paragraph). The honest point: to output a sentence it wasn't taught verbatim, the
  system has to *compose* one — and composing language inherently means choosing what comes
  next in *some* form. That isn't "LLM internals"; it's intrinsic to generating language.
  But it needn't be done the LLM way (token statistics). This architecture's natural route
  is **template / fragment composition**: assemble a response from learned phrase-fragments
  with the blanks filled by the concept graph + copy head — novel-but-structured answers
  with no next-token machinery. A first version of this is now ✅ **built** (`note-template`
  / `compose` in `attention.lisp`; `Plan.md` §3.7): taught "what is a dog/cat → a … is an
  animal", it answers "what is a horse?" with "a horse is an animal", never seen verbatim.
  Free-form creative writing would still stay weak.
- **Broad world knowledge.** Users ask about anything; LLMs got that by reading the web.
  This system knows only what's taught or imported — no shortcut but to acquire a large
  knowledge base (teach it, or import structured knowledge). The bottleneck is the *volume
  of knowledge*, not the mechanism.
- **Deep, novel reasoning** beyond chaining stored facts — hard for any associative system.

## The one unavoidable trade

The two things that most define the LLM *experience* — "answers anything, fluently" — come
from **(a) vast knowledge** and **(b) fluent composition**. (a) is just a matter of feeding
it enough; (b) needs *some* composition step. Everything else — memory, paraphrase, Q&A,
simple instructions, reasoning chains — this continual/associative approach can deliver in
its own way, and would feel markedly more LLM-like to a user while staying true to the
design (continual, local, no backprop).

## Highest-leverage next steps — now built

1. **Conversation memory** — ✅ **built.** `ask` / `resolve-followup` (`ai.lisp`) fold a
   follow-up into the previous turn by concept-similarity ("and cats?" → "do cats have
   legs?") and answer via the concept graph. See `Plan.md` §3.7.
2. **Template / fragment answer composition** — ✅ **built.** `note-template` / `compose`
   (`attention.lisp`) learn `frame → template-with-slot` and fill the slot by reference, so
   a reply can be assembled from fragments and never seen verbatim ("what is a horse?" → "a
   horse is an animal"). See `Plan.md` §3.7.

Both are user-facing, both fit the architecture, and together they move the *feel* of the
system toward an LLM without abandoning what makes it different. Natural follow-ons:
pronoun / coreference resolution (only ellipsis is handled), paraphrase tolerance, and
composing from several fragments rather than one template.
