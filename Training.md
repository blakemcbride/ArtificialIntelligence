# Training the system

This guide covers how to teach the system at scale: how to configure training, how
to ingest very large files, and how to switch the configuration back for everyday use.

It assumes you are in the interactive teaching loop (`main`). The quickest way in is the
launcher script at the repo root (16 GB heap, loads everything, enters `main`):

```sh
./sbcl-llm
```

or, by hand, from `src/`:

```lisp
(load "load.lisp")
(main)
```

Everything below is typed **inside** the `main` loop, where commands begin with a period
(`.set`, `.read`, …). They do **not** work at the bare SBCL `*` prompt — that is SBCL's own
REPL, not the teaching loop. (See `tutorial/tutorial.md` and `CLAUDE.md` for the full command
reference; this file focuses on the training workflow.)

The system learns two ways, and you rarely need to care which:

- **Supervised pairs** — `input => answer` lines (exact stimulus → response).
- **Prose** — ordinary sentences, interpreted best-effort.

`.read FILE` auto-routes each line to the right mode, so one file may mix both.

---

## a. Training configuration

Training behavior is controlled by a handful of live parameters. View them any time with
**`.config`** (it also shows current model sizes); change one with **`.set NAME VALUE`**
(`off`/`nil` = unlimited/disabled, `on`/`t` = enabled). Every one of these is saved in the
`.kb` and restored on `.load` — see the note at the end of this section.

### What each parameter does

| Parameter | Default | Meaning |
|---|---|---|
| `read-extract` | on | Run the **heavy supervised path** (builds the input neuron network, concept graph, associations) for each sentence read. Turn **off** for bulk reading — it is the biggest per-sentence cost and is meant for curated teaching, not firehose ingestion. |
| `read-cooccur` | on | Run the **co-occurrence / similarity** learner. This one is **O(words²)** per sentence — the single biggest cost on long, messy web sentences. First thing to drop for huge corpora. |
| `read-relations` | on | Run **relation discovery** (learns "is a"-style connectors and membership facts). |
| `read-facts` | on | Run the **declarative-fact** extractor (the `(subject relation object)` triples behind "tell me about X"). |
| `read-transitions` | on | Run the **next-word transition** model. This is the learner that genuinely *improves with scale* — keep it on. |
| `read-workers` | 1 | **Parallel** worker threads for a bulk read (SBCL only; 1 = sequential). See section (b). |
| `read-max-mb` | off | Read at most ~this many MB per `.read` (off = whole file in one pass). See section (b). |
| `max-vocab` / `max-cooccur` / `max-transitions` / `max-facts` | off | **Caps** on the largest stores. When set, an over-cap store is pruned to its strongest entries, bounding memory no matter how much you read. |
| `prune-every` | 5000 | Enforce the caps once per this many learned sentences. |

### Which learners are worth running on raw web text?

Turning a learner off does mean the system won't learn *that kind* of thing **from this
corpus** — but it does **not** erase anything already learned, and the learners still work for
everything else. More importantly, the simple Hebbian learners were built for *clean, curated*
teaching. On raw web text (fragments, boilerplate, tables) the expensive ones mostly accumulate
**noise**:

- **co-occurrence** builds muddier (not sharper) similarity vectors, and it's O(n²) + unbounded;
- **relation discovery** mines plenty of garbage "is a" facts from malformed sentences;
- **transitions**, by contrast, genuinely benefit from scale.

So for a huge web dump, *"transitions only"* is usually the right trade — not a lobotomy.
Keep every learner on for small, high-signal, curated material where speed isn't the issue.

### Recommended bulk-training configuration

```text
input> .set read-extract off       ; skip the heavy supervised path
input> .set read-cooccur off       ; drop the O(n^2) similarity learner
input> .set read-workers 8         ; use 8 cores (set to your core count, SBCL only)
input> .set max-transitions 2000000  ; (optional) cap the store you ARE growing
input> .set read-max-mb 1000       ; read ~1 GB per .read (see section b)
input> .config                     ; confirm the settings
```

Adjust to taste: keep `read-relations`/`read-facts` on if you want the system to learn category
structure and describable facts from the corpus and can afford the noise; drop them for the
leanest, fastest pass.

### These settings persist in the `.kb`

`.save` records the settings that were in effect, and `.load` re-applies them. That is convenient
during training (set once, they survive `.save`/`.quit`/restart) — but it means a knowledge base
trained in "bulk mode" will **reload in bulk mode**. Section (c) is about flipping them back.

---

## b. Dealing with very large files

A file is always read in **bounded-memory chunks** — it is never loaded whole — so a 100 GB+
corpus streams fine. What grows is the *learned model*, which the caps above bound. Four tools
make a huge ingest practical:

### 1. A big heap (do this first)

SBCL's default Lisp heap is ~1 GB, and that — not the file size — is the usual cause of a
`Heap exhausted` crash. `./sbcl-llm` already starts SBCL with `--dynamic-space-size 16384`
(16 GB). By hand: `sbcl --dynamic-space-size 16384`. The caps bound the model; the heap is the
absolute ceiling — you need both.

### 2. Read in slices — `.read` resumes automatically

`.set read-max-mb N` caps how much each `.read` ingests. `.read FILE` **resumes where it last
stopped** (a per-file byte offset, saved in the `.kb`), so just call it again for the next slice:

```text
input> .set read-max-mb 1000
input> .read fineweb-edu.txt       ; first ~1 GB
input> .save                       ; checkpoint (offset is saved too)
input> .read fineweb-edu.txt       ; the NEXT ~1 GB -- automatically
input> .save
input> ...                         ; repeat until it reports "reached end of file"
```

A robust loop is `.read` → `.save` → repeat, so a crash never costs more than one slice.
`.rewind FILE` forgets the offset and starts that file over. While a `.read` runs it shows a
**live status line** that updates in place: `reading FILE: X MB / Y MB (Z%)`.

### 3. Use more cores — `read-workers`

`.set read-workers N` (SBCL only; only when `read-extract` is off) fans the per-sentence learning
across N threads: one thread streams + tokenizes the file, N workers learn in parallel into
private tables that are summed at the end. The result is **exact** for the counting learners —
identical to a single-threaded run — and roughly N× faster (minus I/O and merge overhead). On
non-SBCL builds (CLISP/CCL/ECL) this **silently falls back to single-threaded**, so the same
commands work everywhere.

> Memory note: parallel mode prunes the caps **once after** the merge, so peak memory is higher
> than the sequential path. If memory gets tight, lower `read-workers` or `read-max-mb`.

### 4. Shard across processes — `.merge`

Because the bulk learners are additive counters, you can also split a corpus across **separate
runs** and combine them. Read different slices in different processes (each saving its own
`.kb`), then load one and merge the rest:

```text
input> .load shard-1.kb
input> .merge shard-2.kb           ; sum in another shard's learned counts
input> .merge shard-3.kb
input> .save combined.kb
```

`.merge FILE` adds another `.kb`'s co-occurrence / transitions / sentence-starts / facts /
relation counts into the current model **without resetting**. It does *not* merge the neuron /
concept graph (those aren't additive) — only the statistical stores.

---

## c. After training: configuration for regular use

Two things to know:

1. The training toggles affect **learning**, not **answering**. Asking questions
   (`tell me about X`, `is a cat an animal?`, `say dog`, …) does not depend on any of them, so a
   bulk-trained knowledge base already answers as well as it ever will the moment training ends.
2. But the settings **persist in the `.kb`** (section a), so a knowledge base saved in bulk mode
   will *reload* in bulk mode. If you intend to keep **teaching the system interactively** (new
   `input => answer` pairs, new prose), turn the learners back on so that future teaching builds
   the full structures again — then `.save` so the everyday config sticks.

```text
input> .set read-extract on        ; full supervised learning for new lessons
input> .set read-cooccur on        ; restore the similarity learner
input> .set read-relations on
input> .set read-facts on
input> .set read-transitions on
input> .set read-workers 1         ; parallelism only helps bulk .read; not needed live
input> .save mybrain.kb            ; persist the everyday configuration
```

Notes:

- **Caps** (`max-*`) can stay set — they only prune when exceeded and protect memory as you keep
  teaching. Lift them (`.set max-cooccur off`, etc.) only if you want unbounded growth in regular
  use.
- **`read-max-mb`** is irrelevant to everyday use (it only affects `.read`); leave it or set `off`.
- If you would rather not re-flip these by hand, keep **two files**: a training `.kb` (bulk
  config) and an everyday `.kb` (the config above). `.save`/`.load` make the named file the
  active (auto-saved) one.

That's it: train lean and parallel, then flip the learners back on and `.save` for everyday use.
