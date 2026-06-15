# Tutorial — Teaching the Hebbian AI

This is a hands-on guide to using the system: loading it, teaching it, asking what it
learned, and saving a knowledge base. Every command below is real and runnable.

If you want the *why* rather than the *how*, read `Plan.md` (design + rationale) and
`notes/Overview.txt` (the original vision) alongside this.

---

## What this is, in one paragraph

This is a small, experimental AI that **learns while you use it** — there is no separate
training phase. You teach it **input → response** relationships (interactively or from a
file); it stores them in a network of neuron-like nodes connected by weighted edges, and it
**generalizes**: shown a new input similar to things it already knows, it responds
sensibly; shown something unrelated, it doesn't. It deliberately uses only **local,
Hebbian-style learning** ("cells that fire together wire together," strengthen what works,
let the unused fade) — **no backpropagation**. Those two properties — continual learning
and no backprop — are the whole point.

---

## Quick start (60 seconds)

From a shell, in the `src/` directory, start SBCL and paste these:

```lisp
;; 1. load the system (run this from inside src/)
(dolist (f '("data-structures" "line-input" "input" "output"
             "concepts" "processing" "persist" "ai"))
  (load (format nil "~a.lisp" f)))

;; 2. import the starter knowledge base (105 facts)
(train-from-file "training-set.txt")

;; 3. ask something it was taught
(infer-p "Do dogs walk on their legs?" "yes")    ; => T

;; 4. ask about a creature it was NEVER taught to walk -- it generalizes
(infer-p "Do horses walk on their legs?" "yes")  ; => T

;; 5. ask about a legless animal -- it is excluded
(infer-p "Do snakes walk on their legs?" "yes")  ; => NIL
```

If those five lines behaved as shown, everything works. The rest of this tutorial explains
each piece.

---

## 1. Loading the system

The code is eight source files in `src/`, each its own package, plus `ai.lisp` (the
top-level). They must load in dependency order. **Start your Lisp from inside `src/`** so
that relative paths (the training file, the saved knowledge base) resolve there.

**SBCL (the primary target):**

```lisp
(dolist (f '("data-structures" "line-input" "input" "output"
             "concepts" "processing" "persist" "ai"))
  (load (format nil "~a.lisp" f)))
```

**CLISP** can use the one-liner `(load "ai.lisp")` (it finds the other files
automatically); SBCL/CCL/ECL need the explicit list above.

After loading, all the functions below are available at the REPL (they are imported into
`COMMON-LISP-USER`). Note: there is no `ai` package — call `(main)`, not `(ai::main)`.

---

## 2. Your first conversation — the teaching loop

`(main)` starts an interactive teaching session. Each turn it reads a sentence, tells you
what it would answer, then reads your correction (or a confirmation) and learns from it.

```text
* (main)
Teaching loop -- type a sentence, then on the next line the correct
response (or yes/y/right/correct/ok to confirm a correct guess).  Stop with quit.

input> hello.
  guess: (I don't know)
teach> hi there.
  (learned)

input> hello.
  guess: hi there
teach> yes.
  (reinforced)

input> quit.
(memory saved to ai-network.sexp)
```

Things to know:

- **Every line must end with `.`, `!`, or `?`** — including `quit.` / `exit.`. Input is
  lowercased automatically.
- The first turn it has no idea, so it says **"(I don't know)"**; you supply the answer and
  it **learns**. The second time you say `hello.` it **recalls** "hi there"; typing a
  confirmation word (`yes`, `ok`, …) **reinforces** that pathway instead of retyping the
  answer.
- On `quit.` it **saves everything** to `ai-network.sexp` in the current directory and
  prints its internal neuron tree. Next time you run `(main)` it **loads that file back**,
  so the system remembers across sessions.

You can call `(main)` again and it will pick up where it left off.

---

## 3. Teaching from a file

Typing facts one at a time is slow. A **training set** is a plain text file with one
relationship per line:

```text
input phrase => answer
```

Blank lines and lines starting with `#` or `;` are ignored, so you can comment and group.
A starter set ships in `src/training-set.txt` (about 105 facts across animals, vehicles,
furniture, and objects). Import it with:

```lisp
(train-from-file "training-set.txt")
;; trained on 105 relationships from training-set.txt
;; => 105
```

Each line is learned exactly as a teaching-loop turn would be. To author your own set, just
make a file like:

```text
# my facts
is the sky blue => yes
do birds fly => yes
do fish fly => no
```

…and `(train-from-file "my-facts.txt")`.

You can also teach a single pair from code — `learn` takes an input and an answer (each a
sentence string or a word list) and returns what the system *would* have answered *before*
this lesson (handy for scoring). On a blank system the first answer is `NIL`:

```lisp
(reset)
(learn "Do cats purr?" "yes")   ; => NIL     (a blank system knows nothing yet)
(learn "Do cats purr?" "yes")   ; => ("yes")  (now it recalls it -- and reinforces)
```

---

## 4. Asking what it learned

There are two ways to query, and they answer different questions.

### Recall / direct response — `respond`

`respond` gives the system's actual answer to an input, the way the teaching loop does
(driven by the direct input→output associations). Continuing from the step above (where we
taught `cats purr`):

```lisp
(respond "Do cats purr?")     ; => ("yes")
(respond "What is this?")     ; => NIL   (no idea)
```

It returns the answer as a list of words, or `NIL` for "I don't know."

### Generalization — `infer-p` and `infer-strength`

This is the interesting part. `infer-p` asks **"does this generalize?"** — it works even for
inputs the system was *never directly taught*, by reasoning over the **concept graph** (see
§6). You give it the question and a candidate answer:

```lisp
;; horses were taught they have legs / feet / are animals -- but NEVER that they walk.
(infer-p "Do horses walk on their legs?" "yes")   ; => T   (generalized!)

;; snakes were taught they are legless -- so they are excluded
(infer-p "Do snakes walk on their legs?" "yes")   ; => NIL
```

`infer-strength` returns the underlying number (higher = stronger membership) and, as a
second value, which word it treated as the subject:

```lisp
(infer-strength "Do horses walk on their legs?" "yes")
;; => 0.025...   and second value "horses"
```

You don't tell it which word is the "subject" — it tries every word as the subject and
keeps the strongest reading. That is what "slot-free" means here.

---

## 5. A complete worked example

Start fresh and train on the starter set:

```lisp
(reset)                              ; wipe memory
(train-from-file "training-set.txt") ; => 105
```

Now ask the same question about several things. Only `dogs/cats/goats/cows/lions/people`
were taught they walk on their legs; `horses` and `birds` were taught everything *except*
that:

```lisp
(infer-p "Do dogs walk on their legs?"   "yes")  ; T  (recall)
(infer-p "Do horses walk on their legs?" "yes")  ; T  (generalized in)
(infer-p "Do birds walk on their legs?"  "yes")  ; T  (generalized in)
(infer-p "Do snakes walk on their legs?" "yes")  ; NIL (legless animal)
(infer-p "Do tables walk on their legs?" "yes")  ; NIL (has legs, but not an animal, doesn't move)
(infer-p "Do rocks walk on their legs?"  "yes")  ; NIL
(infer-p "Do glorps walk on their legs?" "yes")  ; NIL (unknown word)
```

That is the goal in action: it **teases out the general idea** ("legged, walking, moving
animals") and includes new members of that category while excluding non-members — **even
snakes**, which are animals but lack legs.

---

## 6. The "even snakes" idea — how generalization works

(Conceptual; skip if you only want to use it.)

- Teaching `do dogs have legs => yes` records a weighted edge between the concept **dog**
  and the **state** `has-legs:yes`. The answer is part of the state, so `has-legs:yes` and
  `has-legs:no` are *different* nodes.
- Many such facts make **dog**, **goat**, **horse**… all connect to the same states
  (`has-legs:yes`, `is-animal:yes`, …). **Sharing those neighbours is what makes them
  similar** — there is no separate "category" object; the category *is* the shared wiring.
- To answer "do horses walk on their legs?", activation **spreads** from `horse` through the
  states it shares with the taught walkers and reaches the `walks-on-legs:yes` state.
  `horse` reaches it (shares legs/feet/running with the walkers); `snake` barely does (it
  shares only "animal/moves/breathes" and is explicitly *legless*); `rock` and unknown
  words reach nothing.
- A word counts as "in" when its strength clears an **adaptive** bar — a fraction of the
  category's own taught members' strength — so there is no magic global cutoff.

The crucial, honest point (and your own principle): the system can only **exclude** snakes
because it has **learned snakes' distinctive traits** (legless, can't run) from other
facts. Generalization is a property of the *whole* knowledge base, not of three sentences.
The richer and more discriminating the training, the sharper the categories.

---

## 7. Saving and loading a knowledge base

The whole knowledge base — input network, responses, associations, **and** the concept
graph — round-trips through a single file.

```lisp
(export-kb "my-kb.sexp")    ; write everything to my-kb.sexp
(reset)                     ; wipe memory
(import-kb "my-kb.sexp")    ; load it back; => T  (NIL if the file is missing)
```

`(main)` does this automatically for the default file `ai-network.sexp` (loads on entry,
saves on exit), so an interactive session persists on its own. Use `export-kb`/`import-kb`
when you want to manage named knowledge bases yourself. The file is a human-readable
s-expression — you can peek at it.

---

## 8. Useful functions at a glance

| You want to… | Call |
|---|---|
| Start an interactive teaching session | `(main)` |
| Teach one pair from code | `(learn "Cats purr." "yes")` |
| Bulk-teach from a file | `(train-from-file "file.txt")` |
| Get the system's answer to an input | `(respond "Do cats purr?")` |
| Ask if an input generalizes to an answer | `(infer-p "Do horses walk on their legs?" "yes")` |
| Get the generalization strength (+ subject) | `(infer-strength "Do horses walk on their legs?" "yes")` |
| Save / load a knowledge base | `(export-kb "f.sexp")` / `(import-kb "f.sexp")` |
| Wipe all memory | `(reset)` |
| Print the internal neuron tree | `(dump-dictionary)` |

Inputs and answers can be plain **sentence strings** — the system tokenizes them
(lowercases, drops `.` `!` `?`) — or, if you prefer, lists of lowercase words. So
`(infer-p "Do horses run?" "yes")` and `(infer-p '("do" "horses" "run") '("yes"))` are
equivalent.

---

## 9. Tuning knobs

These are special variables you can `setf`/`let`-bind to change behavior (defaults shown):

- `*save-file*` (`"ai-network.sexp"`) — the file `(main)` auto-loads/saves.
- `*concept-fraction*` (`0.15`) — how close to the taught members a new word must be to
  count as a category member. Raise it to be stricter (fewer things generalize in), lower
  it to be looser.
- `*concept-hops*` (`5`), `*concept-decay*` (`0.6`) — how far / how strongly activation
  spreads through the concept graph.
- `*assoc-decay*` (`0.02`) — how fast unused input→output associations fade between turns.

Example: make generalization stricter just for one query:

```lisp
(let ((*concept-fraction* 0.3))
  (infer-p "Do horses walk on their legs?" "yes"))
```

---

## 10. Running the test suite

From `src/`:

```sh
make test
```

It runs ~100 assertions under SBCL covering every component (input structure, output
generation, associations, inference, reinforcement/decay, the teaching loop, persistence,
and concept-graph generalization). A green run means the whole system is healthy.

---

## 11. Where to go next

- **`Plan.md`** — the full design: the three-part architecture, the learning rule, and
  §3.5 / Phase 7 on how generalization-with-exclusion actually works.
- **`CLAUDE.md`** — a concise map of the code: every file, package, and convention.
- **`notes/Overview.txt`** — the original three-component vision in Blake's words.
- **`src/concept-graph-experiment.lisp`** — a standalone experiment you can `sbcl --script`
  to see the generalization numbers directly.

Happy teaching.
