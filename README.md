# Artificial Intelligence

This is an experimental **LLM (large language model) design** — an attempt at a
fundamentally different kind of language-learning AI, built to pursue two goals that
today's mainstream LLMs do **not** meet:

1. **Continual learning — no separate, nonvolatile pre-training phase.** A conventional LLM
   is frozen once it is trained; using it never changes it. This system learns *while it is
   used*: you teach it one interaction at a time and it keeps what works. Learning *is*
   using it.
   *Why:* so the system gets **continually smarter** over time, instead of holding a fixed,
   static body of knowledge that only ever changes when someone retrains it from scratch.

2. **Hebbian learning instead of backpropagation.** Mainstream LLMs adjust their weights
   with backpropagation. This system uses only **local, Hebbian-style rules** ("cells that
   fire together wire together"; strengthen what succeeds, let the unused fade).
   *Why — two reasons, and note neither is "Hebbian learns better":*
   - **Biological plausibility.** Backpropagation is, as a pure learning algorithm, almost
     certainly *better* than Hebbian learning — but the human brain cannot plausibly be
     running it (there is no global backward pass or exact gradient transport in neural
     tissue). The brain much more likely uses something local, like Hebbian learning. The
     aim here is a brain-like mechanism, not the best optimizer.
   - **Simplicity and cost.** Hebbian learning is *far, far* simpler and far less
     computationally expensive than backpropagation — a few local weight updates per fact,
     with no backward pass over the whole network.

It is a research prototype, not a production model — an alternative architecture rather
than a transformer. Concretely, it is a network of neuron-like nodes that you teach
**input → response** pairs, and that **generalizes** an idea from examples: taught that
dogs, people, and goats "walk on their legs," it will recognize that a never-taught
*horse* does too — while a *snake* does not.

## Background

I've had a life-long interest in AI. In 1994 I began developing ideas on how such a system
could be built; in 2014 I started implementing it in C++. The C++ version became unwieldy
around memory management, so I dropped it and rewrote it in Common Lisp, which proved far
simpler and shorter for the same functionality.

The architecture has three major components:

1. **Input** — turn a sentence into a network of neurons.
2. **Processing** — associate inputs with outputs.
3. **Output** — roughly the inverse of the input component: turn an idea back into words.

For a visual overview of how these fit together — the knowledge sources, the learning and
inference paths, the memory stores, and persistence — see the block diagram in
[`notes/BlockDiagram.tex`](notes/BlockDiagram.tex) (a compiled
[`notes/BlockDiagram.pdf`](notes/BlockDiagram.pdf) is included).

For many years my implementation covered only the input component; I never reached the
other two, though the input section worked as a first pass.

## Where it stands now

Rather than let the ideas disappear — and given the rise of ChatGPT and my advancing years
— I continued the project with the help of **Claude Code**, Anthropic's command-line coding
agent. **Claude Code wrote much of the current code**, working from my design and under my
direction: it built out the Processing and Output components, a Hebbian *concept graph*
that produces the category generalization described above, a transformer-like *attention
head* — built as Hebbian fast weights — that handles copy/binding (the original goal of
`say X → X` for a word never seen), *conversation memory* that resolves follow-ups against
the previous turn, *template composition* that assembles novel replies from learned
fragments, *learned operations* over its own knowledge (taught once, "how many animals do
you know" counts them — generically, for any category — and the answer rises as it learns),
*distributed concept vectors* that give non-brittle similarity ("what is similar to a lion"
→ other animals), an interactive teaching loop, knowledge-base persistence, an importable
training set, and an automated test suite.

The system is now functional end to end. You can teach it, and it learns, generalizes to
things it was never directly taught, and remembers across sessions — all through local
Hebbian learning, with no backpropagation and no separate training phase. It also ships
with a broad **starter knowledge base** (`src/knowledge-base.txt`, ~2,100 facts — including world geography and history) that `main`
learns automatically the first time it runs, so a fresh system already answers questions,
copies, and composes out of the box.

The design and rationale are documented in `Plan.md`; a map of the code is in `CLAUDE.md`.

## Getting started

The code is Common Lisp (developed primarily with SBCL). For a hands-on walkthrough, read
**`tutorial/tutorial.md`**. In brief, from a REPL started in the `src/` directory:

```lisp
;; load the system (run this from inside src/)
(load "load.lisp")

(train-from-file "generalization-test.txt")               ; learn a starter knowledge base
(infer-p "Do horses walk on their legs?" "yes")    ; => T   (generalized category)
(infer-p "Do snakes walk on their legs?" "yes")    ; => NIL (excluded)

;; copy / binding -- the original goal, generalizing to a word never seen:
(learn "say dog" "dog") (learn "say cat" "cat") (learn "say house" "house")
(respond "say car")                                ; => ("car")

;; conversation memory -- a follow-up leans on the previous turn:
(ask "do dogs have legs?")                         ; => ("yes")
(ask "and cats?")                                  ; => ("yes")   (i.e. "do cats have legs?")

;; composition -- a reply assembled from learned fragments, never seen verbatim:
(learn "what is a dog" "a dog is an animal") (learn "what is a cat" "a cat is an animal")
(respond "what is a horse")                        ; => ("a" "horse" "is" "an" "animal")

;; learned operations -- teach what the question MEANS once; the answer is computed over
;; current knowledge and generalizes to any category:
(learn "how many animals do you know" "count animals")
(respond "how many animals do you know")           ; => ("110")   (rises as you teach more)
(respond "how many colors do you know")            ; => ("11")    (generalized via the slot)

;; distributed similarity, learned from co-occurrence (meaning as geometry):
(learn "what is similar to dog" "similar dog")
(respond "what is similar to lion")                ; => (monkey kangaroo dolphin gorilla ...)

(main)   ; interactive session; on a fresh start it auto-learns knowledge-base.txt first
```

Run the test suite from `src/` with `make test`.

## Repository structure

* `src/` — the Common Lisp implementation (the live system), plus `knowledge-base.txt` (the
  broad starter KB auto-learned on first run) and `generalization-test.txt` (the focused
  generalization demo used by the tests).
* `tutorial/` — a hands-on tutorial.
* `notes/` — my original, unfiltered notes, an overview of the intended design, and a
  system block diagram (`BlockDiagram.tex` / `.pdf`).
* `Plan.md` — the design and the phased build plan.
* `CLAUDE.md` — a map of the codebase and its conventions.

## Notes

The `notes/` directory contains my contemporaneous notes, largely in no particular order —
earlier notes often follow later ones. It also contains an overview of the intended design.

## License

Released into the public domain.

Blake McBride
blake@mcbridemail.com
1994–present
