# Artificial Intelligence

This is an experimental **LLM (large language model) design** — an attempt at a
fundamentally different kind of language-learning AI, built to pursue two goals that
today's mainstream LLMs do **not** meet:

1. **Continual learning — no separate training phase.** A conventional LLM is frozen once
   it is trained; using it never changes it. This system learns *while it is used*: you
   teach it one interaction at a time and it keeps what works. Learning *is* using it.

2. **No backpropagation.** Mainstream LLMs adjust their weights with backpropagation — a
   global error signal the human brain almost certainly does not use. This system learns
   with only **local, Hebbian-style rules** ("cells that fire together wire together";
   strengthen what succeeds, let the unused fade), which is closer to how real neurons are
   thought to work.

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

For many years my implementation covered only the input component; I never reached the
other two, though the input section worked as a first pass.

## Where it stands now

Rather than let the ideas disappear — and given the rise of ChatGPT and my advancing years
— I continued the project with the help of **Claude Code**, Anthropic's command-line coding
agent. **Claude Code wrote much of the current code**, working from my design and under my
direction: it built out the Processing and Output components, a Hebbian *concept graph*
that produces the generalization described above, an interactive teaching loop,
knowledge-base persistence, an importable training set, and an automated test suite.

The system is now functional end to end. You can teach it, and it learns, generalizes to
things it was never directly taught, and remembers across sessions — all through local
Hebbian learning, with no backpropagation and no separate training phase.

The design and rationale are documented in `Plan.md`; a map of the code is in `CLAUDE.md`.

## Getting started

The code is Common Lisp (developed primarily with SBCL). For a hands-on walkthrough, read
**`tutorial/tutorial.md`**. In brief, from a REPL started in the `src/` directory:

```lisp
;; load the system
(dolist (f '("data-structures" "line-input" "input" "output"
             "concepts" "processing" "persist" "ai"))
  (load (format nil "~a.lisp" f)))

(train-from-file "training-set.txt")               ; learn a starter knowledge base
(infer-p "Do horses walk on their legs?" "yes")    ; => T   (generalized)
(infer-p "Do snakes walk on their legs?" "yes")    ; => NIL (excluded)

(main)   ; or start an interactive teaching session
```

Run the test suite from `src/` with `make test`.

## Repository structure

* `src/` — the Common Lisp implementation (the live system), plus `training-set.txt`, a
  starter knowledge base.
* `tutorial/` — a hands-on tutorial.
* `notes/` — my original, unfiltered notes and an overview of the intended design.
* `C++/` — the original, incomplete C++ attempt (abandoned; kept for history).
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
