
(defpackage "output"
  (:use "COMMON-LISP")
  (:export "BUILD-OUTPUT-STRUCTURE"
	   "PRODUCE-OUTPUT"
	   "OUTPUT-SENTENCE"))

(in-package "output")
(provide "output")

(require "data-structures")
(use-package "data-structures")

;;; The Output component (Phase 1 of Plan.md) is the mirror image of the input
;;; component.  The input network is convergent -- many word neurons fan IN to a
;;; "meaning" neuron.  An output structure is divergent -- one root "idea" neuron
;;; fans OUT into a word sequence:
;;;
;;;   [root] --extender--> [w1] --extender--> [w2] --extender--> ... --> nil
;;;                        "hi"               "there"
;;;
;;; The root is a plain neuron; each chain node is a named-neuron carrying the word
;;; to emit.  Generation just walks the extender links forward emitting names -- the
;;; same primitive the input side reads backward.  Output nodes are fresh neurons,
;;; deliberately NOT the input word neurons in *dictionary*, so the input and output
;;; networks never share an extender chain.

(defun response-key (words)
  "Canonical *responses* hash key for a response WORD list."
  (format nil "~{~a~^ ~}" words))

(defun link-extender (prior next)
  "Make NEXT the extender of PRIOR, also recording it as an outgoing dendrite (per
   the extender convention used throughout the network).  Returns NEXT."
  (setf (neuron-extender prior) next)
  (push (make-dendrite :neuron next :kind :extender) (neuron-axon prior))
  next)

(defun build-output-structure (words)
  "Build -- or reuse -- an output chain that generates the response WORDS (a list of
   strings) and return its root neuron.  Identical responses are interned in
   *responses* so they share one chain; every root is registered in *output-roots*."
  (let ((key (response-key words)))
    (or (gethash key *responses*)
	(let ((root (make-neuron))
	      (node nil))
	  (setq node root)
	  (dolist (w words)
	    (setq node (link-extender node (make-named-neuron :name w))))
	  (setf (gethash key *responses*) root)
	  (push root *output-roots*)
	  root))))

(defun produce-output (root)
  "Walk ROOT's extender chain and return the list of emitted word strings."
  (let (words)
    (do ((n (neuron-extender root) (neuron-extender n)))
	((null n) (nreverse words))
      (when (typep n 'named-neuron)
	(push (named-neuron-name n) words)))))

(defun output-sentence (root)
  "Produce ROOT's response as a single space-separated display string."
  (format nil "~{~a~^ ~}" (produce-output root)))
