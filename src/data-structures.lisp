
(defpackage "data-structures"
  (:use "COMMON-LISP")
  (:export "*DICTIONARY*"
	   "*NEXT-NEURON-ID*"
	   "RESET"
	   "NEURON"
	   "NEURON-ID"
	   "NEURON-AXON"
	   "NEURON-EXTENDER"
	   "NAMED-NEURON"
	   "NAMED-NEURON-NAME"
	   "MAKE-NEURON"
	   "MAKE-NAMED-NEURON"
	   "DENDRITE"
	   "DENDRITE-NEURON"
	   "DENDRITE-WEIGHT"
	   "DENDRITE-KIND"
	   "DENDRITE-FROM"
	   "MAKE-DENDRITE"
	   "NEURON-THRESHOLD"
	   "NEURON-CURRENT-VALUE"
	   "*OUTPUT-ROOTS*"
	   "*RESPONSES*"
	   "*ASSOCIATIONS*"
	   "*CONCEPTS*"
	   "*CONCEPT-GRAPH*"
	   "*COPY-CUES*"
	   "*TEMPLATES*"
	   "*LAST-TURN*"
	   "DUMP-DICTIONARY"))

(in-package "data-structures")
(provide "data-structures")

(defparameter *dictionary* (make-hash-table :test 'equal))

(defparameter *next-neuron-id* 0)

;; Registries populated by later phases (Output / Processing); declared here so
;; reset can clear them and so the other packages share one canonical list.
(defparameter *output-roots* nil)   ; output "idea" neurons that generate a response when fired
(defparameter *responses* (make-hash-table :test 'equal)) ; response text -> output root, for chain reuse
(defparameter *associations* nil)   ; association dendrites (kind :association), for fast decay/pruning
(defparameter *concepts* (make-hash-table :test 'equal)) ; "predicate:answer" string -> state neuron (Phase 7)
(defparameter *concept-graph* (make-hash-table :test 'eq)) ; concept neuron -> (neighbor neuron -> weight): the Hebbian concept graph
(defparameter *copy-cues* (make-hash-table :test 'equal)) ; cue word -> strength: learned "copy the next word" triggers (attention head)
(defparameter *templates* (make-hash-table :test 'equal)) ; frame string -> alist of (template . strength): learned response templates (composition)
(defparameter *last-turn* nil) ; the previous resolved input words: conversation memory (transient; not persisted)

(defun reset ()
  (setq *dictionary* (make-hash-table :test 'equal))
  (setq *next-neuron-id* 0)
  (setq *output-roots* nil)
  (setq *responses* (make-hash-table :test 'equal))
  (setq *associations* nil)
  (setq *concepts* (make-hash-table :test 'equal))
  (setq *concept-graph* (make-hash-table :test 'eq))
  (setq *copy-cues* (make-hash-table :test 'equal))
  (setq *templates* (make-hash-table :test 'equal))
  (setq *last-turn* nil))

(defstruct neuron
  "An unnamed neuron"
  (threshold 0.0 :type single-float)
  (current-value 0.0 :type single-float)
  (id (incf *next-neuron-id*) :type fixnum :read-only t) ; only used for debugging purposes
  (extender nil) ; a particular neuron that acts as an extender to this neuron
  (axon nil)) ; dendrite - linked list of weighted output neurons (including the extender)

(defstruct (named-neuron (:include neuron))
  "An named neuron - just adds name to a regular neuron"
  (name "" :type string))

(defstruct dendrite
  "A Dendrite"
  (neuron nil)
  (weight 0.0 :type single-float)
  (kind :sequence :type keyword) ; :sequence | :extender | :association
  (from nil)) ; source neuron, set for :association dendrites so decay/prune can find it

(defun dump-neuron (n level)
  (dotimes (var level)
    (princ "    "))
  (if (named-neuron-p n)
      (if (neuron-extender n)
	  (format t "~d (~a) (E-~d)~%"
		  (neuron-id n)
		  (named-neuron-name n)
		  (neuron-id (neuron-extender n)))
	  (format t "~d (~a)~%"
		  (neuron-id n)
		  (named-neuron-name n)))
      (if (neuron-extender n)
	  (format t "~d (E-~d)~%"
		  (neuron-id n)
		  (neuron-id (neuron-extender n)))
	  (format t "~d~%"
		  (neuron-id n))))
  (dolist (den (neuron-axon n))
    (dump-neuron (dendrite-neuron den) (1+ level))))

(defun dump-dictionary ()
  (loop for value being the hash-values of *dictionary*
     do (dump-neuron value 0)))

