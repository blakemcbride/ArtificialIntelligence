
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
	   "*OP-TEMPLATES*"
	   "*COOCCUR*"
	   "*VCACHE*"
	   "*VEC-MEAN*"
	   "*FACTS-LEARNED*"
	   "*TRANSITIONS*"
	   "*SENTENCE-STARTS*"
	   "*FACTS*"
	   "*REL-LINKS*"
	   "*REL-HEAD*"
	   "*REL-FREQ*"
	   "*REL-SENTENCES*"
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
(defparameter *op-templates* (make-hash-table :test 'equal)) ; (frame . op) -> strength: learned question->operation mappings
(defparameter *cooccur* (make-hash-table :test 'equal)) ; word -> (hash word -> count): co-occurrence counts behind the distributed concept vectors
(defparameter *vcache* (make-hash-table :test 'equal)) ; derived concept-vector cache (rebuilt from *cooccur*; cleared on reset/reload)
(defparameter *vec-mean* nil)                          ; cached global mean of concept vectors
(defparameter *facts-learned* 0)                       ; cumulative count of facts learned (any source); persisted
(defparameter *transitions* (make-hash-table :test 'equal)) ; word -> (next-word -> count): sequential model for generation (Phase 8)
(defparameter *sentence-starts* (make-hash-table :test 'equal)) ; word -> count: how often a word starts a sentence
(defparameter *facts* (make-hash-table :test 'equal)) ; (subject relation object) -> strength: declarative triples for generation
(defparameter *rel-links* (make-hash-table :test 'equal)) ; connector -> (hash "subj|cat" -> (subj . cat)): learned relation discovery (Phase 9)
(defparameter *rel-head* (make-hash-table :test 'equal))  ; word -> times it served as a subject/category head
(defparameter *rel-freq* (make-hash-table :test 'equal))  ; word -> # sentences it appears in (for function-word discovery)
(defparameter *rel-sentences* 0)                          ; sentences fed to the relation learner

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
  (setq *last-turn* nil)
  (setq *op-templates* (make-hash-table :test 'equal))
  (setq *cooccur* (make-hash-table :test 'equal))
  (setq *vcache* (make-hash-table :test 'equal))
  (setq *vec-mean* nil)
  (setq *facts-learned* 0)
  (setq *transitions* (make-hash-table :test 'equal))
  (setq *sentence-starts* (make-hash-table :test 'equal))
  (setq *facts* (make-hash-table :test 'equal))
  (setq *rel-links* (make-hash-table :test 'equal))
  (setq *rel-head* (make-hash-table :test 'equal))
  (setq *rel-freq* (make-hash-table :test 'equal))
  (setq *rel-sentences* 0))

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

