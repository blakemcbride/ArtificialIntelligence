
(defpackage "concepts"
  (:use "COMMON-LISP")
  (:export "RELATE"
	   "NOTE-RELATIONSHIP"
	   "CATEGORY-STRENGTH"
	   "MEMBER-BASELINE"
	   "RECOGNIZED-STRENGTH-P"
	   "RECOGNIZES-P"
	   "INFER-STRENGTH"
	   "INFER-P"
	   "*CONCEPT-HOPS*"
	   "*CONCEPT-DECAY*"
	   "*CONCEPT-FRACTION*"
	   "*CONCEPT-THRESHOLD*"))

(in-package "concepts")
(provide "concepts")

(require "data-structures")
(use-package "data-structures")
(require "line-input")
(use-package "line-input")   ; intern-word

;;; The Processing component's CONCEPT GRAPH (Phase 7) -- where generalization lives.
;;;
;;; A concept is a neuron.  A word concept is the shared *dictionary* neuron (so "dog"
;;; heard, said, or reasoned about is one node).  A "state" concept is a (predicate,
;;; answer) the input/output relationship co-activated -- e.g. has-legs:yes -- interned
;;; in *concepts*.  The ANSWER is part of the state, so has-legs:yes and has-legs:no are
;;; different nodes (a snake's leglessness is genuinely in the data, not inferred).
;;;
;;; `relate' is the Hebbian step: a taught relationship co-activates a subject and a
;;; state, strengthening the (undirected, weighted) edge between them in *concept-graph*.
;;; Two subjects become similar by sharing state neighbours; that shared-neighbour
;;; structure IS the "general idea".
;;;
;;; Generalization = degree-normalized, decayed spreading activation: how strongly does a
;;; subject's activation reach a (predicate:answer) state?  A novel word reaches it via
;;; the states it shares with the taught members; a dissimilar word doesn't.  Degree
;;; normalization dilutes promiscuous hubs (is-animal:yes, shared by everything) so the
;;; discriminating states (has-legs:yes) decide -- which is what excludes "even snakes".
;;;
;;; NOTE: edges are kept in *concept-graph* (a dedicated weighted table over concept
;;; neurons), not in the neurons' axon slots, so the input network's traversal/dump are
;;; untouched.  It is mathematically a Hebbian synaptic graph.  Auto-populating it from
;;; raw sentences (slot-free subject/predicate discovery) and persisting it are the next
;;; steps -- see Plan.md Phase 7.

(defparameter *concept-hops*      5   "How many hops spreading activation propagates.")
(defparameter *concept-decay*     0.6 "Per-hop activation decay.")
(defparameter *concept-fraction* 0.15
  "Adaptive membership ratio: a subject is recognized for a category if its strength is at
   least this fraction of the category's directly-taught members' mean strength -- so the
   cutoff scales with each category instead of being a fixed number.")
(defparameter *concept-threshold* 0.005
  "Absolute floor: strengths below this never count, even if a category's baseline is tiny.")

(defun state-key (predicate answer)
  (concatenate 'string predicate ":" answer))

(defun intern-state (key)
  "The concept neuron for a (predicate:answer) state, interned by KEY in *concepts*."
  (or (gethash key *concepts*)
      (setf (gethash key *concepts*) (make-named-neuron :name key))))

(defun cg-edge (a b w)
  "Strengthen the undirected concept edge A <-> B by W in *concept-graph*."
  (flet ((bump (x y)
	   (let ((tab (or (gethash x *concept-graph*)
			  (setf (gethash x *concept-graph*) (make-hash-table :test 'eq)))))
	     (incf (gethash y tab 0.0) w))))
    (bump a b)
    (bump b a)))

(defun cg-degree (n)
  (let ((tab (gethash n *concept-graph*)) (sum 0.0))
    (when tab (maphash (lambda (k v) (declare (ignore k)) (incf sum v)) tab))
    sum))

(defun relate (subject predicate answer)
  "Hebbian: record that word SUBJECT had relationship PREDICATE with ANSWER, by
   strengthening the edge between the subject concept and the (predicate:answer) state.
   The concept-graph counterpart of one taught input->output relationship."
  (cg-edge (intern-word subject)
	   (intern-state (state-key predicate answer))
	   1.0))

(defun spread (seed)
  "Degree-normalized, decayed spreading activation from SEED over *concept-graph*.
   Returns a hash node -> total activation reached."
  (let ((current (make-hash-table :test 'eq))
	(total   (make-hash-table :test 'eq)))
    (setf (gethash seed current) 1.0)
    (dotimes (h *concept-hops*)
      (let ((next (make-hash-table :test 'eq)))
	(maphash
	 (lambda (m amt)
	   (let ((tab (gethash m *concept-graph*)) (deg (cg-degree m)))
	     (when (and tab (plusp deg))
	       (maphash (lambda (n w)
			  (incf (gethash n next 0.0) (* amt (/ w deg) *concept-decay*)))
			tab))))
	 current)
	(maphash (lambda (n amt) (incf (gethash n total 0.0) amt)) next)
	(setf current next)))
    total))

(defun strength-between (subj-neuron state-neuron)
  "Spreading-activation strength from SUBJ-NEURON reaching STATE-NEURON (0.0 if either nil)."
  (if (and subj-neuron state-neuron)
      (gethash state-neuron (spread subj-neuron) 0.0)
      0.0))

(defun category-strength (subject predicate answer)
  "How strongly does word SUBJECT belong to the category defined by (PREDICATE ANSWER)?
   = spreading activation from the subject concept reaching the (predicate:answer) state.
   Generalizes through shared concept neighbours; returns 0.0 for unknown subjects."
  (strength-between (gethash subject *dictionary*)
		    (gethash (state-key predicate answer) *concepts*)))

(defun member-baseline (predicate answer)
  "Mean strength of the (predicate:answer) category's directly-taught members -- the
   yardstick the adaptive threshold scales.  0.0 if the category has no members.  (The
   graph is bipartite subjects<->states, so a state's neighbours ARE its members.)"
  (let ((st (gethash (state-key predicate answer) *concepts*)))
    (if (null st)
	0.0
	(let ((tab (gethash st *concept-graph*)) (sum 0.0) (n 0))
	  (when tab
	    (maphash (lambda (m w) (declare (ignore w))
		       (incf sum (strength-between m st))
		       (incf n))
		     tab))
	  (if (plusp n) (/ sum n) 0.0)))))

(defun recognized-strength-p (strength predicate answer)
  "Adaptive test: is STRENGTH enough to count as membership in (PREDICATE ANSWER)?
   Cutoff = max(absolute floor, *concept-fraction* x the category's member baseline)."
  (let ((base (member-baseline predicate answer)))
    (and (plusp base)
	 (>= strength (max *concept-threshold* (* *concept-fraction* base))))))

(defun recognizes-p (subject predicate answer)
  "Does SUBJECT generalize into the (PREDICATE ANSWER) category (adaptive cutoff)?"
  (recognized-strength-p (category-strength subject predicate answer) predicate answer))

;;; --- Auto-population from raw sentences + slot-free queries (Phase 7 #1) ---------
;;; We never decide which word is "the subject".  Teaching a relationship relates EVERY
;;; input word to the frame made of the OTHER input words + the answer; shared frames
;;; then make similar subjects converge.  A query tries every word-as-subject and takes
;;; the strongest -- the real subject wins because only its frame is one the taught
;;; members share.  (Function words land in many frames -> high degree -> diluted.)

(defun frame-without (words i)
  "WORDS joined into a frame string with the i-th word removed."
  (format nil "~{~a~^ ~}"
	  (loop for j from 0 for x in words unless (= j i) collect x)))

(defun note-relationship (input-words output-words)
  "Auto-populate the concept graph from one taught input->output pair: relate each input
   word to (the complement frame : the answer).  Called by processing:learn."
  (when (and input-words output-words)
    (let ((answer (format nil "~{~a~^ ~}" output-words)))
      (loop for i from 0 for w in input-words
	    do (relate w (frame-without input-words i) answer)))))

(defun infer-strength (input-words answer-words)
  "Slot-free query: treating each input word in turn as the subject, how strongly does
   the rest-of-sentence frame categorize it as ANSWER?  Returns (values best-strength
   winning-subject winning-frame)."
  (let ((answer (format nil "~{~a~^ ~}" answer-words))
	(best 0.0) (who nil) (frame nil))
    (loop for i from 0 for w in input-words
	  for fr = (frame-without input-words i)
	  for s = (category-strength w fr answer)
	  when (> s best) do (setf best s who w frame fr))
    (values best who frame)))

(defun infer-p (input-words answer-words)
  "Adaptive, slot-free: does the best word-as-subject decomposition clear its category's
   member baseline?"
  (multiple-value-bind (s who frame) (infer-strength input-words answer-words)
    (declare (ignore who))
    (and frame
	 (recognized-strength-p s frame (format nil "~{~a~^ ~}" answer-words)))))
