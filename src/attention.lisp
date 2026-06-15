
(defpackage "attention"
  (:use "COMMON-LISP")
  (:export "NOTE-COPY"
	   "COPY-RESPONSE"
	   "NOTE-TEMPLATE"
	   "COMPOSE"
	   "*ATTN-DIM*"
	   "*COPY-THRESHOLD*"
	   "*COMPOSE-THRESHOLD*"))

(in-package "attention")
(provide "attention")

(require "data-structures")
(use-package "data-structures")

;;; Attention as Hebbian fast weights (the transformer-like facility).  This is the copy/
;;; binding capability the associative net and concept graph lack: the project's original
;;; FIRST GOAL, "say X" -> X for a NOVEL X, by routing the filler BY REFERENCE.
;;;
;;; Each token gets a deterministic random unit vector; each sequence position a "role"
;;; vector.  Binding a sentence is the Hebbian fast-weight matrix  M = sum_i role_i (x)
;;; token_i  (outer products).  Retrieving the token at a role r is  r . M = sum_i
;;; (r . role_i) token_i -- attention with keys = roles, values = token vectors -- then
;;; decode to the nearest token in the sentence.  No backprop.
;;;
;;; What is *learned* (slow, Hebbian, persisted in *copy-cues*): that a cue word triggers
;;; "copy the next word".  When a taught pair's one-word output equals the word right after
;;; some input word, that input word's copy-cue is strengthened.  When a cue is strong
;;; enough, `copy-response' attends to the word after it and emits it -- generalizing to
;;; fillers never seen, exactly as a transformer "induction head" does.

(defparameter *attn-dim* 256 "Dimensionality of the token/role vector codes.")
(defparameter *copy-threshold* 2.0
  "A copy cue must reach this strength (roughly, be confirmed on this many examples)
   before it fires -- so one coincidental match never triggers copying.")

(defparameter *word-vectors* (make-hash-table :test 'equal)) ; token -> unit vector (deterministic cache)
(defparameter *role-vectors* (make-hash-table :test 'eql))   ; position -> role vector (deterministic cache)

(defun str-hash (s)
  "Deterministic FNV-1a hash of string S (so a token always gets the same vector)."
  (let ((h 2166136261))
    (loop for ch across s
	  do (setf h (logand (* (logxor h (char-code ch)) 16777619) #xFFFFFFFF)))
    h))

(defun gen-vector (seed)
  "A deterministic random unit vector from integer SEED (portable LCG -- no dependence on
   the implementation's random number generator, so codes are stable across runs)."
  (let ((state (logand seed #x7FFFFFFF))
	(v (make-array *attn-dim* :element-type 'double-float)))
    (when (zerop state) (setf state 1))
    (flet ((nextf ()
	     (setf state (logand (+ (* state 1103515245) 12345) #x7FFFFFFF))
	     (- (* 2.0d0 (/ state #x7FFFFFFF)) 1.0d0)))
      (dotimes (i *attn-dim*) (setf (aref v i) (nextf)))
      (let ((norm (sqrt (loop for x across v sum (* x x)))))
	(dotimes (i *attn-dim*) (setf (aref v i) (/ (aref v i) norm))))
      v)))

(defun word-vector (word)
  (or (gethash word *word-vectors*)
      (setf (gethash word *word-vectors*) (gen-vector (str-hash word)))))

(defun role (i)
  (or (gethash i *role-vectors*)
      (setf (gethash i *role-vectors*) (gen-vector (+ 777000 i)))))

(defun dot (a b) (loop for x across a for y across b sum (* x y)))

(defun bind-sequence (words)
  "Bind WORDS to positional roles via Hebbian outer products; return the fast-weight
   memory as a closure  query-vector -> retrieved-vector."
  (let ((pairs (loop for w in words for i from 0 collect (cons (role i) (word-vector w)))))
    (lambda (query)
      (let ((out (make-array *attn-dim* :element-type 'double-float :initial-element 0.0d0)))
	(dolist (p pairs out)
	  (let ((wt (dot query (car p))) (tv (cdr p)))
	    (dotimes (i *attn-dim*) (incf (aref out i) (* wt (aref tv i))))))))))

(defun decode (v words)
  "Nearest token in WORDS to vector V (cosine)."
  (let ((best nil) (best-score -2.0d0))
    (dolist (w words best)
      (let* ((wv (word-vector w))
	     (s  (/ (dot v wv) (max 1d-12 (sqrt (dot v v))))))
	(when (> s best-score) (setf best-score s best w))))))

(defun note-copy (input-words output-words)
  "Learn copy cues: if the one-word OUTPUT equals the word right after some input word,
   strengthen that input word as a 'copy the next word' cue.  Reinforced across examples
   (\"say dog\"->dog, \"say cat\"->cat, ...), the cue (\"say\") becomes strong."
  (when (and input-words output-words (null (cdr output-words)))   ; single-word output only
    (let ((w (car output-words)))
      (loop for p from 0 for x in input-words
	    when (and (> p 0) (string= x w))
	    do (incf (gethash (nth (1- p) input-words) *copy-cues* 0.0) 1.0)))))

(defun copy-response (input-words)
  "If a strongly-learned copy cue (>= *copy-threshold*) appears in INPUT-WORDS with a word
   after it, attend to that next word (fast-weight retrieval) and return it as a one-word
   list -- the induction-head / 'say X -> X'.  Otherwise NIL.  Generalizes to fillers
   never seen before, because the filler is routed by reference."
  (let ((best-weight 0.0) (best-pos nil))
    (loop for c from 0 for cue in input-words
	  for weight = (gethash cue *copy-cues* 0.0)
	  when (and (nth (1+ c) input-words)
		    (>= weight *copy-threshold*)
		    (> weight best-weight))
	  do (setf best-weight weight best-pos c))
    (when best-pos
      (let ((memory (bind-sequence input-words)))
	(list (decode (funcall memory (role (1+ best-pos))) input-words))))))

;;; --- Fragment / template composition (Future.md item 2) ------------------------------
;;; So replies aren't canned: learn response TEMPLATES with a slot, and compose novel
;;; sentences by filling the slot with an input word (copy by reference -- the binding
;;; mechanism above).  A template is the taught output with the reused input word replaced
;;; by :slot, keyed by the rest of the input (the frame).  The genuine slot recurs across
;;; examples ("what is a dog"->"a dog is an animal", ...cat...->...cat...), so its template
;;; accumulates strength; coincidental reuses (e.g. a function word) scatter across frames
;;; and stay weak -- the same recurrence-wins trick the concept graph uses for subjects.

(defparameter *compose-threshold* 2.0
  "A template must recur on at least this many examples before it is used to compose, so a
   one-off coincidence never fires.")

(defun frame-string (words omit)
  "WORDS joined into a frame string with the first occurrence of OMIT removed."
  (format nil "~{~a~^ ~}" (remove omit words :test #'string= :count 1)))

(defun note-template (input-words output-words)
  "Learn a response template: if a multi-word OUTPUT reuses an input word, record
   `input-minus-that-word' (frame) -> `output-with-that-word-as-:slot'."
  (when (and input-words output-words (cdr output-words))      ; multi-word output only
    (dolist (w (remove-duplicates input-words :test #'string=))
      (when (member w output-words :test #'string=)
	(let* ((frame (frame-string input-words w))
	       (template (substitute :slot w output-words :test #'string=))
	       (cell (assoc template (gethash frame *templates*) :test #'equal)))
	  (if cell
	      (incf (cdr cell))
	      (push (cons template 1) (gethash frame *templates*))))))))

(defun compose (input-words)
  "Compose a reply from a learned template: for each input word, if removing it yields a
   frame with a template at or above *compose-threshold*, fill that template's :slot with
   the word (copy by reference).  Returns the strongest composed word list, or NIL -- a
   sentence that may never have been seen verbatim, e.g. \"what is a horse\" ->
   (a horse is an animal) from templates taught only on dogs and cats."
  (let ((best 0.0) (out nil))
    (dolist (w input-words)
      (dolist (cell (gethash (frame-string input-words w) *templates*))
	(when (and (>= (cdr cell) *compose-threshold*) (> (cdr cell) best))
	  (setf best (cdr cell) out (substitute w :slot (car cell))))))
    out))
