
(defpackage "induction"
  (:use "COMMON-LISP")
  (:export "COMPLETE"
	   "CONTINUE-SEQUENCE"
	   "INDUCTION-REQUEST-P"
	   "RESPOND-INDUCTION"
	   "*INDUCTION-DIM*"
	   "*INDUCTION-MARGIN*"))

(in-package "induction")
(provide "induction")

(require "data-structures")
(use-package "data-structures")
(require "line-input")
(use-package "line-input")   ; as-words

;;; IN-CONTEXT LEARNING via a two-layer INDUCTION HEAD (productised from
;;; induction-head-experiment.lisp).  The defining trick of modern LLMs is in-context
;;; learning: shown a pattern in the prompt, continue it -- with no weight update.  The
;;; mechanism (from interpretability) is a two-layer attention circuit:
;;;   layer 1  (previous-token head): each position copies in its PREDECESSOR token;
;;;   layer 2  (induction head): given the current token, emit the token that FOLLOWED its
;;;            earlier occurrence  ("... A B ... A" -> "B").
;;; Both layers are Hebbian fast-weight associative memories (outer-product binding, no
;;; backprop); tokens and positions are random codes; a cleanup step (decode + re-encode)
;;; is the nonlinearity between layers.  The circuit is built fresh from the prompt each
;;; call -- it is purely in-context, so there is nothing to persist.

(defparameter *induction-dim* 1024 "Width of the token / position codes.")
(defparameter *induction-margin* 1.4
  "Minimum best/second confidence to commit to a prediction (else NIL -- no pattern).")

;;; --- codes ----------------------------------------------------------------------------
(defun str-hash (s)
  (let ((h 2166136261))
    (loop for c across s do (setf h (logand (* (logxor h (char-code c)) 16777619) #xffffffff)))
    h))

(defparameter *icodes* (make-hash-table :test 'equal))
(defun icode (sym)
  (or (gethash sym *icodes*)
      (setf (gethash sym *icodes*)
	    (let ((state (logand (str-hash sym) #x7fffffff))
		  (v (make-array *induction-dim* :element-type 'double-float)))
	      (when (zerop state) (setf state 1))
	      (dotimes (i *induction-dim* v)
		(setf state (logand (+ (* state 1103515245) 12345) #x7fffffff))
		(setf (aref v i) (if (>= state #x40000000) 1d0 -1d0)))))))

(defun ipos (i) (format nil "#p~d" i))

;;; --- one fast-weight head -------------------------------------------------------------
(defun make-head ()
  (make-array (list *induction-dim* *induction-dim*) :element-type 'double-float :initial-element 0d0))

(defun bind! (head value-sym key-sym)
  (let ((v (icode value-sym)) (k (icode key-sym)))
    (dotimes (i *induction-dim*)
      (let ((vi (aref v i)))
	(dotimes (j *induction-dim*) (incf (aref head i j) (* vi (aref k j))))))))

(defun retrieve (head key-vec)
  (let ((out (make-array *induction-dim* :element-type 'double-float :initial-element 0d0)))
    (dotimes (i *induction-dim* out)
      (let ((s 0d0))
	(dotimes (j *induction-dim*) (incf s (* (aref head i j) (aref key-vec j))))
	(setf (aref out i) s)))))

(defun dot (a b) (let ((s 0d0)) (dotimes (i *induction-dim* s) (incf s (* (aref a i) (aref b i))))))

(defun decode (vec vocab)
  "(values token margin): nearest token to VEC, margin = best/second (large when clean)."
  (let (best bd second)
    (dolist (s vocab)
      (let ((d (dot (icode s) vec)))
	(cond ((or (null best) (> d bd)) (setf second bd best s bd d))
	      ((or (null second) (> d second)) (setf second d)))))
    (values best (if (and second (> second 0)) (/ bd second) 999.0))))

;;; --- the two-layer circuit (built per call, in-context) -------------------------------
(defun build-circuit (seq)
  "Build the induction head (layer 2) for token list SEQ.  Returns (values L2 vocab)."
  (let ((n (length seq)) (vocab (remove-duplicates seq :test #'string=))
	(l1 (make-head)) (l2 (make-head)))
    (loop for i from 0 for tok in seq do (bind! l1 tok (ipos i)))   ; layer 1: token at position
    (loop for i from 1 below n
	  for cur = (nth i seq)
	  for prev = (decode (retrieve l1 (icode (ipos (1- i)))) vocab) ; predecessor (cleaned up)
	  do (bind! l2 cur prev))                                       ; layer 2: predecessor -> current
    (values l2 vocab)))

(defun complete (input)
  "Predict the next token after sequence INPUT (string or word list) by induction, or NIL if
   there is no confident in-context pattern.  Needs at least 3 tokens."
  (let ((seq (as-words input)))
    (when (>= (length seq) 3)
      (multiple-value-bind (l2 vocab) (build-circuit seq)
	(multiple-value-bind (nxt margin) (decode (retrieve l2 (icode (car (last seq)))) vocab)
	  (when (>= margin *induction-margin*) nxt))))))

(defun continue-sequence (input &optional (k 5))
  "Autoregressively emit up to K tokens continuing INPUT; stops early on a low-confidence
   step.  Returns the emitted word list (possibly NIL)."
  (let ((seq (as-words input)))
    (when (>= (length seq) 3)
      (multiple-value-bind (l2 vocab) (build-circuit seq)
	(let (out (cur (car (last seq))))
	  (dotimes (i k (nreverse out))
	    (multiple-value-bind (nxt margin) (decode (retrieve l2 (icode cur)) vocab)
	      (when (< margin *induction-margin*) (return (nreverse out)))
	      (push nxt out) (setf cur nxt))))))))

;;; --- request detection / dispatch (wired into processing:respond) ---------------------
(defun induction-request-p (input)
  "Is INPUT a `continue ...' request with at least two sequence tokens after it?"
  (let ((words (as-words input)))
    (and (consp words) (string= (first words) "continue") (cddr words))))

(defun respond-induction (input)
  "Answer a `continue <sequence>' request: continue the pattern in the sequence, or NIL."
  (let ((words (as-words input)))
    (when (induction-request-p words)
      (continue-sequence (rest words) 6))))
