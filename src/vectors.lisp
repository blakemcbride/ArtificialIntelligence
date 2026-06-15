
(defpackage "vectors"
  (:use "COMMON-LISP")
  (:export "NOTE-COOCCURRENCE"
	   "CONCEPT-VECTOR"
	   "SIMILARITY"
	   "NEAREST"
	   "CENTROID-OF"
	   "SIM-TO-VECTOR"
	   "*VEC-DIM*"
	   "*VEC-THRESHOLD*"))

(in-package "vectors")
(provide "vectors")

(require "data-structures")
(use-package "data-structures")

;;; Distributed concept vectors (the CMAC / sparse-distributed-memory / hyperdimensional
;;; direction).  Each concept is a high-dimensional vector built ONLINE by superposing the
;;; random codes of the words it co-occurs with -- learned by accumulation, no backprop,
;;; a few vector adds per fact.  Similar concepts get similar vectors because they share
;;; company; meaning becomes geometry.  The model we keep is the sparse co-occurrence
;;; counts in *cooccur*; vectors are derived from it (and cached), so persistence is cheap.

(defparameter *vec-dim* 2048)
(defparameter *vec-threshold* 0.15
  "Cosine cutoff for geometric category membership -- graded and fuzzy by nature.")

;;; deterministic random unit code for a word (its fixed distributed pattern)
(defparameter *codes* (make-hash-table :test 'equal))
(defun str-hash (s)
  (let ((h 2166136261))
    (loop for ch across s do (setf h (logand (* (logxor h (char-code ch)) 16777619) #xFFFFFFFF)))
    h))
(defun code (word)
  (or (gethash word *codes*)
      (setf (gethash word *codes*)
	    (let ((st (logand (str-hash word) #x7FFFFFFF))
		  (v (make-array *vec-dim* :element-type 'double-float)))
	      (when (zerop st) (setf st 1))
	      (flet ((nx () (setf st (logand (+ (* st 1103515245) 12345) #x7FFFFFFF))
			 (- (* 2.0d0 (/ st #x7FFFFFFF)) 1.0d0)))
		(dotimes (i *vec-dim*) (setf (aref v i) (nx))))
	      v))))

;;; --- online learning: accumulate co-occurrence counts -------------------------------
;; *vcache* and *vec-mean* live in data-structures so `reset' and load-network clear them.

(defun note-cooccurrence (input-words output-words)
  "Continual, local update: each word in this fact (input + answer) records that it
   co-occurred with the others.  Cheap -- just counter bumps."
  (let ((ws (remove-duplicates (append input-words output-words) :test #'string=)))
    (dolist (w ws)
      (let ((tab (or (gethash w *cooccur*) (setf (gethash w *cooccur*) (make-hash-table :test 'equal)))))
	(dolist (o ws)
	  (unless (string= o w)
	    (incf (gethash o tab 0)))))))
  (clrhash *vcache*) (setf *vec-mean* nil))                 ; invalidate derived vectors

;;; --- derived vectors ----------------------------------------------------------------
(defun df (o)
  "How many distinct concepts O co-occurs with (its 'document frequency')."
  (let ((tab (gethash o *cooccur*))) (if tab (hash-table-count tab) 0)))

(defun idf (o)
  "Inverse-frequency weight: a word that co-occurs with EVERYTHING (do, have, is, a, what)
   is uninformative and gets ~0 weight; a rare, discriminating word (fur, wheels) gets a
   high weight.  Derived from the data -- not a hand-written stop list."
  (log (/ (+ 1.0d0 (float (hash-table-count *cooccur*) 1.0d0)) (+ 1.0d0 (float (df o) 1.0d0)))))

(defun raw-vector (w)
  "Concept vector of W = sum over co-occurring words X of count(W,X) * idf(X) * code(X)."
  (or (gethash w *vcache*)
      (let ((tab (gethash w *cooccur*))
	    (v (make-array *vec-dim* :element-type 'double-float :initial-element 0.0d0)))
	(when tab
	  (maphash (lambda (o n)
		     (let ((c (code o)) (k (* (coerce n 'double-float) (idf o))))
		       (dotimes (i *vec-dim*) (incf (aref v i) (* k (aref c i))))))
		   tab))
	(setf (gethash w *vcache*) v))))

(defun mean-vector ()
  (or *vec-mean*
      (let ((m (make-array *vec-dim* :element-type 'double-float :initial-element 0.0d0)) (n 0))
	(maphash (lambda (w tab) (declare (ignore tab))
		   (incf n) (let ((v (raw-vector w)))
			      (dotimes (i *vec-dim*) (incf (aref m i) (aref v i)))))
		 *cooccur*)
	(when (plusp n) (dotimes (i *vec-dim*) (setf (aref m i) (/ (aref m i) n))))
	(setf *vec-mean* m))))

(defun concept-vector (w)
  "Mean-centered concept vector for W (NIL if unknown).  Centering drops the common-mode
   that every word shares, so the discriminating structure decides similarity."
  (when (gethash w *cooccur*)
    (let ((raw (raw-vector w)) (m (mean-vector))
	  (c (make-array *vec-dim* :element-type 'double-float)))
      (dotimes (i *vec-dim* c) (setf (aref c i) (- (aref raw i) (aref m i)))))))

(defun dot (a b) (loop for x across a for y across b sum (* x y)))
(defun vnorm (v) (sqrt (max 1d-12 (dot v v))))

(defun sim-vectors (a b) (/ (dot a b) (* (vnorm a) (vnorm b))))

(defun similarity (a b)
  "Cosine similarity of concepts A and B in the learned space (0.0 if either is unknown)."
  (let ((va (concept-vector a)) (vb (concept-vector b)))
    (if (and va vb) (sim-vectors va vb) 0.0)))

(defun sim-to-vector (w vec)
  (let ((v (concept-vector w))) (if v (sim-vectors v vec) 0.0)))

(defun centroid-of (words)
  "Mean of the (centered) vectors of WORDS -- a category 'region' from examples."
  (let ((m (make-array *vec-dim* :element-type 'double-float :initial-element 0.0d0)) (n 0))
    (dolist (w words)
      (let ((v (concept-vector w))) (when v (incf n) (dotimes (i *vec-dim*) (incf (aref m i) (aref v i))))))
    (when (plusp n) (dotimes (i *vec-dim*) (setf (aref m i) (/ (aref m i) n))))
    m))

(defun nearest (w &optional (k 8))
  "The K concepts most similar to W (excluding W), as (word . score) pairs."
  (let ((target (concept-vector w)) (scored '()))
    (when target
      (maphash (lambda (other tab) (declare (ignore tab))
		 (unless (string= other w)
		   (push (cons other (sim-to-vector other target)) scored)))
	       *cooccur*))
    (subseq (sort scored #'> :key #'cdr) 0 (min k (length scored)))))
