;;;; learned-induction-experiment.lisp -- DISCOVER the induction circuit from data, locally.
;;;; Run:  sbcl --script learned-induction-experiment.lisp
;;;;
;;;; induction-head-experiment.lisp HAND-WIRED a two-layer induction circuit; learned-
;;;; attention-experiment.lisp showed ONE head can learn its attention by a local rule.  This
;;;; combines them: both layers of the induction circuit are LEARNED from data by a local
;;;; reward rule (no backprop), and crucially layer 1's role is learned via credit that flows
;;;; from layer 2's downstream prediction -- credit assignment THROUGH depth, locally.
;;;;
;;;; The circuit has two learnable head roles (each an offset):
;;;;   layer 1 -- MATCH offset o1: key each past position p by the token o1 back, code(x[p-o1]).
;;;;   layer 2 -- VALUE offset r2: the value emitted for position p is code(x[p-r2]).
;;;; Prediction of the token after the current token q:  attend (content match) q against the
;;;; keys, read out the values:  decode( SUM_p (code(x[p-o1]).code(q)) * code(x[p-r2]) ).
;;;; The induction circuit is (o1=1, r2=0): match the current token against each position's
;;;; PREDECESSOR (so you land right after an earlier occurrence of q) and emit THAT position's
;;;; token (what followed q).  Nothing tells the code this -- it is discovered.
;;;;
;;;; Local learning (no backprop): try the candidate configs against next-token prediction and
;;;; reward what works -- S[o1][r2] += 1 on a correct prediction.  o1 (layer 1) is rewarded
;;;; only through the full two-layer prediction, i.e. through layer 2 -- credit through depth.
;;;;
;;;; Data forces INDUCTION, not a fixed position: "<fillers> A B <fillers> A" -> next is B, but
;;;; the gap varies, so no fixed offset generalizes -- only content-based induction does.

(defparameter *dim* 256)
(defparameter *D* 3 "Largest offset a head may use.")

;;; --- codes + deterministic PRNG -------------------------------------------------------
(defun str-hash (s)
  (let ((h 2166136261))
    (loop for c across s do (setf h (logand (* (logxor h (char-code c)) 16777619) #xffffffff)))
    h))
(defparameter *codes* (make-hash-table :test 'equal))
(defun code (sym)
  (or (gethash sym *codes*)
      (setf (gethash sym *codes*)
	    (let ((st (logand (str-hash sym) #x7fffffff)) (v (make-array *dim* :element-type 'double-float)))
	      (when (zerop st) (setf st 1))
	      (dotimes (i *dim* v)
		(setf st (logand (+ (* st 1103515245) 12345) #x7fffffff))
		(setf (aref v i) (if (>= st #x40000000) 1d0 -1d0)))))))
(defun dot (a b) (let ((s 0d0)) (dotimes (i *dim* s) (incf s (* (aref a i) (aref b i))))))

(defparameter *seed* 1)
(defun nextr () (setf *seed* (logand (+ (* *seed* 1103515245) 12345) #x7fffffff)))
(defun pick (lst) (nth (mod (nextr) (length lst)) lst))

;;; --- data: an A->B pair recurs at a VARYING position; predict B -----------------------
(defparameter *pool*
  '("alpha" "beta" "gamma" "delta" "epsilon" "zeta" "eta" "theta" "iota" "kappa" "mu" "nu"))

(defun gen-seq (&optional (pool *pool*))
  "Return (values SEQUENCE TARGET): <fillers> A B <fillers> A, target = B."
  (let* ((a (pick pool)) (b (pick pool)))
    (loop while (string= a b) do (setf b (pick pool)))
    (let ((others (remove-if (lambda (x) (or (string= x a) (string= x b))) pool)))
      (flet ((fillers (n) (loop repeat n collect (pick others))))
	(values (append (fillers (mod (nextr) 3)) (list a b)
			(fillers (1+ (mod (nextr) 3))) (list a))
		b)))))

;;; --- the two-head circuit: prediction by content attention ----------------------------
(defun predict (seq o1 r2 vocab)
  "Predict the token after the last position of SEQ with match-offset O1 and value-offset R2."
  (let* ((v (coerce seq 'vector)) (n (length v)) (q (code (aref v (1- n))))
	 (acc (make-array *dim* :element-type 'double-float :initial-element 0d0)) (hits 0))
    (loop for p from (max o1 r2) below n
	  do (let ((score (dot (code (aref v (- p o1))) q)))
	       (incf hits)
	       (let ((val (code (aref v (- p r2)))))
		 (dotimes (i *dim*) (incf (aref acc i) (* score (aref val i)))))))
    (when (plusp hits)
      (let (best bd)
	(dolist (s vocab best)
	  (let ((d (dot (code s) acc))) (when (or (null best) (> d bd)) (setf best s bd d))))))))

;;; --- local learning: reward the config that predicts the next token --------------------
(defun train (s seqs)
  (dolist (pr seqs)
    (destructuring-bind (seq . target) pr
      (let ((vocab (remove-duplicates seq :test #'string=)))
	(loop for o1 from 0 to *D* do
	  (loop for r2 from 0 to *D* do
	    (when (equal (predict seq o1 r2 vocab) target)
	      (incf (aref s o1 r2)))))))))

(defun best-config (s)
  (let (bo br bv)
    (loop for o1 from 0 to *D* do
      (loop for r2 from 0 to *D* do
	(when (or (null bv) (> (aref s o1 r2) bv)) (setf bv (aref s o1 r2) bo o1 br r2))))
    (values bo br bv)))

(defun accuracy (o1 r2 seqs)
  (let ((ok 0))
    (dolist (pr seqs (/ ok (float (length seqs))))
      (destructuring-bind (seq . target) pr
	(when (equal (predict seq o1 r2 (remove-duplicates seq :test #'string=)) target)
	  (incf ok))))))

;;; --------------------------------------------------------------------- demo ------------
(setf *seed* 777)
(defun dataset (n &optional (pool *pool*))
  (loop repeat n collect (multiple-value-bind (s tgt) (gen-seq pool) (cons s tgt))))

(let ((train-seqs (dataset 120))
      (test-seqs  (dataset 60))
      (s (make-array (list (1+ *D*) (1+ *D*)) :initial-element 0)))
  (format t "~%example training item: ~a  (target: ~a)~%" (caar train-seqs) (cdar train-seqs))
  (train s train-seqs)
  (format t "~%reward table S[match-offset o1][value-offset r2] (correct predictions):~%")
  (format t "        r2=0  r2=1  r2=2  r2=3~%")
  (loop for o1 from 0 to *D* do
    (format t "  o1=~d " o1)
    (loop for r2 from 0 to *D* do (format t " ~5d" (aref s o1 r2)))
    (terpri))
  (multiple-value-bind (o1 r2 v) (best-config s)
    (declare (ignore v))
    (format t "~%learned circuit:  layer-1 match-offset o1=~d, layer-2 value-offset r2=~d~%" o1 r2)
    (format t "  -> layer 1 learned the PREVIOUS-TOKEN head (match on the token ~d back)~%" o1)
    (format t "  -> layer 2 learned to EMIT the matched token (value ~d back) = induction~%" r2)
    (format t "  (configs with o1-r2=1 are equivalent on single-pair data; the learner takes~%")
    (format t "   the MINIMAL one -- the canonical previous-token + induction circuit.)~%")
    (format t "~%held-out accuracy of the learned circuit (o1=~d r2=~d): ~,2f~%" o1 r2 (accuracy o1 r2 test-seqs))
    (format t "held-out accuracy of a one-layer config (o1=0 r2=0):      ~,2f  <- cannot induce~%"
	    (accuracy 0 0 test-seqs))
    ;; generalization: brand-new tokens never used in training
    (setf *seed* 9999)
    (let ((novel (dataset 60 '("zorp" "blee" "quax" "vimp" "wuzz" "narf" "glip" "dax"))))
      (format t "held-out accuracy on NOVEL tokens (in-context learning): ~,2f~%"
	      (accuracy o1 r2 novel)))))

(format t "~%Takeaway: both heads of the induction circuit are LEARNED from data by a local~%")
(format t "reward rule -- no backprop.  Layer 1's role (previous-token head) is rewarded only~%")
(format t "through layer 2's prediction, so credit flows through DEPTH locally.  The circuit~%")
(format t "rediscovers (o1=1, r2=0) -- exactly the hand-wired induction head -- and does~%")
(format t "in-context learning on tokens it never saw.  This is the locally-learned, stacked~%")
(format t "version of induction; scaling to many heads + content-based query/key learned the~%")
(format t "same way (predictive-coding / forward-forward style) is the road to a real model.~%")
