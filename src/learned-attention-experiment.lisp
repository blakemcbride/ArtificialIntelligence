;;;; learned-attention-experiment.lisp -- a head LEARNS what it does, with a local rule.
;;;; Run:  sbcl --script learned-attention-experiment.lisp
;;;;
;;;; induction-head-experiment.lisp HAND-WIRED the layer roles (previous-token head, then
;;;; induction head).  A real transformer LEARNS what each head does.  The open question:
;;;; can that be done with a LOCAL rule instead of backprop?  This PoC answers it for one
;;;; head's attention pattern.
;;;;
;;;; The head attends over recent positions by a learnable profile w[k] (how much to look k
;;;; tokens back) and predicts the next token as the attention-weighted token.  It is trained
;;;; by next-token prediction with a purely LOCAL credit rule (no backprop, no gradient
;;;; through a network):
;;;;     for each position, reward offset k whenever the token k-back EQUALS the actual next
;;;;     token  --  w[k] += 1.
;;;; That is local Hebbian credit: it correlates "where the head could have looked" with
;;;; "what actually came next", using only those two things.  Over data the profile
;;;; concentrates on the offset the data rewards -- the head DISCOVERS its function.
;;;;
;;;; Shown two ways: trained on period-2 data it learns the previous-token offset (k=1, the
;;;; layer-1 head hand-wired in Part 1); trained on period-3 data the SAME architecture
;;;; learns k=2.  Same head, different data, different learned function -- learned locally.

(defparameter *dim* 256)
(defparameter *D* 6 "How many positions back the head may attend.")

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

(defparameter *seed* 1)
(defun nextr () (setf *seed* (logand (+ (* *seed* 1103515245) 12345) #x7fffffff)))

;;; --- periodic data: a repeated window of random tokens --------------------------------
(defparameter *pool* '("alpha" "beta" "gamma" "delta" "epsilon" "zeta" "eta" "theta" "iota" "kappa"))
(defun gen-seq (period len)
  (let ((window (loop repeat period collect (nth (mod (nextr) (length *pool*)) *pool*))))
    (loop for i from 0 below len collect (nth (mod i period) window))))

;;; --- the head: a learnable attention profile over relative offsets ---------------------
(defun train (w seqs)
  "Local credit rule: for each position, reward each offset that pointed at the true next
   token.  Pure Hebbian correlation -- no backprop."
  (dolist (s seqs)
    (let ((v (coerce s 'vector)))
      (loop for tpos from *D* below (1- (length v))
	    for nxt = (aref v (1+ tpos))
	    do (dotimes (k *D*)
		 (when (string= (aref v (- tpos k)) nxt) (incf (aref w k))))))))

(defun attention (w)
  "Normalize the learned profile into attention weights (sum to 1)."
  (let ((sum (reduce #'+ w)))
    (if (zerop sum)
	(make-array *D* :initial-element (/ 1.0 *D*))
	(let ((a (make-array *D*))) (dotimes (k *D* a) (setf (aref a k) (/ (aref w k) sum)))))))

(defun predict (w seq tpos vocab)
  "Attention-weighted prediction of the token after position TPOS."
  (let ((a (attention w)) (acc (make-array *dim* :element-type 'double-float :initial-element 0d0)))
    (dotimes (k *D*)
      (let ((c (code (nth (- tpos k) seq))) (wk (aref a k)))
	(dotimes (i *dim*) (incf (aref acc i) (* wk (aref c i))))))
    (let (best bd)
      (dolist (s vocab best)
	(let ((d (let ((x (code s)) (sm 0d0)) (dotimes (i *dim* sm) (incf sm (* (aref x i) (aref acc i)))))))
	  (when (or (null best) (> d bd)) (setf best s bd d)))))))

(defun accuracy (w seqs)
  (let ((ok 0) (tot 0))
    (dolist (s seqs)
      (let ((vocab (remove-duplicates s :test #'string=)))
	(loop for tpos from *D* below (1- (length s))
	      do (incf tot)
		 (when (string= (predict w s tpos vocab) (nth (1+ tpos) s)) (incf ok)))))
    (if (zerop tot) 0.0 (/ ok (float tot)))))

(defun profile-str (w)
  (let ((a (attention w)))
    (format nil "~{~a~^ ~}"
	    (loop for k below *D* collect (format nil "k=~d:~,2f" k (aref a k))))))

;;; --------------------------------------------------------------------- demo ------------
(defun run (period label)
  (let ((w (make-array *D* :initial-element 0))
	(train-seqs (loop repeat 40 collect (gen-seq period 20)))
	(test-seqs  (loop repeat 20 collect (gen-seq period 20))))
    (format t "~%=== ~a ===~%" label)
    (format t "  before training: profile ~a~%" (profile-str w))
    (format t "  before training: held-out accuracy ~,2f~%" (accuracy w test-seqs))
    (train w train-seqs)
    (format t "  after training:  profile ~a~%" (profile-str w))
    (format t "  after training:  held-out accuracy ~,2f~%" (accuracy w test-seqs))
    (let ((peak (let (b bi) (dotimes (k *D* bi) (when (or (null b) (> (aref w k) b)) (setf b (aref w k) bi k))))))
      (format t "  learned offset: attend k=~d back  (the head discovered this from data)~%" peak))))

(setf *seed* 12345)
(run 2 "trained on period-2 data (should learn the previous-token offset, k=1)")
(run 3 "trained on period-3 data (same head -- should instead learn k=2)")

(format t "~%Takeaway: the SAME attention head learns DIFFERENT functions from different data~%")
(format t "(k=1 vs k=2) by a LOCAL credit rule -- reward the offset that pointed at the actual~%")
(format t "next token -- with NO backprop.  Part 1 hand-wired the previous-token head; here a~%")
(format t "head DISCOVERS it (and others) from data.  Next milestone: stack such learned heads~%")
(format t "so a learned layer-1 feeds a learned layer-2 -- a locally-learned induction circuit,~%")
(format t "and content-based (not just positional) attention learned the same way.~%")
