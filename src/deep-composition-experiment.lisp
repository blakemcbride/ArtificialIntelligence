;;;; deep-composition-experiment.lisp -- COMPOSITION AT DEPTH with local learning (no backprop)
;;;; Run:  sbcl --script deep-composition-experiment.lisp
;;;;
;;;; The remaining frontier (see notes/TransformerFromHebbianParts.md): stack learned layers
;;;; that COOPERATE, trained by a LOCAL rule -- no global backward pass.  This builds a real
;;;; two-layer induction circuit where DEPTH IS REQUIRED and BOTH layers are learned locally.
;;;;
;;;; The honesty knob: a real attention layer reads the value AT the position it attends to
;;;; (token[p]), NOT that position's successor.  So to predict "the token after the previous
;;;; occurrence of the current token", the "+1" must come from a layer BENEATH -- one layer
;;;; cannot do it (it can only return the matched token itself).  Two cooperating layers can:
;;;;
;;;;   layer 1  (previous-token head, LEARNED positional weights): local objective = predict
;;;;            each position's PREVIOUS token.  It learns to attend one back, so its output
;;;;            prev[p] carries token[p-1].
;;;;   layer 2  (content head, LEARNED continuous query-key matrix M2): local objective =
;;;;            predict the NEXT token, KEYING on layer 1's output (prev[p]) and reading the
;;;;            position-local value token[p].  Query = current token A; it matches A against
;;;;            the predecessor field, lands on the position right after an earlier A, and
;;;;            reads that position's token = the successor.  Induction.
;;;;
;;;; Each layer has its OWN local objective (greedy / layer-wise, the Forward-Forward family);
;;;; credit reaches layer 1 from its own target, not by backprop from layer 2.  They are
;;;; trained in sequence and COMPOSE.  A single trained content layer is shown to fail.

(defparameter *dim* 128)
(defparameter *D* 4)

;;; --- codes + PRNG ---------------------------------------------------------------------
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

;;; --- a content head's weight matrix M (learnable query-key form) -----------------------
(defun make-m () (make-array (list *dim* *dim*) :element-type 'double-float :initial-element 0d0))
(defun mscore (m q k)
  (let ((s 0d0))
    (dotimes (a *dim* s)
      (let ((qa (aref q a)) (acc 0d0))
	(dotimes (b *dim*) (incf acc (* (aref m a b) (aref k b))))
	(incf s (* qa acc))))))
(defun hebb! (m q k coeff)
  (dotimes (a *dim*)
    (let ((c (* coeff (aref q a))))
      (unless (zerop c) (dotimes (b *dim*) (incf (aref m a b) (* c (aref k b))))))))

;;; --- data: A->B recurs at a VARYING position; predict B -------------------------------
(defun gen-seq (pool)
  (let* ((a (pick pool)) (b (pick pool)))
    (loop while (string= a b) do (setf b (pick pool)))
    (let ((others (remove-if (lambda (x) (or (string= x a) (string= x b))) pool)))
      (flet ((fillers (n) (loop repeat n collect (pick others))))
	(cons (append (fillers (mod (nextr) 3)) (list a b)
		      (fillers (1+ (mod (nextr) 3))) (list a))
	      b)))))
(defun dataset (n pool) (loop repeat n collect (gen-seq pool)))

;;; --- LAYER 1: a learned positional head, local objective = predict the previous token ---
(defun train-layer1 (w1 seqs)
  (dolist (pr seqs)
    (let ((v (coerce (car pr) 'vector)))
      (loop for p from 1 below (length v)
	    for prev = (aref v (1- p))
	    do (loop for k from 1 to *D*
		     when (and (>= (- p k) 0) (string= (aref v (- p k)) prev))
		       do (incf (aref w1 k)))))))
(defun layer1-offset (w1)
  (let (bo bv) (loop for k from 1 to *D* when (or (null bv) (> (aref w1 k) bv)) do (setf bv (aref w1 k) bo k)) bo))

;;; --- LAYER 2: a learned content head on top of layer 1's output ------------------------
;;; key for position p = layer-1 output prev[p] = code(token[p - k1]); value = token[p].
(defun prevcode (v p k1) (when (>= (- p k1) 0) (code (aref v (- p k1)))))

(defun train-layer2 (m2 seqs k1 &optional (eta 1.0))
  (dolist (pr seqs)
    (destructuring-bind (seq . target) pr
      (let* ((v (coerce seq 'vector)) (n (length v)) (q (code (aref v (1- n)))) (tc (code target))
	     (ps (loop for p from k1 below (1- n) when (prevcode v p k1) collect p)))
	(when ps
	  (let* ((gs (mapcar (lambda (p) (/ (dot (code (aref v p)) tc) (float *dim*))) ps))
		 (gbar (/ (reduce #'+ gs) (length gs))))
	    (loop for p in ps for g in gs
		  do (hebb! m2 q (prevcode v p k1) (* eta (- g gbar))))))))))

(defun predict-deep (m2 seq k1)
  "Two layers: content-match the current token against layer-1's predecessor field, read the
   position-local value (token[p]) of the best match."
  (let* ((v (coerce seq 'vector)) (n (length v)) (q (code (aref v (1- n)))) best bp)
    (loop for p from k1 below (1- n)
	  for kc = (prevcode v p k1)
	  when kc do (let ((s (mscore m2 q kc))) (when (or (null best) (> s best)) (setf best s bp p))))
    (and bp (aref v bp))))

;;; --- single-layer baseline (trained the same way, but no layer 1) ----------------------
(defun train-single (m seqs &optional (eta 1.0))
  (dolist (pr seqs)
    (destructuring-bind (seq . target) pr
      (let* ((v (coerce seq 'vector)) (n (length v)) (q (code (aref v (1- n)))) (tc (code target))
	     (ps (loop for p from 0 below (1- n) collect p)))
	(let* ((gs (mapcar (lambda (p) (/ (dot (code (aref v p)) tc) (float *dim*))) ps))
	       (gbar (/ (reduce #'+ gs) (length gs))))
	  (loop for p in ps for g in gs do (hebb! m q (code (aref v p)) (* eta (- g gbar)))))))))
(defun predict-single (m seq)
  (let* ((v (coerce seq 'vector)) (n (length v)) (q (code (aref v (1- n)))) best bp)
    (loop for p from 0 below (1- n)
	  do (let ((s (mscore m q (code (aref v p))))) (when (or (null best) (> s best)) (setf best s bp p))))
    (and bp (aref v bp))))

(defun acc (predict-fn seqs) (let ((ok 0)) (dolist (pr seqs (/ ok (float (length seqs))))
				 (when (equal (funcall predict-fn (car pr)) (cdr pr)) (incf ok)))))

;;; --------------------------------------------------------------------- demo ------------
(defparameter *train-pool* (loop for i below 30 collect (format nil "tok~d" i)))
(defparameter *novel-pool* (loop for i below 12 collect (format nil "new~d" i)))

(setf *seed* 20240)
(let ((w1 (make-array (1+ *D*) :initial-element 0))
      (m2 (make-m)) (msingle (make-m))
      (train-seqs (dataset 400 *train-pool*))
      (test-seqs  (dataset 100 *train-pool*))
      (novel-seqs (dataset 100 *novel-pool*)))
  ;; layer-wise local training (no backprop, each layer its own objective)
  (train-layer1 w1 train-seqs)
  (let ((k1 (layer1-offset w1)))
    (format t "~%LAYER 1 (local objective: predict previous token) learned offset k1=~d~%" k1)
    (train-layer2 m2 train-seqs k1)
    (train-single msingle train-seqs)
    (format t "~%two-layer circuit (both layers learned locally):~%")
    (format t "  held-out accuracy (trained tokens): ~,2f~%" (acc (lambda (s) (predict-deep m2 s k1)) test-seqs))
    (format t "  held-out accuracy on NOVEL tokens:  ~,2f  <- in-context, composed at depth~%"
	    (acc (lambda (s) (predict-deep m2 s k1)) novel-seqs))
    (format t "~%single trained content layer (no layer 1 beneath):~%")
    (format t "  held-out accuracy: ~,2f  <- cannot do it; depth is required~%"
	    (acc (lambda (s) (predict-single msingle s)) test-seqs))))

(format t "~%Takeaway: a two-layer induction circuit is trained by LOCAL, LAYER-WISE objectives~%")
(format t "(no backprop): layer 1 learns the previous-token head from its OWN target; layer 2~%")
(format t "learns a continuous content query-key matrix ON TOP of layer 1's output.  They~%")
(format t "COMPOSE -- and depth is necessary, since a single trained content layer (which reads~%")
(format t "the value at the position it attends) can only return the matched token, not its~%")
(format t "successor.  This is composition at depth under local learning -- the Forward-Forward~%")
(format t "/ predictive-coding family.  Scaling to many such layers + multi-head is the road on.~%")
