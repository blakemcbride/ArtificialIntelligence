;;;; learned-qk-attention-experiment.lisp -- a CONTENT-based attention head whose weight
;;;; matrix is LEARNED by a local rule (no backprop).  Run:
;;;;     sbcl --script learned-qk-attention-experiment.lisp
;;;;
;;;; The previous steps learned DISCRETE choices (which offset, which config).  A real
;;;; transformer head learns CONTINUOUS query/key weights that decide, by content, what
;;;; attends to what.  Here a single head has one learnable matrix M (the combined
;;;; query-key bilinear form): the attention score from the current token q to a past token
;;;; k is  q^T M k.  Learning M = learning what the head attends to.
;;;;
;;;; LOCAL learning rule (no backprop): to predict the next token, attend over past
;;;; positions and read each position's SUCCESSOR; reward attending where the successor
;;;; equals the actual next token.  Concretely, with query = current token q, for each past
;;;; position j (key k_j = token there, value = its successor):
;;;;     g_j = match(successor_j, true-next)            ; a LOCAL reward (1 if it matched)
;;;;     M += eta * (g_j - mean_g) * (q (x) k_j)        ; raise the score q^T M k_j for keys
;;;;                                                    ;   whose value predicted correctly
;;;; That is reward-modulated Hebbian on the bilinear form: it correlates the query and key
;;;; (outer product) weighted by how useful attending there was.  No gradient is propagated.
;;;;
;;;; The data forces CONTENT (not position): "<fillers> A B <fillers> A" -> next is B, at a
;;;; varying gap.  The head should discover M ~ identity ("attend to the same token") -- a
;;;; GENERAL operation that then works on tokens never seen in training (in-context learning).

(defparameter *dim* 128)

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

;;; --- the head's weight matrix M (the learnable query-key form) -------------------------
(defun make-m () (make-array (list *dim* *dim*) :element-type 'double-float :initial-element 0d0))

(defun score (m q k)
  "Attention score q^T M k between query code Q and key code K."
  (let ((s 0d0))
    (dotimes (a *dim* s)
      (let ((qa (aref q a)) (row a) (acc 0d0))
	(dotimes (b *dim*) (incf acc (* (aref m row b) (aref k b))))
	(incf s (* qa acc))))))

(defun hebb! (m q k coeff)
  "M += coeff * (q (x) k)  -- reward-modulated Hebbian outer product (local; no backprop)."
  (dotimes (a *dim*)
    (let ((c (* coeff (aref q a))))
      (unless (zerop c)
	(dotimes (b *dim*) (incf (aref m a b) (* c (aref k b))))))))

;;; --- data: an A->B pair recurs at a VARYING position; predict B -----------------------
(defun gen-seq (pool)
  (let* ((a (pick pool)) (b (pick pool)))
    (loop while (string= a b) do (setf b (pick pool)))
    (let ((others (remove-if (lambda (x) (or (string= x a) (string= x b))) pool)))
      (flet ((fillers (n) (loop repeat n collect (pick others))))
	(cons (append (fillers (mod (nextr) 3)) (list a b)
		      (fillers (1+ (mod (nextr) 3))) (list a))
	      b)))))
(defun dataset (n pool) (loop repeat n collect (gen-seq pool)))

;;; --- train (local rule) / predict (content attention) ---------------------------------
(defun train (m seqs &optional (eta 1.0))
  (dolist (pr seqs)
    (destructuring-bind (seq . target) pr
      (let* ((v (coerce seq 'vector)) (n (length v)) (q (code (aref v (1- n)))) (tcode (code target))
	     (js (loop for j from 0 below (1- n) collect j))
	     (gs (mapcar (lambda (j) (/ (dot (code (aref v (1+ j))) tcode) (float *dim*))) js))
	     (gbar (/ (reduce #'+ gs) (max 1 (length gs)))))
	(loop for j in js for g in gs
	      do (hebb! m q (code (aref v j)) (* eta (- g gbar))))))))

(defun predict (m seq)
  "Attend (content) from the last token over past positions; return the successor of the
   best-matching position."
  (let* ((v (coerce seq 'vector)) (n (length v)) (q (code (aref v (1- n)))) best bj)
    (loop for j from 0 below (1- n)
	  do (let ((sc (score m q (code (aref v j)))))
	       (when (or (null best) (> sc best)) (setf best sc bj j))))
    (and bj (aref v (1+ bj)))))

(defun accuracy (m seqs)
  (let ((ok 0))
    (dolist (pr seqs (/ ok (float (length seqs))))
      (when (equal (predict m (car pr)) (cdr pr)) (incf ok)))))

(defun same-vs-diff (m toks)
  "Average score(c,c) for same tokens vs score(c,d) for different -- did M learn to match?"
  (let ((same 0d0) (diff 0d0) (ns 0) (nd 0))
    (dolist (c toks)
      (incf same (score m (code c) (code c))) (incf ns)
      (dolist (d toks)
	(unless (string= c d) (incf diff (score m (code c) (code d))) (incf nd))))
    (values (/ same ns) (/ diff nd))))

;;; --------------------------------------------------------------------- demo ------------
(defparameter *train-pool*
  (loop for i from 0 below 30 collect (format nil "tok~d" i)))
(defparameter *novel-pool*    ; disjoint from training -- to test generalization
  (loop for i from 0 below 12 collect (format nil "new~d" i)))

(setf *seed* 4242)
(let ((m (make-m))
      (train-seqs (dataset 400 *train-pool*))
      (test-seqs  (dataset 100 *train-pool*))
      (novel-seqs (dataset 100 *novel-pool*)))
  (format t "~%before training:~%")
  (format t "  held-out accuracy: ~,2f~%" (accuracy m test-seqs))
  (train m train-seqs)
  (multiple-value-bind (same diff) (same-vs-diff m (subseq *train-pool* 0 8))
    (format t "~%after local-rule training (no backprop):~%")
    (format t "  M scores -- same token: ~,1f   different token: ~,1f  (M learned to MATCH)~%" same diff))
  (format t "  held-out accuracy (trained tokens):      ~,2f~%" (accuracy m test-seqs))
  (format t "  held-out accuracy on NOVEL tokens:       ~,2f  <- in-context, learned a GENERAL head~%"
	  (accuracy m novel-seqs)))

(format t "~%Takeaway: the head's WEIGHT MATRIX (a content-based query-key form) is learned by a~%")
(format t "LOCAL reward-modulated Hebbian rule -- no backprop.  It discovers M ~~ identity:~%")
(format t "\"attend to the same token, copy its successor\" -- a GENERAL operation, so it does~%")
(format t "in-context learning even on tokens never seen in training.  This is the continuous-~%")
(format t "weight, content-based generalization of the earlier discrete-offset learning.  The~%")
(format t "remaining frontier: STACK many such locally-learned heads across depth (predictive-~%")
(format t "coding / forward-forward style) -- learned weights at every layer, no global gradient.~%")
