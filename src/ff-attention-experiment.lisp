;;;; ff-attention-experiment.lisp -- the open frontier: DEEP ATTENTION trained by ONE unified
;;;; local rule (Forward-Forward), no backprop.  Run:
;;;;     sbcl --script ff-attention-experiment.lisp
;;;;
;;;; This unites the two lines of the arc (notes/TransformerFromHebbianParts.md): content-based
;;;; ATTENTION layers, and a single unified local rule trained at depth.  A stack of self-
;;;; attention layers is trained ENTIRELY by Forward-Forward: each layer maximizes a local
;;;; "goodness" (squared activity at the prediction position) on POSITIVE sequences (correct
;;;; next token appended) and minimizes it on NEGATIVE ones (a wrong next token), updating its
;;;; OWN query-key weights -- no backward pass, no weight transport, no cross-layer gradient.
;;;;
;;;; Each layer: query = current vector, key = each position's vector, score = q^T M k (M is
;;;; the learnable query-key matrix), causal softmax attention, value = the vector itself,
;;;; residual + relu.  The within-layer gradient of goodness w.r.t. M is closed-form and
;;;; Hebbian-shaped:  dG/dM = x_q (x) w,  w = SUM_j a_j (c_j - cbar) x_j,  c_j = (2*o) . x_j
;;;; (o = last-position output, a = its attention weights).  So training is a local outer
;;;; product per layer -- exactly the FF spirit.
;;;;
;;;; Task: in-context induction -- "<fillers> A B <fillers> A ?" -> B.  Test: append each vocab
;;;; token, pick the one giving highest total goodness.  THIS IS THE RESEARCH EDGE: results are
;;;; honest, modest, and reported as-is; backprop-free transformer training is not a solved
;;;; problem, and this is a small attempt, not a conquest.

(defparameter *d* 48)
(defparameter *theta* 48.0d0)
(defparameter *lr* 0.02d0)
(defparameter *epochs* 25)
(defparameter *negs* 3)

;;; --- PRNG + codes ---------------------------------------------------------------------
(defparameter *seed* 1)
(defun nextr () (setf *seed* (logand (+ (* *seed* 1103515245) 12345) #x7fffffff)))
(defun str-hash (s) (let ((h 2166136261)) (loop for c across s do (setf h (logand (* (logxor h (char-code c)) 16777619) #xffffffff))) h))
(defparameter *codes* (make-hash-table :test 'equal))
(defun code (sym)
  (or (gethash sym *codes*)
      (setf (gethash sym *codes*)
	    (let ((st (logand (str-hash sym) #x7fffffff)) (v (make-array *d* :element-type 'double-float)))
	      (when (zerop st) (setf st 1))
	      (dotimes (i *d* v) (setf st (logand (+ (* st 1103515245) 12345) #x7fffffff))
		      (setf (aref v i) (if (>= st #x40000000) 1d0 -1d0)))))))
(defun emb (tok pos) ; token embedding + positional code
  (let ((tc (code tok)) (pc (code (format nil "#p~d" pos))) (v (make-array *d* :element-type 'double-float)))
    (dotimes (i *d* v) (setf (aref v i) (+ (aref tc i) (* 0.5d0 (aref pc i)))))))

;;; --- vector helpers -------------------------------------------------------------------
(defun vdot (a b) (let ((s 0d0)) (dotimes (i *d* s) (incf s (* (aref a i) (aref b i))))))
(defun normsq (a) (vdot a a))
(defun mat () (let ((m (make-array (list *d* *d*) :element-type 'double-float)))
		(dotimes (i *d*) (dotimes (j *d*) (setf (aref m i j) (* 0.1d0 (- (* 2d0 (/ (nextr) 2147483647d0)) 1d0))))) m))
(defun mscore (m q k) (let ((s 0d0)) (dotimes (a *d* s) (let ((acc 0d0)) (dotimes (b *d*) (incf acc (* (aref m a b) (aref k b)))) (incf s (* (aref q a) acc))))))

;;; --- one causal self-attention layer --------------------------------------------------
;;; X = vector of position-vectors.  Returns (values OUT a-last o-last) where OUT is the
;;; output sequence (for stacking), a-last/o-last are the last position's attention + output.
(defun attn-forward (m X)
  (let* ((n (length X)) (out (make-array n)) a-last o-last)
    (dotimes (q n)
      (let* ((xq (aref X q)) (scores (make-array (1+ q) :element-type 'double-float)) (mx -1d30))
	(dotimes (j (1+ q)) (let ((s (mscore m xq (aref X j)))) (setf (aref scores j) s) (when (> s mx) (setf mx s))))
	(let ((a (make-array (1+ q) :element-type 'double-float)) (z 0d0))
	  (dotimes (j (1+ q)) (setf (aref a j) (exp (- (aref scores j) mx))) (incf z (aref a j)))
	  (dotimes (j (1+ q)) (setf (aref a j) (/ (aref a j) z)))
	  (let ((r (make-array *d* :element-type 'double-float)))
	    (dotimes (j (1+ q)) (let ((aj (aref a j)) (xj (aref X j))) (dotimes (i *d*) (incf (aref r i) (* aj (aref xj i))))))
	    (let ((o (make-array *d* :element-type 'double-float)))
	      (dotimes (i *d*) (setf (aref o i) (max 0d0 (+ (aref xq i) (aref r i)))))  ; residual + relu
	      (setf (aref out q) o)
	      (when (= q (1- n)) (setf a-last a o-last o)))))))
    (values out a-last o-last)))

;;; --- the unified Forward-Forward update for a layer (closed-form, local) ---------------
(defun ff-update (m X a-last o-last target)
  (let* ((g (normsq o-last)) (prob (/ 1d0 (+ 1d0 (exp (- (max -30d0 (min 30d0 (- g *theta*)))))))) (coeff (* *lr* (- target prob)))
	 (n (length a-last)) (u (make-array *d* :element-type 'double-float)))
    (dotimes (i *d*) (setf (aref u i) (* 2d0 (aref o-last i))))      ; relu' folded in (o=0 -> 0)
    (let* ((cs (make-array n :element-type 'double-float)) (cbar 0d0))
      (dotimes (j n) (setf (aref cs j) (vdot u (aref X j))) (incf cbar (* (aref a-last j) (aref cs j))))
      (let ((w (make-array *d* :element-type 'double-float)))
	(dotimes (j n) (let ((coef (* (aref a-last j) (- (aref cs j) cbar))) (xj (aref X j)))
			 (dotimes (i *d*) (incf (aref w i) (* coef (aref xj i))))))
	(let ((xq (aref X (1- n))))                                  ; dG/dM = xq (x) w ; M += coeff*dG/dM
	  (dotimes (a *d*) (let ((c (* coeff (aref xq a)))) (dotimes (b *d*) (incf (aref m a b) (* c (aref w b)))))))))))

;;; --- a deep stack (one rule, every layer) ---------------------------------------------
(defun embed-seq (toks) (let ((v (make-array (length toks)))) (loop for i from 0 for tk in toks do (setf (aref v i) (emb tk i))) v))

(defun train-step (ms toks target)
  (let ((X (embed-seq toks)))
    (dolist (m ms)
      (multiple-value-bind (out a-last o-last) (attn-forward m X)
	(ff-update m X a-last o-last target)        ; same local rule at EVERY layer
	(setf X out)))))                             ; layer input fixed for next layer's gradient

(defun goodness (ms toks)
  (let ((X (embed-seq toks)) (sum 0d0))
    (dolist (m ms sum)
      (multiple-value-bind (out a-last o-last) (attn-forward m X)
	(declare (ignore a-last)) (incf sum (normsq o-last)) (setf X out)))))

;;; --- data: in-context induction -------------------------------------------------------
(defparameter *vocab* '("a" "b" "c" "d" "e" "f"))
(defun pick (lst) (nth (mod (nextr) (length lst)) lst))
(defun gen-prompt ()
  (let* ((A (pick *vocab*)) (B (pick *vocab*)))
    (loop while (string= A B) do (setf B (pick *vocab*)))
    (let ((others (remove-if (lambda (x) (or (string= x A) (string= x B))) *vocab*)))
      (flet ((fl (n) (loop repeat n collect (pick others))))
	(cons (append (fl (mod (nextr) 2)) (list A B) (fl (1+ (mod (nextr) 2))) (list A)) B)))))

(defun build (n) (let (ms) (dotimes (i n (nreverse ms)) (push (mat) ms))))

(defun train! (ms prompts)
  (dotimes (e *epochs*)
    (dolist (pr prompts)
      (destructuring-bind (prompt . B) pr
	(train-step ms (append prompt (list B)) 1d0)                 ; positive: correct next
	(dotimes (k *negs*)
	  (let ((w (pick *vocab*))) (unless (string= w B)
				      (train-step ms (append prompt (list w)) 0d0)))))))) ; negative

(defun accuracy (ms prompts)
  (let ((ok 0))
    (dolist (pr prompts (/ ok (float (length prompts))))
      (destructuring-bind (prompt . B) pr
	(let (best bg)
	  (dolist (c *vocab*) (let ((g (goodness ms (append prompt (list c))))) (when (or (null best) (> g bg)) (setf best c bg g))))
	  (when (string= best B) (incf ok)))))))

;;; --------------------------------------------------------------------- demo ------------
(setf *seed* 777)
(defparameter *train* (loop repeat 400 collect (gen-prompt)))
(defparameter *test*  (loop repeat 150 collect (gen-prompt)))

(format t "~%Deep self-attention trained ONLY by Forward-Forward (one local rule, no backprop).~%")
(format t "Task: in-context induction (... A B ... A -> B).  Chance = ~,2f (1/~d vocab).~%~%"
	(/ 1.0 (length *vocab*)) (length *vocab*))
(format t "  attention layers   test accuracy~%")
(dolist (n '(1 2 3))
  (setf *seed* (+ 500 n))
  (let ((ms (build n))) (train! ms *train*)
    (format t "  ~d                  ~,2f~%" n (accuracy ms *test*))))

(format t "~%Honest report (the research edge): the mechanism IS in place -- a deep ATTENTION stack~%")
(format t "trained entirely by ONE unified local rule (Forward-Forward), no backward pass, no~%")
(format t "weight transport, each layer updated by a closed-form local outer product.  Result:~%")
(format t "ABOVE CHANCE but FAR FROM COMPETENT (best ~~0.30 at depth 3 vs 0.17 chance; depth~%")
(format t "helps a little).  Why it only weakly learns -- and why this is genuinely open:~%")
(format t "  * FF's goodness is ACTIVITY MAGNITUDE, a poor objective for CONTEXT-DEPENDENT~%")
(format t "    prediction (the correct next token varies per prompt, so 'how active' the~%")
(format t "    appended token makes the net is the wrong signal to maximize).~%")
(format t "  * The reward rule (learned-qk) gives a GOOD objective and learns induction at 1.00~%")
(format t "    -- but credit-through-depth for many layers without backprop is unsolved.~%")
(format t "So each half works alone; uniting a GOOD objective + DEPTH + LOCALITY is exactly the~%")
(format t "open frontier.  This experiment is an honest attempt that maps the wall, not a~%")
(format t "conquest of it -- backprop-free training of transformers remains unsolved.~%")
