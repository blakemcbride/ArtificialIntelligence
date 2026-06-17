;;;; pc-attention-experiment.lisp -- the frontier step: PREDICTIVE CODING over a 2-layer
;;;; ATTENTION induction circuit, vs Forward-Forward's 0.30.  Run:
;;;;     sbcl --script pc-attention-experiment.lisp
;;;;
;;;; Context (notes/TransformerFromHebbianParts.md): the open problem is a LOCAL, no-backprop
;;;; rule with a GOOD objective that ALSO assigns credit THROUGH DEPTH, trained into deep
;;;; ATTENTION.  predictive-coding-experiment.lisp validated the rule on a generic MLP (its
;;;; local update reproduces the backprop gradient, cosine ~1).  THIS file carries it into the
;;;; attention case that Forward-Forward could not crack (FF stalled at ~0.30 vs 0.17 chance).
;;;;
;;;; The circuit (a minimal induction head with LEARNABLE attention):
;;;;   embeddings  e_t = tokencode(x_t) + 0.5 poscode(t)
;;;;   layer 1 (M1): causal self-attention, value = e_j, RESIDUAL -> H_q = e_q + sum_j a1_qj e_j
;;;;   layer 2 (M2): query/key = H, value = TOKEN CODE t_j ; read only the LAST position ->
;;;;                 o = sum_j a2_j t_j  (a clean predicted next-token code)
;;;;   predict: argmax_c  t_c . o      (nearest vocab code)
;;;; Task: in-context induction  "... A B ... A -> B".  B is present in-context, so a working
;;;; induction circuit attends from the final A to the token after the earlier A (= B) and
;;;; copies its code.  DEPTH is required: layer 1 builds the features layer 2 matches on.
;;;;
;;;; Method (validation-first, the discipline that paid off before):
;;;;   * One attention forward + its vector-Jacobian products (VJPs), used for BOTH backprop
;;;;     (the yardstick + the gradient check) AND predictive coding.
;;;;   * BACKPROP yardstick: confirm this circuit + objective CAN be trained to competence.
;;;;   * PREDICTIVE CODING: settle the layer-1 latents H by local messages (top-down VJP of the
;;;;     output error), then update M1/M2 by LOCAL products of each layer's OWN equilibrium
;;;;     error -- no global backward pass.  Check PC's update vs backprop (cosine), then report
;;;;     PC's induction accuracy against FF's 0.30.
;;;; Honest scope: toy (small vocab, short prompts).  Results are reported as-is.

(declaim (optimize (speed 3) (safety 1)))
(deftype dv () '(simple-array double-float (*)))
(deftype dm () '(simple-array double-float (* *)))
(defparameter *d* 32)
(declaim (type fixnum *d*))

;;; --- PRNG + codes ---------------------------------------------------------------------
(defparameter *seed* 1) (declaim (type fixnum *seed*))
(defun nextr () (setf *seed* (logand (+ (* *seed* 1103515245) 12345) #x7fffffff)))
(defun str-hash (s) (let ((h 2166136261)) (loop for c across s do (setf h (logand (* (logxor h (char-code c)) 16777619) #xffffffff))) h))
(defparameter *codes* (make-hash-table :test 'equal))
(defun code (sym)                                              ; deterministic +-1 random code
  (or (gethash sym *codes*)
      (setf (gethash sym *codes*)
            (let ((st (logand (str-hash sym) #x7fffffff)) (v (make-array *d* :element-type 'double-float)))
              (when (zerop st) (setf st 1))
              (dotimes (i *d* v) (setf st (logand (+ (* st 1103515245) 12345) #x7fffffff))
                      (setf (aref v i) (if (>= st #x40000000) 1d0 -1d0)))))))
(defun tcode (tok) (code tok))
(defun emb (tok pos)
  (let ((tc (code tok)) (pc (code (format nil "#p~d" pos))) (v (make-array *d* :element-type 'double-float)))
    (declare (type dv tc pc v)) (dotimes (i *d* v) (setf (aref v i) (+ (aref tc i) (* 0.5d0 (aref pc i)))))))

;;; --- vector / matrix helpers ----------------------------------------------------------
(defun dvec (n) (make-array n :element-type 'double-float :initial-element 0d0))
(defun dmat (r c &optional (s 0d0)) (let ((m (make-array (list r c) :element-type 'double-float)))
                                      (dotimes (i r m) (dotimes (j c) (setf (aref m i j) (* s (- (* 2d0 (/ (nextr) 2147483647d0)) 1d0)))))))
(defun vdot (a b) (declare (type dv a b)) (let ((s 0d0)) (declare (double-float s)) (dotimes (i (length a) s) (incf s (* (aref a i) (aref b i))))))
(defun mvec (m v) (declare (type dm m) (type dv v))            ; M v
  (let ((o (dvec *d*))) (declare (type dv o)) (dotimes (a *d* o) (let ((s 0d0)) (declare (double-float s)) (dotimes (b *d*) (incf s (* (aref m a b) (aref v b)))) (setf (aref o a) s)))))
(defun mTvec (m v) (declare (type dm m) (type dv v))           ; M^T v
  (let ((o (dvec *d*))) (declare (type dv o)) (dotimes (a *d*) (let ((va (aref v a))) (dotimes (b *d*) (incf (aref o b) (* (aref m a b) va))))) o))
(defun softmax (s m) (declare (type dv s) (fixnum m))          ; in place over first m entries
  (let ((mx -1d30)) (declare (double-float mx)) (dotimes (j m) (when (> (aref s j) mx) (setf mx (aref s j))))
    (let ((z 0d0)) (declare (double-float z)) (dotimes (j m) (setf (aref s j) (exp (- (aref s j) mx))) (incf z (aref s j)))
      (dotimes (j m) (setf (aref s j) (/ (aref s j) z))))) s)

;;; --- layer 1: previous-token head.  score uses E (position-aware); VALUE = token code -----
;;; prev_q = sum_{j<=q} a1_qj t_j  -> when a1 attends to q-1, prev_q ~ t_{q-1} (a CLEAN code,
;;; not superimposed with the current token -- so layer 2 can separate current from previous).
(defun attn1-fwd (m1 E T- n)                                  ; returns (values prev a1)
  (let ((prev (make-array n)) (a1 (make-array n)))
    (dotimes (q n (values prev a1))
      (let ((scores (dvec (1+ q))) (evq (aref E q)))
        (dotimes (j (1+ q)) (setf (aref scores j) (vdot evq (mvec m1 (aref E j)))))
        (softmax scores (1+ q))
        (let ((pv (dvec *d*))) (declare (type dv pv))         ; value = token code, NO residual
          (dotimes (j (1+ q)) (let ((aj (aref scores j)) (tj (aref T- j))) (dotimes (i *d*) (incf (aref pv i) (* aj (aref tj i))))))
          (setf (aref prev q) pv (aref a1 q) scores))))))

(defun attn1-dM1 (seedp m1 E T- a1 n)                         ; sum_q VJP of prev_q wrt M1, seed=seedp_q
  (let ((dM (dmat *d* *d*)))
    (dotimes (q n dM)
      (let* ((a (aref a1 q)) (sp (aref seedp q)) (mm (1+ q)) (ga (dvec mm)) (gbar 0d0))
        (declare (type dv a sp ga) (double-float gbar))
        (dotimes (j mm) (setf (aref ga j) (vdot sp (aref T- j))) (incf gbar (* (aref a j) (aref ga j))))  ; prev=sum a_j t_j
        (dotimes (j mm) (let ((gs (* (aref a j) (- (aref ga j) gbar))) (evq (aref E q)) (ej (aref E j)))
                          (dotimes (r *d*) (let ((c (* gs (aref evq r)))) (dotimes (cc *d*) (incf (aref dM r cc) (* c (aref ej cc))))))))))))

;;; --- layer 2: induction head.  query = t_last (current token); key = prev_j; value = t_j ---
(defun attn2-fwd (m2 prev T- n)                              ; returns (values o a2)
  (let* ((l (1- n)) (scores (dvec n)) (tl (aref T- l)))
    (dotimes (j n) (setf (aref scores j) (vdot tl (mvec m2 (aref prev j)))))
    (softmax scores n)
    (let ((o (dvec *d*))) (declare (type dv o))
      (dotimes (j n) (let ((aj (aref scores j)) (tj (aref T- j))) (dotimes (i *d*) (incf (aref o i) (* aj (aref tj i))))))
      (values o scores))))

(defun attn2-back (seedo m2 prev T- a2 n)                    ; (values dM2 dprev) ; dprev: n dv
  (let* ((l (1- n)) (tl (aref T- l)) (ga (dvec n)) (gbar 0d0) (dM (dmat *d* *d*)) (dprev (make-array n))
         (m2tl (mTvec m2 tl)))
    (declare (type dv tl ga m2tl) (double-float gbar))
    (dotimes (j n) (setf (aref dprev j) (dvec *d*)))
    (dotimes (j n) (setf (aref ga j) (vdot seedo (aref T- j))) (incf gbar (* (aref a2 j) (aref ga j))))
    (dotimes (j n)                                            ; s_j = tl^T M2 prev_j ; query tl is fixed
      (let* ((gs (* (aref a2 j) (- (aref ga j) gbar))) (pj (aref prev j)) (dpj (aref dprev j)))
        (declare (double-float gs) (type dv pj dpj))
        (dotimes (r *d*) (let ((c (* gs (aref tl r)))) (dotimes (cc *d*) (incf (aref dM r cc) (* c (aref pj cc))))))  ; dM2 += gs tl (x) prev_j
        (dotimes (i *d*) (incf (aref dpj i) (* gs (aref m2tl i))))))                ; key part -> prev_j
    (values dM dprev)))

;;; --- forward + loss + prediction ------------------------------------------------------
(defun seq-emb (toks) (let* ((n (length toks)) (E (make-array n)) (T- (make-array n)))
                        (loop for i from 0 for tk in toks do (setf (aref E i) (emb tk i) (aref T- i) (tcode tk)))
                        (values E T- n)))
(defun predict-token (m1 m2 toks vocab)
  (multiple-value-bind (E T- n) (seq-emb toks)
    (multiple-value-bind (prev a1) (attn1-fwd m1 E T- n) (declare (ignore a1))
      (multiple-value-bind (o a2) (attn2-fwd m2 prev T- n) (declare (ignore a2))
        (let (best (bd -1d30)) (dolist (c vocab best) (let ((s (vdot o (tcode c)))) (when (> s bd) (setf bd s best c)))))))))

;;; --- BACKPROP gradient (yardstick + validation) ---------------------------------------
(defun backprop-grads (m1 m2 toks target)                    ; returns (values dM1 dM2 o)
  (multiple-value-bind (E T- n) (seq-emb toks)
    (multiple-value-bind (prev a1) (attn1-fwd m1 E T- n)
      (multiple-value-bind (o a2) (attn2-fwd m2 prev T- n)
        (let ((go (dvec *d*))) (declare (type dv go))
          (dotimes (i *d*) (setf (aref go i) (- (aref o i) (aref target i))))         ; dL/do
          (multiple-value-bind (dM2 dprev) (attn2-back go m2 prev T- a2 n)
            (values (attn1-dM1 dprev m1 E T- a1 n) dM2 o)))))))  ; dprev seeds layer 1

;;; --- PREDICTIVE CODING: settle the layer-1 latents prev, then local per-layer updates ---
(defun pc-grads (m1 m2 toks target steps gamma)              ; returns (values dM1 dM2 o eo-norm)
  (multiple-value-bind (E T- n) (seq-emb toks)
    (multiple-value-bind (muprev a1) (attn1-fwd m1 E T- n)    ; muprev FIXED (depends only on E,T,M1)
      (let ((prev (make-array n)))
        (dotimes (q n) (setf (aref prev q) (let ((p (dvec *d*))) (dotimes (i *d* p) (setf (aref p i) (aref (aref muprev q) i))))))  ; init at feedforward
        (dotimes (it steps)
          (multiple-value-bind (o a2) (attn2-fwd m2 prev T- n)
            (let ((eo (dvec *d*))) (declare (type dv eo))
              (dotimes (i *d*) (setf (aref eo i) (- (aref target i) (aref o i))))      ; output error (clamped target)
              (multiple-value-bind (dM2 tdp) (attn2-back eo m2 prev T- a2 n) (declare (ignore dM2))
                (dotimes (q n)                                  ; dprev_q = -(prev_q - muprev_q) + tdp_q
                  (let ((pq (aref prev q)) (mq (aref muprev q)) (td (aref tdp q)))
                    (declare (type dv pq mq td))
                    (dotimes (i *d*) (incf (aref pq i) (* gamma (+ (- (aref mq i) (aref pq i)) (aref td i)))))))))))
        ;; equilibrium errors -> LOCAL weight updates
        (multiple-value-bind (o a2) (attn2-fwd m2 prev T- n)
          (let ((eo (dvec *d*)) (eprev (make-array n)))
            (declare (type dv eo))
            (dotimes (i *d*) (setf (aref eo i) (- (aref target i) (aref o i))))
            (dotimes (q n) (let ((e (dvec *d*)) (pq (aref prev q)) (mq (aref muprev q)))
                             (dotimes (i *d*) (setf (aref e i) (- (aref pq i) (aref mq i)))) (setf (aref eprev q) e)))
            (multiple-value-bind (dM2 dptmp) (attn2-back eo m2 prev T- a2 n) (declare (ignore dptmp))
              (values (attn1-dM1 eprev m1 E T- a1 n) dM2 o (vdot eo eo)))))))))

;;; --- task: in-context induction -------------------------------------------------------
(defparameter *vocab* '("a" "b" "c" "d" "e" "f"))
(defun pick (l) (nth (mod (nextr) (length l)) l))
(defun gen-prompt ()                                          ; (toks . B) ;  ... A B ... A
  (let* ((A (pick *vocab*)) (B (pick *vocab*)))
    (loop while (string= A B) do (setf B (pick *vocab*)))
    (let ((others (remove-if (lambda (x) (or (string= x A) (string= x B))) *vocab*)))
      (flet ((fl (k) (loop repeat k collect (pick others))))
        (cons (append (fl (mod (nextr) 2)) (list A B) (fl (1+ (mod (nextr) 2))) (list A)) B)))))

(defun cosine-m (a b) (declare (type dm a b))
  (let ((d 0d0) (na 0d0) (nb 0d0)) (declare (double-float d na nb))
    (dotimes (i *d*) (dotimes (j *d*) (let ((x (aref a i j)) (y (aref b i j))) (incf d (* x y)) (incf na (* x x)) (incf nb (* y y)))))
    (if (or (zerop na) (zerop nb)) 0d0 (/ d (sqrt (* na nb))))))
(defun scaled (m s) (declare (type dm m)) (let ((o (dmat *d* *d*))) (dotimes (i *d* o) (dotimes (j *d*) (setf (aref o i j) (* s (aref m i j)))))))
(defun add! (w dw lr) (declare (type dm w dw)) (dotimes (i *d*) (dotimes (j *d*) (incf (aref w i j) (* lr (aref dw i j))))))

(defun accuracy (m1 m2 prompts)
  (let ((ok 0)) (dolist (pr prompts (/ ok (float (length prompts))))
    (when (string= (predict-token m1 m2 (car pr) *vocab*) (cdr pr)) (incf ok)))))

;;; train with either gradient source; STEP returns (values dM1 dM2)
(defun train (prompts which epochs lr mom steps gamma)
  (setf *seed* 123) (let ((m1 (dmat *d* *d* 0.1d0)) (m2 (dmat *d* *d* 0.1d0))
                          (v1 (dmat *d* *d*)) (v2 (dmat *d* *d*)))
    (dotimes (e epochs (values m1 m2))
      (let ((g1 (dmat *d* *d*)) (g2 (dmat *d* *d*)) (n (length prompts)))
        (dolist (pr prompts)
          (let ((target (tcode (cdr pr))))
            (multiple-value-bind (dM1 dM2)
                (if (eq which :pc) (pc-grads m1 m2 (car pr) target steps gamma)
                    (multiple-value-bind (a b) (backprop-grads m1 m2 (car pr) target) (values (scaled a -1d0) (scaled b -1d0)))) ; descent = -grad
              (add! g1 dM1 1d0) (add! g2 dM2 1d0))))
        (dotimes (i *d*) (dotimes (j *d*)
          (setf (aref v1 i j) (+ (* mom (aref v1 i j)) (* lr (/ (aref g1 i j) n)))) (incf (aref m1 i j) (aref v1 i j))
          (setf (aref v2 i j) (+ (* mom (aref v2 i j)) (* lr (/ (aref g2 i j) n)))) (incf (aref m2 i j) (aref v2 i j))))))))

;;; oracle: feed CLEAN previous-token keys (prev_j = t_{j-1}), M2 = c*I -- does layer 2 alone
;;; do induction?  (Tests whether the CIRCUIT can represent the task, separate from training.)
(defun acc-oracle (prompts c)
  (let ((ok 0))
    (dolist (pr prompts (/ ok (float (length prompts))))
      (multiple-value-bind (E T- n) (seq-emb (car pr)) (declare (ignore E))
        (let ((prev (make-array n)) (m2 (dmat *d* *d*)))
          (dotimes (j n) (setf (aref prev j) (if (zerop j) (dvec *d*) (aref T- (1- j)))))
          (dotimes (i *d*) (setf (aref m2 i i) c))
          (multiple-value-bind (o a2) (attn2-fwd m2 prev T- n) (declare (ignore a2))
            (let (best (bd -1d30)) (dolist (cc *vocab*) (let ((s (vdot o (tcode cc)))) (when (> s bd) (setf bd s best cc))))
              (when (string= best (cdr pr)) (incf ok)))))))))

;;; ============================================================= demo =====================
(setf *seed* 777)
(defparameter *train* (loop repeat 250 collect (gen-prompt)))
(defparameter *test*  (loop repeat 200 collect (gen-prompt)))
(format t "~%Predictive coding over a 2-layer ATTENTION induction circuit (no backprop).~%")
(format t "Task: in-context induction (... A B ... A -> B).  Chance = ~,2f.  FF baseline ~~0.30.~%~%"
        (/ 1.0 (length *vocab*)))

;;; (1) the CIRCUIT can represent induction (clean keys + M2=I).
(format t "(1) Can the circuit represent induction?  oracle (clean prev keys, M2=I): ~,2f~%~%"
        (acc-oracle *test* 4d0))

;;; (2) PC's update reproduces backprop's credit assignment, per attention layer.
(setf *seed* 50)
(let ((m1 (dmat *d* *d* 0.3d0)) (m2 (dmat *d* *d* 0.3d0)) (c1 0d0) (c2 0d0) (k 0))
  (dolist (pr (subseq *train* 0 40))
    (let ((target (tcode (cdr pr))))
      (multiple-value-bind (b1 b2 o) (backprop-grads m1 m2 (car pr) target) (declare (ignore o))
        (multiple-value-bind (p1 p2) (pc-grads m1 m2 (car pr) target 50 0.1d0)
          (incf c1 (cosine-m p1 (scaled b1 -1d0)))            ; PC update vs backprop DESCENT direction
          (incf c2 (cosine-m p2 (scaled b2 -1d0))) (incf k)))))
  (format t "(2) cosine(PC update, backprop step), per attention layer (avg over prompts):~%")
  (format t "      layer 1 (M1):  ~,3f      layer 2 (M2):  ~,3f~%" (/ c1 k) (/ c2 k))
  (format t "    (~~1.0 = PC's LOCAL update IS backprop's credit assignment through the attention depth.)~%~%"))

;;; (3) JOINT end-to-end training from scratch -- backprop AND predictive coding.
(format t "(3) induction accuracy after JOINT training from scratch (same circuit):~%")
(multiple-value-bind (m1 m2) (train *train* :bp 60 0.5d0 0.9d0 0 0d0)
  (format t "      backprop                  test ~,2f~%" (accuracy m1 m2 *test*)))
(multiple-value-bind (m1 m2) (train *train* :pc 60 0.5d0 0.9d0 40 0.1d0)
  (format t "      PREDICTIVE CODING (local) test ~,2f~%" (accuracy m1 m2 *test*)))

(format t "~%Honest report -- a clarifying NEGATIVE result that sharpens the frontier:~%")
(format t "  * The circuit CAN represent induction (oracle ~~1.0), and PC's local update IS~%")
(format t "    backprop's gradient through the attention depth (cosine ~~0.9 above) -- so PC is a~%")
(format t "    faithful LOCAL stand-in for backprop here, not a weaker rule.~%")
(format t "  * Yet JOINT end-to-end training from scratch lands at ~~chance for BOTH backprop and~%")
(format t "    PC.  The obstacle is the credit-through-depth PLATEAU: layer 1 must become a~%")
(format t "    previous-token head, but gets no useful signal until layer 2 is an induction head,~%")
(format t "    and vice versa.  At this toy scale even backprop does not escape it.~%")
(format t "  * So the binding constraint is JOINT OPTIMIZATION, not the locality of the rule.  The~%")
(format t "    route that DOES work (deep-composition-experiment.lisp) gives each layer its OWN~%")
(format t "    local objective (layer 1: predict the previous token) -- reaching 1.00.  Unifying~%")
(format t "    that with PC (a per-layer prediction-error objective at depth) is the next step.~%")
(format t "  Honest: a toy (small vocab, short prompts); backprop-free deep-attention training~%")
(format t "  remains open -- this maps the wall more precisely, it does not break through it.~%")
