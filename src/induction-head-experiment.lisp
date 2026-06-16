;;;; induction-head-experiment.lisp -- IN-CONTEXT LEARNING from a 2-layer attention circuit,
;;;; built with Hebbian fast weights, no backprop.  Run:
;;;;     sbcl --script induction-head-experiment.lisp
;;;;
;;;; Develops attention-stack-experiment.lisp toward something genuinely LLM-like.  The
;;;; defining trick of modern LLMs is IN-CONTEXT LEARNING: shown a pattern in the prompt, the
;;;; model continues it -- with no weight update.  Mechanistic-interpretability work traced
;;;; this to the INDUCTION HEAD, a two-layer circuit:
;;;;     layer 1  (previous-token head): each position copies in its PREDECESSOR token.
;;;;     layer 2  (induction head): given the current token, find where it occurred before
;;;;              and emit the token that FOLLOWED it.
;;;; Net effect: "... A B ... A ?"  ->  predict B.  It needs DEPTH -- a single attention
;;;; layer cannot express "the token after the previous occurrence of X".
;;;;
;;;; Here both layers are fast-weight associative memories (outer-product binding, Hebbian),
;;;; tokens and positions are random codes (embeddings), and a cleanup step (decode + re-
;;;; encode) is the nonlinearity between layers.  Nothing is trained; the sequence is the
;;;; only input -- so a correct continuation of NOVEL tokens is in-context learning.

(defparameter *dim* 2048)

;;; --- codes (token & position "embeddings") --------------------------------------------
(defun str-hash (s)
  (let ((h 2166136261))
    (loop for c across s do (setf h (logand (* (logxor h (char-code c)) 16777619) #xffffffff)))
    h))

(defparameter *codes* (make-hash-table :test 'equal))
(defun code (sym)
  (or (gethash sym *codes*)
      (setf (gethash sym *codes*)
	    (let ((state (logand (str-hash sym) #x7fffffff))
		  (v (make-array *dim* :element-type 'double-float)))
	      (when (zerop state) (setf state 1))
	      (dotimes (i *dim* v)
		(setf state (logand (+ (* state 1103515245) 12345) #x7fffffff))
		(setf (aref v i) (if (>= state #x40000000) 1d0 -1d0)))))))

(defun pos (i) (format nil "#p~d" i))   ; positional embedding key

;;; --- fast-weight head: a Hebbian associative memory (one attention layer) --------------
(defun make-head () (make-array (list *dim* *dim*) :element-type 'double-float :initial-element 0d0))

(defun bind! (head value-sym key-sym)
  "Hebbian store: key -> value, by adding the outer product code(value) (x) code(key)."
  (let ((v (code value-sym)) (k (code key-sym)))
    (dotimes (i *dim*)
      (let ((vi (aref v i)))
	(dotimes (j *dim*) (incf (aref head i j) (* vi (aref k j))))))))

(defun retrieve (head key-vec)
  "head . key-vec -- attend and read out a (noisy) value vector."
  (let ((out (make-array *dim* :element-type 'double-float :initial-element 0d0)))
    (dotimes (i *dim* out)
      (let ((s 0d0))
	(dotimes (j *dim*) (incf s (* (aref head i j) (aref key-vec j))))
	(setf (aref out i) s)))))

(defun dot (a b) (let ((s 0d0)) (dotimes (i *dim* s) (incf s (* (aref a i) (aref b i))))))

(defun decode (vec vocab)
  "Nearest token in VOCAB to VEC; (values token best-dot second-dot)."
  (let (best bd second)
    (dolist (s vocab)
      (let ((d (dot (code s) vec)))
	(cond ((or (null best) (> d bd)) (setf second bd best s bd d))
	      ((or (null second) (> d second)) (setf second d)))))
    (values best bd (or second 0d0))))

(defun margin-str (best second)
  "Readable separation: best/second, or 'clean' when nothing else scores positive."
  (if (> second 0) (format nil "~,1fx" (/ best second)) "clean"))

;;; --- the two-layer induction circuit ---------------------------------------------------
(defun build-circuit (seq)
  "Build the previous-token head (L1) and the induction head (L2) from SEQ.
   Returns (values L2 vocab)."
  (let ((n (length seq)) (vocab (remove-duplicates seq :test #'string=))
	(l1 (make-head)) (l2 (make-head)))
    ;; layer 1 -- previous-token head: store token at each position (value=token, key=position)
    (loop for i from 0 for tok in seq do (bind! l1 tok (pos i)))
    ;; for each position, ATTEND to the previous position to recover the predecessor token,
    ;; CLEAN it up (decode + re-encode = the nonlinearity), and feed it to layer 2:
    ;; layer 2 -- induction head: key = predecessor token, value = current token
    (loop for i from 1 below n
	  for cur = (nth i seq)
	  for prev-vec = (retrieve l1 (code (pos (1- i))))
	  for prev = (decode prev-vec vocab)          ; cleanup nonlinearity between layers
	  do (bind! l2 cur prev))
    (values l1 l2 vocab)))

(defun predict-next (l2 last-token vocab)
  "Induction prediction: the token that followed LAST-TOKEN's earlier occurrence."
  (decode (retrieve l2 (code last-token)) vocab))

(defun continue-seq (l2 last-token vocab k)
  "Autoregressively emit K tokens by iterating the induction head."
  (let (out (cur last-token))
    (dotimes (i k (nreverse out))
      (setf cur (predict-next l2 cur vocab))
      (push cur out))))

;;; --------------------------------------------------------------------- demo ------------
(defun demo (seq label)
  (format t "~%=== ~a ===~%" label)
  (format t "  prompt (novel tokens, never \"trained\"): ~{~a~^ ~}~%" seq)
  (multiple-value-bind (l1 l2 vocab) (build-circuit seq)
    ;; layer 1 works: recover the predecessor of a middle position
    (let ((i (floor (length seq) 2)))
      (multiple-value-bind (p bd sd) (decode (retrieve l1 (code (pos i))) vocab)
	(format t "  layer 1 (previous-token head): token at position ~d is ~a (~a)~%"
		i p (margin-str bd sd))))
    ;; layer 2: in-context next-token prediction
    (let ((last (car (last seq))))
      (multiple-value-bind (nxt bd sd) (predict-next l2 last vocab)
	(format t "  layer 2 (induction head): after final \"~a\" -> predict \"~a\" (~a)~%"
		last nxt (margin-str bd sd)))
      (format t "  autoregressive continuation: ~{~a~^ ~}~%" (continue-seq l2 last vocab 6)))))

(demo '("zon" "qix" "vel" "zon" "qix" "vel" "zon") "in-context learning: a length-3 motif")
(demo '("fip" "wug" "fip" "wug" "fip") "generalization: different novel tokens, length-2 motif")
(demo '("blue" "seven" "tree" "blue" "seven" "tree" "blue" "seven") "another: predict the 3rd of the cycle")

(format t "~%Takeaway: a TWO-LAYER attention circuit -- a previous-token head feeding an~%")
(format t "induction head, with a cleanup nonlinearity between -- does IN-CONTEXT LEARNING:~%")
(format t "it continues a pattern of tokens it never saw, from the prompt alone, with NO~%")
(format t "weight training and NO backprop (pure Hebbian outer-product binding).  This is the~%")
(format t "interpretability \"induction head\" -- the circuit behind LLM in-context learning --~%")
(format t "realized with local fast weights.  Depth is essential, not optional: layer 2's keys~%")
(format t "ARE the predecessor tokens that only layer 1 (positional attention) can supply, so~%")
(format t "the induction head literally cannot be built without the layer beneath it.  Toward a~%")
(format t "fuller model: many such layers with distinct learned weights, a value/output~%")
(format t "projection per head, and a local learning rule for those weights.~%")
