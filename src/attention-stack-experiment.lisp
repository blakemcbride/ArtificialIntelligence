;;;; attention-stack-experiment.lisp -- can Hebbian attention STACK?  (a step toward depth)
;;;; Run:  sbcl --script attention-stack-experiment.lisp
;;;;
;;;; attention.lisp realises ONE attention head as Hebbian fast weights: bind associations
;;;; into a matrix by outer products, M = SUM value (x) key, then retrieve by M . key.  That
;;;; is a single layer -- one "hop".  Modern LLMs get their power from STACKING many such
;;;; layers: each layer's output feeds the next, so meaning is composed across the whole
;;;; context, and multi-step (multi-hop) relationships become reachable.
;;;;
;;;; This PoC tests the smallest version of that question: compose two heads so the output of
;;;; head 1 is the query for head 2.  With this representation, stacking k heads = applying M
;;;; k times = M^k, i.e. following a relation k hops.  The task is transitive containment:
;;;;     cat in box,  box in room,  room in house
;;;; A single head answers only the IMMEDIATE container (cat -> box).  Reaching "the cat is
;;;; ultimately in the house" REQUIRES feeding one head's output into the next -- exactly
;;;; what a single layer cannot do.  All Hebbian (outer-product binding), no backprop.
;;;;
;;;; It also shows the honest catch: raw linear stacking (M^k) accumulates noise, so deep
;;;; stacks blur -- a CLEANUP step between layers (decode to the nearest symbol and re-encode,
;;;; a nonlinearity) restores sharpness, which is why real layers interleave a nonlinearity.

(defparameter *dim* 1024)

;;; --- deterministic random +/-1 code per symbol (the fixed "pattern" it drives) ---------
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

;;; --- the fast-weight associative memory M (one attention head) --------------------------
(defun make-m () (make-array (list *dim* *dim*) :element-type 'double-float :initial-element 0d0))

(defun bind! (m key value)
  "Hebbian: store key -> value by adding the outer product value (x) key into M."
  (let ((k (code key)) (v (code value)))
    (dotimes (i *dim*)
      (let ((vi (aref v i)))
	(dotimes (j *dim*) (incf (aref m i j) (* vi (aref k j))))))))

(defun mv (m x)
  "M . x  -- one attention hop: retrieve what X points to (a noisy value vector)."
  (let ((out (make-array *dim* :element-type 'double-float :initial-element 0d0)))
    (dotimes (i *dim* out)
      (let ((s 0d0))
	(dotimes (j *dim*) (incf s (* (aref m i j) (aref x j))))
	(setf (aref out i) s)))))

(defun dot (a b) (let ((s 0d0)) (dotimes (i *dim* s) (incf s (* (aref a i) (aref b i))))))

(defun decode (v symbols)
  "Nearest symbol to vector V by dot product; returns (values symbol margin) where margin is
   best/second (>1 means a clean win)."
  (let (best best-d second-d)
    (dolist (s symbols)
      (let ((d (dot (code s) v)))
	(cond ((or (null best) (> d best-d)) (setf second-d best-d best s best-d d))
	      ((or (null second-d) (> d second-d)) (setf second-d d)))))
    (values best (if (and second-d (> second-d 0)) (/ best-d second-d) most-positive-double-float))))

;;; --- stacking k heads, two ways --------------------------------------------------------
(defun stack-linear (m start k symbols)
  "Apply M k times raw (M^k . start) -- pure linear stacking.  Returns (values symbol margin)."
  (let ((v (code start)))
    (dotimes (i k) (setf v (mv m v)))
    (decode v symbols)))

(defun stack-cleanup (m start k symbols)
  "Apply M k times, but DECODE to the nearest symbol and re-encode between hops -- a cleanup
   nonlinearity that stops noise from compounding.  Returns (values symbol margin)."
  (let ((v (code start)) (sym start))
    (dotimes (i k) (setf v (mv m v)) (setf sym (decode v symbols)) (setf v (code sym)))
    (decode v symbols)))

;;; --------------------------------------------------------------------- demo ------------
(defparameter *containment*
  '(("cat" . "box") ("box" . "room") ("room" . "house")     ; chain 1: cat in box in room in house
    ("pen" . "drawer") ("drawer" . "desk") ("desk" . "office"))) ; chain 2

(defparameter *symbols*
  '("cat" "box" "room" "house" "pen" "drawer" "desk" "office"))

(defparameter *m* (make-m))
(dolist (p *containment*) (bind! *m* (car p) (cdr p)))   ; Hebbian: store each "in" relation

(format t "~%Stored containment (Hebbian outer-product binding): ~{~a~^, ~}~%"
	(mapcar (lambda (p) (format nil "~a in ~a" (car p) (cdr p))) *containment*))

(format t "~%=== one head = one hop (the immediate container) ===~%")
(dolist (start '("cat" "pen"))
  (multiple-value-bind (s m) (stack-linear *m* start 1 *symbols*)
    (format t "  ~a -> ~a   (margin ~,1f)~%" start s m)))

(format t "~%=== STACKING heads = multi-hop (composition one layer cannot do) ===~%")
(format t "    raw linear stack M^k (noise compounds with depth):~%")
(dolist (start '("cat" "pen"))
  (format t "  ~a:" start)
  (dotimes (k 3)
    (multiple-value-bind (s mg) (stack-linear *m* start (1+ k) *symbols*)
      (format t "  [~d head~:p -> ~a, margin ~,1f]" (1+ k) s mg)))
  (terpri))

(format t "~%    with a cleanup nonlinearity between heads (sharp at every depth):~%")
(dolist (start '("cat" "pen"))
  (format t "  ~a:" start)
  (dotimes (k 3)
    (multiple-value-bind (s mg) (stack-cleanup *m* start (1+ k) *symbols*)
      (format t "  [~d head~:p -> ~a, margin ~,1f]" (1+ k) s mg)))
  (terpri))

(format t "~%=== the single-layer limitation, concretely ===~%")
(format t "  Q: what large place is the cat ultimately inside? (a 3-hop question)~%")
(format t "    1 head : ~a   <- only the immediate container; WRONG for the question~%"
	(stack-cleanup *m* "cat" 1 *symbols*))
(format t "    3 heads: ~a   <- correct -- required feeding each head's output to the next~%"
	(stack-cleanup *m* "cat" 3 *symbols*))

(format t "~%Takeaway: composing Hebbian attention heads -- output of one as the query of the~%")
(format t "next -- DOES stack: k heads follow a relation k hops, reaching answers no single~%")
(format t "head can.  Raw linear stacking blurs with depth; a cleanup nonlinearity between~%")
(format t "heads keeps it sharp -- the local-learning echo of why transformer layers~%")
(format t "interleave a nonlinearity.  Next step toward real depth: DIFFERENT learned weights~%")
(format t "per layer (not the same M), still trained by a local rule.~%")
