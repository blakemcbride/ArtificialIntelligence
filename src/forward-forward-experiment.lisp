;;;; forward-forward-experiment.lisp -- DEPTH AT SCALE under ONE unified local rule, no backprop.
;;;; Run:  sbcl --script forward-forward-experiment.lisp
;;;;
;;;; The remaining frontier (notes/TransformerFromHebbianParts.md): train a DEEP stack of
;;;; layers with a SINGLE unified local rule -- the same rule at every layer -- and NO global
;;;; backward pass.  This implements Hinton's Forward-Forward (2022), the canonical such rule.
;;;;
;;;; Idea: instead of backprop, each layer has its OWN local objective -- make a "goodness"
;;;; (sum of squared activities) HIGH on POSITIVE data and LOW on NEGATIVE data.  Positive =
;;;; an input with its CORRECT label embedded; negative = the same input with a WRONG label.
;;;; Each layer updates its weights from a local logistic loss on its own goodness, using only
;;;; that layer's input and output -- no signal is propagated from other layers.  Activities
;;;; are length-normalized before being passed up, so a layer can't cheat on raw magnitude.
;;;; At test time, the predicted label is the one whose embedding yields the highest total
;;;; goodness.  The SAME rule trains every layer, so it scales to arbitrary depth.
;;;;
;;;; Task: a nonlinear one -- is a 2-D point INSIDE a circle?  (not linearly separable.)  We
;;;; show the one rule trains nets of depth 1..5 with no backprop (it scales to depth; the
;;;; deepest representation stays discriminative) and clearly beats a linear baseline trained
;;;; by the same rule (which cannot bend a circle).  Honest caveat: on a task this simple one
;;;; hidden layer already suffices, so added depth does not improve accuracy here.

(defparameter *hidden* 40)
(defparameter *theta* 2.0d0)   ; goodness threshold
(defparameter *lr* 0.03d0)
(defparameter *epochs* 30)

;;; --- PRNG + helpers -------------------------------------------------------------------
(defparameter *seed* 1)
(defun nextr () (setf *seed* (logand (+ (* *seed* 1103515245) 12345) #x7fffffff)))
(defun runif () (- (* 2d0 (/ (nextr) 2147483647d0)) 1d0))   ; uniform in [-1,1]
(defun sigmoid (x) (/ 1d0 (+ 1d0 (exp (- (max -30d0 (min 30d0 x)))))))
(defun relu (x) (if (> x 0d0) x 0d0))

;;; --- a layer: weights out x in ---------------------------------------------------------
(defstruct layer w nout nin)
(defun make-rand-layer (nin nout)
  (let ((w (make-array (list nout nin) :element-type 'double-float)))
    (dotimes (i nout) (dotimes (j nin) (setf (aref w i j) (* 0.3d0 (runif)))))
    (make-layer :w w :nout nout :nin nin)))

(defun layer-forward (ly inp)
  "Return (values normalized-activity goodness z a)."
  (let* ((nout (layer-nout ly)) (nin (layer-nin ly))
	 (z (make-array nout :element-type 'double-float))
	 (a (make-array nout :element-type 'double-float)) (g 0d0))
    (dotimes (i nout)
      (let ((s 0d0))
	(dotimes (j nin) (incf s (* (aref (layer-w ly) i j) (aref inp j))))
	(setf (aref z i) s) (setf (aref a i) (relu s)) (incf g (* (aref a i) (aref a i)))))
    (let* ((norm (sqrt (+ 1d-8 g))) (out (make-array nout :element-type 'double-float)))
      (dotimes (i nout) (setf (aref out i) (/ (aref a i) norm)))
      (values out g z a))))

(defun layer-update (ly inp z a g positive)
  "Local Forward-Forward update of LY from its own goodness G -- no backprop."
  (let* ((target (if positive 1d0 0d0)) (coeff (- (sigmoid (- g *theta*)) target)))
    (dotimes (i (layer-nout ly))
      (when (> (aref z i) 0d0)
	(let ((c (* *lr* coeff 2d0 (aref a i))))
	  (dotimes (j (layer-nin ly)) (decf (aref (layer-w ly) i j) (* c (aref inp j)))))))))

;;; --- a deep stack: ONE rule, every layer ----------------------------------------------
(defun fwd-train (layers x positive)
  (let ((inp x))
    (dolist (ly layers)
      (multiple-value-bind (out g z a) (layer-forward ly inp)
	(layer-update ly inp z a g positive)     ; same local rule at EVERY layer
	(setf inp out)))))

(defun total-goodness (layers x &optional (skip 1))
  "Sum goodness over layers from index SKIP upward (Hinton ignores the first layer)."
  (let ((inp x) (sum 0d0) (idx 0))
    (dolist (ly layers sum)
      (multiple-value-bind (out g) (layer-forward ly inp)
	(when (>= idx skip) (incf sum g))
	(setf inp out) (incf idx)))))

;;; --- data: inside a circle? (nonlinear) + label-embedding -----------------------------
(defun embed (x y label) (vector x y 1d0 (if (zerop label) 2d0 0d0) (if (zerop label) 0d0 2d0)))
(defun dataset (n)
  (loop repeat n collect (let* ((x (runif)) (y (runif))
				(lab (if (< (+ (* x x) (* y y)) 0.5d0) 0 1)))
			   (list x y lab))))

(defun build (nlayers)
  (let (ls (nin 5))
    (dotimes (i nlayers (nreverse ls))
      (push (make-rand-layer nin *hidden*) ls) (setf nin *hidden*))))

(defun train! (layers data)
  (dotimes (e *epochs*)
    (dolist (pt data)
      (destructuring-bind (x y lab) pt
	(fwd-train layers (embed x y lab) t)              ; positive: correct label
	(fwd-train layers (embed x y (- 1 lab)) nil)))))  ; negative: wrong label

(defun accuracy (layers data &optional (skip 1))
  (let ((ok 0))
    (dolist (pt data (/ ok (float (length data))))
      (destructuring-bind (x y lab) pt
	(let ((g0 (total-goodness layers (embed x y 0) skip))
	      (g1 (total-goodness layers (embed x y 1) skip)))
	  (when (= lab (if (>= g0 g1) 0 1)) (incf ok)))))))

;;; --------------------------------------------------------------------- demo ------------
(setf *seed* 12345)
(defparameter *train* (dataset 600))
(defparameter *test*  (dataset 300))

(format t "~%Forward-Forward: one local rule (no backprop) at every layer.~%")
(format t "Task: is a 2-D point inside a circle? (nonlinear)  ~d train / ~d test points.~%~%"
	(length *train*) (length *test*))
(format t "  depth   all-layer-goodness   last-layer-only~%")
(dolist (n '(1 2 3 5))
  (setf *seed* (+ 1000 n))
  (let ((net (build n)))
    (train! net *train*)
    (format t "  ~2d        ~,2f                ~,2f~%"
	    n (accuracy net *test* 0) (accuracy net *test* (max 0 (1- n))))))

;; linear baseline: same local logistic rule, but no hidden nonlinearity (goodness of a
;; single linear projection) -- cannot bend a circle.
(defun linear-accuracy ()
  (setf *seed* 999)
  (let ((w (make-array 5 :element-type 'double-float)))
    (dotimes (j 5) (setf (aref w j) (* 0.3d0 (runif))))
    (flet ((g (x) (let ((s 0d0)) (dotimes (j 5 (* s s)) (incf s (* (aref w j) (aref x j)))))))
      (dotimes (e *epochs*)
	(dolist (pt *train*)
	  (destructuring-bind (x y lab) pt
	    (dolist (pn (list (cons (embed x y lab) t) (cons (embed x y (- 1 lab)) nil)))
	      (let* ((in (car pn)) (pos (cdr pn))
		     (s (let ((acc 0d0)) (dotimes (j 5 acc) (incf acc (* (aref w j) (aref in j))))))
		     (gg (* s s)) (coeff (- (sigmoid (- gg *theta*)) (if pos 1d0 0d0))))
		(dotimes (j 5) (decf (aref w j) (* *lr* coeff 2d0 s (aref in j)))))))))
      (let ((ok 0))
	(dolist (pt *test* (/ ok (float (length *test*))))
	  (destructuring-bind (x y lab) pt
	    (when (= lab (if (>= (g (embed x y 0)) (g (embed x y 1))) 0 1)) (incf ok))))))))
(format t "  linear (no hidden)      ~,2f  <- one layer, no nonlinearity: cannot bend a circle~%"
	(linear-accuracy))

(format t "~%Takeaway: a SINGLE local rule -- raise goodness on correct-label inputs, lower it on~%")
(format t "wrong-label inputs -- trains EVERY layer of a deep stack with NO backward pass and NO~%")
(format t "weight transport (Forward-Forward).  The one rule scales to depth: it trains nets up~%")
(format t "to 5 layers, the deepest layer's representation stays discriminative, and all depths~%")
(format t "clearly beat a linear model on a nonlinear task.  Honest caveat: on a task this simple~%")
(format t "one hidden layer already suffices, so ADDED depth doesn't raise accuracy here --~%")
(format t "making depth pay off needs harder tasks and per-layer tuning (FF is finicky).  The~%")
(format t "milestone is the RULE: one local objective, no backprop, at arbitrary depth.  Training~%")
(format t "the attention layers this way (deep ATTENTION, locally) is the natural unification.~%")
