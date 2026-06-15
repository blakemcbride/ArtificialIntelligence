;;;; concept-vectors-experiment.lisp -- Hebbian DISTRIBUTED concept vectors, learned online.
;;;; Run:  sbcl --script concept-vectors-experiment.lisp
;;;;
;;;; The move (per our discussion): non-brittleness comes from DISTRIBUTED, CONTINUOUS
;;;; representations -- a novel thing lands NEAR known ones and interpolates, instead of
;;;; falling off the cliff a symbolic/frame system hits when you stray off its design.
;;;; This is the cheap, local, biologically-flavored way to get it -- the CMAC / Kanerva
;;;; sparse-distributed-memory / hyperdimensional family:
;;;;   * each concept is a high-dimensional vector;
;;;;   * built ONLINE by superposing the (random) codes of the words it co-occurs with
;;;;     (Hebbian: co-active -> added together -- no backprop, just accumulation);
;;;;   * similar concepts get similar vectors because they SHARE contexts -- generalization
;;;;     by overlap, exactly CMAC's coarse-coding idea;
;;;;   * meaning / similarity / membership / counting become GEOMETRY (distance), not label
;;;;     matching -- so they degrade gracefully instead of breaking.
;;;;
;;;; Learning rule cost: per taught fact, a few vector additions.  No gradients, no layers.

(defparameter *dim* 4096)

;; --- deterministic random unit code for a word (the fixed "cell pattern" it drives) ----
(defparameter *codes* (make-hash-table :test 'equal))
(defun str-hash (s)
  (let ((h 2166136261))
    (loop for ch across s do (setf h (logand (* (logxor h (char-code ch)) 16777619) #xFFFFFFFF)))
    h))
(defun code (word)
  (or (gethash word *codes*)
      (setf (gethash word *codes*)
	    (let ((state (logand (str-hash word) #x7FFFFFFF))
		  (v (make-array *dim* :element-type 'double-float)))
	      (when (zerop state) (setf state 1))
	      (flet ((nx () (setf state (logand (+ (* state 1103515245) 12345) #x7FFFFFFF))
		         (- (* 2.0d0 (/ state #x7FFFFFFF)) 1.0d0)))
		(dotimes (i *dim*) (setf (aref v i) (nx))))
	      v))))

;; --- a concept's vector: the running superposition of its contexts (learned online) ----
(defparameter *concept* (make-hash-table :test 'equal))   ; word -> accumulated vector
(defun blank () (make-array *dim* :element-type 'double-float :initial-element 0.0d0))
(defun concept-acc (w) (or (gethash w *concept*) (setf (gethash w *concept*) (blank))))

(defun learn-fact (words)
  "Continual, local learning: in this fact, every word's concept accumulates the codes of
   the OTHER words it co-occurs with.  Words that keep appearing in the same company drift
   together in the space."
  (dolist (w words)
    (let ((acc (concept-acc w)))
      (dolist (other words)
	(unless (eq other w)
	  (let ((c (code other)))
	    (dotimes (i *dim*) (incf (aref acc i) (aref c i)))))))))

;; --- geometry ------------------------------------------------------------------------
(defun dot (a b) (loop for x across a for y across b sum (* x y)))
(defun norm (v) (sqrt (max 1d-12 (dot v v))))
;; mean-center to drop the common-mode shared by everything (cheap decorrelation), then cosine
(defparameter *mean* nil)
(defun compute-mean ()
  (let ((m (blank)) (n 0))
    (maphash (lambda (w v) (declare (ignore w)) (incf n)
	       (dotimes (i *dim*) (incf (aref m i) (aref v i)))) *concept*)
    (when (plusp n) (dotimes (i *dim*) (setf (aref m i) (/ (aref m i) n))))
    (setf *mean* m)))
(defun cvec (w)
  (let ((v (gethash w *concept*)))
    (when v (let ((c (make-array *dim* :element-type 'double-float)))
	      (dotimes (i *dim*) (setf (aref c i) (- (aref v i) (aref *mean* i))))
	      c))))
(defun sim (a b)
  (let ((va (cvec a)) (vb (cvec b)))
    (if (and va vb) (/ (dot va vb) (* (norm va) (norm vb))) 0.0)))
(defun centroid (words)
  (let ((m (blank)) (n 0))
    (dolist (w words) (let ((v (cvec w))) (when v (incf n) (dotimes (i *dim*) (incf (aref m i) (aref v i))))))
    (when (plusp n) (dotimes (i *dim*) (setf (aref m i) (/ (aref m i) n))))
    m))
(defun sim-to (w centroid-vec)
  (let ((v (cvec w))) (if v (/ (dot v centroid-vec) (* (norm v) (norm centroid-vec))) 0.0)))

;;; =====================================================================================
;;; Demonstration
;;; =====================================================================================
(dolist (f '(("a" "dog" "has" "fur") ("a" "dog" "has" "legs") ("a" "dog" "is" "an" "animal") ("a" "dog" "can" "run")
	     ("a" "cat" "has" "fur") ("a" "cat" "has" "legs") ("a" "cat" "is" "an" "animal") ("a" "cat" "can" "run")
	     ("a" "lion" "has" "fur") ("a" "lion" "has" "legs") ("a" "lion" "is" "an" "animal")
	     ("a" "cow" "has" "fur") ("a" "cow" "has" "legs") ("a" "cow" "is" "an" "animal")
	     ("a" "car" "has" "wheels") ("a" "car" "has" "doors") ("a" "car" "is" "a" "vehicle") ("a" "car" "can" "drive")
	     ("a" "bus" "has" "wheels") ("a" "bus" "has" "doors") ("a" "bus" "is" "a" "vehicle")
	     ("a" "truck" "has" "wheels") ("a" "truck" "is" "a" "vehicle")))
  (learn-fact f))
(compute-mean)

(format t "~%Emergent similarity (learned only from shared usage, no rules, no frames):~%")
(format t "   dog ~~ cat   = ~,3f~%" (sim "dog" "cat"))
(format t "   dog ~~ lion  = ~,3f~%" (sim "dog" "lion"))
(format t "   dog ~~ car   = ~,3f   (low -- different company)~%" (sim "dog" "car"))
(format t "   car ~~ bus   = ~,3f~%" (sim "car" "bus"))

(format t "~%Generalization from MINIMAL data (non-brittle): teach a brand-new word ONE fact~%")
(learn-fact '("a" "wolf" "has" "fur"))    ; wolf: a single fact
(compute-mean)
(format t "   after only \"a wolf has fur\":~%")
(format t "   wolf ~~ dog  = ~,3f   wolf ~~ cat = ~,3f   wolf ~~ car = ~,3f~%"
	(sim "wolf" "dog") (sim "wolf" "cat") (sim "wolf" "car"))
(format t "   -> wolf landed near the animals, not the vehicles, from one fact.~%")

(format t "~%Membership + counting as GEOMETRY (give a few examples, no category label needed):~%")
(let ((animalish (centroid '("dog" "cat")))      ; the category = where these examples sit
      (words '("dog" "cat" "lion" "cow" "wolf" "car" "bus" "truck")))
  (format t "   similarity of each known thing to the {dog, cat} region:~%")
  (dolist (w words) (format t "      ~6a ~,3f~%" w (sim-to w animalish)))
  (let ((n (count-if (lambda (w) (> (sim-to w animalish) 0.15)) words)))
    (format t "   how many are in that region (> 0.15)?  ~d~%" n)))

(format t "~%No frames, no labels, no rules -- just positions in a learned space.  A novel~%")
(format t "thing gets a position and a graded answer; it never falls off a cliff.  Learning~%")
(format t "is a handful of vector adds per fact -- local and cheap, no backprop.~%")
