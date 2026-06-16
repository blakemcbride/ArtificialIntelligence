
(defpackage "relations"
  (:use "COMMON-LISP")
  (:export "OBSERVE"
	   "RELATION-OF"
	   "MEMBERSHIP-CONNECTORS"
	   "RELATIONS-REPORT"))

(in-package "relations")
(provide "relations")

(require "data-structures")
(use-package "data-structures")
(require "line-input")
(use-package "line-input")   ; tokenize, as-words

;;; LEARNED RELATION DISCOVERY (Plan.md Phase 9) -- the live-system version of
;;; relation-discovery-experiment.lisp.  The prose reader's "is a", "is the Y of", ... and
;;; its fixed subject/category positions are HARDCODED (extract-fact / note-facts).  This
;;; layer instead LEARNS, by Hebbian counting (no backprop, continual):
;;;
;;;   * CONNECTORS -- a MEMBERSHIP connector ("is a") links many subjects to few category
;;;     hubs (high subjects-per-category); a relational connector / verb is ~one-to-one.
;;;   * FUNCTION WORDS -- frequent tokens that are rarely a head (the/a/is); the glue.
;;;   * HEADS -- the real subject/category in a span is the word that most often serves as a
;;;     head, so modifiers and relative-clause words fall away.
;;;
;;; It runs ALONGSIDE the hardcoded patterns, not as a hard replacement: a learned
;;; recognizer needs accumulated evidence (*rel-min-support*), so the hardcoded path is the
;;; cold-start floor and this layer takes over / extends it as more is read (e.g. it parses
;;; "a small brown dog that barks is a friendly animal" -> dog / is a / animal, and learns
;;; novel connectors like "is a kind of", which the patterns cannot).  Stores live in
;;; data-structures and persist with the network.

(defparameter *rel-min-support* 3   "Distinct links a connector needs before it is judged.")
(defparameter *rel-hub* 1.8         "subjects-per-category at/above this = membership connector.")
(defparameter *rel-fw* 0.12         "Appear in >= this fraction of sentences (and rarely a head) => function word.")
(defparameter *rel-short* 5         "<= this many words (after a leading article) => seed positionally.")

;;; ------------------------------------------------------------------- helpers ---
(defun jw (ws) (format nil "~{~a~^ ~}" ws))
(defun hc (w) (gethash w *rel-head* 0))

(defun strip-lead-article (ws)
  (if (member (first ws) '("a" "an" "the") :test #'string=) (rest ws) ws))

(defun find-sub (sub seq)
  (let ((ls (length sub)) (ln (length seq)))
    (loop for i from 0 to (- ln ls)
	  when (loop for j from 0 below ls always (string= (nth j sub) (nth (+ i j) seq)))
	    do (return i))))

(defun link! (c a b)
  (when (and (plusp (length c)) (plusp (length a)) (plusp (length b)) (not (string= a b)))
    (let ((tab (or (gethash c *rel-links*)
		   (setf (gethash c *rel-links*) (make-hash-table :test 'equal)))))
      (setf (gethash (format nil "~a|~a" a b) tab) (cons a b)))))

(defun conn-stats (c)
  (let ((tab (gethash c *rel-links*)) (cats (make-hash-table :test 'equal)) (pairs 0))
    (when tab
      (maphash (lambda (k v) (declare (ignore k)) (incf pairs) (setf (gethash (cdr v) cats) t)) tab))
    (let ((dc (hash-table-count cats)))
      (values pairs dc (if (plusp dc) (/ pairs dc) 0)))))

(defun classify (c)
  (multiple-value-bind (pairs dc hub) (conn-stats c)
    (declare (ignore dc))
    (cond ((< pairs *rel-min-support*) :unknown)
	  ((>= hub *rel-hub*) :membership)
	  (t :relational))))

(defun function-word-p (w)
  "Frequent AND rarely a head -- so frequent CONTENT words (a common category like
   \"animal\") are not mistaken for glue, while the/a/is (frequent, never heads) are."
  (let ((f (gethash w *rel-freq* 0)))
    (and (plusp *rel-sentences*)
	 (>= (/ f *rel-sentences*) *rel-fw*)
	 (< (hc w) (* 0.3 f)))))

(defun known-connectors ()
  (let (cs)
    (maphash (lambda (c tab) (declare (ignore tab))
	       (when (>= (conn-stats c) *rel-min-support*) (push c cs)))
	     *rel-links*)
    (sort cs #'> :key (lambda (c) (length (tokenize c))))))

(defun find-connector (tokens)
  (dolist (c (known-connectors))
    (let* ((ct (tokenize c)) (i (find-sub ct tokens)))
      (when i (return-from find-connector
		(values (subseq tokens 0 i) c (subseq tokens (+ i (length ct))))))))
  nil)

(defun head-of (span)
  (let ((content (remove-if #'function-word-p span)))
    (when (null content) (setf content span))
    (let ((best (first content)) (bc (hc (first content))))
      (dolist (w (rest content) best)
	(when (>= (hc w) bc) (setf best w bc (hc w)))))))

(defun analyze (tokens)
  "Parse TOKENS (a word list) with the current learned model: (values SUBJECT CONNECTOR
   CATEGORY CLASS), or (values nil nil nil :unknown)."
  (let ((ws (strip-lead-article tokens)))
    (multiple-value-bind (left c right) (find-connector ws)
      (if (and c left right)
	  (values (head-of left) c (head-of right) (classify c))
	  (values nil nil nil :unknown)))))

;;; --------------------------------------------------------- learning + query ---
(defun observe (input)
  "Learn from one declarative sentence (a string or word list), continually: update token
   frequency, then either seed positionally (short, clean sentences) or extract heads with
   the current model (longer sentences -- online bootstrapping).  No corpus is stored."
  (let ((words (as-words input)))
    (when (>= (length words) 3)
      (incf *rel-sentences*)
      (dolist (w (remove-duplicates words :test #'string=)) (incf (gethash w *rel-freq* 0)))
      (let ((ws (strip-lead-article words)))
	(cond
	  ((<= (length ws) *rel-short*)             ; short -> trust positions: a <c> b
	   (when (>= (length ws) 3)
	     (let ((a (first ws)) (b (car (last ws))) (c (jw (subseq ws 1 (1- (length ws))))))
	       (incf (gethash a *rel-head* 0)) (incf (gethash b *rel-head* 0))
	       (link! c a b))))
	  (t                                        ; longer -> use what we already know
	   (multiple-value-bind (s c cat) (analyze words)
	     (when (and s c cat)
	       (incf (gethash s *rel-head* 0)) (incf (gethash cat *rel-head* 0))
	       (link! c s cat)))))))))

(defun relation-of (input)
  "The learned (SUBJECT CONNECTOR CATEGORY CLASS) of INPUT (string or word list)."
  (analyze (as-words input)))

(defun membership-connectors ()
  "Connectors learned to be membership-forming (\"is a\"-like), strongest hub first."
  (let (rows)
    (maphash (lambda (c tab) (declare (ignore tab))
	       (when (eq (classify c) :membership)
		 (multiple-value-bind (p d h) (conn-stats c) (declare (ignore d))
		   (push (cons c (cons p h)) rows))))
	     *rel-links*)
    (mapcar #'car (sort rows #'> :key #'cddr))))

(defun relations-report (&optional (stream t))
  "Print the connectors discovered so far and their learned class."
  (let (rows)
    (maphash (lambda (c tab) (declare (ignore tab))
	       (multiple-value-bind (p d h) (conn-stats c)
		 (when (>= p *rel-min-support*)
		   (push (list c p d (coerce h 'float) (classify c)) rows))))
	     *rel-links*)
    (setf rows (sort rows #'> :key #'fourth))
    (format stream "~&  ~24a ~6@a ~5@a ~9@a  ~a~%" "connector" "links" "cats" "subj/cat" "class")
    (dolist (r rows)
      (format stream "  ~24a ~6d ~5d ~9,1f  ~a~%" (first r) (second r) (third r) (fourth r)
	      (case (fifth r) (:membership "MEMBERSHIP") (:relational "relational") (t "unknown"))))
    rows))
