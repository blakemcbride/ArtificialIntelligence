
(defpackage "operations"
  (:use "COMMON-LISP")
  (:export "NOTE-OPERATION"
	   "RUN-OPERATION"
	   "OPERATION-ANSWER-P"
	   "CATEGORY-MEMBERS"
	   "RECOGNIZED-MEMBER-P"
	   "*OP-KEYWORDS*"))

(in-package "operations")
(provide "operations")

(require "data-structures")
(use-package "data-structures")
(require "vectors")
(use-package "vectors")        ; nearest -- similarity in the distributed concept space

;;; Learned OPERATIONS over the system's own knowledge -- the start of "understanding as a
;;; learned procedure", not a stored fact.  An operation's answer is COMPUTED over the
;;; current knowledge, so it changes as the system learns (teach one more animal and
;;; "how many animals do you know" goes up by one).
;;;
;;; Two parts, mirroring the design discussion:
;;;   * a small set of general OPERATION PRIMITIVES -- the grounded substrate, reused by any
;;;     question.  The first is COUNT: enumerate the members of a category and tally them.
;;;     (It iterates + tallies -- the recurrent/successor substrate of procedure-experiment
;;;     -- here over the concept graph.)  It is NOT a per-question function.
;;;   * a LEARNED mapping from a question phrasing to (operation + category slot), taught in
;;;     the loop: teach "how many animals do you know" => "count animals", and it generalizes
;;;     to "how many <anything> do you know" through the slot -- exactly like composition,
;;;     but the payload is an operation instead of literal words.

(defparameter *op-keywords* '("count" "similar")
  "Operation names a teacher may use in an answer to say what a question MEANS (rather than
   giving a literal reply).  e.g. \"count animals\" or \"similar dog\" define operations.")

(defun operation-answer-p (answer-words)
  "Does a taught ANSWER define an operation, e.g. (\"count\" \"animals\")?"
  (and answer-words (member (first answer-words) *op-keywords* :test #'string=)))

;;; --- the COUNT primitive: members of a category, grounded in the concept graph --------
;;; Convention: category C is the set of subjects taught \"are <subject> C => yes\"; its
;;; members are that state's neighbours in the concept graph.  Counting is thus computed
;;; from what the system actually knows, and grows as it is taught more.
(defun category-variants (c)
  "C with its naive singular/plural forms, so we match a category however it was phrased
   (\"animals\" vs \"animal\", \"color\" vs \"colors\", \"countries\" vs \"country\")."
  (let ((n (length c)) (v (list c (concatenate 'string c "s"))))
    (when (and (> n 1) (char= (char c (1- n)) #\s))
      (push (subseq c 0 (1- n)) v))                                  ; cats -> cat
    (when (and (> n 3) (string= "ies" (subseq c (- n 3))))
      (push (concatenate 'string (subseq c 0 (- n 3)) "y") v))       ; countries -> country
    (when (and (> n 1) (char= (char c (1- n)) #\y))
      (push (concatenate 'string (subseq c 0 (1- n)) "ies") v))      ; country -> countries
    (remove-duplicates v :test #'string=)))

(defun key-parts (key)
  "Split a concept state KEY \"predicate:answer\" into (values predicate answer)."
  (let ((colon (position #\: key :from-end t)))
    (if colon (values (subseq key 0 colon) (subseq key (1+ colon))) (values key ""))))

(defun last-word (s)
  (let ((p (position #\Space s :from-end t))) (if p (subseq s (1+ p)) s)))

(defun state-members (state)
  "Distinct subject-word names connected to STATE in the concept graph."
  (let ((members '()) (tab (gethash state *concept-graph*)))
    (when tab
      (maphash (lambda (n w) (declare (ignore w))
		 (when (typep n 'named-neuron)
		   (pushnew (named-neuron-name n) members :test #'string=)))
	       tab))
    members))

(defun category-members (c)
  "Members of category C, found GENERICALLY -- not by a fixed pattern.  Among every
   positive membership state whose predicate NAMES C (any phrasing: \"are _ animals\",
   \"is _ a color\", \"is a _ a vehicle\", ...), take the one the MOST subjects share --
   that is the real category frame, because a shared frame accumulates members while
   idiosyncratic decompositions (function-word subjects) do not -- and return its members.
   Works for any category the system has been taught, in whatever words."
  (let ((variants (category-variants c)) (best nil) (best-n -1))
    (maphash (lambda (key state)
	       (multiple-value-bind (pred ans) (key-parts key)
		 (when (and (string= ans "yes")
			    (member (last-word pred) variants :test #'string=))
		   (let ((n (length (state-members state))))
		     (when (> n best-n) (setf best-n n best state))))))
	     *concepts*)
    (and best (state-members best))))

(defun stem (w)
  "Naive number-normalisation (strip a trailing s) so 'tiger' matches framed 'tigers'."
  (if (and (> (length w) 3) (char= (char w (1- (length w))) #\s)) (subseq w 0 (1- (length w))) w))

(defun recognized-member-p (word category)
  "Is WORD a member of CATEGORY -- framed (crisp), or, if it was never explicitly told,
   recognized **non-brittlely** by resemblance: a majority of its nearest concepts (in the
   distributed vector space) are framed members of the category.  k-NN, so it degrades
   gracefully and needs no global threshold the way counting would.  (Comparison is
   number-normalised so singular/plural forms match.)"
  (let ((members (mapcar #'stem (category-members category)))
	(ws (stem word)))
    (and members
	 (or (member ws members :test #'string=)
	     (let* ((nbrs (mapcar (lambda (p) (stem (car p))) (nearest word 8)))
		    (hits (count-if (lambda (n) (member n members :test #'string=)) nbrs)))
	       (>= hits 4))))))

(defun execute-op (op c)
  "Run primitive OP over argument C, returning a value (number, list, or NIL)."
  (cond ((string= op "count")   (length (category-members c)))
	((string= op "similar") (mapcar #'car (nearest c 5)))   ; geometry: nearest concepts
	(t nil)))

;;; --- learned mapping: question frame -> operation (the slot is the removed word) -------
(defun frame-without-word (words w)
  (format nil "~{~a~^ ~}" (remove w words :test #'string= :count 1)))

(defun note-operation (input-words answer-words)
  "Teach that INPUT-WORDS denotes an operation.  ANSWER-WORDS is (op ... category); record
   `input minus the category word' -> op, so it generalizes to any category via the slot."
  (when (and (operation-answer-p answer-words) (cdr answer-words))
    (let ((op  (first answer-words))
	  (cat (car (last answer-words))))      ; \"count animals\" or \"count the animals\"
      (when (member cat input-words :test #'string=)
	(incf (gethash (cons (frame-without-word input-words cat) op) *op-templates* 0))))))

(defun run-operation (input-words)
  "If INPUT-WORDS matches a learned operation frame, execute that operation with the slot
   word as its category argument and return the result as a one-word list; else NIL."
  (let ((best-op nil) (best-cat nil) (best-s 0))
    (dolist (w input-words)
      (let ((frame (frame-without-word input-words w)))
	(maphash (lambda (key strength)
		   (when (and (string= (car key) frame) (> strength best-s))
		     (setf best-s strength best-op (cdr key) best-cat w)))
		 *op-templates*)))
    (when best-op
      (let ((result (execute-op best-op best-cat)))
	(cond ((null result) nil)
	      ((listp result) result)                      ; e.g. "similar" -> a list of words
	      (t (list (princ-to-string result))))))))     ; e.g. "count" -> a number
