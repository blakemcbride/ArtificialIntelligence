
(defpackage "generation"
  (:use "COMMON-LISP")
  (:export "NOTE-SEQUENCE"
	   "NOTE-FACTS"
	   "NOTE-FACT"
	   "NOTE-FACT-FROM-QA"
	   "DESCRIBE-WORDS"
	   "WHY"
	   "GENERATION-REQUEST-P"
	   "RESPOND-GENERATION"
	   "GENERATE-CHAIN"))

(in-package "generation")
(provide "generation")

(require "data-structures")
(use-package "data-structures")
(require "line-input")
(use-package "line-input")   ; tokenize, as-words

;;; The Output component, for real (Plan.md Phase 8): GENERATION.  The rest of the system
;;; RETRIEVES (respond walks one stored chain) or INFERS membership (the concept graph); it
;;; cannot assemble a NEW sentence from what it knows.  Generation is two pieces, both pure
;;; online counting (Hebbian, no backprop, continual):
;;;
;;;   * CONTENT SELECTION -- what to say: declarative (subject relation object) facts in
;;;     *facts*, captured at the same parse points the rest of the system already uses
;;;     (extract-fact's prose patterns via read-text, and a few question frames via learn).
;;;   * SURFACE REALIZATION -- how to say it: render each fact through a per-relation frame
;;;     (the slot-fill behind `compose') and aggregate several into one short paragraph
;;;     (`describe-words'); explanations (`why') cite an is-a chain.  A learned next-word
;;;     model (*transitions* / *sentence-starts*) backs the lower-fidelity `generate-chain'.
;;;
;;; Grounding the object in a real fact is what keeps this trustworthy -- it states only what
;;; was actually taught/read, never a free-associated falsehood.  See generation-experiment.lisp
;;; for the proof of concept and the head-to-head against a bare bigram chain.

;;; ------------------------------------------------------------------- small helpers ---
(defun jw (ws) (format nil "~{~a~^ ~}" ws))
(defun firstn (lst n) (subseq lst 0 (min n (length lst))))

(defun cap (s)
  (if (plusp (length s)) (concatenate 'string (string-upcase (subseq s 0 1)) (subseq s 1)) s))

(defun art (word)
  "\"a\" or \"an\" to agree with WORD's first sound (vowel-letter heuristic)."
  (if (and (plusp (length word)) (find (char word 0) "aeiou")) "an" "a"))

(defun human-list (items)
  (cond ((null items) "")
	((null (cdr items)) (car items))
	((null (cddr items)) (format nil "~a and ~a" (first items) (second items)))
	(t (format nil "~{~a~^, ~}, and ~a" (butlast items) (car (last items))))))

(defun split-spaces (s)
  "Split S on single spaces into a list of word-strings (keeps case and punctuation)."
  (let (res (start 0) (n (length s)))
    (dotimes (i n)
      (when (char= (char s i) #\Space)
	(when (> i start) (push (subseq s start i) res))
	(setf start (1+ i))))
    (when (> n start) (push (subseq s start n) res))
    (nreverse res)))

(defun find-sub (sub seq)
  "Index where token list SUB occurs in token list SEQ, else NIL."
  (let ((ls (length sub)) (ln (length seq)))
    (loop for i from 0 to (- ln ls)
	  when (loop for j from 0 below ls always (string= (nth j sub) (nth (+ i j) seq)))
	    do (return i))))

(defun split-on (sep words)
  "If token list SEP occurs in WORDS, return (values LEFT RIGHT) around its first occurrence."
  (let ((i (find-sub sep words)))
    (when i (values (subseq words 0 i) (subseq words (+ i (length sep)))))))

(defun starts-with (words prefix)
  (and (>= (length words) (length prefix))
       (every #'string= prefix (subseq words 0 (length prefix)))))

(defun strip-article (s)
  "Drop a leading a/an/the from string S."
  (let ((ws (tokenize s)))
    (if (member (first ws) '("a" "an" "the") :test #'string=) (jw (rest ws)) s)))

(defparameter *gen-question-words*
  '("what" "who" "where" "when" "why" "how" "is" "are" "do" "does" "can" "will" "did"))

;;; ----------------------------------- the sequential model (surface realization) ---
(defun note-sequence (words)
  "Hebbian: grow the next-word transition model and sentence-start counts from WORDS."
  (when words
    (incf (gethash (first words) *sentence-starts* 0.0) 1.0)
    (loop for (a b) on words do
      (let ((tab (or (gethash a *transitions*)
		     (setf (gethash a *transitions*) (make-hash-table :test 'equal)))))
	(incf (gethash (or b :end) tab 0.0) 1.0)))))

(defun assoc-score (word topic)
  "Co-occurrence count of WORD with TOPIC (topic bias for the chain), 0 if none."
  (let ((tab (gethash topic *cooccur*)))
    (if tab (gethash word tab 0.0) 0.0)))

(defun best-next (current topic used)
  "Greedy next word after CURRENT: argmax count*(1+assoc), skipping used words; :end allowed."
  (let ((tab (gethash current *transitions*)) (best nil) (bw -1.0))
    (when tab
      (maphash (lambda (nw c)
		 (let ((w (if (eq nw :end) c
			      (* c (+ 1.0 (* 4.0 (assoc-score nw topic)))))))
		   (when (and (stringp nw) (gethash nw used)) (setf w (* w 0.02)))
		   (when (> w bw) (setf bw w best nw))))
	       tab))
    best))

(defun generate-chain (topic &optional (maxlen 14))
  "Lower-fidelity realizer: greedily walk the transition chain from TOPIC, biased by
   co-occurrence.  Returns a word list, or NIL if TOPIC has no transitions.  NOT wired into
   respond -- the grounded path is preferred; this is the experimental fallback."
  (let ((tab (gethash topic *transitions*)))
    (when tab
      (let ((used (make-hash-table :test 'equal)) (out (list topic)) (cur topic))
	(setf (gethash topic used) t)
	(loop for i from 1 below maxlen do
	  (let ((nx (best-next cur topic used)))
	    (when (or (null nx) (eq nx :end)) (return))
	    (push nx out) (setf (gethash nx used) t) (setf cur nx)))
	(nreverse out)))))

;;; ---------------------------------------- the fact store (content selection) ---
(defun strip-indef (s)
  "Drop a leading indefinite article (a/an) from string S, so \"a cat\" keys as \"cat\".
   (\"the\" is kept -- \"the louvre\" reads better and isn't a generalization subject.)"
  (let ((ws (tokenize s)))
    (if (member (first ws) '("a" "an") :test #'string=) (jw (rest ws)) s)))

(defun note-fact (subj rel obj)
  "Hebbian: record/strengthen the triple (SUBJ REL OBJ) in *facts*.  REL is a string."
  (let ((subj (strip-indef subj)) (obj (strip-indef obj)))
    (when (and (plusp (length subj)) (plusp (length obj)) (not (string= subj obj)))
      (incf (gethash (list subj rel obj) *facts* 0.0) 1.0))))

(defun note-facts (words)
  "Extract declarative triples from one sentence's WORDS (skips questions).  Mirrors the
   shapes extract-fact recognizes, but keeps the DECLARATIVE form generation renders from."
  (when (and words (not (member (first words) *gen-question-words* :test #'string=)))
    (let (l r)
      (cond
	;; X is/are the R of Z  ->  (Z R X)   e.g. "Paris is the capital of France"
	((or (multiple-value-setq (l r) (split-on '("is" "the") words))
	     (multiple-value-setq (l r) (split-on '("are" "the") words)))
	 (let ((oi (find-sub '("of") r)))
	   (if oi
	       (note-fact (jw (subseq r (1+ oi))) (jw (subseq r 0 oi)) (jw l))
	       (note-fact (jw l) "is-a" (jw r)))))
	;; X was a/an ROLE  ->  (X was ROLE)
	((or (multiple-value-setq (l r) (split-on '("was" "a") words))
	     (multiple-value-setq (l r) (split-on '("was" "an") words)))
	 (note-fact (jw l) "was" (jw r)))
	;; X is a/an Y [in Z]  ->  (X is-a Y) [+ (X in Z), or (X city-of Z) when Y is "city"]
	((or (multiple-value-setq (l r) (split-on '("is" "a") words))
	     (multiple-value-setq (l r) (split-on '("is" "an") words)))
	 (let ((ii (find-sub '("in") r)))
	   (if ii
	       (let ((cls (subseq r 0 ii)) (reg (subseq r (1+ ii))))
		 (if (string= (jw cls) "city")
		     (note-fact (jw l) "city-of" (jw reg))
		     (progn (when cls (note-fact (jw l) "is-a" (jw cls)))
			    (note-fact (jw l) "in" (jw reg)))))
	       (note-fact (jw l) "is-a" (jw r)))))
	;; X is/are in Z  ->  (X in Z)
	((or (multiple-value-setq (l r) (split-on '("is" "in") words))
	     (multiple-value-setq (l r) (split-on '("are" "in") words)))
	 (note-fact (jw l) "in" (jw r)))))))

(defun note-fact-from-qa (input-words answer-words)
  "Derive a clean triple from a learned question->answer pair (so describe works on the
   starter KB), for the high-value question frames."
  (let ((ans (jw answer-words)) (inw input-words))
    (when (and inw answer-words)
      (cond
	;; "what is the R of Z" => A   ->  (Z R A)
	((starts-with inw '("what" "is" "the"))
	 (let* ((rest (subseq inw 3)) (oi (find-sub '("of") rest)))
	   (when oi (note-fact (jw (subseq rest (1+ oi))) (jw (subseq rest 0 oi)) ans))))
	;; "what <kind> is X in" => A   ->  (X in A)   e.g. "what continent is france in"
	((and (string= (first inw) "what") (string= (car (last inw)) "in"))
	 (let ((ci (position "is" inw :test #'string=)))
	   (when (and ci (< (1+ ci) (1- (length inw))))
	     (note-fact (jw (subseq inw (1+ ci) (1- (length inw)))) "in" ans))))
	;; "who was S" => a/an ROLE   ->  (S was ROLE)
	((starts-with inw '("who" "was"))
	 (note-fact (jw (subseq inw 2)) "was" (strip-article ans)))
	;; "is S a/an C" => yes   ->  (S is-a C)
	((and (string= (first inw) "is") (string= ans "yes"))
	 (let* ((r (rest inw)) (ai (or (position "a" r :test #'string=)
				       (position "an" r :test #'string=))))
	   (when (and ai (> ai 0))
	     (note-fact (jw (subseq r 0 ai)) "is-a" (jw (subseq r (1+ ai)))))))))))

;;; ----------------------------------------------------- describe (tell me about X) ---
(defun topic-token-p (topic str) (member topic (tokenize str) :test #'string=))

(defun describe-paragraph (topic)
  "A short factual paragraph (string) assembled from *facts* triples that mention TOPIC,
   or NIL if none.  Several facts are aggregated -- discourse the retrieval path can't make."
  (let (isa was regions attrs valueof cities landmarks)
    (maphash
     (lambda (k v) (declare (ignore v))
       (destructuring-bind (s rel o) k
	 (cond
	   ((topic-token-p topic s)                       ; TOPIC is the subject
	    (cond ((string= rel "is-a")    (unless isa (setf isa o)))
		  ((string= rel "was")     (unless was (setf was o)))
		  ((string= rel "in")      (pushnew o regions :test #'string=))
		  ((string= rel "city-of") nil)
		  (t                       (push (cons rel o) attrs))))
	   ((topic-token-p topic o)                       ; TOPIC is the object
	    (cond ((string= rel "city-of") (pushnew s cities :test #'string=))
		  ((string= rel "in")      (pushnew s landmarks :test #'string=))
		  ((member rel '("is-a" "was") :test #'string=) nil)
		  (t                       (push (cons rel s) valueof)))))))
     *facts*)
    (setf attrs     (remove-duplicates (nreverse attrs)   :key #'car :test #'string= :from-end t)
	  valueof   (remove-duplicates (nreverse valueof) :key #'car :test #'string= :from-end t)
	  ;; drop landmarks already named as the capital/region (avoid "Paris is the capital"
	  ;; and "Paris ... is in france" both)
	  landmarks (set-difference landmarks (append (mapcar #'cdr attrs) regions cities)
				    :test #'string=))
    (let (sents)
      (flet ((say (fmt &rest a) (push (apply #'format nil fmt a) sents)))
	(cond (isa (say "~a is ~a ~a." (cap topic) (art isa) isa))
	      (was (say "~a was ~a ~a." (cap topic) (art was) was)))
	(when regions (say "It is in ~a." (first regions)))
	(dolist (a (firstn attrs 2))   (say "Its ~a is ~a." (car a) (cdr a)))
	(dolist (a (firstn valueof 1)) (say "~a is the ~a of ~a." (cap topic) (car a) (cdr a)))
	(when cities    (say "Cities in ~a include ~a." topic (human-list (firstn cities 3))))
	(when landmarks (say "~a ~a in ~a." (cap (human-list (firstn landmarks 3)))
			     (if (cdr landmarks) "are" "is") topic)))
      (setf sents (nreverse sents))
      (and sents (format nil "~{~a~^ ~}" sents)))))

(defun describe-words (topic)
  "TOPIC's description as a word list (so it flows through respond), or NIL if unknown."
  (let ((para (describe-paragraph topic)))
    (and para (split-spaces para))))

;;; ----------------------------------------------------------------- why (explain) ---
(defun isa-neighbors (x)
  (let (acc)
    (maphash (lambda (k v) (declare (ignore v))
	       (destructuring-bind (s rel o) k
		 (when (and (string= rel "is-a") (string= s x)) (push o acc))))
	     *facts*)
    acc))

(defun extend-path (path seen maxdepth)
  "Enqueue-ready successor paths of PATH (one per unseen is-a neighbour), or NIL if too deep."
  (when (< (length path) maxdepth)
    (let (out)
      (dolist (nb (isa-neighbors (car (last path))) (nreverse out))
	(unless (gethash nb seen)
	  (setf (gethash nb seen) t)
	  (push (append path (list nb)) out))))))

(defun isa-path (x y maxdepth)
  "Shortest is-a path (list of nodes) from X to Y in *facts*, or NIL."
  (let ((q (list (list x))) (seen (make-hash-table :test 'equal)) (result nil))
    (setf (gethash x seen) t)
    (loop while (and q (not result)) do
      (let ((path (pop q)))
	(if (string= (car (last path)) y)
	    (setf result path)
	    (setf q (append q (extend-path path seen maxdepth))))))
    result))

(defun explain-path (x y maxdepth)
  "An is-a path X -> ... -> Y with at least one intermediate (so it explains), or NIL."
  (dolist (m (isa-neighbors x))
    (unless (string= m y)
      (let ((sub (isa-path m y (1- maxdepth))))
	(when sub (return-from explain-path (cons x sub))))))
  nil)

(defun render-why (path)
  (let (clauses)
    (loop for (a b) on path while b
	  do (push (format nil "~a ~a is ~a ~a" (art a) a (art b) b) clauses))
    (format nil "~a ~a is ~a ~a because ~a."
	    (cap (art (first path))) (first path)
	    (art (car (last path))) (car (last path))
	    (human-list (nreverse clauses)))))

(defun why (words)
  "Explain \"why is X a Y\" via an is-a chain in *facts* (X -> ... -> Y).  Returns a word
   list, or NIL when no explanatory chain is known."
  (let ((content (remove-if (lambda (w)
			      (member w '("why" "is" "are" "was" "were" "a" "an" "the")
				      :test #'string=))
			    words)))
    (when (>= (length content) 2)
      (let ((path (explain-path (first content) (car (last content)) 4)))
	(and path (split-spaces (render-why path)))))))

;;; ------------------------------------------------------- request detection / dispatch ---
(defun topic-of (ws) (and ws (car (last ws))))   ; the trailing word is the topic key

(defun generation-request-p (input)
  "If INPUT (string or word list) is a generation request, return (values KIND TOPIC):
   KIND is :describe (TOPIC = the topic word) or :why (TOPIC = the full word list)."
  (let ((words (as-words input)))
    (cond
      ((null words) nil)
      ((string= (first words) "why") (values :why words))
      ((starts-with words '("tell" "me" "about"))
       (values :describe (topic-of (subseq words 3))))
      ((starts-with words '("what" "can" "you" "tell" "me" "about"))
       (values :describe (topic-of (subseq words 6))))
      ((starts-with words '("what" "do" "you" "know" "about"))
       (values :describe (topic-of (subseq words 5))))
      ((and (string= (first words) "describe") (cdr words))
       (values :describe (topic-of (rest words))))
      (t nil))))

(defun respond-generation (input)
  "Answer a generation request (describe / why), or NIL if INPUT isn't one or nothing is
   known.  Called first by processing:respond."
  (multiple-value-bind (kind topic) (generation-request-p input)
    (case kind
      (:describe (and topic (describe-words topic)))
      (:why      (why topic))
      (t nil))))
