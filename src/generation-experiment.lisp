;;;; generation-experiment.lisp -- a topic-conditioned Hebbian text GENERATOR (proof of concept).
;;;; Run:  sbcl --script generation-experiment.lisp        (from src/, so prose.txt resolves)
;;;;
;;;; The point of this PoC: the main system can RETRIEVE answers (respond walks a stored
;;;; output chain) and INFER membership (the concept graph), but it cannot GENERATE -- it
;;;; can't say "tell me about France" by assembling a NEW sentence from what it knows.
;;;; This file shows that generation is buildable within the project's constraints
;;;; (Hebbian, no backprop, continual) and is worth promoting to a real Phase 8.
;;;;
;;;; Generation = two pieces, both just online counting (Hebbian accumulation, no gradients):
;;;;
;;;;   1. CONTENT SELECTION  -- what to say.  Which words are associated with the topic?
;;;;      A co-occurrence table (word -> co-word -> count), IDF-weighted so "the"/"is" don't
;;;;      win.  This is exactly what the real system already keeps in *cooccur* (and the
;;;;      concept graph) -- it is already collected, just never used to generate.
;;;;
;;;;   2. SURFACE REALIZATION -- how to say it.  A learned next-word transition model
;;;;      (word -> next-word -> count: a bigram chain with an :end marker), harvested from
;;;;      the same prose read-text already ingests.  Generation samples this chain forward,
;;;;      BIASED toward the topic's associated words -- grammaticality from the transitions,
;;;;      on-topic-ness from the bias.  This sequential table is the one piece the live
;;;;      system is missing (its *cooccur* is order-less; here *trans* adds the order).
;;;;
;;;; Honest caveats (the reason this is a PoC, not a finished module):
;;;;   * A bigram chain has a low coherence ceiling -- short-range fluency, no long-range
;;;;     structure or guaranteed factual consistency.  Higher-order context (the input
;;;;     network's order-preserving frontier) and learned multi-slot frames (generalizing
;;;;     `compose') would push quality up; that is the real Phase 8 work.
;;;;   * Quality scales with corpus size.  prose.txt is ~700 sentences, so transitions are
;;;;     sparse and output is rough -- but it is GENUINELY GENERATED (novel word sequences),
;;;;     not retrieved (see the (novel)/(seen) tag on each line).

;;; ----------------------------------------------------------------- deterministic PRNG ---
;;; A fixed LCG so a run is reproducible (same style as the other experiment files).
(defparameter *seed* 1)
(defun nextr () (setf *seed* (logand (+ (* *seed* 1103515245) 12345) #x7FFFFFFF)))
(defun rand-unit () (coerce (/ (nextr) #x7FFFFFFF) 'double-float))   ; in [0,1]

;;; ----------------------------------------------------------------------- tokenizing ---
(defun tokenize (s)
  "Lowercase word tokens of S (runs of alphanumerics); everything else is a separator."
  (let (words (start nil) (n (length s)))
    (dotimes (i n)
      (if (alphanumericp (char s i))
	  (unless start (setf start i))
	  (when start (push (string-downcase (subseq s start i)) words) (setf start nil))))
    (when start (push (string-downcase (subseq s start n)) words))
    (nreverse words)))

(defun split-sentences (text)
  "Split TEXT into sentences on . ! ? (terminators dropped)."
  (let (sents (start 0) (n (length text)))
    (dotimes (i n)
      (when (member (char text i) '(#\. #\! #\?))
	(push (subseq text start i) sents)
	(setf start (1+ i))))
    (when (< start n) (push (subseq text start n) sents))
    (nreverse sents)))

;;; ----------------------------------------------------- the learned model (counts) ---
(defparameter *trans* (make-hash-table :test 'equal)) ; word -> (next-word -> count); :end ends a sentence
(defparameter *cooc*  (make-hash-table :test 'equal)) ; word -> (co-word -> count): same-sentence co-occurrence
(defparameter *df*    (make-hash-table :test 'equal)) ; word -> # sentences it appears in (for IDF)
(defparameter *n*     0)                              ; sentences seen
(defparameter *corpus-text* "")                      ; kept only to tag output novel vs. seen

(defun bump (table key) (incf (gethash key table 0)))
(defun bump2 (outer k1 k2)
  (let ((inner (or (gethash k1 outer)
		   (setf (gethash k1 outer) (make-hash-table :test 'equal)))))
    (incf (gethash k2 inner 0))))

(defun learn-sentence (words)
  "One online Hebbian update: grow the transition chain, the co-occurrence table, and DF."
  (when words
    (incf *n*)
    (loop for (a b) on words do (bump2 *trans* a (or b :end)))    ; bigram transitions + :end
    (let ((uniq (remove-duplicates words :test #'string=)))
      (dolist (w uniq) (bump *df* w))                            ; document frequency
      (dolist (a uniq)
	(dolist (b uniq)
	  (unless (string= a b) (bump2 *cooc* a b)))))))           ; unordered co-occurrence

;;; ============ generator #2 substrate: relational facts (subject rel object) ==========
;;; The same idea as the live system's concept graph: pull simple declarative facts out of
;;; the prose as triples, so realization can be GROUNDED in a verified edge instead of a
;;; free word-walk.  This is what kills the bigram's "capital of latvia" class of error.
(defparameter *facts* nil)   ; list of (subject-string relation-keyword object-string)

(defun jw (lst) (format nil "~{~a~^ ~}" lst))

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

(defun extract-facts (words)
  "Pull relational triples out of one sentence's WORDS (most specific patterns first)."
  (let (l r)
    (flet ((emit (s rel o) (when (and s o) (push (list (jw s) rel (jw o)) *facts*))))
      (cond
	((multiple-value-setq (l r) (split-on '("is" "the" "capital" "of") words)) (emit l :capital-of r))
	((multiple-value-setq (l r) (split-on '("is" "the" "language" "of") words)) (emit l :language-of r))
	((multiple-value-setq (l r) (split-on '("is" "a" "city" "in") words)) (emit l :city-of r))
	((multiple-value-setq (l r) (split-on '("is" "a" "city" "of") words)) (emit l :city-of r))
	((multiple-value-setq (l r) (split-on '("is" "a" "country" "in") words))
	 (emit l :is-a '("country")) (emit l :located-in r))
	((multiple-value-setq (l r) (split-on '("was" "a") words)) (emit l :was-a r))
	((multiple-value-setq (l r) (split-on '("was" "an") words)) (emit l :was-a r))
	((multiple-value-setq (l r) (split-on '("is" "a") words)) (emit l :is-a r))
	((multiple-value-setq (l r) (split-on '("is" "an") words)) (emit l :is-a r))
	((multiple-value-setq (l r) (split-on '("is" "in") words)) (emit l :located-in r))))))

(defun read-corpus (path)
  (with-open-file (s path)
    (let ((str (make-string (file-length s))))
      (read-sequence str s)
      (setf *corpus-text* (string-downcase str))
      (dolist (sent (split-sentences str))
	(let ((w (tokenize sent)))
	  (learn-sentence w)        ; bigram chain + co-occurrence (generator #1)
	  (extract-facts w))))))    ; relational triples           (generator #2)

;;; --------------------------------------------------- 1. content selection (topic) ---
(defun idf (w) (log (/ (+ 1.0d0 *n*) (+ 1.0d0 (gethash w *df* 0)))))

(defun topic-scores (topic)
  "Normalized hash co-word -> assoc(topic,co-word) = cooc * idf, scaled to [0,1]."
  (let ((inner (gethash topic *cooc*)) (res (make-hash-table :test 'equal)) (mx 0.0d0))
    (when inner
      (maphash (lambda (w c)
		 (let ((s (* c (idf w)))) (setf (gethash w res) s) (when (> s mx) (setf mx s))))
	       inner))
    (when (> mx 0.0d0) (maphash (lambda (w s) (setf (gethash w res) (/ s mx))) res))
    res))

(defun top-associates (topic k)
  "The K words most associated with TOPIC (the 'what to say'), strongest first."
  (let (pairs)
    (maphash (lambda (w s) (push (cons w s) pairs)) (topic-scores topic))
    (mapcar #'car (subseq (sort pairs #'> :key #'cdr) 0 (min k (length pairs))))))

;;; ------------------------------------------------- 2. surface realization (chain) ---
(defparameter *alpha* 8.0d0 "How hard to bias the next-word choice toward topic words.")

(defun weighted-pick (alist)
  "Sample one car from ALIST of (item . weight) in proportion to weight (deterministic PRNG)."
  (let ((total (reduce #'+ alist :key #'cdr :initial-value 0.0d0)))
    (when (> total 0.0d0)
      (let ((r (* (rand-unit) total)) (acc 0.0d0))
	(dolist (cell alist (car (car (last alist))))
	  (incf acc (cdr cell))
	  (when (>= acc r) (return (car cell))))))))

(defun next-word (current ts used)
  "Pick the word after CURRENT: transition count * (1 + alpha*topic-boost), with already-used
   content words strongly discouraged so it does not loop.  TS is the topic-score table."
  (let ((inner (gethash current *trans*)) cand)
    (when inner
      (maphash (lambda (nw c)
		 (let* ((boost (if (eq nw :end) 0.0d0 (* *alpha* (gethash nw ts 0.0d0))))
			(w     (* c (+ 1.0d0 boost))))
		   (when (and (stringp nw) (gethash nw used)) (setf w (* w 0.03d0)))
		   (push (cons nw w) cand)))
	       inner))
    (weighted-pick cand)))

(defun generate-sentence (topic seed &optional (maxlen 16))
  "Walk the transition chain from SEED, biased toward TOPIC, until :end or MAXLEN."
  (let ((ts (topic-scores topic)) (used (make-hash-table :test 'equal)) (out (list seed))
	(current seed))
    (setf (gethash seed used) t)
    (loop for i from 1 below maxlen do
      (let ((nw (next-word current ts used)))
	(when (or (null nw) (eq nw :end)) (return))
	(push nw out) (setf (gethash nw used) t) (setf current nw)))
    (nreverse out)))

(defun generate-about (topic n)
  "Up to N on-topic sentences: seed the chain with the topic itself, then its top associates."
  (let ((seeds (cons topic (top-associates topic 6))) results)
    (dolist (s seeds)
      (let ((sent (generate-sentence topic s)))
	(when (>= (length sent) 4) (pushnew sent results :test #'equal)))
      (when (>= (length results) n) (return)))
    (nreverse results)))

;;; ============== generator #2: grounded realization (facts + frames) =================
;;; Assemble a description by SELECTING verified triples that mention the topic and
;;; rendering each through a fixed frame (the slot-fill idea behind `compose', one frame per
;;; relation).  Several facts are aggregated into one short paragraph -- discourse the
;;; retrieval system can't produce (it returns a single stored response).  Each object comes
;;; from a real edge, so it can't say "capital of latvia"; the cost is it can only state
;;; what was actually read (no leaps).
(defun cap (s)
  (if (plusp (length s)) (concatenate 'string (string-upcase (subseq s 0 1)) (subseq s 1)) s))

(defun mentions (topic s) (member topic (tokenize s) :test #'string=))

(defun art (word)
  "\"a\" or \"an\" to agree with the first sound of WORD (vowel-letter heuristic)."
  (if (and (plusp (length word)) (find (char word 0) "aeiou")) "an" "a"))

(defun fact-other (topic rel side)
  "OTHER side of every triple with REL where TOPIC is on SIDE (:subj or :obj), de-duped."
  (remove-duplicates
   (loop for (s r o) in *facts*
	 when (and (eq r rel) (mentions topic (if (eq side :subj) s o)))
	   collect (if (eq side :subj) o s))
   :test #'string= :from-end t))

(defun human-list (items)
  (cond ((null items) "")
	((null (cdr items)) (car items))
	((null (cddr items)) (format nil "~a and ~a" (first items) (second items)))
	(t (format nil "~{~a~^, ~}, and ~a"
		   (butlast items) (car (last items))))))

(defun describe-grounded (topic)
  "A short factual paragraph about TOPIC, assembled from triples.  Returns (values list-of-
   sentence-strings facts-used)."
  (let* ((identity (loop for (s r o) in *facts*           ; topic IS-A / WAS-A something
			 when (and (member r '(:is-a :was-a)) (mentions topic s))
			   return (list s r o)))
	 (region    (first (fact-other topic :located-in :subj)))   ; topic is in X
	 (capital   (first (fact-other topic :capital-of :obj)))    ; X is the capital of topic
	 (is-cap-of (first (fact-other topic :capital-of :subj)))   ; topic is the capital of X
	 (language  (first (fact-other topic :language-of :obj)))
	 (cities    (fact-other topic :city-of :obj))
	 (landmarks (set-difference (fact-other topic :located-in :obj) cities :test #'string=))
	 (out nil) (used 0))
    (flet ((say (fmt &rest a) (push (apply #'format nil fmt a) out) (incf used)))
      (when identity
	(destructuring-bind (s r o) identity
	  (say "~a ~a ~a ~a." (cap s) (if (eq r :was-a) "was" "is") (art o) o)))
      (when is-cap-of (say "~a is the capital of ~a." (cap topic) is-cap-of))
      (when region    (say "It is in ~a." region))
      (when capital   (say "Its capital is ~a." capital))
      (when language  (say "~a is its main language." (cap language)))
      (when cities    (say "Cities in ~a include ~a." topic
			   (human-list (subseq cities 0 (min 3 (length cities))))))
      (when landmarks (say "~a ~a in ~a." (cap (human-list (subseq landmarks 0 (min 3 (length landmarks)))))
			   (if (cdr landmarks) "are" "is") topic)))
    (values (nreverse out) used)))

;;; ----------------------------------------------------------- head-to-head demo ---
(defun novel-p (sent)
  "True if SENT (as text) does NOT appear verbatim in the corpus -- i.e. it was generated."
  (not (search (format nil "~{~a~^ ~}" sent) *corpus-text*)))

(defun describe-topic (topic)
  (format t "~%================ tell me about ~a ================~%" (string-upcase topic))
  (let ((assoc (top-associates topic 8)))
    (format t "content selection (top associates): ~a~%"
	    (if assoc (format nil "~{~a~^, ~}" assoc) "(none -- topic not in corpus)"))
    ;; generator #1: bigram chain (novel but ungrounded)
    (format t "~%  [1] bigram chain (fluent-ish, may be FALSE):~%")
    (setf *seed* 1)
    (let ((sents (and assoc (generate-about topic 3))))
      (if sents
	  (dolist (s sents)
	    (format t "      ~a  ~a~%" (cap (format nil "~{~a~^ ~}." s))
		    (if (novel-p s) "(novel)" "(seen verbatim)")))
	  (format t "      (nothing -- too little data)~%")))
    ;; generator #2: grounded facts + frames (true, assembled)
    (format t "~%  [2] grounded facts + frames (TRUE, assembled from the concept store):~%")
    (multiple-value-bind (sents used) (describe-grounded topic)
      (if sents
	  (format t "      ~a  (assembled from ~d fact~:p)~%"
		  (format nil "~{~a~^ ~}" sents) used)
	  (format t "      (no facts extracted about this topic)~%")))))

(read-corpus "prose.txt")
(format t "learned ~:d sentences, vocabulary ~:d words, ~:d transition heads, ~:d facts~%"
	*n* (hash-table-count *df*) (hash-table-count *trans*) (length *facts*))
(dolist (tp '("france" "egypt" "einstein" "rome" "river" "florida"))
  (describe-topic tp))
(format t "~%Takeaway: [1] is novel but unreliable; [2] is reliable but only restates what~%")
(format t "was read.  Phase 8 = [2]'s grounding to choose WHAT is true, [1]'s chain (and~%")
(format t "higher-order context) to vary HOW it is said.  florida: absent from the corpus,~%")
(format t "so both stay silent -- a coverage gap, fixed by reading, not an architecture wall.~%")
