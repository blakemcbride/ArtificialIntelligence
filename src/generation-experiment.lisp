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

(defun read-corpus (path)
  (with-open-file (s path)
    (let ((str (make-string (file-length s))))
      (read-sequence str s)
      (setf *corpus-text* (string-downcase str))
      (dolist (sent (split-sentences str))
	(learn-sentence (tokenize sent))))))

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

;;; ----------------------------------------------------------------------- the demo ---
(defun cap (s)
  (if (plusp (length s)) (concatenate 'string (string-upcase (subseq s 0 1)) (subseq s 1)) s))

(defun novel-p (sent)
  "True if SENT (as text) does NOT appear verbatim in the corpus -- i.e. it was generated."
  (not (search (format nil "~{~a~^ ~}" sent) *corpus-text*)))

(defun describe-topic (topic)
  (format t "~%==== tell me about ~a ====~%" (string-upcase topic))
  (let ((assoc (top-associates topic 8)))
    (cond
      ((null assoc)
       (format t "  (\"~a\" is not in this corpus -- nothing to say about it)~%" topic))
      (t
       (format t "  content selection (top associates): ~{~a~^, ~}~%" assoc)
       (setf *seed* 1)                                ; reset PRNG per topic -> reproducible
       (let ((sents (generate-about topic 3)))
	 (if sents
	     (dolist (s sents)
	       (format t "  -> ~a.  ~a~%" (cap (format nil "~{~a~^ ~}" s))
		       (if (novel-p s) "(novel)" "(seen verbatim)")))
	     (format t "  (too little data after \"~a\" to continue a sentence)~%" topic)))))))

(read-corpus "prose.txt")
(format t "learned ~:d sentences, vocabulary ~:d words, ~:d transition heads~%"
	*n* (hash-table-count *df*) (hash-table-count *trans*))
(dolist (tp '("france" "egypt" "einstein" "rome" "river" "florida"))
  (describe-topic tp))
(format t "~%(florida shows the data-coverage limit: with nothing read about it, there is~%")
(format t " nothing to generate -- exactly what more reading, or a broader corpus, fixes.)~%")
