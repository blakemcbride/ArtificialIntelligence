;;;; relation-discovery-experiment.lisp -- LEARN "is a" (which words form a relationship)
;;;; AND which words are the subject / category, instead of hardcoding any of it.  Run:
;;;;     sbcl --script relation-discovery-experiment.lisp
;;;;
;;;; The live system's prose reader hardcodes "is a", "is the Y of", ... AND assumes the
;;;; subject/category sit in fixed positions.  This PoC shows BOTH can be LEARNED -- pure
;;;; Hebbian counting, no backprop, continual -- given enough data.
;;;;
;;;; Learned signals, all just counts:
;;;;   1. CONNECTORS  -- a MEMBERSHIP connector ("is a") links MANY subjects to FEW category
;;;;      hubs (high subjects-per-category); a relational connector / verb maps each subject
;;;;      to its own object (~1).  Nothing about "is"/"a" is special to the code.
;;;;   2. FUNCTION WORDS -- tokens that occur in a large fraction of sentences (the, a, ...).
;;;;   3. HEADS -- the real subject/category in a span is the word that most often SERVES as
;;;;      a head elsewhere; modifiers ("small brown"), relative-clause words ("that barks"),
;;;;      and articles score low and fall away.
;;;;
;;;; ITERATIVE BOOTSTRAPPING (this version): heads are seeded from short, clean sentences,
;;;; then the model RE-PARSES the whole corpus and re-counts heads from what it discovered,
;;;; looping a few rounds.  So a head that appears ONLY inside modified / relative-clause
;;;; sentences ("...salmon...") is learned from its clean occurrences and then wins in the
;;;; messy ones -- an unsupervised, local self-training loop, no labels, no backprop.

;;; -------------------------------------------------------------------- helpers ---
(defun tokenize (s)
  (let (words (start nil) (n (length s)))
    (dotimes (i n)
      (if (alphanumericp (char s i))
	  (unless start (setf start i))
	  (when start (push (string-downcase (subseq s start i)) words) (setf start nil))))
    (when start (push (string-downcase (subseq s start n)) words))
    (nreverse words)))

(defun jw (ws) (format nil "~{~a~^ ~}" ws))
(defun strip-lead-article (ws)
  (if (member (first ws) '("a" "an" "the") :test #'string=) (rest ws) ws))

(defun find-sub (sub seq)
  (let ((ls (length sub)) (ln (length seq)))
    (loop for i from 0 to (- ln ls)
	  when (loop for j from 0 below ls always (string= (nth j sub) (nth (+ i j) seq)))
	    do (return i))))

(defun parse-acb (sentence)
  "Naive POSITIONAL parse: subject = first token, category = last, connector = the middle."
  (let ((ws (strip-lead-article (tokenize sentence))))
    (if (and (>= (length ws) 3) (not (string= (first ws) (car (last ws)))))
	(values (first ws) (jw (subseq ws 1 (1- (length ws)))) (car (last ws)) t)
	(values nil nil nil nil))))

;;; ------------------------------------------------ the learned model (counts) ---
(defparameter *links* (make-hash-table :test 'equal))      ; connector -> distinct (subj . cat) pairs
(defparameter *head-count* (make-hash-table :test 'equal)) ; word -> times it served as a head
(defparameter *tok-freq* (make-hash-table :test 'equal))   ; word -> # sentences it appears in
(defparameter *sentences* 0)

(defparameter *min-support* 3)
(defparameter *hub-threshold* 1.8)
(defparameter *fw-threshold* 0.12)

(defun hc (w) (gethash w *head-count* 0))

(defun count-freq (sentence)
  "Token frequency over ALL text -- the basis for function-word discovery."
  (incf *sentences*)
  (dolist (w (remove-duplicates (tokenize sentence) :test #'string=))
    (incf (gethash w *tok-freq* 0))))

(defun link! (c a b)
  (let ((tab (or (gethash c *links*) (setf (gethash c *links*) (make-hash-table :test 'equal)))))
    (setf (gethash (format nil "~a|~a" a b) tab) (cons a b))))

(defun seed-positional (sentence)
  "Seed pass on a SHORT/clean sentence: positional subject/category + connector + heads."
  (multiple-value-bind (a c b ok) (parse-acb sentence)
    (when ok
      (incf (gethash a *head-count* 0)) (incf (gethash b *head-count* 0))
      (link! c a b))))

(defun conn-stats (c)
  (let ((tab (gethash c *links*)) (cats (make-hash-table :test 'equal)) (pairs 0))
    (when tab
      (maphash (lambda (k v) (declare (ignore k)) (incf pairs) (setf (gethash (cdr v) cats) t)) tab))
    (let ((dc (hash-table-count cats)))
      (values pairs dc (if (plusp dc) (/ pairs dc) 0)))))

(defun classify (c)
  (multiple-value-bind (pairs dc hub) (conn-stats c)
    (declare (ignore dc))
    (cond ((< pairs *min-support*) :unknown)
	  ((>= hub *hub-threshold*) :membership)
	  (t :relational))))

;;; ----------------------------------- the LEARNED parser (connector + heads) ---
(defun function-word-p (w)
  "Frequent AND rarely a head -- so frequent CONTENT words (e.g. \"animal\", a common
   category) are not mistaken for glue, while \"the\"/\"a\"/\"is\" (frequent, never heads) are."
  (let ((f (gethash w *tok-freq* 0)))
    (and (plusp *sentences*)
	 (>= (/ f *sentences*) *fw-threshold*)
	 (< (hc w) (* 0.3 f)))))

(defun known-connectors ()
  (let (cs)
    (maphash (lambda (c tab) (declare (ignore tab))
	       (when (>= (conn-stats c) *min-support*) (push c cs)))
	     *links*)
    (sort cs #'> :key (lambda (c) (length (tokenize c))))))

(defun find-connector (tokens)
  (dolist (c (known-connectors))
    (let* ((ct (tokenize c)) (i (find-sub ct tokens)))
      (when i (return-from find-connector
		(values (subseq tokens 0 i) c (subseq tokens (+ i (length ct))))))))
  nil)

(defun head-of (span)
  "Drop function words, then the word that most often serves as a head wins; ties / all-novel
   -> the rightmost content word (the word adjacent to the connector -- the usual head slot)."
  (let ((content (remove-if #'function-word-p span)))
    (when (null content) (setf content span))
    (let ((best (first content)) (bc (hc (first content))))
      (dolist (w (rest content) best)
	(when (>= (hc w) bc) (setf best w bc (hc w)))))))

(defun analyze (sentence)
  (let ((tokens (strip-lead-article (tokenize sentence))))
    (multiple-value-bind (left c right) (find-connector tokens)
      (if (and c left right)
	  (values (head-of left) c (head-of right) (classify c))
	  (values nil nil nil :unknown)))))

(defun bootstrap-round ()
  "Re-parse the whole corpus with the current model and rebuild head counts from the
   DISCOVERED heads (links accumulate).  Returns the number of sentences parsed."
  (let ((new-hc (make-hash-table :test 'equal)) (parsed 0))
    (dolist (s (all-sentences))
      (multiple-value-bind (a c b) (analyze s)
	(when (and a c b)
	  (incf parsed)
	  (incf (gethash a new-hc 0)) (incf (gethash b new-hc 0))
	  (link! c a b))))
    (setf *head-count* new-hc)
    parsed))

;;; --------------------------------------------------------------------- demos ---
(defun desc (sentence learnedp)
  (if learnedp
      (multiple-value-bind (a c b cls) (analyze sentence)
	(if c (format nil "subject=~s connector=~s category=~s [~a]" a c b cls)
	    "(no known connector found)"))
      (multiple-value-bind (a c b ok) (parse-acb sentence)
	(if ok (format nil "subject=~s connector=~s category=~s" a c b) "(unparsable)"))))

(defun show-connectors ()
  (let (rows)
    (maphash (lambda (c tab) (declare (ignore tab))
	       (multiple-value-bind (p d h) (conn-stats c)
		 (push (list c p d (coerce h 'float) (classify c)) rows)))
	     *links*)
    (setf rows (sort rows (lambda (x y) (> (fourth x) (fourth y)))))
    (format t "  ~22a ~5@a ~5@a ~9@a  ~a~%" "connector" "links" "cats" "subj/cat" "class")
    (dolist (r rows)
      (format t "  ~22a ~5d ~5d ~9,1f  ~a~%" (first r) (second r) (third r) (fourth r)
	      (case (fifth r) (:membership "MEMBERSHIP") (:relational "relational") (t "unknown"))))))

;;; ----- the corpus: SIMPLE sentences seed the model; COMPLEX ones need bootstrapping -----
(defparameter *simple*
  '("robin is a bird" "eagle is a bird" "owl is a bird" "bird is an animal"
    "rose is a flower" "tulip is a flower" "daisy is a flower"
    "paris is a city" "tokyo is a city" "cairo is a city"
    "water is a liquid" "milk is a liquid"
    "iron is a metal" "gold is a metal" "copper is a metal"
    "george is a person" "mary is a person"
    "dog is an animal" "cat is an animal" "elephant is an animal"
    "ant is an insect" "bee is an insect"
    "car is a vehicle" "truck is a vehicle" "bike is a vehicle"
    "dogs are animals" "cats are animals" "horses are animals"
    "robins are birds" "eagles are birds" "roses are flowers" "tulips are flowers"
    "paris is the capital of france" "tokyo is the capital of japan"
    "cairo is the capital of egypt" "rome is the capital of italy"
    "george likes pizza" "mary likes tea" "john likes music"))

;; "salmon"/"fish" appear ONLY here (never in a short seed sentence) -- so they can only be
;; learned by bootstrapping.  salmon sits next to the connector in 3 of them (clean) and
;; buried behind a relative clause in the 4th (the hard case).
(defparameter *complex*
  '("a silver salmon is a fish"
    "a fresh wild salmon is a fish"
    "the tasty grilled salmon is a fish"
    "the salmon that swims upstream is a fish"          ; <- the hard one
    "a small brown dog is a friendly animal"
    "the dog that barks loudly is an animal"
    "the large gray elephant is a huge animal"
    "a fast red car is a vehicle"
    "the bird that sings sweetly is an animal"))

(defun all-sentences () (append *simple* *complex*))

;; token frequencies over everything; seed the model from the simple sentences only
(dolist (s (all-sentences)) (count-freq s))
(dolist (s *simple*) (seed-positional s))

(defparameter *hard* "the salmon that swims upstream is a fish")

(format t "~%=== iterative bootstrapping: learning a head seen ONLY in complex sentences ===~%")
(format t "  target sentence: ~s~%" *hard*)
(format t "  head-count(salmon) after seed-only: ~d~%" (hc "salmon"))
(format t "  parse after seed-only:  ~a~%" (desc *hard* t))
(dotimes (r 3)
  (let ((n (bootstrap-round)))
    (format t "  -- bootstrap round ~d: parsed ~d sentences; head-count(salmon)=~d, (upstream)=~d~%"
	    (1+ r) n (hc "salmon") (hc "upstream"))))
(format t "  parse after bootstrapping: ~a~%" (desc *hard* t))
(format t "  (salmon was never in a simple sentence; it was learned from its clean complex~%")
(format t "   occurrences, then beat \"upstream\" in the relative-clause sentence.)~%")

(format t "~%=== connectors discovered (post-bootstrap) ===~%")
(show-connectors)

(format t "~%=== function words discovered ===~%")
(let (fw) (maphash (lambda (w n) (declare (ignore n)) (when (function-word-p w) (push w fw))) *tok-freq*)
     (format t "  ~{~a~^, ~}~%" (sort fw #'string<)))

(format t "~%=== subject/category spans with modifiers & relative clauses ===~%")
(format t "    (naive = fixed-position parse; learned = connector + bootstrapped heads)~%")
(dolist (s '("a small brown dog is a friendly animal"
	     "the dog that barks loudly is an animal"
	     "the large gray elephant is a huge animal"
	     "the bird that sings sweetly is an animal"))
  (format t "~%  ~s~%     naive  : ~a~%     learned: ~a~%" s (desc s nil) (desc s t)))

(format t "~%=== tag brand-new sentences (novel subjects/categories) ===~%")
(dolist (s '("a clever little trout is a fish"
	     "sparrows are birds"
	     "george likes chess"
	     "wombat resembles a quokka"))
  (multiple-value-bind (a c b cls) (analyze s)
    (format t "  ~42s -> ~a~%" s
	    (case cls
	      (:membership (format nil "MEMBERSHIP: ~a is a kind of ~a (via ~s)" a b c))
	      (:relational (format nil "relational via ~s: ~a -> ~a" c a b))
	      (t "unknown connector -- not enough data yet")))))

(format t "~%=== continual learning: a brand-new phrasing \"is a kind of\" ===~%")
(format t "  before:  ~a~%" (classify "is a kind of"))
(dolist (s '("sedan is a kind of car" "coupe is a kind of car"
	     "rose is a kind of flower" "tulip is a kind of flower"
	     "salmon is a kind of fish" "trout is a kind of fish"))
  (count-freq s) (seed-positional s))
(format t "  after 6 examples:  ~a~%" (classify "is a kind of"))
(format t "  parse \"a shiny new minivan is a kind of car\": ~a~%"
	(desc "a shiny new minivan is a kind of car" t))

(format t "~%Takeaway: connectors, function words, and span heads are all LEARNED by counting~%")
(format t "-- no grammar, no hardcoded \"is a\", no fixed positions.  Iterative bootstrapping~%")
(format t "lets the model teach itself heads it only ever sees inside modified or~%")
(format t "relative-clause sentences, the same way more reading would, continually.~%")
