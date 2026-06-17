
; Common Lisp version started on 2/16/15 by Blake McBride

;; Top-level entry point.  The input algorithm now lives in input.lisp; this file
;; only holds `main' and pulls the whole system together.  It has no package of its
;; own, so its symbols (main) land in the current package (normally COMMON-LISP-USER)
;; and the components' exported symbols are imported here via use-package -- call
;; (main), not (ai::main).

(require "data-structures")
;; SBCL's COMMON-LISP-USER inherits SB-PROFILE:RESET, which would make the
;; use-package below signal a name conflict.  Shadow our own RESET first.  We look
;; the symbol up with find-symbol because the package name is a lowercase string
;; ("data-structures") that the reader would otherwise upcase in a pkg:sym token.
(shadowing-import (find-symbol "RESET" "data-structures"))
(use-package "data-structures")

(require "line-input")
(use-package "line-input")

(require "input")
(use-package "input")

(require "output")
(use-package "output")

(require "concepts")
(use-package "concepts")

(require "attention")
(use-package "attention")

(require "vectors")
(use-package "vectors")

(require "operations")
(use-package "operations")

(require "generation")
(use-package "generation")
(require "induction")
(use-package "induction")
(require "relations")
(use-package "relations")
(require "processing")
(use-package "processing")

(require "persist")
(use-package "persist")

(require "llm")
(use-package "llm")
(require "controller")
(use-package "controller")

;; Parallel bulk reading uses SBCL threads + a mailbox; load the contrib when available.
#+(and sbcl sb-thread) (require :sb-concurrency)
(defparameter *parallel-ok* #+(and sbcl sb-thread) t #-(and sbcl sb-thread) nil
  "True when parallel bulk reading (SBCL threads) is available.")

(defparameter *confirm-words* '("yes" "y" "right" "correct" "ok")
  "Single-word teacher lines that confirm a correct guess instead of giving a new answer.")

(defparameter *starter-kb* "knowledge-base.txt"
  "Training file `main' learns on first startup when there is no saved memory yet, so a
   fresh system already knows something.  Bind to NIL to start completely blank.")
(declaim (ftype function train-from-file))   ; defined below; main calls it on first startup
(declaim (ftype function read-text-file))    ; defined below
(declaim (ftype function load-knowledge))    ; defined below; the .read command calls it

;;; --- Bounded-model config (so an arbitrarily large corpus fits in fixed memory) -------
;;; The FILE streams in constant memory (stream-chunks); these caps bound the LEARNED model.
;;; All default NIL = unlimited (original behavior).  View/change them live with .config / .set.
(defparameter *read-max-mb*  nil "Stop a .read after about this many MB ingested (NIL = all).")
(defparameter *read-chunk*   262144 "Characters read per chunk while streaming a file.")
(defparameter *read-flush*   4194304 "Force-process a delimiter-free buffer once it exceeds this.")
(defparameter *read-extract* t
  "When NIL, .read does only the lightweight statistical learners (co-occurrence, transitions,
   relations, facts) and SKIPS the heavy supervised path (extract-fact -> build-structure /
   concept graph / associations).  Turn off for bulk corpus ingestion -- far less memory.")
(defparameter *max-vocab*       nil "Cap on distinct words in *dictionary* (the extract path).")
(defparameter *max-cooccur*     nil "Cap on co-occurrence head words (pruned to top mass).")
(defparameter *max-transitions* nil "Cap on transition head words (pruned to top mass).")
(defparameter *max-facts*       nil "Cap on generation facts (pruned to highest-count).")
(defparameter *prune-every*     5000 "Enforce caps once per this many learned sentences.")
(defvar *learned-since-prune* 0)

;;; Per-learner toggles for the bulk (read-extract off) path.  Each statistical learner can be
;;; switched off independently, so a huge/noisy corpus can run only the learners worth running.
;;; *read-cooccur* gates the O(words^2) co-occurrence learner -- the big one to drop for web text.
(defparameter *read-cooccur*     t "Run the (O(n^2)) co-occurrence/similarity learner while reading.")
(defparameter *read-relations*   t "Run the relation-discovery learner (observe + membership) while reading.")
(defparameter *read-facts-p*     t "Run the declarative-fact learner (note-facts) while reading.")
(defparameter *read-transitions-p* t "Run the next-word transition learner while reading.")

;;; Parallel bulk reading (SBCL only): N worker threads share the load via shard-and-merge.
(defparameter *read-workers* 1 "Worker threads for a bulk .read (SBCL only; 1 = sequential).")

(defparameter *tunables*
  '(("read-max-mb"     . *read-max-mb*)     ("read-extract"    . *read-extract*)
    ("read-cooccur"    . *read-cooccur*)    ("read-relations"  . *read-relations*)
    ("read-facts"      . *read-facts-p*)    ("read-transitions" . *read-transitions-p*)
    ("read-workers"    . *read-workers*)
    ("max-vocab"       . *max-vocab*)       ("max-cooccur"     . *max-cooccur*)
    ("max-transitions" . *max-transitions*) ("max-facts"       . *max-facts*)
    ("prune-every"     . *prune-every*))
  "Live-tunable parameters, exposed by .config / .set (name -> special variable).")

(defun tunables->alist ()
  "Snapshot the .set parameters as an alist of (name . value), for saving in a .kb."
  (mapcar (lambda (tn) (cons (car tn) (symbol-value (cdr tn)))) *tunables*))

(defun restore-tunables (alist)
  "Apply an (name . value) alist read back from a .kb to the live tunable variables."
  (dolist (pair alist)
    (let ((tn (assoc (car pair) *tunables* :test #'string=)))
      (when tn (setf (symbol-value (cdr tn)) (cdr pair))))))

;; Let persist (a separate package) save/restore the tunables that live out here.
(setf *tunable-provider* #'tunables->alist
      *tunable-restorer*  #'restore-tunables)

;;; --- pruning (keep only the strongest entries when a store is over its cap) ------------
(defun prune-flat (table cap)
  "Keep only the CAP highest-valued entries of TABLE (key -> number)."
  (when (and cap (> (hash-table-count table) cap))
    (let (pairs)
      (maphash (lambda (k v) (push (cons k v) pairs)) table)
      (setf pairs (sort pairs #'> :key #'cdr))
      (clrhash table)
      (loop for cell in pairs repeat cap do (setf (gethash (car cell) table) (cdr cell))))))

(defun nested-mass (inner) (let ((s 0)) (maphash (lambda (k v) (declare (ignore k)) (incf s v)) inner) s))

(defun prune-nested (table cap inner-cap)
  "Keep the CAP head words with the largest total mass; prune each inner table to INNER-CAP."
  (when (and cap (> (hash-table-count table) cap))
    (let (pairs)
      (maphash (lambda (k inner) (push (cons k (nested-mass inner)) pairs)) table)
      (setf pairs (sort pairs #'> :key #'cdr))
      (let ((keep (make-hash-table :test 'equal)) dead)
	(loop for cell in pairs repeat cap do (setf (gethash (car cell) keep) t))
	(maphash (lambda (k v) (declare (ignore v)) (unless (gethash k keep) (push k dead))) table)
	(dolist (k dead) (remhash k table)))))
  (when inner-cap (maphash (lambda (k inner) (declare (ignore k)) (prune-flat inner inner-cap)) table)))

(defun enforce-caps ()
  "Prune every capped store down to its cap (called periodically while learning)."
  (prune-nested *cooccur* *max-cooccur* 128)
  (prune-nested *transitions* *max-transitions* 64)
  (prune-flat *facts* *max-facts*)
  ;; pruning *cooccur* invalidates the derived vector cache
  (setq *vcache* (make-hash-table :test 'equal) *vec-mean* nil))

(defun cap-tokens (words)
  "Drop words that would grow *dictionary* past *max-vocab* (keeps the extract path bounded)."
  (if (and *max-vocab* (>= (hash-table-count *dictionary*) *max-vocab*))
      (remove-if-not (lambda (w) (gethash w *dictionary*)) words)
      words))

;;; --- resume support: .read picks up where it left off (per-file byte offset, persisted) ---
(defun offset-key (path)
  "Stable key for *read-offsets* (absolute namestring when resolvable, else the path)."
  (or (ignore-errors (namestring (truename path))) (namestring path)))

(defun rewind-file (path)
  "Forget how far PATH has been read, so the next .read starts again from the beginning."
  (remhash (offset-key path) *read-offsets*)
  (format t "  (~a: rewound -- the next .read starts from the beginning)~%" path))

(defun report-slice (path start end eofp)
  (cond ((eql start end)
	 (format t "  (~a: already fully read -- .rewind ~a to read it again)~%" path path))
	(eofp (format t "  (~a: reached end of file)~%" path))
	(t (format t "  (~a: read bytes ~:d..~:d; .read it again for the next slice)~%" path start end))))

;;; --- .config / .set : view and change the tunables live -------------------------------
(defun show-config ()
  (format t "~&Tunable parameters (.set NAME VALUE to change; off/nil = unlimited):~%")
  (dolist (tn *tunables*)
    (format t "  ~16a ~a~%" (car tn) (symbol-value (cdr tn))))
  (format t "Current model sizes:~%")
  (format t "  ~16a ~:d~%" "vocabulary" (hash-table-count *dictionary*))
  (format t "  ~16a ~:d~%" "cooccur-words" (hash-table-count *cooccur*))
  (format t "  ~16a ~:d~%" "transition-heads" (hash-table-count *transitions*))
  (format t "  ~16a ~:d~%" "facts" (hash-table-count *facts*)))

(defun parse-value (s)
  "Parse a .set value: off/nil/none/unlimited -> NIL; on/t/yes -> T; digits -> integer; else string."
  (let ((d (string-downcase s)))
    (cond ((member d '("nil" "off" "none" "unlimited" "false") :test #'string=) nil)
	  ((member d '("t" "on" "yes" "true") :test #'string=) t)
	  ((every #'digit-char-p s) (parse-integer s))
	  (t s))))

(defun set-tunable (arg)
  "Handle `.set NAME VALUE': change a tunable, then re-enforce caps."
  (let* ((sp (position #\Space arg))
	 (name (string-downcase (string-trim " " (if sp (subseq arg 0 sp) arg))))
	 (valstr (and sp (string-trim " " (subseq arg sp))))
	 (tn (assoc name *tunables* :test #'string=)))
    (cond
      ((null tn) (format t "  (unknown parameter ~s -- type .config to list them)~%" name))
      ((or (null valstr) (zerop (length valstr)))
       (format t "  ~a = ~a~%" name (symbol-value (cdr tn))))   ; no value -> just show it
      (t (setf (symbol-value (cdr tn)) (parse-value valstr))
	 (format t "  ~a = ~a~%" name (symbol-value (cdr tn)))
	 (enforce-caps)))))

(defun words-of (neurons)
  "The word strings of a create-line result, in order."
  (mapcar #'named-neuron-name neurons))

(defun quit-line-p (words)
  "Is WORDS a lone `quit' or `exit'?"
  (and (consp words) (null (cdr words))
       (or (string= "quit" (car words)) (string= "exit" (car words)))))

(defun print-help ()
  "Print the teaching-loop instructions and the available loop commands."
  (format t "~&Teaching loop -- type a sentence, then on the next line the correct~%")
  (format t "response (or ~{~a~^/~} to confirm a correct guess).~%" *confirm-words*)
  (format t "Commands start with a period and are typed alone on a line:~%")
  (format t "  .help          show this help~%")
  (format t "  .stats         show system statistics~%")
  (format t "  .list [DIR]    list the .kb files in DIR (default: current directory)~%")
  (format t "  .save FILE     save to FILE and make it the active file~%")
  (format t "  .load FILE     clear, then load FILE and make it the active file~%")
  (format t "  .read FILE     learn from FILE (resumes where it left off; honors read-max-mb)~%")
  (format t "  .rewind FILE   forget how far FILE was read (next .read starts over)~%")
  (format t "  .merge FILE    merge another .kb's learned counts into the current model~%")
  (format t "  .config        show tunable parameters (model caps) and current sizes~%")
  (format t "  .set NAME VAL  change a parameter (e.g. .set max-cooccur 200000; off = unlimited)~%")
  (format t "  .quit          save and exit                       (also .exit)~%")
  (format t "The active file is ~a; it is loaded on start and auto-saved on exit.~%"
	  *save-file*))

(defun parse-command (line)
  "If raw input LINE is a leading-period command, return (values NAME ARG): NAME is the
   lowercased command word (e.g. \"save\") and ARG the trimmed remainder, with case and any
   embedded periods preserved (so filenames survive intact) -- ARG is \"\" when none was
   given.  Returns NIL when LINE is not a command (so it is treated as a sentence)."
  (let ((trimmed (string-left-trim '(#\space #\tab #\return #\linefeed) line)))
    (when (and (plusp (length trimmed)) (char= #\. (char trimmed 0)))
      (let* ((body (subseq trimmed 1))                       ; drop the leading period
	     (sp   (position-if (lambda (c) (member c '(#\space #\tab #\return #\linefeed)))
				body))
	     (name (string-downcase (subseq body 0 (or sp (length body)))))
	     (arg  (if sp
		       (string-trim '(#\space #\tab #\return #\linefeed) (subseq body sp))
		       "")))
	;; tolerate a trailing sentence terminator on the argument (.save my.kb.)
	(when (and (plusp (length arg)) (find (char arg (1- (length arg))) ".!?"))
	  (setf arg (string-right-trim '(#\space #\tab #\return #\linefeed)
				       (subseq arg 0 (1- (length arg))))))
	(values name arg)))))

(defun list-kb-files (arg)
  "Print the .kb files in directory ARG (the current directory when ARG is empty), sorted."
  (let* ((dirp   (plusp (length arg)))
	 (where  (if dirp arg "."))
	 (prefix (cond ((not dirp) "")
		       ((char= #\/ (char arg (1- (length arg)))) arg)   ; already ends in /
		       (t (concatenate 'string arg "/"))))
	 (files  (sort (mapcar #'file-namestring
			       (directory (concatenate 'string prefix "*.kb")))
		       #'string<)))
    (if files
	(progn
	  (format t "  (~d .kb file~:p in ~a)~%" (length files) where)
	  (dolist (f files) (format t "    ~a~%" f)))
	(format t "  (no .kb files in ~a)~%" where))))

(defun run-command (name arg)
  "Execute a leading-period loop command.  Return :quit to stop the loop, else NIL."
  (cond
    ((or (string= name "quit") (string= name "exit")) :quit)
    ((string= name "help")   (print-help)        nil)
    ((string= name "stats")  (system-stats)      nil)
    ((string= name "config") (show-config)       nil)
    ((string= name "set")    (set-tunable arg)   nil)
    ((string= name "list")  (list-kb-files arg) nil)
    ((string= name "save")
     (cond ((zerop (length arg)) (format t "  (.save needs a filename, e.g. .save my.kb)~%"))
	   (t (save-network arg)
	      (setf *save-file* arg)              ; FILE becomes the active (auto-saved) file
	      (format t "  (saved to ~a -- now the active file; it will auto-save here on exit)~%"
		      arg)))
     nil)
    ((string= name "load")
     (cond ((zerop (length arg)) (format t "  (.load needs a filename)~%"))
	   ((load-network arg)
	    (setf *save-file* arg)                ; FILE becomes the active (auto-saved) file
	    (format t "  (loaded ~a -- now the active file; it will auto-save here on exit)~%" arg))
	   (t (format t "  (could not load ~a -- file not found; knowledge base unchanged)~%" arg)))
     nil)
    ((string= name "read")
     (cond ((zerop (length arg)) (format t "  (.read needs a filename)~%"))
	   ((probe-file arg)      (load-knowledge arg))   ; resumes from where it left off
	   (t (format t "  (could not read ~a -- file not found)~%" arg)))
     nil)
    ((string= name "rewind")
     (if (zerop (length arg)) (format t "  (.rewind needs a filename)~%") (rewind-file arg))
     nil)
    ((string= name "merge")
     (cond ((zerop (length arg)) (format t "  (.merge needs a .kb filename)~%"))
	   ((merge-kb arg) (format t "  (merged the count stores from ~a into the current model)~%" arg))
	   (t (format t "  (could not merge ~a -- file not found)~%" arg)))
     nil)
    (t (format t "  (unknown command .~a -- type .help for the command list)~%" name)
       nil)))

(defun confirm-p (words)
  "Is WORDS a single confirmation token (accept the last guess)?"
  (and (consp words) (null (cdr words))
       (member (car words) *confirm-words* :test #'string=)))

;;; --- Conversation memory (Future.md item 1) ------------------------------------
;;; A short follow-up leans on the previous turn: its content word fills the previous
;;; sentence's most concept-similar slot.  After "do dogs have legs?", "and cats?" becomes
;;; "do cats have legs?".  Which word to replace is decided by the learned concept graph
;;; (concept-similarity) -- emergent, not a grammar rule.

(defparameter *followup-markers* '("and" "or")
  "Leading words that mark an input as a conversational follow-up fragment.")

(defun followup-p (words)
  "Is WORDS a follow-up fragment (lean on the previous turn) rather than a full sentence?
   True for a single word, a leading `and'/`or', or a leading `what about' / `how about'."
  (and (consp words)
       (or (null (cdr words))                              ; a single word
	   (member (first words) *followup-markers* :test #'string=)
	   (and (cdr words)
		(member (first words) '("what" "how") :test #'string=)
		(string= (second words) "about")))))

(defun resolve-followup (words)
  "If WORDS is a follow-up and a previous turn exists, return the previous sentence with
   its most concept-similar word replaced by this fragment's best-matching word; otherwise
   return WORDS unchanged."
  (if (and (followup-p words) *last-turn*)
      (let ((best 0.0) (fw nil) (lw nil))
	(dolist (a words)
	  (dolist (b *last-turn*)
	    (let ((s (similarity a b)))   ; distributed-vector similarity (smoother than the graph)
	      (when (> s best) (setf best s fw a lw b)))))
	(if fw (substitute fw lw *last-turn* :test #'string=) words))
      words))

(defun remember-turn (words)
  "Record WORDS as the previous (resolved) turn, for the next follow-up to lean on."
  (setf *last-turn* (copy-list words)))

(defun ask (input)
  "Conversational query: resolve a follow-up against the previous turn, answer it, and
   remember the resolved turn.  INPUT is a sentence string or a word list.  Returns the
   answer word list (or NIL)."
  (let* ((words    (as-words input))
	 (resolved (resolve-followup words))
	 (answer   (if (generation-request-p resolved)
		       (respond resolved)         ; generation request (tell me about / why)
		       (or (infer-answer resolved)  ; accurate concept-graph answer first
			   (respond resolved)))))   ; fall back to direct association
    (remember-turn resolved)
    answer))

(defun main ()
  "Interactive continual-learning teaching loop.  Each turn:
     1. read an input sentence (terminated by . ! or ?),
     2. print what the system would answer (or \"I don't know\"),
     3. read the teacher's line -- the correct response, or a confirm token
        (one of *confirm-words*) to accept a correct guess,
     4. learn from it (reinforce a correct guess / teach or correct otherwise).
   Stop with `.quit' / `.exit' (or end-of-input).  On entry it loads saved memory if any,
   else learns *starter-kb* (so a fresh system already knows things); it saves on exit."
  (cond ((load-network)
         (format t "~&(loaded saved memory from ~a)~%" *save-file*))
        ((and *starter-kb* (probe-file *starter-kb*))
         (reset)
         (format t "~&(no saved memory -- learning the starter knowledge base ~a ...)~%"
                 *starter-kb*)
         (let ((n (train-from-file *starter-kb* :verbose nil)))
           (format t "~&(learned ~d facts; teach me more, or just ask)~%" n)))
        (t (reset)))
  (print-help)
  (terpri)
  (loop
     (format t "~&input> ")
     (force-output)
     (let ((line (read-line nil nil)))                 ; raw line: keeps case for filenames
       (when (null line) (return))                     ; end of input
       (multiple-value-bind (cmd arg) (parse-command line)
	 (cond
	   (cmd                                          ; a .command (help/stats/save/load/read/quit)
	    (when (eq :quit (run-command cmd arg)) (return)))
	   (t
	    (let ((in (tokenize (string-downcase line))))
	      (cond
		((null in) nil)                          ; blank line -- just prompt again
		((quit-line-p in) (return))              ; bare quit/exit still stops the loop
		(t
		 (let* ((resolved (resolve-followup in)) ; conversation memory: fold in previous turn
			(guess (if (generation-request-p resolved)
				   (respond resolved)
				   (or (infer-answer resolved) (respond resolved)))))
		   (remember-turn resolved)
		   (format t "  guess: ~a~%"
			   (if guess (format nil "~{~a~^ ~}" guess) "(I don't know)"))
		   (format t "teach> ")
		   (force-output)
		   (let ((tline (read-line nil nil)))
		     (when (null tline) (return))        ; end of input
		     (let ((teacher (tokenize (string-downcase tline))))
		       (cond
			 ((null teacher) nil)            ; blank teacher line -- learn nothing
			 ((confirm-p teacher)
			  (cond (guess
				 (learn resolved guess)
				 (format t "  (reinforced)~%"))
				(t
				 (format t "  (nothing to confirm -- give the answer)~%"))))
			 (t
			  (learn resolved teacher)
			  (format t "  (learned)~%"))))))))))))))
  (save-network)
  (format t "~&(memory saved to ~a)~%" *save-file*))

;;; --- Training-set import -------------------------------------------------------
;;; A training set is a text file of one relationship per line:
;;;     input phrase => answer phrase
;;; Blank lines and lines beginning with # or ; are ignored.  Each line is learned
;;; exactly as a teaching-loop turn would be (associations + concept graph).

(defun find-substring (needle haystack)
  "Index of the first occurrence of NEEDLE in HAYSTACK, or NIL."
  (let ((nl (length needle)) (hl (length haystack)))
    (loop for i from 0 to (- hl nl)
	  when (string= needle haystack :start2 i :end2 (+ i nl)) return i)))

(defun train-from-file (path &key (separator "=>") (verbose t))
  "Learn from a training file: each non-blank, non-comment (# or ;) line is
   `input phrase SEPARATOR answer phrase' (default SEPARATOR \"=>\").  Each pair is
   learned via LEARN (so it grows associations and the concept graph).  Returns the
   number of relationships learned."
  (with-open-file (s path :direction :input)
    (let ((count 0))
      (loop for line = (read-line s nil nil)
	    while line
	    do (let ((trimmed (string-left-trim '(#\Space #\Tab) line)))
		 (unless (or (zerop (length trimmed))
			     (member (char trimmed 0) '(#\# #\;)))
		   (let ((sep (find-substring separator line)))
		     (when sep
		       (let ((in  (tokenize (subseq line 0 sep)))
			     (out (tokenize (subseq line (+ sep (length separator))))))
			 (when (and in out)
			   (learn in out)
			   (incf count))))))))
      (when verbose (format t "~&trained on ~d relationships from ~a~%" count path))
      count)))

;;; --- Learning from raw text (read-text) ----------------------------------------
;;; Two things happen per sentence: (1) its words feed the distributed-vector co-occurrence
;;; -- unsupervised similarity learning, no teacher; (2) if the sentence matches a simple
;;; declarative pattern, a fact is also learned.  Pattern extraction is deliberately light
;;; (regular sentences only); arbitrary prose is the open, hard problem.

(defparameter *question-words*
  '("what" "who" "where" "when" "why" "how" "is" "are" "do" "does" "can" "will" "did")
  "Leading words that mark a sentence as a question, so it isn't mistaken for a stated fact.")

;; (slurp-file removed: files are now streamed in bounded-memory chunks -- see stream-chunks /
;;  read-text-file below.  Loading a whole 100 GB+ corpus into a string is exactly what we avoid;
;;  it was also wrong for UTF-8, where file-length counts bytes, not characters.)

(defun split-sentences (text)
  "Split TEXT into trimmed sentence strings on . ! ? boundaries."
  (let ((res '()) (start 0) (n (length text)))
    (flet ((emit (i) (let ((s (string-trim '(#\Space #\Tab #\Newline #\Return)
					    (subseq text start i))))
		       (when (plusp (length s)) (push s res)))))
      (dotimes (i n) (when (member (char text i) '(#\. #\! #\?)) (emit i) (setf start (1+ i))))
      (emit n))
    (nreverse res)))

(defun join-words (ws) (format nil "~{~a~^ ~}" ws))

(defun superlative-p (w)
  "Is W a superlative adjective (largest, tallest, best, most, ...)?  Used to read
   'X is the largest Y' as 'what is the largest Y => X'."
  (or (and (> (length w) 4) (string= "est" (subseq w (- (length w) 3))))   ; -est: largest, tallest
      (member w '("most" "least" "best" "worst") :test #'string=)))

(defun extract-fact (words)
  "If WORDS is a simple declarative sentence, learn a fact and return T.  Handles
   'X is the Y of Z' (relational) and 'X is/are Y' (membership/copula).  Skips questions."
  (when (and words (not (member (first words) *question-words* :test #'string=)))
    (let ((cp (or (position "is" words :test #'string=)
		  (position "are" words :test #'string=)
		  (position "was" words :test #'string=)
		  (position "were" words :test #'string=))))
      (when (and cp (> cp 0) (< cp (1- (length words))))
	(let ((cop (nth cp words)) (before (subseq words 0 cp)) (after (subseq words (1+ cp))))
	  (cond
	    ;; "X is the Y of Z"  ->  what is the Y of Z => X
	    ((and (string= (first after) "the") (position "of" after :test #'string=))
	     (let* ((op (position "of" after :test #'string=))
		    (y (subseq after 1 op)) (z (subseq after (1+ op))))
	       (when (and y z before)
		 (learn (format nil "what is the ~a of ~a" (join-words y) (join-words z))
			(join-words before))
		 t)))
	    ;; "the Y of Z is X"  ->  what is the Y of Z => X
	    ((and (string= (first before) "the") (position "of" before :test #'string=))
	     (let* ((op (position "of" before :test #'string=))
		    (y (subseq before 1 op)) (z (subseq before (1+ op))))
	       (when (and y z after)
		 (learn (format nil "what is the ~a of ~a" (join-words y) (join-words z))
			(join-words after))
		 t)))
	    ;; "X is the <superlative> Y"  ->  what is the <superlative> Y => X
	    ((and (string= (first after) "the") (cdr after) (superlative-p (second after)))
	     (learn (format nil "what is ~a" (join-words after)) (join-words before))
	     t)
	    ;; "the <superlative> Y is X"  ->  what is the <superlative> Y => X
	    ((and (string= (first before) "the") (cdr before) (superlative-p (second before)))
	     (learn (format nil "what is ~a" (join-words before)) (join-words after))
	     t)
	    ;; "X was a/an Y"  ->  who was X => a/an Y   (a person and their role)
	    ((and (string= cop "was") (member (first after) '("a" "an") :test #'string=))
	     (learn (format nil "who was ~a" (join-words before)) (join-words after))
	     t)
	    ;; "X is/are/was/were Y"  ->  is/are/... X Y => yes
	    (t (learn (format nil "~a ~a ~a" cop (join-words before) (join-words after)) "yes")
	       t)))))))

;; (read/cap parameters and pruning are defined near the top, with the .config/.set machinery)

(defun split-lines (text)
  "Split TEXT on newlines into a list of lines (delimiters dropped)."
  (let (res (start 0) (n (length text)))
    (dotimes (i n) (when (char= (char text i) #\Newline)
		     (push (subseq text start i) res) (setf start (1+ i))))
    (push (subseq text start n) res)
    (nreverse res)))

(defun process-sentence (words &optional (extract *read-extract*))
  "Learn from one already-tokenized sentence in a single online pass.  Honors the model caps:
   the heavy extract path is bounded by *max-vocab*, and capped stores are pruned every
   *prune-every* sentences.  Returns 1 if a fact was extracted, else 0."
  (let ((nf 0))
    (when words
      (when *read-relations* (observe words))      ; relations: online connector/head discovery (Phase 9)
      (when *read-cooccur* (note-cooccurrence words nil))  ; distributed-vector similarity (O(n^2))
      (when *read-transitions-p* (note-sequence words))   ; generation transition model (Phase 8)
      (when *read-facts-p*
	(note-facts words)               ; generation declarative triples (hardcoded)
	(when *read-relations*
	  (multiple-value-bind (subj conn cat cls) (relation-of words)
	    (declare (ignore conn))
	    (when (and (eq cls :membership) subj cat) (note-fact subj "is-a" cat)))))
      (when (and extract (extract-fact (cap-tokens words))) (setf nf 1)))   ; vocab-capped
    (when (>= (incf *learned-since-prune*) *prune-every*)
      (setf *learned-since-prune* 0) (enforce-caps))
    nf))

(defun read-text (text &key (extract *read-extract*) (verbose t))
  "Learn from raw TEXT (one or more sentences) in a single online pass.  Returns (values
   sentences-read facts-learned)."
  (let ((ns 0) (nf 0))
    (dolist (s (split-sentences text))
      (let ((words (tokenize s))) (when words (incf ns) (incf nf (process-sentence words extract)))))
    (when verbose (format t "~&read ~:d sentences, learned ~:d facts~%" ns nf))
    (values ns nf)))

(defun stream-chunks (path drain &key (start 0) (max-bytes nil) (verbose t))
  "Stream PATH in bounded-memory chunks (the whole file is never held in memory; UTF-8 safe),
   beginning at file position START.  DRAIN is called as (drain buffer) during streaming -- it
   processes the complete units in BUFFER and returns the unprocessed tail -- and once as
   (drain tail t) at end of stream.  Returns (values END-POSITION REACHED-EOF)."
  (with-open-file (s path :direction :input)
    (when (and start (plusp start)) (file-position s start))
    (let ((cbuf (make-string *read-chunk*)) (pending "") (bytes 0) (prog 0) (eofp nil)
	  (total (ignore-errors (file-length s)))   ; whole-file size (for the X-of-Y status)
	  (shown nil))                               ; did we print a live status line?
      (loop
	(let ((n (read-sequence cbuf s)))
	  (when (zerop n) (setf eofp t) (return))   ; consumed to end of file
	  (incf bytes n)
	  (setf pending (funcall drain (concatenate 'string pending (subseq cbuf 0 n))))
	  (when (and verbose (>= bytes prog))       ; live, in-place status (carriage return, no newline)
	    (let ((pos (or (ignore-errors (file-position s)) bytes)))
	      (format t "~C  reading ~a: ~:d MB~@[ / ~:d MB~]~@[ (~d%)~]    "
		      #\Return (file-namestring path)
		      (round pos 1000000)
		      (and total (round total 1000000))
		      (and total (plusp total) (min 100 (round (* 100 pos) total))))
	      (force-output))
	    (setf shown t prog (+ bytes 2000000)))  ; update every ~2 MB
	  (when (and max-bytes (>= bytes max-bytes)) (return))))   ; slice cap reached (not EOF)
      (funcall drain pending t)
      (when (and verbose shown) (terpri))         ; finish the status line
      (values (file-position s) eofp))))

(defun use-parallel-p ()
  "Should the next bulk read fan out across worker threads?"
  (and *parallel-ok* (integerp *read-workers*) (> *read-workers* 1)))

(defun read-text-file (path &key (extract *read-extract*) (verbose t) (max-mb *read-max-mb*))
  "Stream raw text from PATH (bounded memory) and learn from it as prose.  Returns (values
   sentences facts).  When read-extract is off and read-workers > 1 (SBCL), fans the per-sentence
   learning out across worker threads (shard-and-merge); otherwise runs the sequential path."
  (if (and (not extract) (use-parallel-p))
      (read-text-file-parallel path :max-mb max-mb :verbose verbose)
      (read-text-file-sequential path :extract extract :verbose verbose :max-mb max-mb)))

(defun read-text-file-sequential (path &key (extract *read-extract*) (verbose t) (max-mb *read-max-mb*))
  "Single-threaded prose reader (the portable path).  Splits on sentence terminators across the
   whole stream (newlines are whitespace), never loading the file in full.  (values sentences facts)."
  (let* ((ns 0) (nf 0) (key (offset-key path)) (start (gethash key *read-offsets* 0)))
    (flet ((proc (text)
	     (dolist (s (split-sentences text))
	       (let ((w (tokenize s))) (when w (incf ns) (incf nf (process-sentence w extract)))))))
      (multiple-value-bind (end eofp)
	  (stream-chunks
	   path
	   (lambda (buf &optional final)
	     (if final (progn (proc buf) "")
		 (let ((p (position-if (lambda (c) (member c '(#\. #\! #\?))) buf :from-end t)))
		   (cond (p (proc (subseq buf 0 (1+ p))) (subseq buf (1+ p)))
			 ((> (length buf) *read-flush*) (proc buf) "")  ; no terminator: force-flush
			 (t buf)))))
	   :start start :max-bytes (and max-mb (round (* max-mb 1000000))) :verbose verbose)
	(setf (gethash key *read-offsets*) end)
	(when verbose
	  (format t "~&read ~:d sentences, learned ~:d facts from ~a~%" ns nf path)
	  (report-slice path start end eofp))))
    (values ns nf)))

;;; --- Unified loader: one front door, two underlying modes ----------------------
;;; The system learns two ways -- supervised `input => answer' pairs (learn) and raw prose
;;; (read-text) -- and they are genuinely different (a pair states an exact stimulus->
;;; response; prose is interpreted best-effort).  But the *author* shouldn't have to keep
;;; two file formats straight: `load-knowledge' reads a file line by line and ROUTES each
;;; line to the right mode, so one file may freely mix both.  train-from-file / read-text
;;; are unchanged underneath -- this is just the smart front door.

(defun load-knowledge (path &key (verbose t) (separator "=>") (max-mb *read-max-mb*))
  "Stream a knowledge file (bounded memory -- read in chunks, never loaded whole), auto-routing
   each line:
     * a line containing SEPARATOR (default \"=>\") is a supervised pair  -> learn;
     * any other non-comment line is prose                                -> read-text;
     * blank lines and lines beginning with # or ; are ignored.
   One file may mix both formats.  Bind *read-max-mb* (or pass :max-mb) to cap a huge corpus.
   When read-extract is off and read-workers > 1 (SBCL), the prose is learned in parallel worker
   threads while any supervised pairs are applied sequentially after the parallel pass.
   Returns (values pairs-learned prose-sentences prose-facts)."
  (if (and (not *read-extract*) (use-parallel-p))
      (load-knowledge-parallel path :verbose verbose :separator separator :max-mb max-mb)
      (load-knowledge-sequential path :verbose verbose :separator separator :max-mb max-mb)))

(defun load-knowledge-sequential (path &key (verbose t) (separator "=>") (max-mb *read-max-mb*))
  "Single-threaded unified loader (the portable path).  (values pairs prose-sentences prose-facts)."
  (let* ((pairs 0) (psent 0) (pfacts 0) (key (offset-key path)) (start (gethash key *read-offsets* 0)))
    (flet ((route (line)
	     (let ((trimmed (string-left-trim '(#\Space #\Tab #\Return) line)))
	       (cond
		 ((or (zerop (length trimmed)) (member (char trimmed 0) '(#\# #\;))) nil)
		 ((find-substring separator line)        ; supervised input => answer
		  (let* ((sep (find-substring separator line))
			 (in  (tokenize (subseq line 0 sep)))
			 (out (tokenize (subseq line (+ sep (length separator))))))
		    (when (and in out) (learn in out) (incf pairs))))
		 (t (multiple-value-bind (n f) (read-text trimmed :verbose nil)
		      (incf psent n) (incf pfacts f)))))))
      (multiple-value-bind (end eofp)
	  (stream-chunks
	   path
	   (lambda (buf &optional final)
	     (if final (progn (when (plusp (length buf)) (route buf)) "")
		 (let ((p (position #\Newline buf :from-end t)))
		   (cond (p (dolist (ln (split-lines (subseq buf 0 p))) (route ln)) (subseq buf (1+ p)))
			 ((> (length buf) *read-flush*) (route buf) "")  ; pathological long line
			 (t buf)))))
	   :start start :max-bytes (and max-mb (round (* max-mb 1000000))) :verbose verbose)
	(setf (gethash key *read-offsets*) end)
	(when verbose
	  (format t "~&loaded ~a: ~:d supervised pair~:p, ~:d prose sentence~:p (~:d fact~:p)~%"
		  path pairs psent pfacts)
	  (report-slice path start end eofp))))
    (values pairs psent pfacts)))

;;; --- Parallel bulk reading (SBCL): shard-and-merge --------------------------------
;;; The prose learners are all additive counters, so the work splits cleanly: ONE reader
;;; thread streams + tokenizes the file (the cheap, I/O-bound part) and hands batches of
;;; tokenized sentences to N worker threads; each worker learns into its OWN private count
;;; stores (dynamic rebindings -- zero contention), and we sum the partials at the end
;;; (persist:merge-worker-stores).  A semaphore bounds in-flight batches so the fast reader
;;; can't outrun the workers and blow up memory.  Parallel runs only in the read-extract-off
;;; (lean) path -- the supervised path mutates a shared, non-mergeable neuron graph.
;;; On a non-thread build these fall back to the sequential reader, so the code stays portable.

#+(and sbcl sb-thread)
(defun spawn-prose-worker (mailbox sem)
  "Start a worker that learns batches from MAILBOX into PRIVATE stores and returns them."
  (sb-thread:make-thread
   (lambda ()
     (let ((*cooccur* (make-hash-table :test 'equal))     ; private per-worker count stores
	   (*transitions* (make-hash-table :test 'equal))
	   (*sentence-starts* (make-hash-table :test 'equal))
	   (*facts* (make-hash-table :test 'equal))
	   (*rel-links* (make-hash-table :test 'equal))
	   (*rel-head* (make-hash-table :test 'equal))
	   (*rel-freq* (make-hash-table :test 'equal))
	   (*rel-sentences* 0)
	   (*vcache* (make-hash-table :test 'equal))
	   (*vec-mean* nil)
	   (*max-vocab* nil) (*max-cooccur* nil)          ; no per-worker pruning; prune once after merge
	   (*max-transitions* nil) (*max-facts* nil) (*learned-since-prune* 0))
       (loop for batch = (sb-concurrency:receive-message mailbox)
	     until (eq batch :done)
	     do (dolist (w batch) (process-sentence w nil))   ; extract off
		(sb-thread:signal-semaphore sem))               ; release one in-flight slot
       (list *cooccur* *transitions* *sentence-starts* *facts*
	     *rel-links* *rel-head* *rel-freq* *rel-sentences*)))
   :name "prose-worker"))

(defun read-text-file-parallel (path &key (max-mb *read-max-mb*) (verbose t))
  "Parallel prose reader (SBCL).  Returns (values sentences facts)."
  #-(and sbcl sb-thread)
  (read-text-file-sequential path :extract nil :verbose verbose :max-mb max-mb)
  #+(and sbcl sb-thread)
  (let* ((key (offset-key path)) (start (gethash key *read-offsets* 0))
	 (nw *read-workers*) (batch-size 512)
	 (mailbox (sb-concurrency:make-mailbox))
	 (sem (sb-thread:make-semaphore :count (* nw 4)))   ; backpressure: <= 4 batches/worker in flight
	 (workers (loop repeat nw collect (spawn-prose-worker mailbox sem)))
	 (batch nil) (nb 0) (ns 0))
    (labels ((send-batch ()
	       (when batch
		 (sb-thread:wait-on-semaphore sem)              ; block if too many batches queued
		 (sb-concurrency:send-message mailbox (nreverse batch))
		 (setf batch nil nb 0)))
	     (enq (text)
	       (dolist (s (split-sentences text))
		 (let ((w (tokenize s)))
		   (when w (push w batch) (incf ns)
			 (when (>= (incf nb) batch-size) (send-batch)))))))
      (multiple-value-bind (end eofp)
	  (stream-chunks
	   path
	   (lambda (buf &optional final)
	     (if final (progn (enq buf) "")
		 (let ((p (position-if (lambda (c) (member c '(#\. #\! #\?))) buf :from-end t)))
		   (cond (p (enq (subseq buf 0 (1+ p))) (subseq buf (1+ p)))
			 ((> (length buf) *read-flush*) (enq buf) "")
			 (t buf)))))
	   :start start :max-bytes (and max-mb (round (* max-mb 1000000))) :verbose verbose)
	(send-batch)
	(dotimes (i nw) (sb-concurrency:send-message mailbox :done))   ; one stop signal per worker
	(dolist (th workers) (merge-worker-stores (sb-thread:join-thread th)))
	(clrhash *vcache*) (setf *vec-mean* nil)
	(enforce-caps)                                                 ; prune once over the merged model
	(setf (gethash key *read-offsets*) end)
	(when verbose
	  (format t "~&read ~:d sentences via ~d worker~:p from ~a~%" ns nw path)
	  (report-slice path start end eofp))
	(values ns 0)))))

(defun load-knowledge-parallel (path &key (verbose t) (separator "=>") (max-mb *read-max-mb*))
  "Parallel unified loader (SBCL): prose is learned in worker threads; supervised pairs are
   collected and applied sequentially after the parallel pass (they mutate the shared graph)."
  #-(and sbcl sb-thread)
  (load-knowledge-sequential path :verbose verbose :separator separator :max-mb max-mb)
  #+(and sbcl sb-thread)
  (let* ((key (offset-key path)) (start (gethash key *read-offsets* 0))
	 (nw *read-workers*) (batch-size 512)
	 (mailbox (sb-concurrency:make-mailbox))
	 (sem (sb-thread:make-semaphore :count (* nw 4)))
	 (workers (loop repeat nw collect (spawn-prose-worker mailbox sem)))
	 (batch nil) (nb 0) (psent 0) (deferred nil))
    (labels ((send-batch ()
	       (when batch
		 (sb-thread:wait-on-semaphore sem)
		 (sb-concurrency:send-message mailbox (nreverse batch))
		 (setf batch nil nb 0)))
	     (enq-prose (text)
	       (dolist (s (split-sentences text))
		 (let ((w (tokenize s)))
		   (when w (push w batch) (incf psent)
			 (when (>= (incf nb) batch-size) (send-batch))))))
	     (route (line)
	       (let ((trimmed (string-left-trim '(#\Space #\Tab #\Return) line)))
		 (cond
		   ((or (zerop (length trimmed)) (member (char trimmed 0) '(#\# #\;))) nil)
		   ((find-substring separator line) (push line deferred))   ; pair -> defer to sequential
		   (t (enq-prose trimmed))))))
      (multiple-value-bind (end eofp)
	  (stream-chunks
	   path
	   (lambda (buf &optional final)
	     (if final (progn (when (plusp (length buf)) (route buf)) "")
		 (let ((p (position #\Newline buf :from-end t)))
		   (cond (p (dolist (ln (split-lines (subseq buf 0 p))) (route ln)) (subseq buf (1+ p)))
			 ((> (length buf) *read-flush*) (route buf) "")
			 (t buf)))))
	   :start start :max-bytes (and max-mb (round (* max-mb 1000000))) :verbose verbose)
	(send-batch)
	(dotimes (i nw) (sb-concurrency:send-message mailbox :done))
	(dolist (th workers) (merge-worker-stores (sb-thread:join-thread th)))
	(clrhash *vcache*) (setf *vec-mean* nil)
	(let ((pairs 0))                                  ; now the deferred supervised pairs, in order
	  (dolist (line (nreverse deferred))
	    (let* ((sep (find-substring separator line))
		   (in  (tokenize (subseq line 0 sep)))
		   (out (tokenize (subseq line (+ sep (length separator))))))
	      (when (and in out) (learn in out) (incf pairs))))
	  (enforce-caps)
	  (setf (gethash key *read-offsets*) end)
	  (when verbose
	    (format t "~&loaded ~a: ~:d supervised pair~:p, ~:d prose sentence~:p via ~d worker~:p~%"
		    path pairs psent nw)
	    (report-slice path start end eofp))
	  (values pairs psent 0))))))
