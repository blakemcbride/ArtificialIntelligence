;;;; procedure-experiment.lisp -- learning a PROCEDURE (not a fact) as a trajectory of
;;;; Hebbian state-transitions over persistent working memory, executed by recurrence.
;;;; Standalone proof-of-concept.   Run:  sbcl --script procedure-experiment.lisp
;;;;
;;;; The bet (from our design discussion): the concept graph showed *categories* emerge
;;;; from shared static structure.  A *procedure* (e.g. "how many X do you know") is the
;;;; procedural analog -- it should emerge as a learned chain of state transitions over a
;;;; small, general, grounded substrate.  NOT a hard-coded count() verb, NOT a per-question
;;;; rule.
;;;;
;;;; The substrate -- the irreducible, GENERAL floor, reused by every procedure (the analog
;;;; of innate neural micro-ops; none of these is a task verb):
;;;;   * persistent working-memory STATE (a few registers built from grounded concepts)
;;;;   * SELECT/PEEK one item from a set            <- the attention head, already built
;;;;   * SUCCESSOR, the next number on the number line <- already learned in the KB
;;;;   * RECURRENCE: apply the transition repeatedly (a clock / loop)
;;;;   * HEBBIAN transition learning: condition -> action strengths (the same learning rule)
;;;;
;;;; What is LEARNED -- the procedure itself -- is only this: which ACTION follows which
;;;; state-CONDITION.  Counting is never coded.  It is induced from a few demonstrations and
;;;; then GENERALIZES to collection sizes, compositions, and even CATEGORIES it was never
;;;; shown -- exactly as categories generalized to never-taught words.
;;;;
;;;; (Note on the "logic system" worry: yes, at bottom this is conditional transitions --
;;;; but so is every procedure a brain runs.  The difference from symbolic AI is that here
;;;; the conditionals are LEARNED, graded by strength, and generalize -- not hand-written
;;;; brittle rules.  Conditionals aren't the enemy; hand-coded brittle ones are.)

;;; ---- persistent working-memory state (the "registers") ----------------------------
(defstruct st (count 0) (remaining nil) (output nil) (halted nil))

;;; ---- grounded substrate ops (general; not specific to counting) --------------------
(defun successor (n) (1+ n))                 ; the learned number line (here: integers)
(defun peek (s) (first (st-remaining s)))    ; attention: look at one item of the set
(defun item-category (item) (cdr item))      ; an item is (name . category)

(defun advance (s &key tally)
  "Consume one item; optionally step the count to its successor."
  (make-st :count     (if tally (successor (st-count s)) (st-count s))
           :remaining (rest (st-remaining s))
           :output    (st-output s)
           :halted    nil))

;;; The CONDITION read off the current state, relative to the queried category *target*.
(defparameter *target* nil)
(defun condition-of (s)
  (cond ((null (st-remaining s)) :none)                              ; set exhausted
        ((eq (item-category (peek s)) *target*) :match)              ; next item is one we want
        (t :other)))                                                 ; next item is not

;;; ACTIONS over state (each a grounded micro-op).
(defun act-tally  (s) (advance s :tally t))   ; count this one and move on
(defun act-pass   (s) (advance s))            ; skip this one and move on
(defun act-finish (s) (make-st :count (st-count s) :remaining (st-remaining s)
                               :output (st-count s) :halted t))
(defparameter *actions*
  (list (cons :tally #'act-tally) (cons :pass #'act-pass) (cons :finish #'act-finish)))

;;; ---- the LEARNED part: condition -> (action -> strength), Hebbian --------------------
(defparameter *transitions* (make-hash-table :test 'eq))
(defun strengthen (cnd act)
  (let ((tab (or (gethash cnd *transitions*)
                 (setf (gethash cnd *transitions*) (make-hash-table :test 'eq)))))
    (incf (gethash act tab 0))))
(defun chosen-action (cnd)                       ; winner-take-all over learned strengths
  (let ((tab (gethash cnd *transitions*)) (best nil) (bs -1))
    (when tab (maphash (lambda (a s) (when (> s bs) (setf bs s best a))) tab))
    best))
(defun learn-trace (trace)                       ; teach by demonstration: (condition action)*
  (dolist (step trace) (strengthen (first step) (second step))))

;;; ---- RECURRENCE: run the learned procedure to completion ----------------------------
(defun run-procedure (items target &optional (max 100000))
  (let ((*target* target) (s (make-st :remaining items)))
    (dotimes (i max (st-output s))
      (when (st-halted s) (return (st-output s)))
      (let ((a (chosen-action (condition-of s))))
        (unless a (return :no-procedure-learned))      ; it has no procedure for this state
        (setf s (funcall (cdr (assoc a *actions*)) s))))))

;;; =====================================================================================
;;; Demonstration
;;; =====================================================================================
(defun nshow (label items target)
  (format t "   ~28a -> ~a~%" label (run-procedure items target)))

(let ((zoo (list (cons "dog" :animal) (cons "car" :vehicle) (cons "cat" :animal)
                 (cons "rose" :plant)  (cons "lion" :animal) (cons "bus" :vehicle)
                 (cons "owl" :animal)  (cons "oak" :plant)   (cons "fox" :animal))))

  (format t "~%BEFORE teaching -- it has no procedure:~%")
  (nshow "count animals in the zoo" zoo :animal)

  ;; Teach by DEMONSTRATION: show the step-by-step traces of counting the wanted items in a
  ;; few small, mixed collections.  Each step is just (state-condition  action-taken).
  (learn-trace '((:match :tally) (:other :pass) (:none :finish)))            ; counted {a v}
  (learn-trace '((:match :tally) (:match :tally) (:none :finish)))           ; counted {a a}
  (learn-trace '((:other :pass) (:match :tally) (:other :pass) (:none :finish))) ; {v a v}

  (format t "~%The learned procedure (condition -> action it now takes):~%")
  (dolist (c '(:match :other :none))
    (format t "   when ~6a do ~a~%" c (chosen-action c)))

  (format t "~%AFTER teaching -- the SAME learned procedure, run by recurrence:~%")
  (nshow "count animals  (size 9 set)" zoo :animal)     ; 5 -- size/composition never shown
  (nshow "count vehicles (same set)"   zoo :vehicle)    ; 2 -- a category never demonstrated
  (nshow "count plants   (same set)"   zoo :plant)      ; 2

  ;; a bigger, different collection -> generalizes over length too
  (let ((big (loop for i from 1 to 20
                   collect (cons (format nil "x~d" i)
                                 (if (evenp i) :animal :vehicle)))))
    (nshow "count animals in a size-20 set" big :animal))) ; 10

(format t "~%The point: counting was LEARNED from demonstration, not coded; it generalizes~%")
(format t "to new sizes, new mixes, and new categories, because the learned control is over~%")
(format t "match/other/none -- not over 'animals'.  Teach the operation once; it applies to~%")
(format t "any category.  A one-shot associative lookup cannot do this: the answer is not~%")
(format t "stored, it is computed over state by iterating.~%")
