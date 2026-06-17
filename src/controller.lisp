;;;; controller.lisp -- Part 5 of SystemAnalysis.md: this system as the CONTROLLER, the LLM as a
;;;; tool.  The loop:  PROPOSE (ask the LLM for K candidate actions)  ->  SELECT (a LEARNED policy
;;;; picks among them, conditioned on persistent memory and past outcomes)  ->  EXECUTE (ask the
;;;; LLM to carry out the chosen one)  ->  REINFORCE (the outcome updates the policy).
;;;;
;;;; Why this split: generating a good plan is open-ended (this system is weak at it -> delegate
;;;; to the LLM), but CHOOSING among a few proposals is a small, bounded, learnable problem.  The
;;;; learning rule is the project's own reward-modulated Hebbian update, w <- w(1-lambda) + eta*r,
;;;; applied to a (context, candidate) association in *selector* (which persists with the KB).  So
;;;; what becomes durably learnable is JUDGMENT/POLICY over a fixed reasoning engine -- the form of
;;;; "really learning" that RAG and context-stuffing cannot provide.
;;;;
;;;; The LLM is reached through llm.lisp (local Ollama/llama.cpp or cloud Anthropic/OpenAI); with
;;;; llm:*provider* :mock the whole loop runs offline (used by the tests).

(defpackage "controller"
  (:use "COMMON-LISP" "data-structures" "line-input" "llm")
  (:export "CONTROLLER-RESPOND" "CONTROLLER-REWARD" "CONTROLLER-PROPOSE" "CONTROLLER-SELECT"
	   "CONTROLLER-EXECUTE" "CONTROLLER-REINFORCE" "CONTROLLER-SCORE" "CONTROLLER-STATS"
	   "*SELECT-ETA*" "*SELECT-DECAY*" "*SELECT-PRUNE*" "*CONTROLLER-CANDIDATES*" "*LAST-DECISION*"))
(in-package "controller")
(provide "controller")

(require "data-structures") (require "line-input") (require "llm")

(defparameter *select-eta* 0.5 "Learning rate for the reward-modulated selector update.")
(defparameter *select-decay* 0.02 "Decay (lambda) applied to a weight when it is updated.")
(defparameter *select-prune* 0.01 "Drop a selector weight whose magnitude falls below this.")
(defparameter *controller-candidates* 3 "How many candidates to ask the LLM to propose.")
(defparameter *last-decision* nil "(context-key . chosen-candidate) of the most recent respond, for reward.")

;;; --- selector keys: a coarse, shareable signature of context and candidate -------------
(defun ctx-key (input)
  "A normalized signature of the input (sorted unique lowercased words), so similar inputs share
   policy.  INPUT may be a string or a list of word strings."
  (let* ((text (if (stringp input) input (format nil "~{~a~^ ~}" input)))
	 (ws (sort (remove-duplicates (tokenize (string-downcase text)) :test #'string=) #'string<)))
    (format nil "~{~a~^ ~}" ws)))
(defun cand-key (candidate) (string-downcase (string-trim '(#\Space #\Tab #\Newline #\Return) candidate)))
(defun sel-key (ctxkey candidate) (concatenate 'string ctxkey (string #\Tab) (cand-key candidate)))

(defun controller-score (input candidate)
  "Learned preference for CANDIDATE in INPUT's context (0.0 if never reinforced).  INPUT may be raw
   text or an already-normalized context key (normalization is idempotent)."
  (gethash (sel-key (ctx-key input) candidate) *selector* 0.0))

(defun controller-select (input candidates)
  "Pick the highest-scoring candidate (stable: first of equal-best).  Unseen candidates score 0,
   so they outrank punished ones -- giving natural exploration: try the unknown, drop the bad,
   keep the rewarded."
  (let ((best nil) (bw nil))
    (dolist (c candidates best)
      (let ((s (controller-score input c)))
	(when (or (null bw) (> s bw)) (setf bw s best c))))))

(defun reinforce-key (ctxkey candidate reward)
  (let* ((k (sel-key ctxkey candidate))
	 (w (+ (* (- 1.0 *select-decay*) (gethash k *selector* 0.0)) (* *select-eta* reward))))
    (if (< (abs w) *select-prune*) (progn (remhash k *selector*) 0.0) (setf (gethash k *selector*) w))))

(defun controller-reinforce (input candidate reward)
  "Reward-modulated Hebbian update of the (context, candidate) preference: w <- w(1-lambda)+eta*r."
  (reinforce-key (ctx-key input) candidate (float reward 1.0)))

;;; --- talking to the LLM (propose / execute) -------------------------------------------
(defun split-lines (text)
  (let (out (start 0))
    (dotimes (i (length text))
      (when (char= (char text i) #\Newline) (push (subseq text start i) out) (setf start (1+ i))))
    (push (subseq text start) out)
    (nreverse out)))
(defun strip-bullet (line)
  "Remove a leading list marker like '1.', '2)', '-', '*' and surrounding space."
  (let ((s (string-trim '(#\Space #\Tab #\Return) line)))
    (let ((i 0) (n (length s)))
      (loop while (and (< i n) (digit-char-p (char s i))) do (incf i))         ; digits
      (when (and (> i 0) (< i n) (member (char s i) '(#\. #\) #\:))) (incf i))   ; 1. / 2) / 3:
      (when (and (= i 0) (< i n) (member (char s i) '(#\- #\*))) (incf i))       ; - / *
      (string-trim '(#\Space #\Tab #\Return) (subseq s i)))))

(defun controller-propose (input &key (n *controller-candidates*) context)
  "Ask the LLM for N distinct candidate responses; return them as a list of strings."
  (let* ((system "You are the planning advisor for a controller. Propose distinct candidate responses to the user input. Reply with exactly one option per line, numbered, and no other commentary.")
	 (prompt (format nil "User input: ~a~@[~%Relevant memory: ~a~]~%Propose ~d distinct candidate responses, one per line:"
			 input context n))
	 (text (llm-complete prompt :system system))
	 (cands (remove "" (mapcar #'strip-bullet (split-lines text)) :test #'string=)))
    (or cands (list (string-trim '(#\Space #\Tab #\Newline #\Return) text)))))

(defun controller-execute (input chosen &key context)
  "Ask the LLM to carry out the CHOSEN approach for INPUT; return the final answer."
  (let ((system "You are the executor. Carry out the chosen approach and produce only the final answer.")
	(prompt (format nil "User input: ~a~@[~%Relevant memory: ~a~]~%Chosen approach: ~a~%Final answer:"
			input context chosen)))
    (llm-complete prompt :system system)))

;;; --- the full loop --------------------------------------------------------------------
(defun controller-respond (input &key context (n *controller-candidates*))
  "PROPOSE -> SELECT -> EXECUTE.  Returns (values result chosen candidates).  Records the decision
   in *last-decision* so a later CONTROLLER-REWARD can reinforce it."
  (let* ((candidates (controller-propose input :n n :context context))
	 (chosen (controller-select input candidates))
	 (result (controller-execute input chosen :context context)))
    (setf *last-decision* (cons (ctx-key input) chosen))
    (values result chosen candidates)))

(defun controller-reward (reward)
  "Apply REWARD (e.g. +1 good / -1 bad, from user feedback or a task-success check) to the last
   decision, so the selector learns which proposal to prefer next time in this context."
  (when *last-decision*
    (reinforce-key (car *last-decision*) (cdr *last-decision*) (float reward 1.0))))

(defun controller-stats ()
  "Print and return the size of the learned controller policy."
  (let ((n (hash-table-count *selector*)) (pos 0) (neg 0))
    (maphash (lambda (k w) (declare (ignore k)) (if (plusp w) (incf pos) (incf neg))) *selector*)
    (format t "~&controller policy: ~:d learned (context,candidate) preferences (~:d positive, ~:d negative)~%"
	    n pos neg)
    (list :preferences n :positive pos :negative neg)))
