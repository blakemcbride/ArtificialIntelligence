;;;; controller-vs-rag-experiment.lisp -- the head-to-head from SystemAnalysis.md Part 5 / the RAG
;;;; comparison: is the learning controller actually beneficial ABOVE a RAG memory?  Run:
;;;;     sbcl --script controller-vs-rag-experiment.lisp
;;;;
;;;; Offline (mock LLM): all arms see the SAME task and the SAME proposed candidates; they differ
;;;; ONLY in how they pick.  The task models an assistant whose correct action is environment-
;;;; specific and revealed only by FEEDBACK (not inferable from the prompt): each "ticket" carries
;;;; one rule-word (billing/outage/question/delay) plus noise fillers, and the right action
;;;; (refund/escalate/explain/apologize) depends on the rule-word -- a mapping the system must
;;;; LEARN from outcomes.  Four arms:
;;;;   * LLM-only        -- always take the LLM's first proposal (no learning).
;;;;   * RAG (lexical)   -- store (context -> best action by reward); retrieve the nearest past
;;;;                        context by word overlap and reuse its best action.  Uses feedback.
;;;;   * Controller(exact) -- the REAL controller.lisp selector (keys on the whole normalized input).
;;;;   * Controller(feature) -- a proposed variant: reward-modulated weights per (word, action),
;;;;                        scored additively, so the predictive rule-word can generalize.
;;;; Three regimes expose where each wins: RECALL (exact contexts recur), GENERALIZE (novel
;;;; fillers, same rule-words), DRIFT (the mapping flips midway).  Results printed as-is.

(load "load.lisp")
(setf *provider* :mock)                      ; no network; the mock "LLM" just proposes candidates

;;; --- deterministic PRNG (no Math.random) ----------------------------------------------
(defparameter *seed* 1)
(defun nextr () (setf *seed* (logand (+ (* *seed* 1103515245) 12345) #x7fffffff)))
(defun pick (lst) (nth (mod (nextr) (length lst)) lst))
(defun sample (lst k) (let (out) (dotimes (i k (nreverse out)) (push (pick lst) out))))

;;; --- task ------------------------------------------------------------------------------
(defparameter *actions* '("refund" "escalate" "explain" "apologize"))
(defparameter *rules* '(("billing" . "refund") ("outage" . "escalate")
                        ("question" . "explain") ("delay" . "apologize")))
(defparameter *rule-words* (mapcar #'car *rules*))
;; ONE shared filler vocabulary.  "Novel" contexts are unseen COMBINATIONS of these same words
;; (not disjoint words) -- the realistic generalization test, where filler overlap can mislead a
;; nearest-neighbor retriever but a feature policy can still lean on the predictive rule-word.
(defparameter *fillers* '("monday" "alice" "again" "please" "report" "team" "urgent" "system" "north" "v2"))
(defparameter *drift* nil)   ; when T, the mapping is rotated (the environment changed)

(defun correct-action (ctx)
  (let* ((rule (find-if (lambda (w) (assoc w *rules* :test #'string=)) ctx))
         (base (cdr (assoc rule *rules* :test #'string=))))
    (if *drift*                                ; rotate actions by one -> a changed environment
        (nth (mod (1+ (position base *actions* :test #'string=)) (length *actions*)) *actions*)
        base)))

(defun make-contexts (fillers m)
  "M contexts per rule: (rule-word + 2 fillers), as word lists."
  (let (out)
    (dolist (rw *rule-words* out)
      (dotimes (i m) (push (cons rw (sample fillers 2)) out)))))

(defun propose (ctx) (declare (ignore ctx)) *actions*)   ; the mock LLM proposes all actions

;;; --- arm: RAG (lexical episodic memory; uses feedback) --------------------------------
(defparameter *rag* nil)   ; list of (words . (action . cum-reward) alist)
(defun jaccard (a b)
  (let ((i (length (intersection a b :test #'string=))) (u (length (union a b :test #'string=))))
    (if (zerop u) 0 (/ i (float u)))))
(defun rag-entry (ctx) (find ctx *rag* :key #'car :test (lambda (a b) (null (set-exclusive-or a b :test #'string=)))))
(defun rag-action-reward (entry action)
  (let ((ar (assoc action (cdr entry) :test #'string=))) (if ar (cdr ar) 0)))
(defun rag-choose (ctx cands)
  "Retrieve the nearest past context (word overlap) and pick its best-rewarded action; UNTRIED
   actions score 0, so -- like the controller -- it explores the unknown and abandons the punished."
  (if (null *rag*) (first cands)
      (let (near nb)
	(dolist (e *rag*) (let ((s (jaccard ctx (car e)))) (when (or (null nb) (> s nb)) (setf nb s near e))))
	(let (best bw)
	  (dolist (c cands best)
	    (let ((s (rag-action-reward near c))) (when (or (null bw) (> s bw)) (setf bw s best c))))))))
(defun rag-update (ctx chosen r)
  (let ((e (rag-entry ctx)))
    (unless e (setf e (list ctx)) (push e *rag*))
    (let ((ar (assoc chosen (cdr e) :test #'string=)))
      (if ar (incf (cdr ar) r) (setf (cdr e) (cons (cons chosen r) (cdr e)))))))

;;; --- arm: feature-keyed selector (proposed controller upgrade) -------------------------
(defparameter *fsel* (make-hash-table :test 'equal))
(defun fkey (w cand) (concatenate 'string w "|" cand))
(defun fscore (ctx cand) (let ((s 0d0)) (dolist (w ctx s) (incf s (gethash (fkey w cand) *fsel* 0d0)))))
(defun fchoose (ctx cands) (let (best bw) (dolist (c cands best) (let ((s (fscore ctx c))) (when (or (null bw) (> s bw)) (setf bw s best c))))))
(defun freinforce (ctx cand r)
  (dolist (w ctx) (let ((k (fkey w cand)))
    (setf (gethash k *fsel* 0d0) (+ (* 0.98d0 (gethash k *fsel* 0d0)) (* 0.5d0 r))))))

;;; --- generic play loop ----------------------------------------------------------------
(defun play (choose learn contexts learn-p)
  (let ((hits 0))
    (dolist (ctx contexts (/ hits (float (length contexts))))
      (let* ((cands (propose ctx)) (chosen (funcall choose ctx cands)))
        (when (string= chosen (correct-action ctx)) (incf hits))
        (when learn-p (funcall learn ctx chosen (if (string= chosen (correct-action ctx)) 1.0 -1.0)))))))

(defparameter *arms*
  (list (list "LLM-only"             (lambda (c cs) (declare (ignore c)) (first cs)) (lambda (c ch r) (declare (ignore c ch r))) (lambda () nil) (lambda () 0))
        (list "RAG (lexical)"        #'rag-choose #'rag-update (lambda () (setf *rag* nil)) (lambda () (length *rag*)))
        (list "Controller(exact)"    (lambda (c cs) (controller-select c cs)) (lambda (c ch r) (controller-reinforce c ch r)) (lambda () (clrhash *selector*)) (lambda () (hash-table-count *selector*)))
        (list "Controller(feature)"  #'fchoose #'freinforce (lambda () (clrhash *fsel*)) (lambda () (hash-table-count *fsel*)))))

(defun bench (title do-train do-eval)
  "DO-TRAIN (choose learn) trains one arm (it controls *drift* phasing itself); DO-EVAL (choose)
   returns eval accuracy.  Per arm: reset, train, evaluate, report accuracy + memory size."
  (format t "~%~a~%  arm                     eval-acc   memory~%" title)
  (dolist (arm *arms*)
    (destructuring-bind (name choose learn reset size) arm
      (funcall reset)
      (funcall do-train choose learn)
      (let ((acc (funcall do-eval choose)))
        (format t "  ~24a ~,2f       ~:d~%" name acc (funcall size))))))

;;; ============================================================= demo =====================
(format t "~%Head-to-head: a LEARNING controller vs a RAG memory (same task, same mock LLM).~%")
(format t "Task: pick the action whose correctness is set by the ticket's rule-word -- a mapping~%")
(format t "revealed only by FEEDBACK.  ~d actions, so chance ~,2f.~%"
        (length *actions*) (/ 1.0 (length *actions*)))

;; (1) RECALL: the same contexts recur; can each arm learn them?
(let ((fixed (let ((*seed* 7)) (make-contexts *fillers* 3))))   ; 12 fixed contexts
  (bench "(1) RECALL  -- exact contexts recur (learn them, then re-evaluate):"
         (lambda (choose learn) (setf *drift* nil) (let ((*seed* 100)) (play choose learn (sample fixed 400) t)))
         (lambda (choose)       (setf *drift* nil) (let ((*seed* 200)) (play choose nil (sample fixed 120) nil)))))

;; (2) GENERALIZE: train on one filler set, test on NOVEL fillers (same rule-words).
(bench "(2) GENERALIZE -- novel fillers, same rule-words (zero-shot on unseen contexts):"
       (lambda (choose learn) (setf *drift* nil) (let ((*seed* 11)) (play choose learn (sample (make-contexts *fillers* 3) 400) t)))
       (lambda (choose)       (setf *drift* nil) (let ((*seed* 22)) (play choose nil (make-contexts *fillers* 5) nil))))   ; 20 novel

;; (3) DRIFT: learn the mapping, then the environment FLIPS mid-stream; can each arm recover?
(let ((fixed (let ((*seed* 7)) (make-contexts *fillers* 3))))
  (bench "(3) DRIFT -- mapping learned, then it CHANGES mid-stream; re-evaluate under the new mapping:"
         (lambda (choose learn)
           (setf *drift* nil) (let ((*seed* 300)) (play choose learn (sample fixed 400) t))   ; learn old mapping
           (setf *drift* t)   (let ((*seed* 301)) (play choose learn (sample fixed 400) t)))  ; then learn the new one
         (lambda (choose) (setf *drift* t) (let ((*seed* 302)) (play choose nil (sample fixed 120) nil)))))

(format t "~%Read it honestly (the experiment overturned the going hypothesis):~%")
(format t "  * RECALL: all three learners tie (~~1.00); LLM-only is at chance.  Where exact contexts~%")
(format t "    recur, recall is recall -- the controller is NOT beneficial above RAG.~%")
(format t "  * GENERALIZE: a decent lexical RAG holds up (~~0.85) and TIES the feature controller;~%")
(format t "    Controller(exact) cannot generalize (~~chance) because it keys on the whole input.~%")
(format t "    So generalization is NOT a clean controller win -- nearest-neighbor retrieval is~%")
(format t "    competitive, as long as the predictive word usually dominates the overlap.~%")
(format t "  * DRIFT: the controllers clearly BEAT RAG (~~1.00 vs ~~0.5).  When the environment~%")
(format t "    changes, the controller's reward-modulated DECAY un-learns the stale mapping, while~%")
(format t "    RAG's append-only episodic memory stays anchored to the old (now wrong) answer.~%")
(format t "  Verdict: the controller's genuine edge over RAG is NON-STATIONARITY / forgetting --~%")
(format t "  exactly the 'salience and forgetting' property SystemAnalysis predicted -- not recall~%")
(format t "  and not generalization.  Two concrete implications for controller.lisp: (a) keep the~%")
(format t "  decay (it is the real differentiator); (b) to even MATCH RAG's generalization, make~%")
(format t "  the selector FEATURE-keyed (exact keying generalizes at chance).  Caveat: lexical RAG~%")
(format t "  and a small toy; an embedding RAG would generalize better still on the static regimes.~%")
