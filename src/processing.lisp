
(defpackage "processing"
  (:use "COMMON-LISP")
  (:export "ASSOCIATE"
	   "FIND-ASSOCIATION"
	   "RESPOND"
	   "LEARN"
	   "WEAKEN"
	   "DECAY-ASSOCIATIONS"
	   "*ASSOC-INITIAL-WEIGHT*"
	   "*ASSOC-STRENGTHEN*"
	   "*ASSOC-MAX-WEIGHT*"
	   "*ASSOC-DECAY*"
	   "*ASSOC-PRUNE-THRESHOLD*"
	   "*THRESHOLD-RATE*"
	   "*THRESHOLD-FRACTION*"))

(in-package "processing")
(provide "processing")

(require "data-structures")
(use-package "data-structures")
(require "line-input")
(use-package "line-input")   ; intern-words
(require "input")
(use-package "input")        ; build-structure
(require "output")
(use-package "output")       ; produce-output
(require "concepts")
(use-package "concepts")     ; note-relationship (auto-populate the concept graph)
(require "attention")
(use-package "attention")    ; note-copy / copy-response (the fast-weight attention head)
(require "operations")
(use-package "operations")   ; run-operation / note-operation (learned operations over knowledge)
(require "vectors")
(use-package "vectors")      ; note-cooccurrence (distributed concept vectors, similarity by geometry)
(require "generation")
(use-package "generation")   ; respond-generation / note-sequence / note-fact-from-qa (Phase 8)

;;; The Processing component (Plan.md) bridges the input and output networks.  After
;;; an input sentence is built, build-structure returns a set of "meaning" neurons
;;; (the active frontier); a response is built into an output root (output.lisp).
;;; `associate' is the supervised wiring step -- it links every meaning neuron to the
;;; output root with an :association dendrite.  These are the only edges that later
;;; phases reinforce and decay; structural (:sequence / :extender) edges are permanent
;;; memory and are deliberately left untouched.
;;;
;;; Phase 2 implements only the wiring.  Inference (spreading activation to choose a
;;; root) is Phase 3; reward-modulated strengthening and decay are Phase 4.  The decay
;;; constants below live here so Phase 4 can use the same registry of associations.

(defparameter *assoc-initial-weight* 0.5 "w0: weight of a brand-new association.")
(defparameter *assoc-strengthen*     0.2 "eta: added to an existing association on re-training.")
(defparameter *assoc-max-weight*     1.0 "Soft cap so repeated training can't grow a weight without bound.")
(defparameter *assoc-decay*          0.02 "lambda: fraction of its weight each association loses per decay step.")
(defparameter *assoc-prune-threshold* 0.01 "epsilon: associations at or below this weight are pruned.")
(defparameter *threshold-rate*       0.3 "Homeostatic threshold EMA rate (how fast a root's threshold tracks its drive).")
(defparameter *threshold-fraction*   0.1 "Homeostatic threshold target as a fraction of a root's activation.")

(defun find-association (from to)
  "Return the :association dendrite FROM -> TO if one exists, else nil."
  (dolist (den (neuron-axon from))
    (when (and (eq :association (dendrite-kind den))
	       (eq to (dendrite-neuron den)))
      (return den))))

(defun associate-one (ending root)
  "Create the association ENDING -> ROOT (at *assoc-initial-weight*) or, if it already
   exists, strengthen it by *assoc-strengthen* (clamped to *assoc-max-weight*).
   New associations are registered in *associations* for later decay/pruning.
   Returns the dendrite."
  (let ((den (find-association ending root)))
    (cond (den
	   (setf (dendrite-weight den)
		 (min *assoc-max-weight*
		      (+ (dendrite-weight den) *assoc-strengthen*))))
	  (t
	   (setq den (make-dendrite :neuron root
				    :weight *assoc-initial-weight*
				    :kind :association
				    :from ending))
	   (push den (neuron-axon ending))
	   (push den *associations*)))
    den))

(defun associate (input-endings output-root)
  "Wire every (distinct) meaning neuron in INPUT-ENDINGS to OUTPUT-ROOT with an
   :association dendrite, creating or strengthening as needed.  This is the entire
   supervised learning step for one (input, output) pair.  Returns the dendrites."
  (mapcar (lambda (e) (associate-one e output-root))
	  (remove-duplicates input-endings)))

;;; --- Inference (Phase 3): given an input sentence, produce a response -----------
;;;
;;; Discrete spreading activation, one hop: fire the input's meaning neurons, push
;;; their association weights into the target output roots' current-value, then take
;;; the highest-scoring root above its threshold (winner-take-all).  An input that
;;; shares no meaning neurons with anything taught lands no activation anywhere, so
;;; the winner is nil -- "I don't know".  Partial overlap (a shared subset neuron)
;;; lands partial activation, giving similarity-based recall.  No weights change
;;; here; reward-modulated strengthening and decay are Phase 4.

(defun spread-activation (endings)
  "Zero every output root's current-value, then add each fired ENDING's association
   weights into its target roots.  Returns the roots that received any activation."
  (dolist (root *output-roots*)
    (setf (neuron-current-value root) 0.0))
  (let (touched)
    (dolist (e endings)
      (dolist (den (neuron-axon e))
	(when (eq :association (dendrite-kind den))
	  (let ((root (dendrite-neuron den)))
	    (incf (neuron-current-value root) (dendrite-weight den))
	    (pushnew root touched)))))
    touched))

(defun select-winner (roots)
  "Return the root in ROOTS with the highest current-value strictly above its own
   threshold (winner-take-all), or nil if none qualifies."
  (let (winner)
    (dolist (root roots winner)
      (when (and (> (neuron-current-value root) (neuron-threshold root))
		 (or (null winner)
		     (> (neuron-current-value root) (neuron-current-value winner))))
	(setq winner root)))))

(defun respond (input)
  "Given an input sentence (a string, e.g. \"Do cats purr?\", or a list of words), return
   the word list of the best response, or NIL (\"I don't know\").  A strongly-learned
   attention copy cue fires first (e.g. \"say X\" -> X, even for a novel X); otherwise the
   answer comes from spreading activation across the input->output associations; and if
   nothing is recalled, a learned template may compose one.  Secondary values are the
   winning root and its activation.  No weights change here."
  (let ((input-words (as-words input)))
    (if (generation-request-p input-words)
	;; A recognized generation request (tell me about X / why ...) is OWNED by generation:
	;; return its answer or NIL ("I don't know") -- never fall through to brittle recall.
	(values (respond-generation input-words) nil 0.0)
	(let ((copied (copy-response input-words))         ; 1. attention copy head (say X -> X)
	      (op (run-operation input-words)))            ; 0. learned operation over knowledge
	  (cond
	    (op (values op nil 0.0))
	    (copied (values copied nil 0.0))
	    (t (let ((composed (compose input-words)))      ; 2. compose from a learned template
		 (cond
		   (composed (values composed nil 0.0))
		   (t (let* ((endings (build-structure (intern-words input-words)))
			     (winner  (select-winner (spread-activation endings)))) ; 3. direct recall
			(if winner
			    (values (produce-output winner) winner (neuron-current-value winner))
			    (values nil nil 0.0))))))))))))

;;; --- Reinforcement & decay (Phase 4): the continual-learning dynamics ----------
;;;
;;; Reward-modulated Hebbian learning with weight decay (Plan.md S3.4):
;;;   dw = eta * pre * post * r   -   lambda * w
;;; `associate' is the positive co-activation (r>0): create or strengthen.  `weaken'
;;; is the negative update (r<0) for a wrong guess.  `decay-associations' applies the
;;; global -lambda*w term every step and prunes / garbage-collects what fades away.
;;; `learn' ties them together for one teacher-confirmed turn.

(defun prune-association (source den)
  "Remove association dendrite DEN (whose source is SOURCE) from the network."
  (when source
    (setf (neuron-axon source) (delete den (neuron-axon source))))
  (setf *associations* (delete den *associations*)))

(defun weaken (endings root)
  "Negative-reward update: reduce each ENDINGS->ROOT association by *assoc-strengthen*,
   pruning any that fall to/below *assoc-prune-threshold*.  Used when ROOT was a wrong
   guess.  Returns the surviving dendrites."
  (let (survivors)
    (dolist (e (remove-duplicates endings) survivors)
      (let ((den (find-association e root)))
	(when den
	  (decf (dendrite-weight den) *assoc-strengthen*)
	  (if (<= (dendrite-weight den) *assoc-prune-threshold*)
	      (prune-association e den)
	      (push den survivors)))))))

(defun forget-response (root)
  "Remove ROOT's entry (by value) from the *responses* reuse table."
  (let (dead-keys)
    (maphash (lambda (k v) (when (eq v root) (push k dead-keys))) *responses*)
    (dolist (k dead-keys) (remhash k *responses*))))

(defun gc-orphan-roots ()
  "Drop output roots that have no incoming associations left -- a fully-decayed
   response -- from *output-roots* and *responses*.  Their chains become unreachable
   and are reclaimed by the Lisp GC."
  (let ((live (remove-duplicates (mapcar #'dendrite-neuron *associations*))))
    (dolist (root *output-roots*)
      (unless (member root live)
	(forget-response root)))
    (setf *output-roots*
	  (remove-if-not (lambda (root) (member root live)) *output-roots*))))

(defun decay-associations ()
  "Global decay step: shrink every association weight by *assoc-decay*, prune any that
   reach *assoc-prune-threshold*, then garbage-collect orphaned roots.  Returns the
   number of associations pruned."
  (dolist (den *associations*)
    (setf (dendrite-weight den) (* (dendrite-weight den) (- 1.0 *assoc-decay*))))
  (let ((pruned 0))
    (dolist (den (copy-list *associations*))
      (when (<= (dendrite-weight den) *assoc-prune-threshold*)
	(prune-association (dendrite-from den) den)
	(incf pruned)))
    (gc-orphan-roots)
    pruned))

(defun adapt-threshold (root activation)
  "Homeostatically drift ROOT's firing threshold toward *threshold-fraction* of
   ACTIVATION (an EMA at *threshold-rate*) -- a small confidence floor that scales with
   the root's typical drive without suppressing partial (subset) recall."
  (setf (neuron-threshold root)
	(+ (* (- 1.0 *threshold-rate*) (neuron-threshold root))
	   (* *threshold-rate* *threshold-fraction* activation))))

(defun learn (input correct)
  "One teacher-confirmed turn of continual learning.  INPUT and CORRECT may each be a
   sentence string (e.g. \"Do cats purr?\" / \"yes\") or a list of words.  Build the
   input, see what the system WOULD answer, then apply the reward-modulated update toward
   CORRECT and a global decay step.  Returns the system's guess (word list or nil) BEFORE
   the update -- what it would have said.
     correct guess   -> reinforce that pathway (associate)
     wrong / unknown -> weaken the wrong pathway (if any) and teach the correct one
   Finally the correct root's threshold adapts and unused associations decay.
   If CORRECT is an operation answer (e.g. \"count animals\"), instead learn what the
   question MEANS (note-operation) and return early -- no literal answer is stored."
  (incf *facts-learned*)            ; count every fact learned, from any source
  (when (operation-answer-p (as-words correct))
    (let ((guess (respond (as-words input))))
      (note-operation (as-words input) (as-words correct))
      (return-from learn guess)))
  (let* ((input-words   (as-words input))
	 (correct-words (as-words correct))
	 (endings       (build-structure (intern-words input-words)))
	 (guess-root    (select-winner (spread-activation endings)))
	 (guess         (and guess-root (produce-output guess-root)))
	 (correct-root  (build-output-structure correct-words)))
    (unless (eq guess-root correct-root)
      (when guess-root (weaken endings guess-root)))
    (associate endings correct-root)
    (spread-activation endings)
    (adapt-threshold correct-root (neuron-current-value correct-root))
    (decay-associations)
    (note-relationship input-words correct-words)   ; also grow the concept graph (Phase 7)
    (note-copy input-words correct-words)           ; also learn copy cues (attention head)
    (note-template input-words correct-words)       ; also learn response templates (composition)
    (note-cooccurrence input-words correct-words)    ; also grow the distributed concept vectors (polarity-bound)
    (note-sequence input-words)                      ; also grow the generation transition model (Phase 8)
    (note-sequence correct-words)
    (note-fact-from-qa input-words correct-words)    ; also derive a declarative fact for describe
    guess))
