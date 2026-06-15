
(defpackage "input"
  (:use "COMMON-LISP")
  (:export "BUILD-STRUCTURE"
	   "CONNECT"
	   "CONNECTS-TO"
	   "FIND-CONNECTING-NEURON"))

(in-package "input")
(provide "input")

(require "data-structures")
(use-package "data-structures")

(require "line-input")
(use-package "line-input")   ; for the `add' macro used by build-structure

;;; The input algorithm, extracted from ai.lisp (Phase 3) so that both the top-level
;;; `main' loop and the `processing' component can call build-structure without a
;;; circular dependency.  See CLAUDE.md / Plan.md for how build-structure turns a
;;; sentence into a network covering the full sequence and every order-preserving
;;; subset, reusing existing nodes across sentences via find-connecting-neuron.

(defun connect (prior-neuron next-neuron &optional is-extender)
  "Connect two neurons with a dendrite, return next-neuron"
  (setf (neuron-axon prior-neuron)
	(cons (make-dendrite :neuron next-neuron
				     :kind (if is-extender :extender :sequence))
	      (neuron-axon prior-neuron)))
  (if is-extender
       (setf (neuron-extender prior-neuron) next-neuron))
  next-neuron)

(defmacro new-next-neuron (neuron &optional is-extender)
  "Create a new neuron to follow 'neuron' and return it"
  `(connect ,neuron (make-neuron) ,is-extender))

(defmacro get-extender (neuron)
  "Re-use an existing extender neuron to follow 'neuron' or create a new one if necessary.
   Return the extender neuron."
  `(or (neuron-extender ,neuron)
       (new-next-neuron ,neuron t)))

(defun connects-to (a b)
  "Does neuron a directly connect to neuron b through a structural edge?
   :association edges are skipped -- they are cross-component links (Phase 2+),
   not part of the input sequence network."
  (dolist (den (neuron-axon a))
    (if (and (not (eq :association (dendrite-kind den)))
	     (eq b (dendrite-neuron den)))
	(return t))))

(defun find-connecting-neuron (a b)
  "Find a neuron that both neuron a and b already connect to (structural edges only;
   :association edges are ignored)."
  (let ((extender (neuron-extender a)))
    (dolist (den (neuron-axon a))
      (let ((n (dendrite-neuron den)))
	(if (and (not (eq n extender))
		 (not (eq :association (dendrite-kind den)))
		 (connects-to b n))
	    (return n))))))

(defun build-structure (inp)
  "This takes our input list and generates 'every possible combination' into our net.
   Returns the final `active' frontier -- the candidate meaning neurons."
  (let (active)
    (dolist (neuron inp)
      (let ((next-active (cons (get-extender neuron) nil)))
	(dolist (pn active)
	  (add next-active (get-extender pn))
	  (add next-active (or (find-connecting-neuron pn neuron)
			       (connect pn (new-next-neuron neuron)))))
	(setq active next-active)))
    active))
