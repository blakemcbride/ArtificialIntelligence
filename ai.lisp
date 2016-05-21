
; Common Lisp version started on 2/16/15 by Blake McBride


(require "data-structures")
(use-package "data-structures")

(require "line-input")
(use-package "line-input")

(defun main ()
  (reset)
  (loop
     (let ((inp (create-line)))
       (and (and (consp inp)
		 (not (consp (cdr inp)))
		 (or (string= "quit" (named-neuron-name (car inp)))
		     (string= "exit" (named-neuron-name (car inp)))))
	    (return))
       (build-structure inp)))
  (dump-dictionary))

(defun connect (prior-neuron next-neuron &optional is-extender)
  "Connect two neurons with a dendrite, return next-neuron"
  (setf (neuron-axon prior-neuron)
	(cons (make-dendrite :neuron next-neuron)
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
  "Does neuron a directly connect to neuron b?"
  (dolist (den (neuron-axon a))
    (if (eq b (dendrite-neuron den))
	(return t))))

(defun find-connecting-neuron (a b)
  "Find a neuron that both neuron a and b already connect to."
  (let ((extender (neuron-extender a)))
    (dolist (den (neuron-axon a))
      (let ((n (dendrite-neuron den)))
	(if (and (not (eq n extender))
		 (connects-to b n))
	    (return n))))))

(defun build-structure (inp)
  "This takes our input list and generates 'every possible combination' into our net"
  (let (active)
    (dolist (neuron inp)
      (let ((next-active (cons (get-extender neuron) nil)))
	(dolist (pn active)
	  (add next-active (get-extender pn))
	  (add next-active (or (find-connecting-neuron pn neuron)
			       (connect pn (new-next-neuron neuron)))))
	(setq active next-active)))))

