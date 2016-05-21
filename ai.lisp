
; Common Lisp version started on 2/16/15 by Blake McBride


(require "data-structures")
(use-package "data-structures")

(require "line-input")
(use-package "line-input")

(defmacro add (var cell)
  "Add cell to the beginning of the list contained in variable var"
  `(setq ,var (cons ,cell ,var)))

(defun main ()
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

(defun build-structure (inp)
  "This takes our input list and generates 'every possible combination' into our net"
  (let (active)
    (dolist (neuron inp)
      (let ((next-active (cons (new-next-neuron neuron t) nil)))
	(dolist (pn active)
	  (add next-active (new-next-neuron pn t))
	  (add next-active (connect pn (new-next-neuron neuron))))
	(setq active next-active)))))

