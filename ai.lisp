
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
     (let (inp)
       (setq inp (create-line))
       (and (and (consp inp)
		 (not (consp (cdr inp)))
		 (or (string= "quit" (named-neuron-name (car inp)))
		     (string= "exit" (named-neuron-name (car inp)))))
	    (return))
       (build-structure inp)))
  (dump-dictionary))

(defun connect (prior-neuron next-neuron)
  "Connect two neurons with a dendrite, return next-neuron"
  (let ((den (make-dendrite :neuron next-neuron)))
    (setf (neuron-axon prior-neuron)
	  (cons den (neuron-axon prior-neuron)))
    next-neuron))

(defmacro new-next-neuron (neuron)
  "Create a new neuron to follow 'neuron' and return it"
  `(connect ,neuron (make-neuron)))

(defun build-structure (inp)
  "This takes our input list and generates 'every possible combination' into our net"
  (let (active vn)
    (dolist (neuron inp)
      (let ((next-active (cons (new-next-neuron neuron) nil)))
	(dolist (pn active)
	  (add next-active (new-next-neuron pn))
	  (let ((nn (new-next-neuron neuron)))
	    (add next-active nn)
	    (connect pn nn)))
	(setq active next-active)))))

