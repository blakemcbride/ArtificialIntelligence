
(defpackage "data-structures"
  (:use "COMMON-LISP")
  (:export "*DICTIONARY*"
	   "*NEXT-NEURON-ID*"
	   "RESET"
	   "NEURON"
	   "NEURON-AXON"
	   "NEURON-EXTENDER"
	   "NAMED-NEURON"
	   "NAMED-NEURON-NAME"
	   "MAKE-NEURON"
	   "MAKE-NAMED-NEURON"
	   "DENTRIDE"
	   "MAKE-DENDRITE"
	   "DUMP-DICTIONARY"))

(in-package "data-structures")
(provide "data-structures")

(defparameter *dictionary* (make-hash-table :test 'equal))

(defparameter *next-neuron-id* 0)

(defun reset ()
  (setq *dictionary* (make-hash-table :test 'equal))
  (setq *next-neuron-id* 0))

(defstruct neuron
  "An unnamed neuron"
  (threshold 0 :type fixnum)
  (current-value 0 :type fixnum)
  (id (incf *next-neuron-id*) :type fixnum :read-only) ; only used for debugging purposes
  (extender nil) ; a particular neuron that acts as an extender to this neuron
  (axon nil)) ; dendrite - linked list of weighted output neurons (including the extender)

(defstruct (named-neuron (:include neuron))
  "An named neuron - just adds name to a regular neuron"
  (name "" :type string))

(defstruct dendrite
  "A Dendrite"
  (neuron nil)
  (weight 0.0 :type single-float))

(defun dump-neuron (n level)
  (dotimes (var level)
    (princ "    "))
  (if (named-neuron-p n)
      (if (neuron-extender n)
	  (format t "~d (~a) (E-~d)~%"
		  (neuron-id n)
		  (named-neuron-name n)
		  (neuron-id (neuron-extender n)))
	  (format t "~d (~a)~%"
		  (neuron-id n)
		  (named-neuron-name n)))
      (if (neuron-extender n)
	  (format t "~d (E-~d)~%"
		  (neuron-id n)
		  (neuron-id (neuron-extender n)))
	  (format t "~d~%"
		  (neuron-id n))))
  (dolist (den (neuron-axon n))
    (dump-neuron (dendrite-neuron den) (1+ level))))

(defun dump-dictionary ()
  (loop for value being the hash-values of *dictionary*
     do (dump-neuron value 0)))

