
; Common Lisp version started on 2/16/15 by Blake McBride

(defparameter *dictionary* (make-hash-table :test 'equal))

(defstruct neuron
  "An unnamed neuron"
  (threshold 0 :type fixnum)
  (current-value 0 :type fixnum)
  (id 0 :type fixnum) ; only used for debugging purposes
  (axon nil))

(defstruct (named-neuron (:include neuron))
  "An named neuron - just adds name to a regular neuron"
  (name "" :type string))

(defstruct dendrite
  "A Dendrite"
  (neuron nil)
  (weight 0 :type single-float))

(defun create-line ()
  (let (word neuron (res nil))
    (loop
       (setq word (getword))
       (cond ((not word)
	      (return nil))
	     ((eq word 'eol)
	      (return (nreverse res))))
       (setq neuron (gethash word *dictionary*))
       (cond ((null neuron)
	      (setq neuron (make-named-neuron :name word))
	      (setf (gethash word *dictionary*) neuron)))
       (setq res (cons neuron res)))))
       


(defun abc ()
  (let (x)
    (princ "prompt: ")
    (finish-output)
    (setq x (read-line))
    (princ "XXXXX")
    (terpri)
    (princ x)
    (terpri)
    (princ "MMMMM")
    (terpri)))



(abc)

