
; Common Lisp version started on 2/16/15 by Blake McBride

(defparameter *dictionary* (make-hash-table :test 'equal))

(defparameter *next-neuron-id* 0)

(defstruct neuron
  "An unnamed neuron"
  (threshold 0 :type fixnum)
  (current-value 0 :type fixnum)
  (id (incf *next-neuron-id*) :type fixnum :read-only) ; only used for debugging purposes
  (axon nil))

(defstruct (named-neuron (:include neuron))
  "An named neuron - just adds name to a regular neuron"
  (name "" :type string))

(defstruct dendrite
  "A Dendrite"
  (neuron nil)
  (weight 0.0 :type single-float))

(defun create-line ()
  "Returns a list of named neurons"
  (let (word neuron (res nil))
    (loop
       (setq word (getword))
       (cond ((not word)
	      (return nil))
	     ((string-eol word)
	      (return (nreverse res))))
       (setq neuron (gethash word *dictionary*))
       (cond ((null neuron)
	      (setq neuron (make-named-neuron :name word))
	      (setf (gethash word *dictionary*) neuron)))
       (setq res (cons neuron res)))))

(defparameter *input-line* "")
(defparameter *current-position* 0)

(defun isspace (c)
  (or (eql c #\space)
      (eql c #\tab)
      (eql c #\return)
      (eql c #\linefeed)))

(defun eol (c)
  (or (eql c #\.)
      (eql c #\!)
      (eql c #\?)))

(defun string-eol (s)
  (and (eql 1 (length s))
       (eol (char s 0))))

(defun getword ()
  "Get and return a single word from the input stream, return as a string"
					; if no words, read a non-blank line
  (loop while (>= *current-position* (length *input-line*)) do
       (setq *current-position* 0)
       (setq *input-line* (string-downcase (read-line)))
       (loop while (and (< *current-position* (length *input-line*))
			(isspace (char *input-line* *current-position*))) do
	    (incf *current-position*)))
					; get all non-space characters
  (let* ((c (char *input-line* *current-position*))
	 (res (list c)))
    (incf *current-position*)
    (cond ((not (eol c))
	   (loop while (and (< *current-position* (length *input-line*))
			    (not (isspace (char *input-line* *current-position*)))
			    (not (eol (char *input-line* *current-position*)))) do
		(setq res (cons (char *input-line* *current-position*) res))
		(incf *current-position*))))
					; skip remainder of spaces for next time
    (loop while (and (< *current-position* (length *input-line*))
		     (isspace (char *input-line* *current-position*))) do
	 (incf *current-position*))
    (coerce (nreverse res) 'string)))

  

(defun main ()
  (loop
     (let (inp)
       (setq inp (create-line))
       (and (and (consp inp)
		 (not (consp (cdr inp)))
		 (string= "quit" (named-neuron-name (car inp))))
	    (return))
       (build-structure inp)))
  (dump-dictionary))

(defun connect (prior-neuron next-neuron)
  "Connect two neurons with a dendrite, return next-neuron"
  (let ((den (make-dendrite :neuron next-neuron)))
    (setf (neuron-axon prior-neuron)
	  (cons den (neuron-axon prior-neuron)))
    next-neuron))

(defun new-next-neuron (neuron)
  "Create a new neuron to follow 'neuron' and return it"
  (connect neuron (make-neuron)))


(defun build-structure (inp)
  "This takes our input list and generates 'every possible combination' into our net"
  (let (active vn)
    (dolist (neuron inp)
	 (let ((next-active (cons (new-next-neuron neuron) nil)))
	   (setq vn nil) ; it would be better if vn were a fixed size array
	   ; create extension neurons (ones that represent prior activity) and load up the neuron list (vn)
	   (dolist (pn active)
		(setq next-active (cons (new-next-neuron pn) next-active))
		(setq vn (cons pn vn)))

	   (dolist (pn vn)
		(let ((nn (new-next-neuron neuron)))
		  (setq next-active (cons nn next-active))
		  (connect pn nn)))

	   (setq active next-active)))))

(defun dump-neuron (n level)
  (dotimes (var level)
    (princ "    "))
  (if (named-neuron-p n)
      (format t "~d (~a)~%"
	      (neuron-id n)
	      (named-neuron-name n))
      (format t "~d~%"
	     (neuron-id n)))
  (dolist (den (neuron-axon n))
    (dump-neuron (dendrite-neuron den) (1+ level))))
  

(defun dump-dictionary ()
  (loop for value being the hash-values of *dictionary*
       do (dump-neuron value 0)))

