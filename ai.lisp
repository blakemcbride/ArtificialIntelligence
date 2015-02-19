
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

(defun getword ()
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

