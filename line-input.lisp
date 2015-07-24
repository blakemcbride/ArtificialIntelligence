
(defpackage "line-input"
  (:use "COMMON-LISP")
  (:export "CREATE-LINE"))
(in-package "line-input")
(provide "line-input")

(require "data-structures")
(use-package "data-structures")

(defmacro add (var cell)
  "Add cell to the beginning of the list contained in variable var"
  `(setq ,var (cons ,cell ,var)))

(defun create-line ()
  "Returns a list of named neurons"
  (let (res)
    (loop
       (let ((word (getword)))
	 (cond ((not word)
		(return nil))
	       ((string-eol word)
		(return (nreverse res))))
	 (let ((neuron (gethash word *dictionary*)))
	   (cond ((null neuron)
		  (setq neuron (make-named-neuron :name word))
		  (setf (gethash word *dictionary*) neuron)))
	   (add res neuron))))))

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
		(add res (char *input-line* *current-position*))
		(incf *current-position*))))
					; skip remainder of spaces for next time
    (loop while (and (< *current-position* (length *input-line*))
		     (isspace (char *input-line* *current-position*))) do
	 (incf *current-position*))
    (coerce (nreverse res) 'string)))

  
