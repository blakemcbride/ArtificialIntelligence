
(defpackage "line-input"
  (:use "COMMON-LISP")
  (:export "CREATE-LINE" "ADD" "INTERN-WORD" "INTERN-WORDS" "TOKENIZE" "AS-WORDS"))
(in-package "line-input")
(provide "line-input")

(require "data-structures")
(use-package "data-structures")

(defparameter *input-line* "")
(defparameter *current-position* 0)

(defmacro add (var cell)
  "Add cell to the beginning of the list contained in variable var"
  `(setq ,var (cons ,cell ,var)))

(defun intern-word (word)
  "Return the named-neuron for WORD, creating and interning it in *dictionary* if new."
  (or (gethash word *dictionary*)
      (setf (gethash word *dictionary*) (make-named-neuron :name word))))

(defun intern-words (words)
  "Intern each string in WORDS into *dictionary*; return the named-neurons in order.
   The non-stdin counterpart to create-line, used by inference (processing:respond)."
  (mapcar #'intern-word words))

(defun tokenize (string)
  "Split a sentence STRING into lowercase word strings, the way create-line tokenizes a
   line of input: whitespace and the terminators . ! ? are separators (the terminators
   themselves are dropped).  E.g. \"Do horses walk?\" -> (\"do\" \"horses\" \"walk\")."
  (let ((res '()) (start nil) (n (length string)))
    (flet ((boundary-p (c)
	     (or (eql c #\space) (eql c #\tab) (eql c #\return) (eql c #\linefeed)
		 (eql c #\.) (eql c #\!) (eql c #\?))))
      (dotimes (i n)
	(if (boundary-p (char string i))
	    (when start
	      (push (string-downcase (subseq string start i)) res)
	      (setf start nil))
	    (unless start (setf start i))))
      (when start (push (string-downcase (subseq string start n)) res)))
    (nreverse res)))

(defun as-words (x)
  "Coerce X to a list of word strings: tokenize a sentence string, or pass a word list
   (or nil) through unchanged.  This lets the public API accept either form."
  (if (stringp x) (tokenize x) x))

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
  "Get and return a single word from the input stream as a string, or NIL at end of input"
					; if no words, read a non-blank line
  (loop while (>= *current-position* (length *input-line*)) do
       (setq *current-position* 0)
       (setq *input-line* (string-downcase (or (read-line nil nil)
						(return-from getword nil))))
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

(defun create-line ()
  "Returns a list of named neurons"
  (setq *input-line* "")
  (setq *current-position* 0)
  (let (res)
    (loop
       (let ((word (getword)))
	 (cond ((not word)
		(return nil))
	       ((string-eol word)
		(return (nreverse res))))
	 (add res (intern-word word))))))
