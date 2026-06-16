
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

(defun isspace (c)
  (or (eql c #\space)
      (eql c #\tab)
      (eql c #\return)
      (eql c #\linefeed)))

(defun eol (c)
  "Is C one of the sentence terminators (. ! ?), regardless of context?"
  (or (eql c #\.)
      (eql c #\!)
      (eql c #\?)))

(defun terminator-p (line pos)
  "Does the character at POS in LINE end a sentence?  ! and ? always do; a period does
   only at the end of the line or when followed by whitespace.  This way an embedded
   period -- as in a filename like file.kb, or a decimal like 3.14 -- stays inside its
   word instead of splitting it."
  (let ((c (char line pos)))
    (cond ((or (eql c #\!) (eql c #\?)) t)
	  ((eql c #\.)
	   (or (>= (1+ pos) (length line))            ; period at end of line
	       (isspace (char line (1+ pos)))))       ; period before a space
	  (t nil))))

(defun string-eol (s)
  "Is S a lone terminator token (a single . ! or ?) -- i.e. the end of a sentence?"
  (and (eql 1 (length s))
       (eol (char s 0))))

(defun tokenize (string)
  "Split a sentence STRING into lowercase word strings, the way create-line tokenizes a
   line of input: whitespace and the sentence terminators are separators (and are dropped).
   A period only separates at end-of-string or before whitespace, so embedded periods are
   kept (\"file.kb\" stays one token).  E.g. \"Do horses walk?\" -> (\"do\" \"horses\" \"walk\")."
  (let ((res '()) (start nil) (n (length string)))
    (flet ((boundary-at (i)
	     (or (isspace (char string i))
		 (terminator-p string i))))
      (dotimes (i n)
	(if (boundary-at i)
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

(defun getword ()
  "Get and return a single word from the input stream as a string, or NIL at end of input.
   A sentence terminator comes back as its own one-character word; a period that is not at
   end-of-line and not followed by a space stays inside the surrounding word (see
   terminator-p), so e.g. \"file.kb\" reads as one word."
					; if the buffer is exhausted, read the next non-blank line
  (loop while (>= *current-position* (length *input-line*)) do
       (setq *current-position* 0)
       (setq *input-line* (string-downcase (or (read-line nil nil)
					       (return-from getword nil))))
       (loop while (and (< *current-position* (length *input-line*))
			(isspace (char *input-line* *current-position*))) do
	    (incf *current-position*)))
					; *current-position* now sits on a non-space character
  (let ((start *current-position*))
    (cond
      ((terminator-p *input-line* *current-position*)   ; a lone terminator token
       (incf *current-position*))
      (t					        ; gather to the next space / terminator
       (incf *current-position*)
       (loop while (and (< *current-position* (length *input-line*))
			(not (isspace (char *input-line* *current-position*)))
			(not (terminator-p *input-line* *current-position*))) do
	    (incf *current-position*))))
    (prog1 (subseq *input-line* start *current-position*)
					; skip remainder of spaces for next time
      (loop while (and (< *current-position* (length *input-line*))
		       (isspace (char *input-line* *current-position*))) do
	   (incf *current-position*)))))

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
