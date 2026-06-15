
; Common Lisp version started on 2/16/15 by Blake McBride

;; Top-level entry point.  The input algorithm now lives in input.lisp; this file
;; only holds `main' and pulls the whole system together.  It has no package of its
;; own, so its symbols (main) land in the current package (normally COMMON-LISP-USER)
;; and the components' exported symbols are imported here via use-package -- call
;; (main), not (ai::main).

(require "data-structures")
;; SBCL's COMMON-LISP-USER inherits SB-PROFILE:RESET, which would make the
;; use-package below signal a name conflict.  Shadow our own RESET first.  We look
;; the symbol up with find-symbol because the package name is a lowercase string
;; ("data-structures") that the reader would otherwise upcase in a pkg:sym token.
(shadowing-import (find-symbol "RESET" "data-structures"))
(use-package "data-structures")

(require "line-input")
(use-package "line-input")

(require "input")
(use-package "input")

(require "output")
(use-package "output")

(require "concepts")
(use-package "concepts")

(require "processing")
(use-package "processing")

(require "persist")
(use-package "persist")

(defparameter *confirm-words* '("yes" "y" "right" "correct" "ok")
  "Single-word teacher lines that confirm a correct guess instead of giving a new answer.")

(defun words-of (neurons)
  "The word strings of a create-line result, in order."
  (mapcar #'named-neuron-name neurons))

(defun quit-line-p (words)
  "Is WORDS a lone `quit' or `exit'?"
  (and (consp words) (null (cdr words))
       (or (string= "quit" (car words)) (string= "exit" (car words)))))

(defun confirm-p (words)
  "Is WORDS a single confirmation token (accept the last guess)?"
  (and (consp words) (null (cdr words))
       (member (car words) *confirm-words* :test #'string=)))

(defun main ()
  "Interactive continual-learning teaching loop.  Each turn:
     1. read an input sentence (terminated by . ! or ?),
     2. print what the system would answer (or \"I don't know\"),
     3. read the teacher's line -- the correct response, or a confirm token
        (one of *confirm-words*) to accept a correct guess,
     4. learn from it (reinforce a correct guess / teach or correct otherwise).
   Stop with `quit.' / `exit.' (or end-of-input).  Loads saved memory on entry (if any),
   saves it on exit, and prints the input neuron tree."
  (if (load-network)
      (format t "~&(loaded saved memory from ~a)~%" *save-file*)
      (reset))
  (format t "~&Teaching loop -- type a sentence, then on the next line the correct~%")
  (format t "response (or ~{~a~^/~} to confirm a correct guess).  Stop with quit.~%~%"
	  *confirm-words*)
  (loop
     (format t "~&input> ")
     (force-output)
     (let ((in (words-of (create-line))))
       (when (or (null in) (quit-line-p in))
	 (return))
       (let ((guess (respond in)))
	 (format t "  guess: ~a~%"
		 (if guess (format nil "~{~a~^ ~}" guess) "(I don't know)"))
	 (format t "teach> ")
	 (force-output)
	 (let ((teacher (words-of (create-line))))
	   (cond ((null teacher)
		  (return))
		 ((confirm-p teacher)
		  (cond (guess
			 (learn in guess)
			 (format t "  (reinforced)~%"))
			(t
			 (format t "  (nothing to confirm -- give the answer)~%"))))
		 (t
		  (learn in teacher)
		  (format t "  (learned)~%")))))))
  (save-network)
  (format t "~&(memory saved to ~a)~%" *save-file*)
  (terpri)
  (dump-dictionary))

;;; --- Training-set import -------------------------------------------------------
;;; A training set is a text file of one relationship per line:
;;;     input phrase => answer phrase
;;; Blank lines and lines beginning with # or ; are ignored.  Each line is learned
;;; exactly as a teaching-loop turn would be (associations + concept graph).

(defun find-substring (needle haystack)
  "Index of the first occurrence of NEEDLE in HAYSTACK, or NIL."
  (let ((nl (length needle)) (hl (length haystack)))
    (loop for i from 0 to (- hl nl)
	  when (string= needle haystack :start2 i :end2 (+ i nl)) return i)))

(defun train-from-file (path &key (separator "=>") (verbose t))
  "Learn from a training file: each non-blank, non-comment (# or ;) line is
   `input phrase SEPARATOR answer phrase' (default SEPARATOR \"=>\").  Each pair is
   learned via LEARN (so it grows associations and the concept graph).  Returns the
   number of relationships learned."
  (with-open-file (s path :direction :input)
    (let ((count 0))
      (loop for line = (read-line s nil nil)
	    while line
	    do (let ((trimmed (string-left-trim '(#\Space #\Tab) line)))
		 (unless (or (zerop (length trimmed))
			     (member (char trimmed 0) '(#\# #\;)))
		   (let ((sep (find-substring separator line)))
		     (when sep
		       (let ((in  (tokenize (subseq line 0 sep)))
			     (out (tokenize (subseq line (+ sep (length separator))))))
			 (when (and in out)
			   (learn in out)
			   (incf count))))))))
      (when verbose (format t "~&trained on ~d relationships from ~a~%" count path))
      count)))
