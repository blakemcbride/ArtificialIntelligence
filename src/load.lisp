;;;; One-call loader for the whole system.
;;;; From a REPL whose working directory is src/:
;;;;
;;;;     (load "load.lisp")    ; loads everything, and defines (load-system)
;;;;     (load-system)         ; reload after editing a source file
;;;;
;;;; Each component file calls (provide ...), so the internal (require ...) forms become
;;;; no-ops; loading by pathname here works on every implementation (SBCL/CLISP/CCL/ECL).

(defparameter *system-files*
  '("data-structures" "line-input" "input" "output"
    "concepts" "attention" "vectors" "operations" "generation" "induction" "processing"
    "persist" "relations" "llm" "controller" "ai")
  "The system's source files, in load (dependency) order.")

(defun load-system ()
  "Load (or reload) the whole AI system from the current directory, in dependency order."
  (dolist (f *system-files* t)
    (load (concatenate 'string f ".lisp"))))

(defun describe-knowledge-base ()
  "Report which knowledge base (main) will use on startup, given the files on disk now.
   (main) loads the saved memory file if it exists, otherwise learns the starter KB."
  (let* ((save-file   (symbol-value (find-symbol "*SAVE-FILE*" "persist")))
	 (starter-sym (find-symbol "*STARTER-KB*"))   ; package-less ai.lisp -> current package
	 (starter     (and starter-sym (boundp starter-sym) (symbol-value starter-sym))))
    (format t "~&;;~%;; AI system loaded.  On (main), the knowledge base will be:~%")
    (cond
      ((probe-file save-file)
       (format t ";;   saved memory ~s (it exists -- it will be loaded).~%" save-file))
      ((and starter (probe-file starter))
       (format t ";;   the starter KB ~s (no saved memory ~s yet -- it will be learned).~%"
	       starter save-file))
      (starter
       (format t ";;   none: starter KB ~s is set but the file is missing -- starting blank.~%"
	       starter))
      (t
       (format t ";;   none: no saved memory and no starter KB -- starting blank.~%")))
    (format t ";;~%")
    (values)))

(load-system)
(describe-knowledge-base)
