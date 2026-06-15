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
    "concepts" "attention" "processing" "persist" "ai")
  "The system's source files, in load (dependency) order.")

(defun load-system ()
  "Load (or reload) the whole AI system from the current directory, in dependency order."
  (dolist (f *system-files* t)
    (load (concatenate 'string f ".lisp"))))

(load-system)
