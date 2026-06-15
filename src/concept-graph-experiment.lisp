;;;; Phase-7 investigation -- STANDALONE experiment, not part of the loaded system.
;;;; Run:  sbcl --script concept-graph-experiment.lisp
;;;;
;;;; Question: can a pure-Hebbian concept graph + spreading activation produce
;;;; generalization WITH exclusion -- include a novel "horse", exclude "snake",
;;;; "ruler", and a nonsense "blicket" -- from input->output relationships alone?
;;;;
;;;; Model (faithful to the proposed architecture, no slots / no logic):
;;;;  * Each learned fact co-activates a filler concept and a (predicate,answer)
;;;;    state -- "do dogs have legs -> yes" co-activates  dog <-> has-legs:YES .
;;;;    The answer is part of what's co-active, so has-legs:YES and has-legs:NO are
;;;;    different states (snake's leglessness is genuinely in the data).
;;;;  * Hebbian: co-activation strengthens the undirected edge (weight = count).
;;;;  * Similarity / generalization = degree-normalized, decayed spreading
;;;;    activation through the graph (a random walk).  "Does X walk on its legs?"
;;;;    becomes: how strongly does X's activation reach the  walks-on-legs:YES  node?
;;;;
;;;; The one non-obvious ingredient is DEGREE NORMALIZATION: a node's activation is
;;;; divided among its neighbours, so a promiscuous hub like is-animal:YES (shared by
;;;; everything) carries little, while a discriminating state like has-legs:YES
;;;; carries a lot.  We run WITH and WITHOUT it to see what does the work.

(defparameter *g* (make-hash-table :test 'equal))   ; concept -> (neighbor -> weight)

(defun edge (a b w)
  (let ((ta (or (gethash a *g*) (setf (gethash a *g*) (make-hash-table :test 'equal))))
        (tb (or (gethash b *g*) (setf (gethash b *g*) (make-hash-table :test 'equal)))))
    (incf (gethash b ta 0.0) w)
    (incf (gethash a tb 0.0) w)))

(defun fact (filler state) (edge filler state 1.0))   ; one filler<->state co-activation

(defun wdeg (n)
  (let ((s 0.0) (tab (gethash n *g*)))
    (when tab (maphash (lambda (k v) (declare (ignore k)) (incf s v)) tab))
    s))

(defun spread (seed normalize &key (hops 5) (decay 0.6))
  "Decayed spreading activation from SEED; NORMALIZE t = divide a node's output by its
   weighted degree (random walk).  Returns total activation reached at every node."
  (let ((current (make-hash-table :test 'equal))
        (total   (make-hash-table :test 'equal)))
    (setf (gethash seed current) 1.0)
    (dotimes (h hops)
      (let ((next (make-hash-table :test 'equal)))
        (maphash
         (lambda (m amt)
           (let ((tab (gethash m *g*)))
             (when tab
               (let ((deg (if normalize (wdeg m) 1.0)))
                 (when (plusp deg)
                   (maphash (lambda (n w)
                              (incf (gethash n next 0.0) (* amt (/ w deg) decay)))
                            tab))))))
         current)
        (maphash (lambda (n amt) (incf (gethash n total 0.0) amt)) next)
        (setf current next)))
    total))

(defun sim (a b normalize) (gethash b (spread a normalize) 0.0))

;;; ---------------- corpus: filler <-> (predicate,answer) states ----------------
(dolist (f '("dog" "goat" "person"))                  (fact f "walks-on-legs:YES"))
(dolist (f '("dog" "goat" "person" "horse"))          (fact f "has-legs:YES"))
(dolist (f '("dog" "goat" "person" "horse" "snake"))  (fact f "is-animal:YES"))
(fact "snake" "has-legs:NO")    (fact "snake" "slithers:YES")
(fact "ruler" "has-legs:NO")    (fact "ruler" "is-tool:YES")
;; horse: taught has-legs:YES + is-animal:YES, NEVER the walk question.
;; blicket: taught nothing at all.

(defparameter *subjects* '("dog" "goat" "person" "horse" "snake" "ruler" "blicket"))
(defparameter *query* "walks-on-legs:YES")

(defun states-of (f)
  (let (acc (tab (gethash f *g*)))
    (when tab (maphash (lambda (k v) (declare (ignore v)) (push k acc)) tab))
    (sort acc #'string<)))

(format t "~%what each subject was taught:~%")
(dolist (f *subjects*)
  (format t "   ~9a ~{~a~^, ~}~%" f (states-of f)))

(format t "~%=== 'do X walk on their legs?'  ->  activation reaching ~a ===~%" *query*)
(format t "~9a  ~12a  ~12a~%" "subject" "WITH deg-norm" "WITHOUT")
(dolist (f *subjects*)
  (format t "   ~9a  ~,5f       ~,5f~%"
          f (sim f *query* t) (sim f *query* nil)))
