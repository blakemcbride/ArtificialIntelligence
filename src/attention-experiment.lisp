;;;; Transformer-like attention as Hebbian fast weights -- STANDALONE experiment.
;;;; Run:  sbcl --script attention-experiment.lisp
;;;;
;;;; Claim from our discussion: attention -- the transformer's core operation,
;;;; query . key -> value retrieval -- is, in its linear form, just a HEBBIAN
;;;; outer-product memory ("fast weights"), with NO backpropagation.  This shows it,
;;;; and shows what it buys: the project's original FIRST GOAL (notes/Overview.txt),
;;;;   "say X" -> X  for a NOVEL X,
;;;; by copying the filler BY REFERENCE -- something a learned lookup table cannot do,
;;;; and something the associative net / concept graph also can't (they map to learned
;;;; outputs, never to an arbitrary new input word).
;;;;
;;;; Representation: each token is a random unit vector; each position is a "role" vector.
;;;; Binding a sequence = sum of role (x) token outer products = M (the fast weights, all
;;;; Hebbian).  Retrieving the token at a role r = r . M = sum_k (r . role_k) token_k
;;;; -- attention with keys = roles, values = token vectors -- then decode to the nearest
;;;; known token.  (Softmax/Hopfield would sharpen the retrieval; the linear form already
;;;; suffices here.)

(setf *random-state* (sb-ext:seed-random-state 1234))   ; reproducible

(defparameter *dim* 256)
(defparameter *vectors* (make-hash-table :test 'equal))   ; token -> unit vector
(defparameter *roles*   (make-hash-table :test 'eql))     ; position -> role vector

(defun rand-unit ()
  (let ((v (make-array *dim* :element-type 'double-float)))
    (dotimes (i *dim*) (setf (aref v i) (- (random 2.0d0) 1.0d0)))
    (let ((norm (sqrt (loop for x across v sum (* x x)))))
      (dotimes (i *dim*) (setf (aref v i) (/ (aref v i) norm))))
    v))

(defun vec (token)
  "Unit vector for TOKEN (same token -> same vector; a novel token gets a fresh one)."
  (or (gethash token *vectors*) (setf (gethash token *vectors*) (rand-unit))))

(defun role (i)
  "Unit vector for sequence position I."
  (or (gethash i *roles*) (setf (gethash i *roles*) (rand-unit))))

(defun dot (a b) (loop for x across a for y across b sum (* x y)))

;;; --- Hebbian fast-weight attention --------------------------------------------------
(defun bind-sequence (tokens)
  "Bind TOKENS to positional roles via Hebbian outer products and return the memory as a
   closure  query-vector -> retrieved-vector  (r . M = sum_k (r.role_k) token_k)."
  (let ((pairs (loop for tok in tokens for i from 0 collect (cons (role i) (vec tok)))))
    (lambda (query)
      (let ((out (make-array *dim* :element-type 'double-float :initial-element 0.0d0)))
        (dolist (p pairs out)
          (let ((w (dot query (car p))) (tv (cdr p)))
            (dotimes (i *dim*) (incf (aref out i) (* w (aref tv i))))))))))

(defun decode (v)
  "Nearest known token to vector V (cosine; vectors are unit length so dot = cosine)."
  (let ((best nil) (best-score -2.0d0))
    (maphash (lambda (tok tv)
               (let ((s (/ (dot v tv) (max 1d-12 (sqrt (dot v v))))))
                 (when (> s best-score) (setf best-score s best tok))))
             *vectors*)
    best))

(defun copy-after-cue (sentence)
  "The induction-head behaviour: bind SENTENCE, then retrieve the token at role 1 -- the
   word right after the cue -- and emit it.  SENTENCE is a list like (\"say\" \"horse\")."
  (decode (funcall (bind-sequence sentence) (role 1))))

;;; --- demo --------------------------------------------------------------------------
;; A starting vocabulary (these tokens have been "seen"); horse and zorp have NOT.
(dolist (w '("say" "dog" "cat" "bird" "tree" "blue" "run" "hello")) (vec w))

(format t "~%Hebbian fast-weight attention -- copy the word after the cue 'say':~%")
(dolist (s '(("say" "dog") ("say" "cat") ("say" "bird")
             ("say" "horse")     ; horse: NOVEL  (never in the vocabulary above)
             ("say" "zorp")))    ; zorp:  nonsense (never seen anywhere)
  (format t "   ~14a ->  ~a~%" (format nil "~{~a~^ ~}" s) (copy-after-cue s)))

(format t "~%For contrast, a plain learned association (lookup table), same task:~%")
(let ((table (make-hash-table :test 'equal)))
  (dolist (p '(("say dog" . "dog") ("say cat" . "cat") ("say bird" . "bird")))
    (setf (gethash (car p) table) (cdr p)))
  (dolist (s '("say dog" "say horse"))
    (format t "   ~14a ->  ~a~%" s (or (gethash s table) "(I don't know)"))))

(format t "~%The attention head copies ANY filler (even never-seen words) because it~%")
(format t "routes the filler by reference; the lookup can only repeat what it memorised.~%")
