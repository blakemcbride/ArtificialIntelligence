
(defpackage "persist"
  (:use "COMMON-LISP")
  (:export "SAVE-NETWORK" "LOAD-NETWORK" "EXPORT-KB" "IMPORT-KB" "*SAVE-FILE*" "SYSTEM-STATS"
	   "*TUNABLE-PROVIDER*" "*TUNABLE-RESTORER*"))

(in-package "persist")
(provide "persist")

(require "data-structures")
(use-package "data-structures")

;;; Persistence (Phase 6): serialise the whole neuron network to one readable
;;; s-expression and reload it, so learning survives across runs.  The network is a
;;; shared, partly-cyclic graph (dendrites point at neurons; :association dendrites also
;;; keep a `from' back-pointer; extender links; the *dictionary* / *output-roots* /
;;; *responses* / *associations* registries).  We therefore serialise neurons FLAT --
;;; keyed by their unique id -- writing every reference AS an id, then rebuild in two
;;; passes (create all neurons, then wire links) through an id -> neuron map.  ids are
;;; debug-only, so reloaded neurons get fresh ids; the id MAP is what preserves sharing.

(defparameter *save-file* "auto-save.kb"
  "Default file the teaching loop loads on startup and saves on exit.")

;; The .set tunables (model caps etc.) live in the package-less top level (ai.lisp), so
;; persist can't name them directly.  ai.lisp installs these hooks: the provider returns an
;; alist of (name . value) to write into the .kb; the restorer applies a read-back alist.
;; When unset (e.g. persist loaded standalone) save/load simply omit the tunables.
(defparameter *tunable-provider* nil "() -> alist of (name . value), or NIL.")
(defparameter *tunable-restorer* nil "(alist) -> apply restored tunables, or NIL.")

(defun dendrite->list (d)
  (list (neuron-id (dendrite-neuron d))
	(dendrite-weight d)
	(dendrite-kind d)
	(and (dendrite-from d) (neuron-id (dendrite-from d)))))

(defun neuron->list (n)
  (list (neuron-id n)
	(if (typep n 'named-neuron) :named :plain)
	(if (typep n 'named-neuron) (named-neuron-name n) nil)
	(neuron-threshold n)
	(neuron-current-value n)
	(and (neuron-extender n) (neuron-id (neuron-extender n)))
	(mapcar #'dendrite->list (neuron-axon n))))

(defun collect-neurons ()
  "Every neuron reachable from the registries, as a list (each once)."
  (let ((seen (make-hash-table)) (result nil))
    (labels ((visit (n)
	       (when (and n (not (gethash n seen)))
		 (setf (gethash n seen) t)
		 (push n result)
		 (visit (neuron-extender n))
		 (dolist (d (neuron-axon n))
		   (visit (dendrite-neuron d))
		   (visit (dendrite-from d))))))
      (maphash (lambda (k v) (declare (ignore k)) (visit v)) *dictionary*)
      (dolist (r *output-roots*) (visit r))
      (maphash (lambda (k v) (declare (ignore k)) (visit v)) *responses*)
      (maphash (lambda (k v) (declare (ignore k)) (visit v)) *concepts*))  ; concept state neurons
    result))

(defun concept-edges->list ()
  "Flatten *concept-graph* to a list of (a-id b-id weight) directed entries."
  (let (acc)
    (maphash (lambda (a tab)
	       (maphash (lambda (b w) (push (list (neuron-id a) (neuron-id b) w) acc)) tab))
	     *concept-graph*)
    acc))

(defun copy-cues->alist ()
  "Flatten *copy-cues* (cue word -> strength) to an alist."
  (let (acc)
    (maphash (lambda (cue w) (push (cons cue w) acc)) *copy-cues*)
    acc))

(defun templates->alist ()
  "Flatten *templates* (frame -> alist of (template . strength)) to a serialisable alist."
  (let (acc)
    (maphash (lambda (frame cells) (push (cons frame cells) acc)) *templates*)
    acc))

(defun op-templates->alist ()
  "Flatten *op-templates* ((frame . op) -> strength) to a serialisable alist."
  (let (acc)
    (maphash (lambda (k v) (push (cons k v) acc)) *op-templates*)
    acc))

(defun cooccur->alist ()
  "Flatten *cooccur* (word -> word -> count) to a nested alist (word . ((other . n) ...))."
  (let (acc)
    (maphash (lambda (w tab)
	       (let (inner) (maphash (lambda (o n) (push (cons o n) inner)) tab)
		    (push (cons w inner) acc)))
	     *cooccur*)
    acc))

(defun transitions->alist ()
  "Flatten *transitions* (word -> next-word -> count) to a nested alist."
  (let (acc)
    (maphash (lambda (w tab)
	       (let (inner) (maphash (lambda (nw n) (push (cons nw n) inner)) tab)
		    (push (cons w inner) acc)))
	     *transitions*)
    acc))

(defun starts->alist ()
  "Flatten *sentence-starts* (word -> count) to an alist."
  (let (acc) (maphash (lambda (w n) (push (cons w n) acc)) *sentence-starts*) acc))

(defun facts->alist ()
  "Flatten *facts* ((subject relation object) -> strength) to an alist."
  (let (acc) (maphash (lambda (k n) (push (cons k n) acc)) *facts*) acc))

(defun rel-links->alist ()
  "Flatten *rel-links* (connector -> distinct (subj . cat) pairs) to (connector . pairs)."
  (let (acc)
    (maphash (lambda (c tab)
	       (let (pairs) (maphash (lambda (k v) (declare (ignore k)) (push v pairs)) tab)
		    (push (cons c pairs) acc)))
	     *rel-links*)
    acc))

(defun hash->plain-alist (table)
  (let (acc) (maphash (lambda (k v) (push (cons k v) acc)) table) acc))

(defun hash->id-alist (table)
  "Alist of (key . neuron-id) for a string -> neuron hash table."
  (let (acc)
    (maphash (lambda (k v) (push (cons k (neuron-id v)) acc)) table)
    acc))

(defun save-network (&optional (path *save-file*))
  "Serialise the whole network to PATH as one readable s-expression.  Returns PATH."
  (with-open-file (s path :direction :output :if-exists :supersede :if-does-not-exist :create)
    (with-standard-io-syntax
      (prin1 (list :format :ai-network-v1
		   :next-neuron-id *next-neuron-id*
		   :neurons (mapcar #'neuron->list (collect-neurons))
		   :dictionary (hash->id-alist *dictionary*)
		   :output-roots (mapcar #'neuron-id *output-roots*)
		   :responses (hash->id-alist *responses*)
		   :concept-states (hash->id-alist *concepts*)
		   :concept-edges (concept-edges->list)
		   :copy-cues (copy-cues->alist)
		   :templates (templates->alist)
		   :op-templates (op-templates->alist)
		   :cooccur (cooccur->alist)
		   :facts-learned *facts-learned*
		   :transitions (transitions->alist)
		   :sentence-starts (starts->alist)
		   :facts (facts->alist)
		   :rel-links (rel-links->alist)
		   :rel-head (hash->plain-alist *rel-head*)
		   :rel-freq (hash->plain-alist *rel-freq*)
		   :rel-sentences *rel-sentences*
		   :read-offsets (hash->plain-alist *read-offsets*)
		   :tunables (and *tunable-provider* (funcall *tunable-provider*)))
	     s)
      (terpri s)))
  path)

(defun rebuild-network (data)
  "Reset, then rebuild the network from DATA (the read s-expression)."
  (reset)
  (let ((by-id (make-hash-table)))
    ;; pass 1: create every neuron (links filled in pass 2)
    (dolist (rec (getf data :neurons))
      (destructuring-bind (id kind name threshold current-value extender-id dendrites) rec
	(declare (ignore extender-id dendrites))
	(let ((n (if (eq kind :named) (make-named-neuron :name name) (make-neuron))))
	  (setf (neuron-threshold n) threshold
		(neuron-current-value n) current-value)
	  (setf (gethash id by-id) n))))
    ;; pass 2: wire extender links and axons by resolving ids
    (dolist (rec (getf data :neurons))
      (destructuring-bind (id kind name threshold current-value extender-id dendrites) rec
	(declare (ignore kind name threshold current-value))
	(let ((n (gethash id by-id)))
	  (when extender-id
	    (setf (neuron-extender n) (gethash extender-id by-id)))
	  (setf (neuron-axon n)
		(mapcar (lambda (drec)
			  (destructuring-bind (target weight dkind from) drec
			    (make-dendrite :neuron (gethash target by-id)
					   :weight weight
					   :kind dkind
					   :from (and from (gethash from by-id)))))
			dendrites)))))
    ;; registries
    (dolist (pair (getf data :dictionary))
      (setf (gethash (car pair) *dictionary*) (gethash (cdr pair) by-id)))
    (setf *output-roots*
	  (mapcar (lambda (id) (gethash id by-id)) (getf data :output-roots)))
    (dolist (pair (getf data :responses))
      (setf (gethash (car pair) *responses*) (gethash (cdr pair) by-id)))
    ;; concept graph: state registry + weighted edges
    (dolist (pair (getf data :concept-states))
      (setf (gethash (car pair) *concepts*) (gethash (cdr pair) by-id)))
    (dolist (e (getf data :concept-edges))
      (destructuring-bind (a-id b-id w) e
	(let ((a (gethash a-id by-id)) (b (gethash b-id by-id)))
	  (when (and a b)
	    (let ((tab (or (gethash a *concept-graph*)
			   (setf (gethash a *concept-graph*) (make-hash-table :test 'eq)))))
	      (setf (gethash b tab) w))))))
    (dolist (pair (getf data :copy-cues))
      (setf (gethash (car pair) *copy-cues*) (cdr pair)))
    (dolist (pair (getf data :templates))
      (setf (gethash (car pair) *templates*) (cdr pair)))
    (dolist (pair (getf data :op-templates))
      (setf (gethash (car pair) *op-templates*) (cdr pair)))
    (dolist (pair (getf data :cooccur))
      (let ((tab (make-hash-table :test 'equal)))
	(dolist (oc (cdr pair)) (setf (gethash (car oc) tab) (cdr oc)))
	(setf (gethash (car pair) *cooccur*) tab)))
    (setf *facts-learned* (or (getf data :facts-learned) 0))   ; restore the learned-facts count
    ;; generation (Phase 8): transition model, sentence starts, declarative facts
    (dolist (pair (getf data :transitions))
      (let ((tab (make-hash-table :test 'equal)))
	(dolist (nc (cdr pair)) (setf (gethash (car nc) tab) (cdr nc)))
	(setf (gethash (car pair) *transitions*) tab)))
    (dolist (pair (getf data :sentence-starts))
      (setf (gethash (car pair) *sentence-starts*) (cdr pair)))
    (dolist (pair (getf data :facts))
      (setf (gethash (car pair) *facts*) (cdr pair)))
    ;; learned relation discovery (Phase 9)
    (dolist (pair (getf data :rel-links))
      (let ((tab (make-hash-table :test 'equal)))
	(dolist (ab (cdr pair))
	  (setf (gethash (format nil "~a|~a" (car ab) (cdr ab)) tab) ab))
	(setf (gethash (car pair) *rel-links*) tab)))
    (dolist (p (getf data :rel-head)) (setf (gethash (car p) *rel-head*) (cdr p)))
    (dolist (p (getf data :rel-freq)) (setf (gethash (car p) *rel-freq*) (cdr p)))
    (setf *rel-sentences* (or (getf data :rel-sentences) 0))
    (dolist (p (getf data :read-offsets)) (setf (gethash (car p) *read-offsets*) (cdr p)))
    (when (and *tunable-restorer* (getf data :tunables))   ; restore the .set parameters saved with this KB
      (funcall *tunable-restorer* (getf data :tunables)))
    ;; *associations* = every :association dendrite, recollected from the rebuilt axons
    (dolist (rec (getf data :neurons))
      (dolist (d (neuron-axon (gethash (first rec) by-id)))
	(when (eq :association (dendrite-kind d))
	  (push d *associations*))))
    (setf *next-neuron-id* (getf data :next-neuron-id)))
  t)

(defun load-network (&optional (path *save-file*))
  "If PATH exists, rebuild the network from it and return T; otherwise return NIL."
  (when (probe-file path)
    (rebuild-network (with-open-file (s path :direction :input)
		       (with-standard-io-syntax (read s))))
    t))

;;; --- User-facing knowledge-base export / import ---------------------------------
;;; The whole knowledge base (input network, output chains, associations, AND the
;;; concept graph) round-trips through one file.

(defun export-kb (path)
  "Write the entire knowledge base to PATH.  Returns PATH."
  (save-network path))

(defun import-kb (path)
  "Replace the current knowledge base with the one stored at PATH.  T if loaded, NIL if
   the file is missing."
  (load-network path))

;;; --- System stats ----------------------------------------------------------------
(defun system-stats ()
  "Print a summary of the system's size and contents, and return it as a plist.
   Numbers: facts learned (cumulative), vocabulary, neurons, dendrites, output roots,
   associations, responses, concept states, concept-graph edges, copy cues, templates,
   learned operations, and distributed-vector vocabulary."
  (let* ((neurons (collect-neurons))
	 (dendrites (reduce #'+ neurons :key (lambda (n) (length (neuron-axon n))) :initial-value 0))
	 (cg 0))
    (maphash (lambda (k tab) (declare (ignore k)) (incf cg (hash-table-count tab))) *concept-graph*)
    (let ((stats (list :facts-learned    *facts-learned*
		       :vocabulary       (hash-table-count *dictionary*)
		       :neurons          (length neurons)
		       :dendrites        dendrites
		       :output-roots     (length *output-roots*)
		       :associations     (length *associations*)
		       :responses        (hash-table-count *responses*)
		       :concept-states   (hash-table-count *concepts*)
		       :concept-edges    (floor cg 2)
		       :copy-cues        (hash-table-count *copy-cues*)
		       :templates        (hash-table-count *templates*)
		       :operations       (hash-table-count *op-templates*)
		       :vector-words     (hash-table-count *cooccur*)
		       :facts            (hash-table-count *facts*)
		       :transition-heads (hash-table-count *transitions*)
		       :relation-connectors (hash-table-count *rel-links*))))
      (format t "~&--- system stats ---~%")
      (loop for (k v) on stats by #'cddr
	    do (format t "  ~16a ~:d~%" (string-downcase (symbol-name k)) v))
      stats)))
