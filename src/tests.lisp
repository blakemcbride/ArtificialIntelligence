;;;; Phase 0 smoke tests (see Plan.md).
;;;;
;;;; Run from the src/ directory:
;;;;
;;;;   Interactive:        (load "tests.lisp")
;;;;   Batch (exit code):  make test
;;;;
;;;; Like ai.lisp, this file does NOT create its own package; it defines its
;;;; symbols in the current package (normally COMMON-LISP-USER) and relies on the
;;;; symbols ai.lisp imports via (use-package "data-structures").

;; Load the components explicitly by pathname so the suite does not depend on a
;; given implementation's (require ...) search behaviour -- CLISP resolves a bare
;; module name against the current directory, but SBCL/CCL/ECL do not.  Each file
;; calls (provide ...), so the (require ...) forms inside ai.lisp become no-ops.
(load "data-structures.lisp")
(load "line-input.lisp")
(load "input.lisp")
(load "output.lisp")
(load "concepts.lisp")
(load "attention.lisp")
(load "vectors.lisp")
(load "operations.lisp")
(load "processing.lisp")
(load "persist.lisp")
(load "ai.lisp")   ; loaded last; its use-package forms bring every component's
                   ; exported symbols into this package (COMMON-LISP-USER)

(defvar *tests-run* 0)
(defvar *tests-failed* 0)

(defun check (name ok)
  "Record and print the result of one assertion."
  (incf *tests-run*)
  (cond (ok
	 (format t "  ok   ~a~%" name))
	(t
	 (incf *tests-failed*)
	 (format t "  FAIL ~a~%" name))))

(defun words->neurons (words)
  "Intern WORDS (a list of strings) into *dictionary* and return their
   named-neurons in order -- the same interning create-line does, but without
   reading stdin.  Test helper."
  (mapcar (lambda (w)
	    (or (gethash w *dictionary*)
		(setf (gethash w *dictionary*)
		      (make-named-neuron :name w))))
	  words))

(defun axon-dendrite-to (from to)
  "Return the dendrite in FROM's axon that points at TO, or nil."
  (dolist (den (neuron-axon from))
    (when (eq to (dendrite-neuron den))
      (return den))))

(defun some-dendrite-kind-p (neuron kind)
  "Does NEURON have at least one axon dendrite of the given KIND?"
  (dolist (den (neuron-axon neuron))
    (when (eq kind (dendrite-kind den))
      (return t))))

(defun run-tests ()
  (setq *tests-run* 0 *tests-failed* 0)
  (format t "~%Phase 0 tests~%")

  ;; --- a fresh reset clears every global ---
  (reset)
  (check "reset empties *dictionary*"     (zerop (hash-table-count *dictionary*)))
  (check "reset zeroes *next-neuron-id*"  (eql 0 *next-neuron-id*))
  (check "reset clears *output-roots*"    (null *output-roots*))
  (check "reset clears *associations*"    (null *associations*))

  ;; --- activation slots are single-floats defaulting to 0.0 ---
  (let ((n (make-neuron)))
    (check "neuron threshold is single-float"     (typep (neuron-threshold n) 'single-float))
    (check "neuron current-value is single-float" (typep (neuron-current-value n) 'single-float))
    (check "neuron threshold defaults to 0.0"     (= 0.0 (neuron-threshold n)))
    (check "neuron current-value defaults to 0.0" (= 0.0 (neuron-current-value n))))

  ;; --- dendrites carry a kind, defaulting to :sequence ---
  (check "dendrite kind defaults to :sequence" (eq :sequence (dendrite-kind (make-dendrite))))

  ;; --- build-structure now RETURNS the active "meaning" neurons ---
  (reset)
  (let* ((words  (words->neurons '("the" "dog" "runs")))
	 (the    (first words))
	 (dog    (second words))
	 (runs   (third words))
	 (active (build-structure words)))
    (check "build-structure returns a non-nil active set" (consp active))
    (check "every returned element is a neuron"
	   (every (lambda (x) (typep x 'neuron)) active))
    (check "active set includes the last word's extender"
	   (and (neuron-extender runs)
		(member (neuron-extender runs) active)))

    ;; --- structural edges are tagged by kind ---
    (let ((ext-den (axon-dendrite-to the (neuron-extender the))))
      (check "first word's extender edge is tagged :extender"
	     (and ext-den (eq :extender (dendrite-kind ext-den)))))
    (check "a structural join edge tagged :sequence exists (on 'dog')"
	   (some-dendrite-kind-p dog :sequence))
    (check "build-structure creates no :association edges"
	   (null *associations*))

    ;; --- dump-dictionary still works over the new structs ---
    (check "dump-dictionary runs without error and produces output"
	   (handler-case
	       (plusp (length (with-output-to-string (*standard-output*)
				(dump-dictionary))))
	     (error () nil))))

  ;; ===================== Phase 1 -- Output component =====================
  (format t "~%Phase 1 tests~%")
  (reset)
  (let* ((words '("hi" "there"))
         (root  (build-output-structure words)))
    (check "build-output-structure returns a neuron root" (typep root 'neuron))
    (check "root is a plain neuron, not a named-neuron" (not (typep root 'named-neuron)))
    (check "produce-output reproduces the response words verbatim"
           (equal words (produce-output root)))
    (check "output-sentence joins the words" (string= "hi there" (output-sentence root)))
    (check "root is registered in *output-roots*" (and (member root *output-roots*) t))
    (check "response is interned in *responses*" (eql 1 (hash-table-count *responses*)))
    (let ((n1 (neuron-extender root)))
      (check "first chain node is a named-neuron named 'hi'"
             (and (typep n1 'named-neuron) (string= "hi" (named-neuron-name n1))))
      (let ((d (first (neuron-axon root))))
        (check "root's extender edge is tagged :extender and points at it"
               (and d (eq :extender (dendrite-kind d)) (eq n1 (dendrite-neuron d)))))
      (let ((n2 (neuron-extender n1)))
        (check "second node is 'there' and terminates the chain"
               (and (typep n2 'named-neuron)
                    (string= "there" (named-neuron-name n2))
                    (null (neuron-extender n2)))))))
  (let ((r1 (build-output-structure '("ok")))
        (r2 (build-output-structure '("ok"))))
    (check "identical responses reuse one root (eq)" (eq r1 r2))
    (check "reuse interns the response only once" (eql 2 (hash-table-count *responses*))))
  (let ((ra (build-output-structure '("bye" "now")))
        (rb (build-output-structure '("see" "you"))))
    (check "distinct responses produce distinct roots" (not (eq ra rb))))
  (reset)
  (check "reset clears *responses*" (zerop (hash-table-count *responses*)))
  (check "reset clears *output-roots*" (null *output-roots*))

  ;; ===================== Phase 2 -- Association bridge =====================
  (format t "~%Phase 2 tests~%")
  (reset)
  (let* ((in-words (words->neurons '("the" "dog" "barks")))
         (endings  (build-structure in-words))
         (root     (build-output-structure '("woof")))
         (dends    (associate endings root)))
    (check "associate returns one dendrite per distinct ending"
           (eql (length dends) (length (remove-duplicates endings))))
    (check "every input ending now has an :association dendrite to the root"
           (every (lambda (e) (find-association e root)) endings))
    (check "new associations carry the initial weight w0"
           (every (lambda (d) (= *assoc-initial-weight* (dendrite-weight d))) dends))
    (check "every association is registered in *associations*"
           (every (lambda (d) (and (member d *associations*) t)) dends))
    (check "association count matches *associations* length"
           (eql (length dends) (length *associations*)))
    ;; the Phase 0 deferral: structural traversal must ignore association edges
    (check "connects-to ignores the association edge (ending !-> root structurally)"
           (notany (lambda (e) (connects-to e root)) endings))
    ;; re-training the same pair strengthens rather than duplicating
    (let* ((before (length *associations*))
           (d2     (associate endings root))
           (after  (length *associations*)))
      (check "re-associating adds no new dendrites" (eql before after))
      (check "re-associating strengthens the weight by eta"
             (every (lambda (d) (= (min *assoc-max-weight*
                                        (+ *assoc-initial-weight* *assoc-strengthen*))
                                   (dendrite-weight d)))
                    d2)))
    ;; rebuilding identical input is unaffected by the now-present associations
    (let ((endings2 (build-structure in-words)))
      (check "rebuilding identical input reuses the same endings"
             (and (eql (length endings) (length endings2))
                  (null (set-difference endings endings2))))
      (check "the output root never appears among input endings"
             (not (member root endings2)))))
  (reset)
  (check "reset clears *associations* after wiring" (null *associations*))

  ;; ===================== Phase 3 -- Inference (responding) =====================
  (format t "~%Phase 3 tests~%")
  (reset)
  ;; teach: "the dog barks" -> "woof"
  (associate (build-structure (words->neurons '("the" "dog" "barks")))
             (build-output-structure '("woof")))
  (check "respond reproduces the taught response"
         (equal '("woof") (respond '("the" "dog" "barks"))))
  (check "unrelated input (disjoint vocabulary) yields nil -- I don't know"
         (null (respond '("birds" "fly" "south"))))
  ;; teach a second pair: "the cat meows" -> "purr"
  (associate (build-structure (words->neurons '("the" "cat" "meows")))
             (build-output-structure '("purr")))
  (check "second taught pair is retrievable"
         (equal '("purr") (respond '("the" "cat" "meows"))))
  (check "first taught pair still retrievable after the second"
         (equal '("woof") (respond '("the" "dog" "barks"))))
  (check "a subset of a taught input retrieves its learned response"
         (equal '("woof") (respond '("dog" "barks"))))
  (check "respond's winner is an output root with positive activation"
         (multiple-value-bind (words root score) (respond '("the" "cat" "meows"))
           (declare (ignore words))
           (and (member root *output-roots*) (> score 0.0))))

  ;; ===================== Phase 4 -- Reinforcement & decay =====================
  (format t "~%Phase 4 tests~%")

  ;; reinforcement stabilizes: repeated associate clamps at the max weight
  (reset)
  (let ((endings (build-structure (words->neurons '("a" "b"))))
        (root    (build-output-structure '("x"))))
    (dotimes (i 50) (associate endings root))
    (check "repeated reinforcement clamps at *assoc-max-weight* (stabilizes)"
           (every (lambda (e) (let ((d (find-association e root)))
                                (and d (= (dendrite-weight d) *assoc-max-weight*))))
                  endings)))

  ;; decay shrinks weights; unused associations are pruned and roots GC'd
  (reset)
  (let* ((endings (build-structure (words->neurons '("a" "b"))))
         (root    (build-output-structure '("x"))))
    (associate endings root)
    (let ((w0 (dendrite-weight (find-association (first endings) root))))
      (let ((*assoc-decay* 0.1)) (decay-associations))
      (check "one decay step shrinks an association's weight"
             (< (dendrite-weight (find-association (first endings) root)) w0)))
    (let ((*assoc-decay* 0.9))
      (dotimes (i 20) (decay-associations)))
    (check "unused associations decay below epsilon and are pruned"
           (null *associations*))
    (check "pruned associations are removed from their source axons"
           (every (lambda (e) (null (find-association e root))) endings))
    (check "a fully-decayed root is GC'd from *output-roots*" (null *output-roots*))
    (check "a fully-decayed response is forgotten from *responses*"
           (zerop (hash-table-count *responses*))))

  ;; learn: novel -> nil + teaches; correct guess -> returns it; wrong -> corrects
  (reset)
  (let ((*assoc-decay* 0.0))
    (check "learn on a novel pair returns nil and teaches it"
           (and (null (learn '("ping") '("pong")))
                (equal '("pong") (respond '("ping")))))
    (check "learn returns the prior correct guess (reinforcing it)"
           (equal '("pong") (learn '("ping") '("pong"))))
    (check "learn corrects a previously-wrong mapping"
           (progn (dotimes (i 8) (learn '("ping") '("dong")))
                  (equal '("dong") (respond '("ping"))))))

  ;; homeostatic threshold rises but does not suppress retrieval
  (reset)
  (let ((*assoc-decay* 0.0))
    (dotimes (i 5) (learn '("foo") '("bar")))
    (let ((root (gethash "bar" *responses*)))
      (check "homeostatic threshold rises above zero after repeated learning"
             (and root (> (neuron-threshold root) 0.0)))
      (check "retrieval still works despite the raised threshold"
             (equal '("bar") (respond '("foo"))))))

  ;; bounded size: a long script of one-off pairs stays small under decay
  (reset)
  (let ((*assoc-decay* 0.5) (*assoc-prune-threshold* 0.05))
    (dotimes (i 20)
      (learn (list (format nil "in~d" i)) (list (format nil "out~d" i))))
    (check "one-off associations are pruned -- *associations* stays bounded"
           (< (length *associations*) 20))
    (check "one-off roots are GC'd -- *output-roots* stays bounded"
           (< (length *output-roots*) 20)))

  ;; ===================== Phase 5 -- Interactive teaching loop ==================
  (format t "~%Phase 5 tests~%")
  ;; Drive `main' over a scripted session: teach "say hi" -> "hi there" on turn 1,
  ;; then re-present "say hi" on turn 2 and confirm.  main loads/saves *save-file*, so
  ;; bind it to a throwaway path (and remove it) to keep the test self-contained.
  (let ((*save-file* "phase5-temp.kb") (*starter-kb* nil))  ; no auto-train: keep isolated
    (ignore-errors (delete-file *save-file*))
    (let* ((script (format nil "say hi.~%hi there.~%say hi.~%yes.~%quit.~%"))
           (out (with-output-to-string (*standard-output*)
                  (with-input-from-string (*standard-input* script)
                    (main)))))
      (check "interactive loop: the first (untaught) turn says it doesn't know"
             (and (search "I don't know" out) t))
      (check "interactive loop: a later turn answers what an earlier turn taught"
             (and (search "guess: hi there" out) t))
      (check "interactive loop: a confirmed correct guess is reinforced"
             (and (search "(reinforced)" out) t)))
    (ignore-errors (delete-file *save-file*)))

  ;; Leading-period commands: teach a fact, .save to a file, then in a fresh session
  ;; .load it back and confirm the fact survived.  Filenames keep their case/dots.
  (let ((*save-file* "phase5-cmd.kb") (*starter-kb* nil)
        (kb "Phase5-Cmd-Out.kb"))
    (ignore-errors (delete-file *save-file*))
    (ignore-errors (delete-file kb))
    (let ((out (with-output-to-string (*standard-output*)
                 (with-input-from-string
                     (*standard-input*
                      (format nil "say hi.~%hi there.~%.stats~%.save ~a~%.quit~%" kb))
                   (main)))))
      (check ".stats command prints stats inside the loop"
             (and (search "--- system stats ---" out) t))
      (check ".save command writes the knowledge base (case-preserved filename)"
             (and (search "saved knowledge base to" out) (probe-file kb) t)))
    (let ((out (with-output-to-string (*standard-output*)
                 (with-input-from-string
                     (*standard-input*
                      (format nil ".load ~a~%say hi.~%yes.~%.quit~%" kb))
                   (main)))))
      (check ".load command restores a saved knowledge base"
             (and (search "loaded knowledge base from" out)
                  (search "guess: hi there" out) t)))
    (ignore-errors (delete-file *save-file*))
    (ignore-errors (delete-file kb)))

  ;; ===================== Phase 6 -- Persistence =====================
  (format t "~%Phase 6 tests~%")
  (reset)
  (learn '("say" "hi") '("hi" "there"))
  (learn '("the" "dog" "barks") '("woof"))
  (let ((tmp "phase6-temp.kb")
        (hi-before  (respond '("say" "hi")))
        (dog-before (respond '("the" "dog" "barks"))))
    (save-network tmp)
    (reset)
    (check "after a simulated restart (reset) memory is empty"
           (and (zerop (hash-table-count *dictionary*))
                (null *output-roots*) (null *associations*)))
    (check "loading a saved network returns T" (and (load-network tmp) t))
    (check "a reloaded network reproduces a taught response"
           (equal hi-before (respond '("say" "hi"))))
    (check "a reloaded network reproduces a second taught response"
           (equal dog-before (respond '("the" "dog" "barks"))))
    (check "reload restores *output-roots*, *responses*, and *associations*"
           (and (plusp (length *output-roots*))
                (plusp (hash-table-count *responses*))
                (plusp (length *associations*))))
    (check "loading a missing file returns NIL"
           (null (load-network "no-such-file-zzz.kb")))
    (ignore-errors (delete-file tmp)))

  ;; ===================== Phase 7 -- Concept graph (generalization) =============
  (format t "~%Phase 7 tests~%")
  (reset)
  ;; teach relationships (subject, predicate, answer).  dogs/goats/people walk on legs;
  ;; snake is a legless animal that slithers; ruler is a legless tool; horse is taught
  ;; only that it has legs and is an animal -- never that it walks; blicket: nothing.
  (dolist (s '("dog" "goat" "person"))                 (relate s "walks-on-legs" "yes"))
  (dolist (s '("dog" "goat" "person" "horse"))         (relate s "has-legs" "yes"))
  (dolist (s '("dog" "goat" "person" "horse" "snake")) (relate s "is-animal" "yes"))
  (relate "snake" "has-legs" "no")  (relate "snake" "slithers" "yes")
  (relate "ruler" "has-legs" "no")  (relate "ruler" "is-tool"  "yes")
  (let ((dog     (category-strength "dog"     "walks-on-legs" "yes"))
        (horse   (category-strength "horse"   "walks-on-legs" "yes"))
        (snake   (category-strength "snake"   "walks-on-legs" "yes"))
        (ruler   (category-strength "ruler"   "walks-on-legs" "yes"))
        (blicket (category-strength "blicket" "walks-on-legs" "yes")))
    (check "a taught member scores highest (recall > generalization)" (> dog horse))
    (check "a novel similar word generalizes IN (horse walks on legs)"
           (recognizes-p "horse" "walks-on-legs" "yes"))
    (check "a dissimilar animal is excluded (snake)"
           (not (recognizes-p "snake" "walks-on-legs" "yes")))
    (check "an unrelated object is excluded (ruler)"
           (not (recognizes-p "ruler" "walks-on-legs" "yes")))
    (check "an unknown word is excluded (blicket)"
           (not (recognizes-p "blicket" "walks-on-legs" "yes")))
    (check "inclusion clears exclusion by a wide margin (horse > 2x snake)"
           (> horse (* 2.0 snake)))
    (check "a never-taught word has zero category strength (blicket)"
           (= 0.0 blicket)))
  ;; ----- auto-populated from raw learn() calls + slot-free query (Phase 7 #1) -----
  (reset)
  (dolist (s '("dogs" "goats" "cats"))                   (learn (list "do" s "walk" "on" "their" "legs") '("yes")))
  (dolist (s '("dogs" "goats" "cats" "horses"))          (learn (list "do" s "have" "legs") '("yes")))
  (dolist (s '("dogs" "goats" "cats" "horses" "snakes")) (learn (list "are" s "animals") '("yes")))
  (learn '("do" "snakes" "have" "legs") '("no"))
  (learn '("do" "snakes" "slither") '("yes"))
  (learn '("do" "rulers" "have" "legs") '("no"))
  (learn '("are" "rulers" "tools") '("yes"))
  ;; horse: taught has-legs + animal, NEVER the walk question; blicket: nothing
  (let ((h  (infer-strength '("do" "horses"   "walk" "on" "their" "legs") '("yes")))
        (sn (infer-strength '("do" "snakes"   "walk" "on" "their" "legs") '("yes")))
        (ru (infer-strength '("do" "rulers"   "walk" "on" "their" "legs") '("yes")))
        (bl (infer-strength '("do" "blickets" "walk" "on" "their" "legs") '("yes"))))
    (check "auto-pop: a novel word generalizes in (horse, positive strength)" (plusp h))
    (check "auto-pop: horse clears snake by a wide margin (>2x)" (> h (* 2.0 sn)))
    (check "auto-pop: horse beats the unrelated object (ruler)" (> h ru))
    (check "auto-pop: an unknown word scores zero (blicket)" (= 0.0 bl))
    (check "auto-pop: slot-free query identifies the subject (horse case)"
           (string= "horses" (nth-value 1 (infer-strength '("do" "horses" "walk" "on" "their" "legs") '("yes")))))
    (check "auto-pop: infer-p includes horse"
           (infer-p '("do" "horses" "walk" "on" "their" "legs") '("yes")))
    (check "auto-pop: infer-p excludes snake"
           (not (infer-p '("do" "snakes" "walk" "on" "their" "legs") '("yes"))))
    (check "auto-pop: infer-p excludes nonsense"
           (not (infer-p '("do" "blickets" "walk" "on" "their" "legs") '("yes")))))

  ;; ----- the concept graph persists across save/reset/load (Phase 7 #2) -----
  (reset)
  (dolist (s '("dogs" "goats" "cats"))                   (learn (list "do" s "walk" "on" "their" "legs") '("yes")))
  (dolist (s '("dogs" "goats" "cats" "horses"))          (learn (list "do" s "have" "legs") '("yes")))
  (dolist (s '("dogs" "goats" "cats" "horses" "snakes")) (learn (list "are" s "animals") '("yes")))
  (learn '("do" "snakes" "have" "legs") '("no"))
  (let ((tmp "phase7-temp.kb")
        (before (infer-strength '("do" "horses" "walk" "on" "their" "legs") '("yes"))))
    (save-network tmp)
    (reset)
    (check "after reset the concept graph is empty"
           (and (zerop (hash-table-count *concepts*)) (zerop (hash-table-count *concept-graph*))))
    (load-network tmp)
    (check "reloaded concept graph is non-empty"
           (and (plusp (hash-table-count *concepts*)) (plusp (hash-table-count *concept-graph*))))
    (check "reloaded concept graph reproduces horse's generalization strength"
           (= before (infer-strength '("do" "horses" "walk" "on" "their" "legs") '("yes"))))
    (ignore-errors (delete-file tmp)))

  ;; ----- adaptive threshold scales with each category's members (Phase 7 #3) -----
  (reset)
  (dolist (s '("dogs" "goats" "cats"))                   (learn (list "do" s "walk" "on" "their" "legs") '("yes")))
  (dolist (s '("dogs" "goats" "cats" "horses"))          (learn (list "do" s "have" "legs") '("yes")))
  (dolist (s '("dogs" "goats" "cats" "horses" "snakes")) (learn (list "are" s "animals") '("yes")))
  (learn '("do" "snakes" "have" "legs") '("no"))
  (let ((base (member-baseline "do walk on their legs" "yes")))
    (check "adaptive: a taught category has a positive member baseline" (plusp base))
    (check "adaptive: a strength at half the baseline is recognized"
           (recognized-strength-p (* 0.5 base) "do walk on their legs" "yes"))
    (check "adaptive: a strength at a twentieth of the baseline is not"
           (not (recognized-strength-p (* 0.05 base) "do walk on their legs" "yes")))
    (check "adaptive (no hardcoded cutoff): horse still included via the ratio"
           (recognizes-p "horses" "do walk on their legs" "yes"))
    (check "adaptive (no hardcoded cutoff): snake still excluded via the ratio"
           (not (recognizes-p "snakes" "do walk on their legs" "yes"))))

  (reset)
  (check "reset clears the concept graph"
         (and (zerop (hash-table-count *concepts*))
              (zerop (hash-table-count *concept-graph*))))

  ;; ===================== Knowledge base & training =====================
  (format t "~%Knowledge base & training tests~%")
  (reset)
  (let ((n (train-from-file "training-set.txt" :verbose nil)))
    (check "train-from-file loads the large starter set (>100 facts)" (> n 100))
    ;; horses & birds were taught legs/feet/run/animal but NEVER 'walk on their legs'
    (check "imported KB generalizes: horses walk on their legs"
           (infer-p '("do" "horses" "walk" "on" "their" "legs") '("yes")))
    (check "imported KB generalizes: birds walk on their legs"
           (infer-p '("do" "birds" "walk" "on" "their" "legs") '("yes")))
    (check "imported KB excludes legless animals: snakes"
           (not (infer-p '("do" "snakes" "walk" "on" "their" "legs") '("yes"))))
    (check "imported KB excludes legless animals: fish"
           (not (infer-p '("do" "fish" "walk" "on" "their" "legs") '("yes"))))
    (check "imported KB excludes vehicles: cars"
           (not (infer-p '("do" "cars" "walk" "on" "their" "legs") '("yes"))))
    (check "imported KB excludes legged furniture: tables"
           (not (infer-p '("do" "tables" "walk" "on" "their" "legs") '("yes"))))
    (check "imported KB excludes inert objects: rocks"
           (not (infer-p '("do" "rocks" "walk" "on" "their" "legs") '("yes")))))
  ;; export / import the whole KB (input net, outputs, associations, concept graph)
  (let ((tmp "kb-temp.kb"))
    (export-kb tmp)
    (reset)
    (check "after reset the KB is empty" (zerop (hash-table-count *dictionary*)))
    (import-kb tmp)
    (check "exported/imported KB still generalizes (horses walk on their legs)"
           (infer-p '("do" "horses" "walk" "on" "their" "legs") '("yes")))
    (check "exported/imported KB still excludes (snakes)"
           (not (infer-p '("do" "snakes" "walk" "on" "their" "legs") '("yes"))))
    (ignore-errors (delete-file tmp)))

  ;; ===================== String interface (parse sentences) =====================
  (format t "~%String interface tests~%")
  (check "tokenize lowercases and strips terminal punctuation"
         (equal '("do" "cats" "purr") (tokenize "Do cats purr?")))
  (check "tokenize keeps an embedded period (filename stays one token)"
         (equal '("file.kb") (tokenize "file.kb")))
  (check "tokenize keeps a decimal but drops the terminal period"
         (equal '("the" "value" "is" "3.14") (tokenize "The value is 3.14.")))
  (check "tokenize splits on a period followed by a space"
         (equal '("save" "my.kb" "now") (tokenize "save my.kb. now")))
  (reset)
  (train-from-file "training-set.txt" :verbose nil)
  (check "infer-p accepts a sentence string (horse generalizes in)"
         (infer-p "Do horses walk on their legs?" "yes"))
  (check "infer-p (string) still excludes snakes"
         (not (infer-p "Do snakes walk on their legs?" "yes")))
  (reset)
  (check "learn accepts strings (nil on a blank system)"
         (null (learn "Do cats purr?" "yes")))
  (check "respond accepts a string and recalls"
         (equal '("yes") (respond "Do cats purr?")))

  ;; ===================== Attention copy head (say X -> X) =====================
  (format t "~%Attention (copy / binding) tests~%")
  (reset)
  (learn "say dog" "dog")
  (learn "say cat" "cat")
  (learn "say bird" "bird")
  (check "copy: a taught filler is recalled (say dog)"
         (equal '("dog") (respond "say dog")))
  (check "copy: a NOVEL filler is copied (say horse -> horse)"
         (equal '("horse") (respond "say horse")))
  (check "copy: nonsense is copied too (say zorp -> zorp)"
         (equal '("zorp") (respond "say zorp")))
  (check "copy: an input with no learned cue is unaffected"
         (null (respond "the moon glows")))
  (reset)
  (learn "echo fish" "fish")
  (check "copy: a single example does not yet trigger copying"
         (null (copy-response '("echo" "whale"))))
  (check "copy cue persists across save / reload"
         (let ((tmp "copy-temp.kb"))
           (reset)
           (dolist (s '("dog" "cat" "bird")) (learn (format nil "say ~a" s) s))
           (save-network tmp) (reset) (load-network tmp)
           (prog1 (equal '("horse") (respond "say horse"))
             (ignore-errors (delete-file tmp)))))

  ;; ===================== Conversation memory (follow-ups) =====================
  (format t "~%Conversation memory tests~%")
  (reset)
  (train-from-file "training-set.txt" :verbose nil)
  (check "ask answers a direct question (do dogs have legs -> yes)"
         (equal '("yes") (ask "do dogs have legs?")))
  (check "follow-up 'and cats?' resolves against the previous turn"
         (equal '("do" "cats" "have" "legs") (resolve-followup '("and" "cats"))))
  (check "ask resolves + answers a follow-up (and cats? -> yes)"
         (equal '("yes") (ask "and cats?")))
  (check "ask resolves + answers 'what about snakes?' -> no"
         (equal '("no") (ask "what about snakes?")))
  (check "a full sentence is not treated as a follow-up"
         (progn (remember-turn '("do" "dogs" "have" "legs"))
                (equal '("the" "moon" "glows") (resolve-followup '("the" "moon" "glows")))))

  ;; ===================== Template / fragment composition =====================
  (format t "~%Composition tests~%")
  (reset)
  (learn "what is a dog" "a dog is an animal")
  (learn "what is a cat" "a cat is an animal")
  (check "composes a novel sentence for an unseen subject (horse)"
         (equal '("a" "horse" "is" "an" "animal") (respond "what is a horse")))
  (check "composes for another unseen subject (llama)"
         (equal '("a" "llama" "is" "an" "animal") (compose '("what" "is" "a" "llama"))))
  (check "a single example does not yet compose"
         (progn (reset) (learn "what is a dog" "a dog is an animal")
                (null (compose '("what" "is" "a" "horse")))))
  (check "templates persist across save / reload"
         (let ((tmp "tmpl-temp.kb"))
           (reset)
           (learn "what is a dog" "a dog is an animal")
           (learn "what is a cat" "a cat is an animal")
           (save-network tmp) (reset) (load-network tmp)
           (prog1 (equal '("a" "horse" "is" "an" "animal") (respond "what is a horse"))
             (ignore-errors (delete-file tmp)))))

  ;; ===================== Starter knowledge base =====================
  (format t "~%Starter knowledge-base tests~%")
  (reset)
  (let ((n (train-from-file "knowledge-base.txt" :verbose nil)))
    (check "knowledge-base.txt loads a large starter set (> 300 facts)" (> n 300))
    (check "starter KB: concept-graph recall (what color is the sky -> blue)"
           (equal '("blue") (ask "what color is the sky?")))
    (check "starter KB: copy head fires (say platypus -> platypus, never taught)"
           (equal '("platypus") (respond "say platypus")))
    (check "starter KB: composes for an unseen animal (what is a hippo)"
           (equal '("a" "hippo" "is" "an" "animal") (respond "what is a hippo")))
    (check "starter KB: a trait question generalizes (are tigers animals -> yes)"
           (equal '("yes") (ask "are tigers animals?"))))

  ;; ===================== Learned operations (how many X do you know) =====================
  (format t "~%Learned-operation tests~%")
  (reset)
  (train-from-file "knowledge-base.txt" :verbose nil)
  (check "operation: unknown before it is taught"
         (null (run-operation '("how" "many" "animals" "do" "you" "know"))))
  (learn "how many animals do you know" "count animals")     ; teach what the question MEANS
  (let ((animals (parse-integer (first (ask "how many animals do you know?")))))
    (check "operation: how many animals -> a positive count (computed)" (> animals 5))
    (check "operation generalizes to a new category via the slot (mammals)"
           (> (parse-integer (first (ask "how many mammals do you know?"))) 0))
    (check "operation: the count rises as the system learns a new member"
           (progn (learn "are dragons animals" "yes")
                  (> (parse-integer (first (ask "how many animals do you know?"))) animals)))
    (check "operation mapping persists across save / reload"
           (let ((tmp "op-temp.kb"))
             (save-network tmp) (reset) (load-network tmp)
             (prog1 (and (run-operation '("how" "many" "animals" "do" "you" "know")) t)
               (ignore-errors (delete-file tmp))))))

  ;; ===================== Distributed concept vectors (similarity by geometry) ===========
  (format t "~%Distributed-vector tests~%")
  (reset)
  (train-from-file "knowledge-base.txt" :verbose nil)
  (check "vectors: same-category similarity beats cross-category (dog~cat > dog~car)"
         (> (similarity "dog" "cat") (similarity "dog" "car")))
  (check "vectors: nearest neighbour of dog is an animal"
         (member (car (first (nearest "dog" 1)))
                 '("cat" "horse" "cow" "lion" "tiger" "wolf" "fox" "pig" "monkey" "kangaroo")
                 :test #'string=))
  (learn "what is similar to dog" "similar dog")     ; teach the 'similar' operation (geometry)
  (check "operation: 'what is similar to X' returns related words"
         (let ((r (ask "what is similar to horse"))) (and r (> (length r) 1))))
  (check "vectors persist across save / reload"
         (let ((tmp "vec-temp.kb") (before (similarity "dog" "cat")))
           (save-network tmp) (reset) (load-network tmp)
           (prog1 (> (similarity "dog" "cat") (* 0.9 before))
             (ignore-errors (delete-file tmp)))))
  ;; non-brittle membership recognition (k-NN over the vector space)
  (reset)
  (train-from-file "knowledge-base.txt" :verbose nil)
  (check "membership: a framed member is recognized (tiger -> animals)"
         (recognized-member-p "tiger" "animals"))
  (check "membership: a non-member is rejected (car -> animals)"
         (not (recognized-member-p "car" "animals")))
  ;; a novel word with several animal traits, but NEVER told "are zebus animals":
  (dolist (f '("do zebus have fur" "do zebus have legs" "can zebus run"
               "do zebus eat grass" "do zebus have a tail"))
    (learn f "yes"))
  (check "membership is non-brittle: a novel word is recognized by resemblance"
         (recognized-member-p "zebus" "animals"))
  (check "membership stays discriminating (the novel word is not a vehicle)"
         (not (recognized-member-p "zebus" "vehicles")))

  ;; ===================== Learning from raw text (read-text) =====================
  (format t "~%Read-text tests~%")
  (reset)
  (train-from-file "knowledge-base.txt" :verbose nil)
  (read-text "Quito is the capital of Ecuador. Ecuador is a country." :verbose nil)
  (check "read-text learns a relational fact from prose (capital of ecuador -> quito)"
         (equal '("quito") (ask "what is the capital of ecuador")))
  (check "read-text learns a membership fact from prose (ecuador is a country -> yes)"
         (equal '("yes") (ask "is ecuador a country")))
  (check "read-text reports how many sentences it read"
         (= 2 (read-text "The sky is high. Birds sing." :verbose nil)))
  ;; feed a whole prose corpus from a file; facts below are ONLY in prose.txt, not the KB
  (read-text-file "prose.txt" :verbose nil)
  (check "read-text-file: a new capital learned from the prose corpus (iceland -> reykjavik)"
         (equal '("reykjavik") (ask "what is the capital of iceland")))
  (check "read-text-file: 'who was X' learned from prose (ada lovelace -> a mathematician)"
         (equal '("a" "mathematician") (ask "who was ada lovelace")))
  (check "read-text: superlative fact (largest planet -> jupiter)"
         (equal '("jupiter") (ask "what is the largest planet")))
  (check "read-text: particle surname kept (van gogh -> a painter)"
         (equal '("a" "painter") (ask "who was van gogh")))

  ;; ===================== Facts counter + system stats =====================
  (format t "~%Stats tests~%")
  (reset)
  (check "facts-learned starts at 0 after reset" (= 0 *facts-learned*))
  (learn "do cats purr" "yes")
  (learn "do dogs bark" "yes")
  (check "facts-learned increments on every learn" (= 2 *facts-learned*))
  (let ((s (system-stats)))
    (check "system-stats reports the facts count" (= 2 (getf s :facts-learned)))
    (check "system-stats reports neurons and dendrites"
           (and (> (getf s :neurons) 0) (> (getf s :dendrites) 0))))
  (check "facts-learned persists across save / reload"
         (let ((tmp "stats-temp.kb"))
           (save-network tmp) (reset)
           (prog1 (progn (load-network tmp) (= 2 *facts-learned*))
             (ignore-errors (delete-file tmp)))))

  (format t "~%~d run, ~d failed -- ~a~%~%"
	  *tests-run* *tests-failed*
	  (if (zerop *tests-failed*) "ALL TESTS PASSED" "SOME TESTS FAILED"))
  *tests-failed*)

(run-tests)
