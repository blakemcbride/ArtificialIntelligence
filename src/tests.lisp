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
  (let ((*save-file* "phase5-temp.sexp"))
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

  ;; ===================== Phase 6 -- Persistence =====================
  (format t "~%Phase 6 tests~%")
  (reset)
  (learn '("say" "hi") '("hi" "there"))
  (learn '("the" "dog" "barks") '("woof"))
  (let ((tmp "phase6-temp.sexp")
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
           (null (load-network "no-such-file-zzz.sexp")))
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
  (let ((tmp "phase7-temp.sexp")
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
  (let ((tmp "kb-temp.sexp"))
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

  (format t "~%~d run, ~d failed -- ~a~%~%"
	  *tests-run* *tests-failed*
	  (if (zerop *tests-failed*) "ALL TESTS PASSED" "SOME TESTS FAILED"))
  *tests-failed*)

(run-tests)
