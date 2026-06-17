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
(load "generation.lisp")
(load "induction.lisp")
(load "relations.lisp")
(load "processing.lisp")
(load "persist.lisp")
(load "llm.lisp")
(load "controller.lisp")
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
             (and (search "now the active file" out) (probe-file kb) t)))
    (let ((out (with-output-to-string (*standard-output*)
                 (with-input-from-string
                     (*standard-input*
                      (format nil ".load ~a~%say hi.~%yes.~%.quit~%" kb))
                   (main)))))
      (check ".load command restores a saved knowledge base"
             (and (search "now the active file" out)
                  (search "guess: hi there" out) t)))
    (let ((out (with-output-to-string (*standard-output*)
                 (with-input-from-string (*standard-input* (format nil ".list~%.quit~%"))
                   (main)))))
      (check ".list command lists the .kb files in the current directory"
             (and (search ".kb file" out) (search kb out) t)))
    (ignore-errors (delete-file *save-file*))
    (ignore-errors (delete-file kb)))

  ;; .load / .save make FILE the active file: after .load FILE, exiting auto-saves back to
  ;; FILE -- not to the default *save-file* (the bug this guards against).
  (let ((*save-file* "phase5-default.kb") (*starter-kb* nil)
        (active "phase5-active.kb"))
    (ignore-errors (delete-file *save-file*))
    (ignore-errors (delete-file active))
    ;; seed the active file with one fact, via .save
    (with-output-to-string (*standard-output*)
      (with-input-from-string (*standard-input*
                               (format nil "alpha.~%one.~%.save ~a~%.quit~%" active))
        (main)))
    ;; a fresh session at the default file: .load the active file, teach, then quit
    (let ((*save-file* "phase5-default.kb"))
      (with-output-to-string (*standard-output*)
        (with-input-from-string (*standard-input*
                                 (format nil ".load ~a~%beta.~%two.~%.quit~%" active))
          (main))))
    (import-kb active)
    (check ".load makes the loaded file the auto-save target (new fact lands in it)"
           (equal '("two") (respond "beta")))
    (check ".load leaves the default save file untouched"
           (not (probe-file "phase5-default.kb")))
    (ignore-errors (delete-file active)))

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
  (let ((n (train-from-file "generalization-test.txt" :verbose nil)))
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
  (train-from-file "generalization-test.txt" :verbose nil)
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
  (train-from-file "generalization-test.txt" :verbose nil)
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

  ;; ===================== Phase 8 -- Generation =====================
  (format t "~%Phase 8 (generation) tests~%")
  (reset)
  ;; Declarative prose -> a fact store generation can describe from.
  (read-text "Paris is the capital of France. France is a country in Europe.
              French is the language of France. Marseille is a city in France.
              The louvre is in France." :verbose nil)
  (let ((desc (join-words (respond "tell me about france"))))
    (check "describe assembles a multi-fact paragraph about a topic"
           (and (search "france is a country" (string-downcase desc))
                (search "capital is paris" (string-downcase desc))))
    (check "describe aggregates several facts (not a single stored response)"
           (and (search "europe" desc) (search "french" desc) (search "marseille" desc))))
  (check "describe works for a different request phrasing"
         (and (search "france" (string-downcase (join-words (respond "describe france")))) t))
  (check "describe returns nil for an unknown topic"
         (null (respond "tell me about florida")))
  ;; QA-derived facts feed describe too (so it works on the starter-KB style)
  (reset)
  (learn "what is the capital of japan" "tokyo")
  (learn "is japan a country" "yes")
  (check "describe uses facts derived from learned question->answer pairs"
         (let ((d (string-downcase (join-words (respond "tell me about japan")))))
           (and (search "japan is a country" d) (search "capital is tokyo" d))))
  ;; why: an is-a chain explanation
  (reset)
  (read-text "A cat is a mammal. A mammal is an animal." :verbose nil)
  (let ((w (string-downcase (join-words (respond "why is a cat an animal")))))
    (check "why explains via an is-a chain"
           (and (search "cat is an animal because" w)
                (search "cat is a mammal" w)
                (search "mammal is an animal" w))))
  (check "why returns nil when no explanatory chain is known"
         (null (respond "why is a cat a planet")))
  ;; the sequential transition model is grown and the fact store persists
  (check "reading text grows the transition model"
         (plusp (hash-table-count *transitions*)))
  (check "generation facts persist across save / reload"
         (let ((tmp "gen-temp.kb"))
           (reset)
           (read-text "A robin is a bird. The robin is in the garden." :verbose nil)
           (save-network tmp) (reset)
           (prog1 (and (load-network tmp)
                       (plusp (hash-table-count *facts*))
                       (search "robin is a bird"
                               (string-downcase (join-words (respond "tell me about robin")))))
             (ignore-errors (delete-file tmp)))))

  ;; ===================== Unified loader (auto-routing) =====================
  (format t "~%Unified loader tests~%")
  (reset)
  (let ((tmp "mixed-temp.txt"))
    (with-open-file (s tmp :direction :output :if-exists :supersede :if-does-not-exist :create)
      (format s "# a mixed knowledge file~%")
      (format s "ping => pong~%")                       ; supervised pair
      (format s "Paris is the capital of France.~%")    ; prose -> fact
      (format s ";; a comment line~%~%"))
    (multiple-value-bind (pairs psent pfacts) (load-knowledge tmp :verbose nil)
      (declare (ignore pfacts))
      (check "load-knowledge routes => lines as supervised pairs" (= 1 pairs))
      (check "load-knowledge routes other lines as prose" (= 1 psent)))
    (check "load-knowledge: the supervised pair is recalled"
           (equal '("pong") (respond "ping")))
    (check "load-knowledge: the prose fact is learned (describe finds it)"
           (search "capital is paris"
                   (string-downcase (join-words (respond "tell me about france")))))
    (ignore-errors (delete-file tmp)))

  ;; ===================== Streaming reader (large-file safe) =====================
  (format t "~%Streaming reader tests~%")
  (reset)
  (let ((tmp "stream-temp.txt"))
    (with-open-file (s tmp :direction :output :if-exists :supersede :if-does-not-exist :create)
      ;; a sentence that SPANS a newline -- the chunked reader treats newlines as whitespace
      ;; and splits only on . ! ?, so the file is never loaded whole.
      (format s "Reykjavik is the~%capital of Iceland.~%Quito is a city.~%"))
    (check "read-text-file streams and splits sentences across newlines"
           (progn (read-text-file tmp :verbose nil)
                  (equal '("reykjavik") (ask "what is the capital of iceland"))))
    (ignore-errors (delete-file tmp)))

  ;; ===================== Model caps + .config / .set =====================
  (format t "~%Model-cap tests~%")
  (reset)
  (check ".set changes a tunable"
         (progn (set-tunable "max-cooccur 1234") (eql 1234 *max-cooccur*)))
  (check ".set off resets a tunable to unlimited"
         (progn (set-tunable "max-cooccur off") (null *max-cooccur*)))
  (check "co-occurrence is pruned to its cap while learning"
         (let ((*prune-every* 1) (*max-cooccur* 3))
           (reset)
           (read-text "alpha beta gamma. delta epsilon zeta. eta theta iota. mu nu xi." :verbose nil)
           (<= (hash-table-count *cooccur*) 3)))

  ;; ===================== Resume: .read picks up where it left off =====================
  (format t "~%Resume / rewind tests~%")
  (reset)
  (let ((tmp "resume-temp.txt"))
    (with-open-file (s tmp :direction :output :if-exists :supersede :if-does-not-exist :create)
      (format s "Reykjavik is the capital of Iceland. Quito is a city.~%"))
    (read-text-file tmp :verbose nil)
    (check "a fully-read file is not re-read on the next .read"
           (= 0 (nth-value 0 (read-text-file tmp :verbose nil))))
    (check "the read offset persists across save / reload"
           (let ((kb "resume-off.kb"))
             (save-network kb) (reset)
             (prog1 (and (load-network kb)
                         (= 0 (nth-value 0 (read-text-file tmp :verbose nil))))  ; still consumed
               (ignore-errors (delete-file kb)))))
    (rewind-file tmp)
    (check "rewind makes the file re-readable from the start"
           (> (nth-value 0 (read-text-file tmp :verbose nil)) 0))
    (ignore-errors (delete-file tmp)))

  (format t "~%Tunable (.set parameter) persistence tests~%")
  (reset)
  (let ((kb "tunables-temp.kb")
        (*max-cooccur* 4242) (*read-extract* nil) (*read-max-mb* 7))
    (save-network kb)                                  ; save with non-default tunables
    (let ((*max-cooccur* nil) (*read-extract* t) (*read-max-mb* nil))  ; clobber to defaults
      (load-network kb)                                ; .load restores the saved settings
      (check "save/load restores a numeric tunable (max-cooccur)" (eql *max-cooccur* 4242))
      (check "save/load restores a boolean tunable (read-extract off)" (null *read-extract*))
      (check "save/load restores read-max-mb" (eql *read-max-mb* 7)))
    (ignore-errors (delete-file kb)))

  (format t "~%Per-learner toggle tests~%")
  (reset)
  (let ((*read-cooccur* nil))                            ; drop the O(n^2) learner
    (process-sentence (tokenize "a cat is an animal") nil)
    (check "read-cooccur off skips the co-occurrence learner" (zerop (hash-table-count *cooccur*)))
    (check "other learners still run with read-cooccur off"   (plusp (hash-table-count *transitions*))))
  (reset)
  (let ((*read-transitions-p* nil))
    (process-sentence (tokenize "a cat is an animal") nil)
    (check "read-transitions off skips the transition learner" (zerop (hash-table-count *transitions*))))

  (format t "~%Count-store merge (shard combine) tests~%")
  (flet ((nt (tab) (let ((s 0)) (maphash (lambda (k v) (declare (ignore k))
                                           (maphash (lambda (a b) (declare (ignore a)) (incf s b)) v)) tab) s)))
    (reset)
    (let ((*read-extract* nil) (txt "a cat is an animal. a dog runs fast. a cat is a mammal.")
          (kb "merge-temp.kb"))
      (read-text txt :verbose nil)
      (let ((co1 (nt *cooccur*)) (f1 (hash-table-count *facts*)))
        (save-network kb)
        (reset) (setf *read-extract* nil)               ; fresh model with the SAME text...
        (read-text txt :verbose nil)
        (merge-kb kb)                                    ; ...then merge the saved shard in
        (check "merge-kb sums co-occurrence counts" (= (nt *cooccur*) (* 2 co1)))
        (check "merge-kb keeps the same fact keys (additive strength)"
               (= (hash-table-count *facts*) f1)))
      (ignore-errors (delete-file kb))))

  #+(and sbcl sb-thread)
  (progn
    (format t "~%Parallel bulk-read tests (SBCL)~%")
    (flet ((nt (tab) (let ((s 0)) (maphash (lambda (k v) (declare (ignore k))
                                             (maphash (lambda (a b) (declare (ignore a)) (incf s b)) v)) tab) s)))
      (let ((tmp "par-temp.txt"))
        (with-open-file (s tmp :direction :output :if-exists :supersede :if-does-not-exist :create)
          (dotimes (i 3000)
            (format s "a cat is an animal. a dog runs fast. paris is the capital of france.~%")))
        (reset) (setf *read-extract* nil *read-workers* 1)   ; sequential baseline
        (read-text-file tmp :verbose nil)
        (let ((seq-co (nt *cooccur*)) (seq-tr (nt *transitions*)))
          (reset) (rewind-file tmp) (setf *read-extract* nil *read-workers* 4)   ; 4 workers
          (read-text-file tmp :verbose nil)
          (setf *read-workers* 1)                            ; restore default
          (check "parallel read yields identical co-occurrence totals" (= (nt *cooccur*) seq-co))
          (check "parallel read yields identical transition totals"    (= (nt *transitions*) seq-tr)))
        (ignore-errors (delete-file tmp)))))

  (format t "~%Part 5 (controller + LLM advisor) tests~%")
  ;; the LLM JSON reader (validates response parsing without a live API)
  (check "llm JSON reader extracts a nested field"
         (string= "hi" (jref (parse-json "{\"choices\":[{\"message\":{\"content\":\"hi\"}}]}")
                              "choices" 0 "message" "content")))
  (check "llm JSON reader decodes escapes"
         (string= (format nil "a~cb" #\Newline) (parse-json "\"a\\nb\"")))
  ;; the controller learns WHICH proposal to pick, from outcome feedback, with a mock LLM
  (reset)
  (let ((*provider* :mock)
        (*mock-fn* (lambda (p s) (declare (ignore s))
                     (if (search "Propose" p) (format nil "1. alpha~%2. beta~%3. gamma") "done")))
        (input "choose the best option for me"))
    (check "controller-propose parses the LLM's candidate list"
           (equal '("alpha" "beta" "gamma") (controller-propose input)))
    (dotimes (r 4)                                   ; reward only "beta"
      (multiple-value-bind (res chosen cands) (controller-respond input)
        (declare (ignore res cands))
        (controller-reward (if (search "beta" chosen) 1.0 -1.0))))
    (check "controller learns to select the rewarded candidate"
           (string= "beta" (controller-select input '("alpha" "beta" "gamma"))))
    (check "selector weight is positive for the rewarded (context,candidate)"
           (> (controller-score input "beta") 0.0))
    (check "selector weight is non-positive for a punished candidate"
           (<= (controller-score input "alpha") 0.0))
    ;; the learned policy persists with the knowledge base
    (let ((kb "controller-temp.kb") (w (controller-score input "beta")))
      (save-network kb)
      (reset)
      (check "controller policy is cleared by reset" (zerop (controller-score input "beta")))
      (load-network kb)
      (check "controller policy persists across save / reload"
             (< (abs (- (controller-score input "beta") w)) 0.001))
      (ignore-errors (delete-file kb))))

  ;; ===================== Phase 9 -- Learned relation discovery =====================
  (format t "~%Phase 9 (learned relations) tests~%")
  (reset)
  ;; enough "is a"/"is an" links to a shared category hub to cross the membership threshold
  (read-text "a robin is a bird. an eagle is a bird. an owl is a bird.
              a dog is an animal. a cat is an animal. a horse is an animal." :verbose nil)
  (check "relation discovery: 'is a' is LEARNED to be a membership connector (not hardcoded)"
         (member "is a" (membership-connectors) :test #'string=))
  (multiple-value-bind (s c cat cls) (relation-of "a sparrow is a bird")
    (declare (ignore c))
    (check "relation-of classifies a novel 'X is a Y' as membership"
           (eq cls :membership))
    (check "relation-of extracts the subject and category heads"
           (and (string= s "sparrow") (string= cat "bird"))))
  (multiple-value-bind (s c cat cls) (relation-of "the bird that sings sweetly is an animal")
    (declare (ignore c cls))
    (check "relation-of finds the head THROUGH a relative clause (bird, not sings)"
           (and (string= s "bird") (string= cat "animal"))))
  (check "learned relation discovery persists across save / reload"
         (let ((tmp "rel-temp.kb"))
           (save-network tmp) (reset)
           (prog1 (and (load-network tmp)
                       (member "is a" (membership-connectors) :test #'string=))
             (ignore-errors (delete-file tmp)))))

  ;; ===================== Phase 10 -- In-context learning (induction head) ==============
  (format t "~%Phase 10 (induction / in-context learning) tests~%")
  (check "induction: complete a repeated motif (a b c a b c a -> b)"
         (string= "b" (complete '("a" "b" "c" "a" "b" "c" "a"))))
  (check "induction: continue a length-2 motif"
         (equal '("y" "x" "y" "x") (continue-sequence '("x" "y" "x" "y" "x") 4)))
  (check "induction: novel tokens work (structural, not memorized)"
         (string= "wug" (complete '("fip" "wug" "fip" "wug" "fip"))))
  (check "induction-request-p detects a continue request"
         (and (induction-request-p '("continue" "red" "green" "red")) t))
  (check "induction-request-p ignores a normal question"
         (null (induction-request-p '("do" "cats" "purr"))))
  (check "respond owns a continue request (in-context continuation)"
         (string= "green" (first (respond "continue red green red green red"))))

  (format t "~%~d run, ~d failed -- ~a~%~%"
	  *tests-run* *tests-failed*
	  (if (zerop *tests-failed*) "ALL TESTS PASSED" "SOME TESTS FAILED"))
  *tests-failed*)

(run-tests)
