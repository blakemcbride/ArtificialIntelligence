;;;; llm.lisp -- a small, dependency-free backend that lets the system call an LLM as a tool,
;;;; either a LOCAL model (Ollama / llama.cpp) or a CLOUD model (Anthropic / OpenAI).  Part 5
;;;; of SystemAnalysis.md (controller + LLM advisor) calls this to PROPOSE and to EXECUTE.
;;;;
;;;; Design: no external Lisp libraries (keeps the project loadable on CLISP/SBCL/CCL/ECL).
;;;; HTTP is done by shelling out to `curl` (universally available), with the request body piped
;;;; on stdin so nothing has to be escaped onto the command line.  A compact JSON reader parses
;;;; the response.  A `:mock` provider returns canned text with no network, so the controller's
;;;; LEARNING can be exercised offline and in the test suite.
;;;;
;;;; Usage:
;;;;   (setf llm:*provider* :ollama    llm:*model* "llama3.2")          ; local
;;;;   (setf llm:*provider* :anthropic llm:*model* "claude-3-5-sonnet-latest")  ; needs ANTHROPIC_API_KEY
;;;;   (setf llm:*provider* :openai    llm:*model* "gpt-4o-mini")       ; needs OPENAI_API_KEY
;;;;   (llm-complete "Say hello in three words." :system "Be terse.")
;;;; Live calls (curl) currently require SBCL's run-program; the :mock provider works everywhere.

(defpackage "llm"
  (:use "COMMON-LISP")
  (:export "LLM-COMPLETE" "*PROVIDER*" "*MODEL*" "*MOCK-FN*" "*TIMEOUT*" "*MAX-TOKENS*"
	   "*OLLAMA-URL*" "*OPENAI-URL*" "*ANTHROPIC-URL*" "*ANTHROPIC-VERSION*"
	   "PARSE-JSON" "JREF" "JSON-ESCAPE"))
(in-package "llm")
(provide "llm")

(defparameter *provider* :mock
  "Which backend LLM-COMPLETE uses: :mock (no network), :ollama, :openai, or :anthropic.")
(defparameter *model* nil "Model name; NIL = a per-provider default (see default-model).")
(defparameter *timeout* 120 "curl --max-time, seconds.")
(defparameter *max-tokens* 1024 "Max tokens to generate (Anthropic requires it).")
(defparameter *ollama-url*    "http://localhost:11434/api/generate")
(defparameter *openai-url*    "https://api.openai.com/v1/chat/completions")
(defparameter *anthropic-url* "https://api.anthropic.com/v1/messages")
(defparameter *anthropic-version* "2023-06-01")
(defparameter *mock-fn*
  (lambda (prompt system) (declare (ignore system))
    (format nil "(:mock -- set llm:*mock-fn*; prompt was ~d chars)" (length prompt)))
  "Function (prompt system) -> string, used when *provider* is :mock.")

(defun default-model ()
  (ecase *provider*
    (:mock "mock")
    (:ollama "llama3.2")
    (:openai "gpt-4o-mini")
    (:anthropic "claude-3-5-sonnet-latest")))

(defun getenv (name)
  #+sbcl (sb-ext:posix-getenv name)
  #+ccl (ccl:getenv name)
  #+clisp (ext:getenv name)
  #+ecl (ext:getenv name)
  #-(or sbcl ccl clisp ecl) nil)

;;; --- JSON: escape (for request bodies) ------------------------------------------------
(defun json-escape (s)
  (with-output-to-string (o)
    (loop for c across s do
      (case c
	(#\" (write-string "\\\"" o)) (#\\ (write-string "\\\\" o))
	(#\Newline (write-string "\\n" o)) (#\Return (write-string "\\r" o))
	(#\Tab (write-string "\\t" o))
	(t (write-char c o))))))
(defun jstr (x) (concatenate 'string "\"" (json-escape (if (stringp x) x (princ-to-string x))) "\""))

;;; --- JSON: a compact recursive-descent reader (objects -> alists, arrays -> lists) -----
(defun parse-json (s)
  (let ((i 0) (n (length s)))
    (labels ((peek () (and (< i n) (char s i)))
	     (adv () (incf i))
	     (ws () (loop while (and (< i n) (member (char s i) '(#\Space #\Tab #\Newline #\Return))) do (incf i)))
	     (expect (c) (if (and (< i n) (char= (char s i) c)) (incf i) (error "JSON: expected ~a at ~a" c i)))
	     (lit (word v) (loop for ch across word do (expect ch)) v)
	     (jstring ()
	       (expect #\") (let ((o (make-string-output-stream)))
		 (loop (let ((c (char s i))) (adv)
			 (cond ((char= c #\") (return))
			       ((char= c #\\)
				(let ((e (char s i))) (adv)
				  (case e (#\" (write-char #\" o)) (#\\ (write-char #\\ o)) (#\/ (write-char #\/ o))
				    (#\n (write-char #\Newline o)) (#\t (write-char #\Tab o)) (#\r (write-char #\Return o))
				    (#\b (write-char #\Backspace o)) (#\f (write-char #\Page o))
				    (#\u (let ((code (parse-integer s :start i :end (+ i 4) :radix 16)))
					   (incf i 4) (write-char (code-char code) o)))
				    (t (write-char e o)))))
			       (t (write-char c o)))))
		 (get-output-stream-string o)))
	     (jnumber ()
	       (let ((start i))
		 (when (and (< i n) (char= (char s i) #\-)) (adv))
		 (loop while (and (< i n) (or (digit-char-p (char s i)) (member (char s i) '(#\. #\e #\E #\+ #\-)))) do (adv))
		 (let ((tok (subseq s start i)))
		   (if (find-if (lambda (c) (member c '(#\. #\e #\E))) tok)
		       (let ((*read-default-float-format* 'double-float)) (values (read-from-string tok)))
		       (parse-integer tok)))))
	     (jobject ()
	       (expect #\{) (ws) (let (acc)
		 (unless (char= (peek) #\})
		   (loop (ws) (let ((k (jstring))) (ws) (expect #\:) (push (cons k (jvalue)) acc))
			 (ws) (if (and (peek) (char= (peek) #\,)) (adv) (return))))
		 (ws) (expect #\}) (nreverse acc)))
	     (jarray ()
	       (expect #\[) (ws) (let (acc)
		 (unless (char= (peek) #\])
		   (loop (push (jvalue) acc) (ws) (if (and (peek) (char= (peek) #\,)) (adv) (return))))
		 (ws) (expect #\]) (nreverse acc)))
	     (jvalue ()
	       (ws) (let ((c (peek)))
		 (cond ((null c) (error "JSON: unexpected end"))
		       ((char= c #\{) (jobject)) ((char= c #\[) (jarray)) ((char= c #\") (jstring))
		       ((or (digit-char-p c) (char= c #\-)) (jnumber))
		       ((char= c #\t) (lit "true" :true)) ((char= c #\f) (lit "false" :false))
		       ((char= c #\n) (lit "null" :null))
		       (t (error "JSON: unexpected ~a at ~a" c i))))))
      (prog1 (jvalue) (ws)))))

(defun jref (obj &rest path)
  "Navigate a parsed-JSON structure: string keys index objects (alists), integers index arrays."
  (dolist (k path obj)
    (setf obj (cond ((integerp k) (nth k obj))
		    ((stringp k) (cdr (assoc k obj :test #'string=)))
		    (t (error "jref: bad key ~a" k))))))

;;; --- HTTP via curl (request body on stdin) --------------------------------------------
(defun run-curl (url body &optional headers)
  "POST BODY (a JSON string) to URL with curl; return the response body string.  Signals on error."
  #-sbcl (declare (ignore url body headers))
  #-sbcl (error "llm: live calls need SBCL's run-program (or add a curl shim for your Lisp); use :provider :mock")
  #+sbcl
  (let ((args (append (list "-sS" "--max-time" (princ-to-string *timeout*)
			    "-X" "POST" url "-H" "Content-Type: application/json")
		      (loop for h in headers append (list "-H" h))
		      (list "--data-binary" "@-")))
	(out (make-string-output-stream)) (err (make-string-output-stream)))
    (let ((proc (handler-case
		    (sb-ext:run-program "curl" args :search t :wait t
					:input (make-string-input-stream body) :output out :error err)
		  (error (e) (error "llm: could not run curl (~a)" e)))))
      (let ((code (sb-ext:process-exit-code proc)))
	(unless (eql code 0)
	  (error "llm: curl exited ~a: ~a" code (get-output-stream-string err)))
	(get-output-stream-string out)))))

;;; --- the one public entry point -------------------------------------------------------
(defun llm-complete (prompt &key system (provider *provider*) (model (or *model* (default-model))))
  "Send PROMPT (with optional SYSTEM instruction) to the configured LLM; return its text reply.
   PROVIDER is :mock | :ollama | :openai | :anthropic."
  (ecase provider
    (:mock (funcall *mock-fn* prompt system))
    (:ollama
     (let* ((body (format nil "{\"model\":~a,\"prompt\":~a,~@[\"system\":~a,~]\"stream\":false}"
			  (jstr model) (jstr prompt) (and system (jstr system))))
	    (data (parse-json (run-curl *ollama-url* body))))
       (or (jref data "response") (error "llm/ollama: no response field: ~a" data))))
    (:openai
     (let* ((key (or (getenv "OPENAI_API_KEY") (error "llm/openai: set OPENAI_API_KEY")))
	    (msgs (format nil "[~@[{\"role\":\"system\",\"content\":~a},~]{\"role\":\"user\",\"content\":~a}]"
			  (and system (jstr system)) (jstr prompt)))
	    (body (format nil "{\"model\":~a,\"messages\":~a}" (jstr model) msgs))
	    (data (parse-json (run-curl *openai-url* body
					(list (format nil "Authorization: Bearer ~a" key))))))
       (or (jref data "choices" 0 "message" "content") (error "llm/openai: no content: ~a" data))))
    (:anthropic
     (let* ((key (or (getenv "ANTHROPIC_API_KEY") (error "llm/anthropic: set ANTHROPIC_API_KEY")))
	    (body (format nil "{\"model\":~a,\"max_tokens\":~a,~@[\"system\":~a,~]\"messages\":[{\"role\":\"user\",\"content\":~a}]}"
			  (jstr model) *max-tokens* (and system (jstr system)) (jstr prompt)))
	    (data (parse-json (run-curl *anthropic-url* body
					(list (format nil "x-api-key: ~a" key)
					      (format nil "anthropic-version: ~a" *anthropic-version*))))))
       (or (jref data "content" 0 "text") (error "llm/anthropic: no text: ~a" data))))))
