
clean:
	rm -f *.fas *.fasl *.lib *.lx64fsl *~

# 'make test' runs the smoke tests with SBCL (Blake's primary Lisp); exits non-zero on
# failure.  The suite itself stays portable -- it also loads cleanly under CLISP/CCL/ECL.
test:
	sbcl --non-interactive --eval '(load "tests.lisp")' --eval '(sb-ext:exit :code *tests-failed*)'

