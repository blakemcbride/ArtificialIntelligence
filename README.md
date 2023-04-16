
# Artificial Intelligence

I've had a life-long interest in AI. In 1994, I started developing
ideas on how an AI system could be built. In 2014 I started to
implement the system in C++. Although I had many years of experience
with C, the C++ system quickly became unwieldy in terms of memory
management. I soon dropped the project and re-wrote it in Common
Lisp. That (partial) implementation was far easier to develop and was
incredibly simpler and shorter for the same functionality.

The architecture I designed had three major components:

1. Input
2. Computation
3. Output

Part #2 was supposed to be a typical neural network of one sort or
another. So, the real innovation was the input and output
components. The output component was supposed to be mostly the inverse
of the input component.

My development was only in the input section. I never got to the other
two. However, at least in terms of a first pass, the input section was
functional.

Given the development of ChatGPT and my advancing years, it is clear
this project will go no further under my steam. I, therefore, decided
to release these thoughts and code into the public domain. That's far
better than having these ideas disappear.

## Repo Structure

* C++ - this is the original C++ code.  It is incomplete and unfunctional.
* src - this is the Common Lisp code for the input module that is (first-pass) complete and functional
* notes - this is where I've dumped my (unfiltered) notes on the system

## Notes

The *notes* directory contains the untouched contemporaneous notes of the time.  They're
largely not in any particular order.  So, earlier notes often come
after later notes and visa versa.  They're in mostly random order.

The *notes* directory also contains an overview of the intended design.


Blake McBride
blake@mcbridemail.com
2023

