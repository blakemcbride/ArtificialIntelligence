/* 
 * File:   List.h
 * Author: Blake McBride
 *
 * Created on July 26, 2009, 12:31 AM
 */

#ifndef _LIST_H
#    define	_LIST_H

template <class T>
class List {
public:
    inline T CAR();
    inline T SETCAR(T dat);
    inline List<T> *CDR();
    inline List<T> *SETCDR(List *nxt);
    inline T CADR();
    static List *FreeListStore;
    static List *consf(T a, List *b);
private:
    static const int BlockSize;
    static int NLists;
    T data;
    List *next;
};

template <class T>
inline T List<T>::CAR()
{
    return data;
}

template <class T>
inline T List<T>::SETCAR(T dat)
{
    return data = dat;
}

template <class T>
inline List<T> *List<T>::CDR()
{
    return next;
}

template <class T>
inline List<T> *List<T>::SETCDR(List *nxt)
{
    return next = nxt;
}

template <class T>
inline T List<T>::CADR()
{
    return next->data;
}

template <class T>
inline List<T> *CONS(T a, List<T> *b)
{
    if (List<T>::FreeListStore) {
	List<T> *res = List<T>::FreeListStore;
	List<T>::FreeListStore = List<T>::FreeListStore->CDR();
	res->SETCAR(a);
	res->SETCDR(b);
	return res;
    } else
	return List<T>::consf(a, b);
}

template <class T>
inline List<T> *FCDR(List<T> *x)
{
    List<T> *cdr = x->CDR();
    x->SETCDR(List<T>::FreeListStore);
    List<T>::FreeListStore = x;
    return cdr;
}

template <class T>
inline List<T> *FreeList(List<T> *x)
{
    for (; x; x = FCDR(x));
    return NULL;
}

template <class T>
inline List<T> *Reverse(List<T> *a)
{
    List<T> *b = NULL;

    for (; a; a = a->CDR())
	b = CONS(a->CAR(), b);
    return b;
}

template <class T>
List<T> *List<T>::FreeListStore = NULL;

template <class T>
const int List<T>::BlockSize = 10000;

template <class T>
int List<T>::NLists = 0;

template <class T>
List<T> *List<T>::consf(T a, List<T> *b)
{
    List<T> *n;

    if (!List<T>::FreeListStore) {
	int i;
	List<T> *p;

	p = List<T>::FreeListStore = new List<T>[BlockSize];
	for (i = 1; i++ < BlockSize; p++)
	    p->SETCDR(p + 1);
	p->SETCDR(NULL);
	NLists += BlockSize;
    }
    n = List<T>::FreeListStore;
    List<T>::FreeListStore = n->CDR();
    n->SETCAR(a);
    n->SETCDR(b);
    return n;
}


#endif	/* _LIST_H */

