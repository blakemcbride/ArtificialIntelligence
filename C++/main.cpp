/* 
 * File:   main.cpp
 * Author: Blake McBride
 *
 * Created on July 26, 2009, 12:32 AM
 */


/*  Project started 12/8/02  */

#include "ai.h"


static	void	build_structure(NeuronList inp);
static	void	print_active(NeuronList active);


int	main(int argc, char *argv[])
{
	NeuronList	inp;

	while (inp = create_line())
		build_structure(inp);
	dump_dictionary();
	return 0;
}

static	NeuronList	extend_neurons(NeuronList active, Neuron neuron)
{
	NeuronList	next = CONS(neuron, NullNeuronList);

	for ( ; active ; active=FCDR(active) ) {
		Neuron		n = active->CAR();
		DendriteList	dl = n->Axon();
		if (!dl)
			dl = n->Add(new NeuronInstance);
		for ( ; dl ; dl=dl->CDR())
			next = CONS(dl->CAR()->GetNeuron(), next);
	}
	return next;
}

//  Find neurons which are common to all neurons on the list

static	NeuronList	findCommonLinks(NeuronList lst)
{
	if (!lst)
		return NULL;
	DendriteList	dl = lst->CAR()->Axon();
	if (!(lst=lst->CDR()))
		return dl ? CONS(dl->CAR()->GetNeuron(), NullNeuronList) : NULL;
	NeuronList	nl = NULL;
	for ( ; dl ; dl = dl->CDR()) {
		Neuron n = dl->CAR()->GetNeuron();
		NeuronList scn = lst;
		for ( ; scn && scn->CAR()->findLink(n) ; scn = scn->CDR());
		if (!scn)   //  links to all neurons
			nl = CONS(n, nl);
	}
	return nl;
}

static	void	link_neurons(NeuronList active)
{
	Neuron	t = new NeuronInstance;

	for ( ; active ; active=active->CDR() ) {
		Neuron		n = active->CAR();
		n->Add(t);
	}
}

static	void	build_structure(NeuronList inp)
{
	NeuronList	active = NULL;

	for (; inp  ; inp=FCDR(inp)) {
		Neuron neuron = inp->CAR();

		active = extend_neurons(active, neuron);

		print_active(active);
	}
	if (active) {
		NeuronList nl = findCommonLinks(active);
		if (nl)
			FreeList(nl);
		else
			link_neurons(active);
		FreeList(active);
	}
}

static	void	print_active(NeuronList active)
{
	cout <<  "Active Neurons" << endl;
	for ( ; active ; active=active->CDR()) {
		Neuron	n = active->CAR();
		if (n->isNamed())
			cout << n->ID() << "  "  <<  ((NamedNeuron)n)->Name()  <<  endl;
		else
			cout << n->ID() << endl;
	}
	cout << endl;
}

