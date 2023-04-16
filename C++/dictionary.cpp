

#include <map>
#include "ai.h"

using namespace std;

typedef map<string,NamedNeuron> Dictionary;

static	Dictionary dict;

NamedNeuron	find_neuron(char *name)
{
	return dict.count(name) ? dict[name] : NULL;
}

void	add_neuron(char *name, NamedNeuron neuron)
{
	dict[name] = neuron;
}

static	void	dump_neuron(Neuron n, int level)
{
	for (int i=level ; i-- ; )
		cout <<  "\t";
	if (n->isNamed())
		cout << n->ID() << " (" << ((NamedNeuron)n)->Name() << ")" << endl;
	else
		cout << n->ID() << endl;
	for (DendriteList dl = n->Axon() ; dl ; dl = dl->CDR()) {
		Dendrite d = dl->CAR();
		Neuron	p = d->GetNeuron();
		dump_neuron(p, level+1);
	}
}

void	dump_dictionary(void)
{
	Dictionary::iterator iter = dict.begin();
	Dictionary::iterator iter_end = dict.end();

	cout <<  "Dictionary dump" <<  endl;
	for ( ; iter != iter_end ; ++iter) {
		Neuron	n = (*iter).second;
		dump_neuron(n, 0);
	}
}
