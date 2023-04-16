
#include <ctype.h>
#include "ai.h"


NeuronList	create_line(void)
{
	char	word[256], *p;
	int	end = 0;
	NamedNeuron	neuron;
	NeuronList	res = NULL, rev;

	while (!end) {
		cin >> word;
		for (p=word ; *p  &&  *(p+1) ; p++)
			*p = tolower(*p);
		if (end = (*p == '.'  ||  *p == '!'  ||  *p == '?')) {
			if (p == word)
				break;
			*p = '\0';
		} else
			*p = tolower(*p);
		if (!(neuron = find_neuron(word))) {
			neuron = new NamedNeuronInstance(word);
			add_neuron(word, neuron);
		}
		res = CONS((Neuron)neuron, res);
	}
	rev = Reverse(res);
	FreeList(res);
	return rev;
}


