/* 
 * File:   ai.h
 * Author: Blake McBride
 *
 * Created on July 26, 2009, 12:12 AM
 */

#ifndef _AI_H
#define	_AI_H


#include <iostream>
#include <string>

using std::string;
using std::cout;
using std::endl;
using std::cin;

class	NeuronInstance;
class	NamedNeuronInstance;
class	DendriteInstance;

typedef	NeuronInstance		*Neuron;
typedef	NamedNeuronInstance	*NamedNeuron;
typedef	DendriteInstance	*Dendrite;



#include "List.h"

typedef	List<Neuron>		*NeuronList;
typedef	List<Dendrite>		*DendriteList;


#define	NullNeuronList		(NeuronList)0
#define	NullDendriteList	(DendriteList)0


#include "Dendrite.h"
#include "Neuron.h"
#include "NamedNeuron.h"

NeuronList	create_line(void);
NamedNeuron	find_neuron(char *name);
void	add_neuron(char *name, NamedNeuron neuron);


void	dump_dictionary(void);


#endif	/* _AI_H */

