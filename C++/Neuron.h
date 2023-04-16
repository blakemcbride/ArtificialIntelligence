/* 
 * File:   Neuron.h
 * Author: blake
 *
 * Created on July 26, 2009, 12:36 AM
 */

#ifndef _NEURON_H
#define	_NEURON_H


class NeuronInstance {
private:
	static	int	cLastID;
	static	int	cNumNeurons;

	int	iID;
	int	iThreshold;
	int	iCurrentValue;
	DendriteList	iAxon;

public:
	NeuronInstance() {
		iID = ++cLastID;
		cNumNeurons++;
	}
	int	ID() { return iID; }
	DendriteList	Axon() { return iAxon; }
	DendriteList	Add(Neuron n) {
		iAxon = CONS(new DendriteInstance(n), iAxon);
		return iAxon;
	}

//  Return dendrite which links neuron a directly to neuron b

	Dendrite   findLink(Neuron b) {
		for (DendriteList dl = iAxon ; dl ; dl = dl->CDR()) {
			Dendrite  d = dl->CAR();
			if (b == d->GetNeuron())
				return d;
		}
		return NULL;
	}

//  Scan dl for first dendrite which links to a neuron which b also links to
//  Return DendriteList whose car is the link

	DendriteList	findCommonLink(DendriteList dl) {
		for ( ; dl ; dl = dl->CDR()) {
			Dendrite  d = dl->CAR();
			if (this->findLink(d->GetNeuron()))
				return dl;
		}
		return NULL;
	}
	virtual	int	isNamed() {  return 0; }
	void	print(void);
};



#endif	/* _NEURON_H */

