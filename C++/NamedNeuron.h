/* 
 * File:   NamedNeuron.h
 * Author: blake
 *
 * Created on July 26, 2009, 12:34 AM
 */

#ifndef _NAMEDNEURON_H
#define	_NAMEDNEURON_H



class  NamedNeuronInstance : public NeuronInstance {
private:
	string	iName;
public:
	NamedNeuronInstance(char *name) {
		iName = name;
	}
	string	Name() {
		return iName;
	}
	int	isNamed() {  return 1; }
};



#endif	/* _NAMEDNEURON_H */

