/* 
 * File:   Dendrite.h
 * Author: blake
 *
 * Created on July 26, 2009, 12:14 AM
 */

#ifndef _DENDRITE_H
#define	_DENDRITE_H

class DendriteInstance {
public:

    DendriteInstance(Neuron n) {
        iID = ++cLastID;
        cNumDendrites++;
        iNeuron = n;
    }

    Neuron GetNeuron() {
        return iNeuron;
    }

private:
    
    int iID;
    float iWeight;
    Neuron iNeuron;
    static int cLastID;
    static int cNumDendrites;
};

#endif	/* _DENDRITE_H */

