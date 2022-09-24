#ifndef NEURON_H
#define NEURON_H

#include <array>
#include <stdlib.h>

template <size_t numWeights>
class Neuron
{
    public:
        std::array<double, numWeights> Weights;
        double Bias;
        Neuron()
        :   Bias{(double)rand() / (double)RAND_MAX}
        {
            for (auto&& Weight : Weights)
            {
                Weight = (double)rand() / (double)RAND_MAX;
            }
        }
        
        
};

#endif // NEURON_H