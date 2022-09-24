#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "Neuron.h"

inline double activation(double value)
{
    return value / (1 + abs(value));
}

template <size_t numInputs, size_t numHiddenNeurons, size_t numOutputs>
class NeuralNetwork
{
    public:
        std::array<Neuron<numInputs>, numHiddenNeurons> HiddenLayer;
        std::array<Neuron<numHiddenNeurons>, numOutputs> OutputLayer;
};

#endif // NEURAL_NETWORK_H