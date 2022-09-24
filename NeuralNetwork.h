#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "Neuron.h"
#include "Dataset.h"

inline double Sigmoid(double value)
{
    return value / (1 + abs(value));
}

inline double deltaSigmoid(double value)
{
    return value * (1 - value);
}

template <size_t numInputs, size_t numHiddenNeurons, size_t numOutputs>
class NeuralNetwork
{
    public:
        std::array<Neuron<numInputs>, numHiddenNeurons> HiddenLayer;
        std::array<Neuron<numHiddenNeurons>, numOutputs> OutputLayer;

        void Train(auto&& dataset, int Epochs, double LearningRate)
        {
            for(int epoch = 0; epoch < Epochs; epoch++)
            {
                dataset.Shuffle();

                for(auto&& data: dataset.Data)
                {
                    // Compute hidden layer activation
                    for(auto& hiddenNeuron: HiddenLayer)
                    {
                        double activation = hiddenNeuron.Bias;
                        for(size_t i = 0; i < numInputs; i++)
                        {
                            activation += data.Inputs[i] * hiddenNeuron.Weights[i];
                        }
                        hiddenNeuron.Value = Sigmoid(activation);
                    }

                    // Compute output layer activation
                    for(auto& outputNeuron: OutputLayer)
                    {
                        double activation = outputNeuron.Bias;
                        
                        for(size_t i = 0; i < numHiddenNeurons; i++)
                        {
                            activation += outputNeuron.Value * outputNeuron.Weights[i];
                        }
                        outputNeuron.Value = Sigmoid(activation); 
                    }

                    printf("Input:");
                    for(auto& input: data.Inputs)
                    {
                        printf(" %g", input);
                    }

                    printf("\tOutput:");
                    for(auto& output: data.Outputs)
                    {
                        printf(" %g", output);
                    }

                    printf("\tPredict:");
                    for(auto& output: OutputLayer)
                    {
                        printf(" %g", output.Value);
                    }
                    printf("\n");

                    // Backprop

                    // Compute change in output weights
                    double deltaOutput[numOutputs];

                    for(int j = 0; j < numOutputs; j++)
                    {
                        double error = data.Outputs[j] - OutputLayer[j].Value;
                        deltaOutput[j] = error * deltaSigmoid(OutputLayer[j].Value);
                    }

                    // Compute change in hidden weights
                    double deltaHidden[numHiddenNeurons];
                    for(int j = 0; j < numHiddenNeurons; j++)
                    {
                        double error = 0.0f;
                        for(int k = 0; k < numOutputs; k++)
                        {
                            error += deltaOutput[k] * OutputLayer[k].Weights[j];
                        }
                        deltaHidden[j] = error * deltaSigmoid(HiddenLayer[j].Value);
                    }

                    // Apply change in output weights
                    for(int j = 0 ; j < numOutputs; j++)
                    {
                        OutputLayer[j].Bias += deltaOutput[j] * LearningRate;
                        for(int k = 0; k < numHiddenNeurons; k++)
                        {
                            OutputLayer[j].Weights[k] += HiddenLayer[k].Value * deltaOutput[j] * LearningRate;
                        }
                    }

                    // Apply change in hidden weights
                    for(int j = 0 ; j < numHiddenNeurons; j++)
                    {
                        HiddenLayer[j].Bias += deltaHidden[j] * LearningRate;
                        for(int k = 0; k < numInputs; k++)
                        {
                            HiddenLayer[j].Weights[k] += data.Inputs[k] * deltaHidden[j] * LearningRate;
                        }
                    }
                    // TODO: PRINT FINAL BIASES / WEIGHTS
                }
            }
        }

        std::array<double, numOutputs> Predict(std::array<double, numInputs> input)
        {
            std::array<double, numOutputs> predict{};
            for(auto& hiddenNeuron: HiddenLayer)
            {
                double activation = hiddenNeuron.Bias;
                for(size_t i = 0; i < numInputs; i++)
                {
                    activation += input[i] * hiddenNeuron.Weights[i];
                }
                hiddenNeuron.Value = Sigmoid(activation);
            }

            // Compute output layer activation
            size_t p = 0;
            for(auto& outputNeuron: OutputLayer)
            {
                double activation = outputNeuron.Bias;
                
                for(size_t i = 0; i < numHiddenNeurons; i++)
                {
                    activation += outputNeuron.Value * outputNeuron.Weights[i];
                }
                outputNeuron.Value = Sigmoid(activation); 
                predict[p] = outputNeuron.Value;
                p++;
            }
            return predict;
        }
};

#endif // NEURAL_NETWORK_H