#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "NeuralNetwork.h"
#include "Dataset.h"

int main()
{
    srand(time(NULL));
    NeuralNetwork<2, 3, 1> network;
    

    Dataset<double, 4, 2, 1> a({1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0});
    for (auto &&data : a.Data)
    {
        for (auto &&input : data.Inputs)
            printf("Input: %lf\n", input);
        for (auto &&output : data.Outputs)
            printf("Output: %lf\n", output);
    }

    a.Shuffle();
    printf("Shuffle\n");
    for (auto &&data : a.Data)
    {
        for (auto &&input : data.Inputs)
            printf("Input: %lf\n", input);
        for (auto &&output : data.Outputs)
            printf("Output: %lf\n", output);
    }

    for (auto &&i : network.HiddenLayer)
    {
        printf("Bias: %lf\n", i.Bias);
    }
    return 0;
}