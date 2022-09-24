#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "NeuralNetwork.h"
#include "Dataset.h"

int main()
{
    srand(time(NULL));
    NeuralNetwork<2, 3, 1> network;
    
    Dataset<double, 4, 2, 1> dataset({1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0});
    network.Train(dataset, 10000, 0.1);

    return 0;
}