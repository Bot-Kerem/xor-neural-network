#ifndef DATASET_H
#define DATASET_H

#include <array>
#include <stdlib.h>

template <typename T, size_t size, size_t InputSize, size_t OutputSize>
class Dataset
{
    public:
        struct _Data
        {
            std::array<T, InputSize> Inputs;
            std::array<T, OutputSize> Outputs;
        };

        std::array<_Data, size> Data;

        Dataset(std::array<_Data, size> dataset)
        :   Data{dataset} {}
        void Shuffle()
        {
            for(size_t i = 0; i < size - 1; i++)
            {
                size_t j = i + rand() / (RAND_MAX / (size - 1) + 1);
                auto t = Data[j];
                Data[j] = Data[i];
                Data[i] = t;
            }
        }
};

#endif // DATASET_H
