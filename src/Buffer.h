#ifndef BUFFER_H
#define BUFFER_H

#pragma once
#include "torch/torch.h"
#include "vector"
#include <algorithm>

namespace ReplayBuffer
{
    struct Transition
    {   
        /*
        * \brief Stores the trasition of the agents on the environment
        */
        torch::Tensor obs;
        torch::Tensor actions;
        torch::Tensor obs_1;
        torch::Tensor rewards;
        bool done;
    };
    class Buffer
    {
        /*
        *   \brief Rplay buffer memory class
        */
    public:
        Buffer(size_t max_memory = 1000000, size_t batch_size = 256);
        ~Buffer();
        void storeTransition(Transition t);
        std::vector<Transition> sampleBuffer();
        bool ready();

    private:
        size_t batch_size;
        size_t max_memory;
        size_t buffer_counter;
        std::vector<Transition> memory;
    };
};
#endif