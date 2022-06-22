#include "Buffer.h"

#include <random>

ReplayBuffer::Buffer::Buffer(size_t max_memory, size_t batch_size)
{

    this->max_memory = max_memory;
    this->batch_size = batch_size;
    this->buffer_counter = 0;
    this->memory = std::vector<ReplayBuffer::Transition>(max_memory);
}

ReplayBuffer::Buffer::~Buffer()
{
}

void ReplayBuffer::Buffer::storeTransition(ReplayBuffer::Transition t)
{
    if (this->buffer_counter < this->max_memory)
    {
        memory[this->buffer_counter] = t;
        this->buffer_counter += 1;
    }
    else
    {
        this->buffer_counter = 0;
        this->storeTransition(t);
    }
}

std::vector<ReplayBuffer::Transition> ReplayBuffer::Buffer::sampleBuffer()
{
    static std::mt19937 mt = std::mt19937{std::random_device{}()};
    std::vector<ReplayBuffer::Transition> sampled;
    sampled.reserve(this->batch_size);

    std::sample(memory.begin(), memory.begin() + this->buffer_counter, std::back_inserter(sampled), this->batch_size, mt);
    
    return sampled;
}

bool ReplayBuffer::Buffer::ready()
{
    return this->buffer_counter >= this->batch_size;
}