#ifndef DDPGAGENT_H
#define DDPGAGENT_H

#pragma once
#include <vector>
#include "torch/torch.h"
#include <string>

struct NNImpl : torch::nn::Module {
	/*
	* \brief Neural network implementation
	* \param NNImpl Class constructors
	* \param forward Foward pass implementation
	*/
	NNImpl(int64_t in_dim, int64_t out_dim, std::vector<int64_t> h_dims) {
		/*
		* \brief Neural network constructor
		* \param in_dim Input layer dimmension -> int
		* \param out_dim Output layer dimmension -> int
		* \param h_dim Hidden layer dimmension -> vector of ints
		*/
		int64_t prev_dim = in_dim; // input layer assingment
		for (size_t i = 0; i < h_dims.size(); ++i) { // For each layer
			layers.emplace_back(prev_dim, h_dims[i]); // Assemble layers
			register_module("linear" + std::to_string(i), layers.back()); // Register layer 

			prev_dim = h_dims[i]; // prev_dim = current hidden layer
		}
		layers.emplace_back(prev_dim, out_dim); // Assembling output layer
		register_module("linear" + std::to_string(h_dims.size()), layers.back());
	};
    

	torch::Tensor forward(torch::Tensor x) {
		/*
		* \brief Forward step implememntation
		* \param x layer to step forward -> tensor
		*/
		torch::Tensor ret = x; // Input reading
		for (size_t i = 0; i < layers.size(); ++i) { 
			ret = layers[i]->forward(ret); // forward hidden layer i
			if (i + 1 < layers.size()) {  // if current layer is not last layer
				ret = torch::relu(ret);  // relu(layer)
			}
		}
		return ret;
	};

	std::vector<torch::nn::Linear> layers; // Model placeholder
};
TORCH_MODULE(NN);
class DDPGAgent
{
	/* 
	*	\brief Single DDPg agent implementation
	*/ 
public:
    DDPGAgent(int64_t Ain_dims, int64_t Aout_dims, std::vector<int64_t> Ah_dims, int64_t Cin_dims, int64_t Cout_dims, std::vector<int64_t> Ch_dims, float gamma=0.9,float tau=1);
    ~DDPGAgent();
	void updateParameters(float tau);
	void saveModel(std::string path, size_t i);
	void loadModel(std::string path, size_t i);
	torch::Tensor sampleAction(torch::Tensor obs, bool use_random, bool use_net=true);
	
	NN a_n,  c_n,  target_a_n,  target_c_n;
    torch::optim::Adam  a_optim;
    torch::optim::Adam  c_optim;
	
private:
    float gamma, tau, ou_theta = 0.15f, ou_sigma = 0.4f;
    int64_t Ain_dims, Aout_dims, Cin_dims, Cout_dims;
	


    

};

#endif