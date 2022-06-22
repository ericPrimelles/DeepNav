#include "DDPGAgent.h"

DDPGAgent::DDPGAgent(int64_t Ain_dims, int64_t Aout_dims, std::vector<int64_t> Ah_dims, int64_t Cin_dims, int64_t Cout_dims, std::vector<int64_t> Ch_dims, float gamma, float tau) : a_n(Ain_dims, Aout_dims, Ah_dims),
                                                                                                                                                                                     c_n(Cin_dims, Cout_dims, Ch_dims),
                                                                                                                                                                                     target_a_n(Ain_dims, Aout_dims, Ah_dims),
                                                                                                                                                                                     target_c_n(Cin_dims, Cout_dims, Ch_dims),
                                                                                                                                                                                     a_optim(a_n->parameters(), torch::optim::AdamOptions(1e-4)),
                                                                                                                                                                                     c_optim(c_n->parameters(), torch::optim::AdamOptions(1e-3))

{   
    this->Ain_dims = Ain_dims;
    this->Aout_dims = Aout_dims;
    this->Ain_dims = Cin_dims;
    this->Aout_dims = Cout_dims;
    this->gamma = gamma;
    this->tau = tau;
}

DDPGAgent::~DDPGAgent()
{
}

void DDPGAgent::loadModel(std::string path, size_t i)
{
    std::string indx = std::to_string(i);
    torch::load(this->a_n, path + "ddpg-action-checkpoint-" + indx + ".pt" );
    torch::load(this->target_a_n, path + "ddpg-action-target-checkpoint-"+ indx + ".pt");
    torch::load(this->a_optim, path + "ddpg-action-optimizer-checkpoint-"+ indx + ".pt");
    torch::load(this->c_n, path + "ddpg-q-checkpoint-"+ indx + "-.pt");
    torch::load(this->target_c_n, path + "ddpg-q-target-checkpoint-"+ indx + ".pt");
    torch::load(this->c_optim, path + "ddpg-q-optimizer-checkpoint-"+ indx + ".pt");
}

void DDPGAgent::saveModel(std::string path, size_t i)
{
    std::string indx = std::to_string(i);
    torch::save(this->a_n, path + "ddpg-action-checkpoint-" + indx + ".pt");
    torch::save(this->target_a_n, path + "ddpg-action-target-checkpoint-" + indx + ".pt");
    torch::save(this->a_optim, path + "ddpg-action-optimizer-checkpoint-" + indx + ".pt");
    torch::save(this->c_n, path + "ddpg-q-checkpoint-" + indx + ".pt");
    torch::save(this->target_c_n, path + "ddpg-q-target-checkpoint-" + indx + ".pt");
    torch::save(this->c_optim, path + "ddpg-q-optimizer-checkpoint-" + indx + ".pt");
}

void DDPGAgent::updateParameters(float tau)
{
    // Updating critic parameters
    torch::autograd::GradMode::set_enabled(false);
    auto c_params = this->c_n->named_parameters(true);
    auto target_c_params = this->target_c_n->named_parameters(true);

    for (auto &val : c_params)
    {
        auto *t = target_c_params.find(val.key());
        t->data() = tau * val.value() + (1.0f - tau) * t->data();
    }
    auto c_buffers = this->c_n->named_buffers(true);
    auto target_c_buffers = this->target_c_n->named_buffers(true);
    for (auto &val : c_buffers)
    {
        auto *t = target_c_buffers.find(val.key());
        t->data() = tau * val.value() + (1.0f - tau) * t->data();
    }

    // Updating actor params (not in original code)
    auto a_params = this->a_n->named_parameters(true);
    auto target_a_params = this->target_a_n->named_parameters(true);

    for (auto &val : a_params)
    {
        auto *t = target_a_params.find(val.key());
        t->data() = tau * val.value() + (1.0f - tau) * t->data();
    }
    auto a_buffers = this->a_n->named_buffers(true);
    auto target_a_buffers = this->target_a_n->named_buffers(true);
    for (auto &val : a_buffers)
    {
        auto *t = target_a_buffers.find(val.key());
        t->data() = tau * val.value() + (1.0f - tau) * t->data();
    }

    torch::autograd::GradMode::set_enabled(true);
}

torch::Tensor DDPGAgent::sampleAction(torch::Tensor obs, bool use_random, bool use_net){
    torch::Tensor a; // Action placeholder
    if (use_net){
        a = this->a_n(obs);
    }
    else{
        a = torch::zeros(this->Ain_dims);
    }
    if (use_random){
        a = a + this->ou_sigma * torch::randn(this->Aout_dims);
    }
    return a;
}