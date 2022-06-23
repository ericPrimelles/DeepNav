#include "MADDPG.h"
#include <iostream>
#include <fstream>

using std::cout, std::endl;

MADDPG::MADDPG(Environment *sim, int64_t Ain_dims, int64_t Aout_dims, std::vector<int64_t> Ah_dims, int64_t Cin_dims, int64_t Cout_dims, std::vector<int64_t> Ch_dims,
               size_t scenario, float alpha, float beta, size_t fc1, size_t fc2, size_t T, float gamma, float tau,
               std::string path, size_t batch_size, size_t max_memory, size_t k_epchos)
{
    this->env = sim;
    this->Ain_dims = Ain_dims;
    this->Aout_dims = Aout_dims;
    this->Cin_dims = Cin_dims;
    this->Cout_dims = Cout_dims;
    this->batch_size = batch_size;
    this->n_agents = this->env->getNAgents();
    this->scenario = scenario;
    this->alpha = alpha;
    this->beta = beta;
    this->fc1 = fc1;
    this->fc2 = fc2;
    this->gamma = gamma;
    this->tau = tau;
    this->path = path;
    this->T = T;
    this->max_memory = max_memory;
    this->k_epochs = k_epchos;

    this->agents.reserve(this->n_agents);
    this->memory = new ReplayBuffer::Buffer(max_memory, batch_size);
    for (size_t i = 0; i < n_agents; i++)
    {
        this->agents.push_back(new DDPGAgent(this->Ain_dims, this->Aout_dims, Ah_dims, this->Cin_dims, this->Cout_dims, Ch_dims,
                                             this->gamma, this->tau));
        this->agents[i]->updateParameters(1.0f);
    }
}

MADDPG::~MADDPG()
{
    for (size_t i = 0; i < this->n_agents; i++)
    {
        delete agents[i];
    }
}

void MADDPG::saveCheckpoint()
{
    cout << "...saving chekpoint..." << endl;

    for (size_t i = 0; i < this->n_agents; i++)
    {
        agents[i]->saveModel(this->path, i);
    }
}

void MADDPG::loadCheckpoint()
{
    cout << "...loading chekpoint..." << endl;

    for (size_t i = 0; i < this->n_agents; i++)
    {
        agents[i]->loadModel(this->path, i);
    }
}

torch::Tensor MADDPG::chooseAction(torch::Tensor obs, bool use_rnd, bool use_net)
{
    torch::Tensor actions = torch::zeros({(int64_t)this->n_agents, 2}, torch::dtype(torch::kFloat32));

    for (size_t i = 0; i < this->n_agents; i++)
    {
        actions[i] = this->agents[i]->sampleAction(obs[i], use_rnd, use_net);

        // std::cout << i << std::endl;
    }
    // std::cout << actions[0].sizes() << endl;
    return actions;
}

void MADDPG::Train()
{

    // Creating a base memory
    ReplayBuffer::Transition a;
    std::vector<ReplayBuffer::Transition> sampledTrans;
    float traj_reward = 0.0f;
    float traj_q_loss = 0.0f;
    float traj_a_loss = 0.0f;
   // Starting training
    float avg_reward = 0.0f;
    for (size_t epochs = 0; epochs < k_epochs; epochs++)
    {
        std::cout << "Training" << std::endl;
        std::cout << "\n\n\n";

        env->reset();
        //  Colecting some new experiences
        float avg_reward = 0.0f;
        float step_rewards = 0.0f;

        
        for (size_t i = 0; i < T && !this->env->isDone(); i++)
        {
            cout << "Epoch: " << epochs << "/" << k_epochs << " ";
            std::cout << "TimeStep:" << i << "/" << T << " Rewards:" << "    " << avg_reward << std::endl;
            a.obs = env->getObservation();
            a.actions = this->chooseAction(a.obs);
            a.rewards = env->step(a.actions);
            a.obs_1 = env->getObservation();
            a.done = env->isDone();
            this->memory->storeTransition(a);

            sampledTrans = this->memory->sampleBuffer();
            this->learn(sampledTrans);
            
            
            
            step_rewards += torch::mean(a.rewards).item<float>();
            avg_reward = step_rewards / (i + 1);
            sampledTrans = memory->sampleBuffer();
        }
        std::ofstream write;
        write.open("rewards.txt", std::ios::out | std::ios::app);
        if (write.is_open())
        {
            write << avg_reward << "\n";
        }

        write.close();

        if (epochs % 10 == 0)
        {
            std::cout << "Saving..." << std::endl;
            this->saveCheckpoint();
        }
        // this->visualize();
    }
}

void MADDPG::Test(size_t epochs)
{

    std::cout << "Testing" << std::endl;
    this->loadCheckpoint();
    for (size_t i = 0; i < epochs; i++)
    {

        this->env->reset();
        for (size_t j = 0; j < this->T; j++)
        {
            this->env->step(this->chooseAction(this->env->getObservation(), false, true));
            // this->chooseAction(this->env->getObservation());
        }
    }
}

void MADDPG::visualize()
{
    for (size_t i = 0; i < this->n_agents; i++)
    {
        std::cout << this->env->getAgentPos(i) << "--";
    }

    std::cout << std::endl;
}

void MADDPG::learn(vector<ReplayBuffer::Transition> sampledTrans)
{

    if (!this->memory->ready())
    {
        return;
    }

    // Cut from here
    for (size_t agent = 0; agent < this->n_agents; agent++)
    {

        this->agents[agent]->a_optim.zero_grad();
        this->agents[agent]->c_optim.zero_grad();
        float memsize_scale = 1.0f / static_cast<float>(this->batch_size);
        torch::Tensor q_loss = torch::zeros({1});
        torch::Tensor a_loss = torch::zeros({1});

        for (auto &t : sampledTrans)
        {
            torch::Tensor target;
            torch::Tensor ret;
            if (!t.done)
            {

                ret = this->gamma * agents[agent]->target_c_n(torch::cat({t.obs_1[agent], this->agents[agent]->target_a_n(t.obs_1[agent])})).detach();
                target = t.rewards[agent] + ret;
            }
            else
            {

                target = t.rewards[agent] * torch::ones({1});
            }
            torch::Tensor seg_loss = this->agents[agent]->target_c_n(torch::cat({t.obs[agent], t.actions[agent]}).detach()) - target;
            q_loss += memsize_scale * seg_loss * seg_loss;
        }
        q_loss.backward();
        this->agents[agent]->c_optim.step();
        //*traj_q_loss += q_loss.item<float>();
        for (auto &t : sampledTrans)
        {
            a_loss -= memsize_scale * this->agents[agent]->c_n(torch::cat({t.obs[agent], this->agents[agent]->a_n(t.obs[agent])}).detach());
        }
        a_loss.backward();
        this->agents[agent]->a_optim.step();
        //*traj_a_loss += a_loss.item<float>();
        this->agents[agent]->updateParameters(tau);
    }
}