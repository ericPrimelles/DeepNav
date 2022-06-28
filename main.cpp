#include <iostream>
#include <vector>
#include <torch/torch.h>
#include <RVO/RVO.h>
#include "Environment.h"
#include "MADDPG.h"
#include "Buffer.h"
using namespace std;
using namespace RVO;

// Environment global varaiables
size_t Agents = 10;
float timestep = 0.25f;
float neighbor_dist = 1.0f;
size_t max_neig = Agents;
float time_horizont = 10.0f;
float time_horizont_obst = 20.0f;
float radius = 2.0f;
float max_speed = 1.5f;

// Train Hyperparameters
size_t k_epochs = 10000;
size_t T = 1000;
size_t batch_size = 256;

size_t scenario = 1, sidesize = 25;
float traj_reward = 0.0f;
float traj_q_loss = 0.0f;
float traj_a_loss = 0.0f;
float avg_reward = 0.0f;
float step_rewards = 0.0f;

// Exec functions

void Train(Environment * env, MADDPG program);

int main(int argc, char **argv)
{
   Environment *env = new Environment(Agents, timestep, neighbor_dist, max_neig, time_horizont, time_horizont_obst, radius, max_speed);
   env->make(1);
   MADDPG program(env, 4, 2, {32, 16, 8}, 6, 1, {32, 16, 8}, 0);
   
   Train(env, program);
   
   return 0;
}

void Train(Environment * env, MADDPG program){
   ReplayBuffer::Buffer *memory = new ReplayBuffer::Buffer();
   ReplayBuffer::Transition a;
   vector<ReplayBuffer::Transition> sampledTrans;
   
   for (size_t i = 0; i < k_epochs; i++)
   {
      env->reset();
      avg_reward = 00.f;
      step_rewards = 00.f;
      for (size_t j = 0; j < T || env->isDone(); j++)
      {
         cout << "Epoch: " << i << "/" << k_epochs << " ";
         std::cout << "TimeStep:" << j << "/" << T << " Rewards:"
                   << "    " << avg_reward << std::endl;
         a.obs = env->getObservation();
         a.actions = program.chooseAction(a.obs);
         a.rewards = env->step(a.actions);
         a.obs_1 = env->getObservation();
         a.done = env->isDone();
         memory->storeTransition(a);

         sampledTrans = memory->sampleBuffer();
         if (memory->ready()) program.Train(sampledTrans);
         
         step_rewards += torch::mean(a.rewards).item<float>();
         avg_reward = step_rewards / (j + 1);
         // sampledTrans = memory->sampleBuffer();
      }
      if (i % 10 == 0)
        {
            std::cout << "Saving..." << std::endl;
            program.saveCheckpoint();
        }

      std::ofstream write;
      write.open("rewards.txt", std::ios::out | std::ios::app);
      if (write.is_open())
      {
         write << avg_reward << "\n";
      }

      write.close();
   }

}