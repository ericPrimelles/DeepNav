#include <iostream>
#include <torch/torch.h>
#include <RVO/RVO.h>
#include "Environment.h"
#include "MADDPG.h"

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





int main(int argc, char** argv) {

   Environment * env = new Environment(Agents, timestep, neighbor_dist, max_neig, time_horizont, time_horizont_obst, radius, max_speed);
   env->make(1);
   MADDPG program(env, 4, 2, {32, 16, 8}, 6, 1, {32, 16, 8}, 0);
   program.Train();
   
   return 0;
}

