#include <iostream>
#include <torch/torch.h>
#include <RVO/RVO.h>
#include "Environment.h"

using namespace std;
using namespace RVO;

size_t Agents = 10;
float timestep = 0.25f;
float neighbor_dist = 1.0f;
size_t max_neig = Agents;
float time_horizont = 10.0f;
float time_horizont_obst = 20.0f;
float radius = 2.0f;
float max_speed = 1.5f;

int main(int, char**) {
    
    int count = 0;
    Environment* env = new Environment(Agents, timestep, neighbor_dist, max_neig, time_horizont, time_horizont_obst, radius, max_speed );
    env->make(1);
    cout << "NAgents: " << env->getNumAgents() << endl;
    do{
        env->sample();
        cout<< env->getAgentPosition(0) << endl;
        count ++;
    }while(count < 10);

}
