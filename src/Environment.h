#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#pragma once


#include<torch/torch.h>
#include<vector>
#include "Circle.h"

#ifdef __APPLE__
#include <RVO/RVO.h>
#else
#include<RVO/RVO.h>
#endif

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#include <GL/gl.h>
#include <GL/glu.h>
#endif

using namespace RVO;
using namespace std;



class Environment
{
public:
    Environment(size_t n_agents, float timestep,float neighbor_dists, size_t max_neig, float time_horizont,
                         float time_horizont_obst, float radius, float max_speeds);
    torch::Tensor step(torch::Tensor actions);
    torch::Tensor sample();
    torch::Tensor getObservation();
    void render();
    void make(size_t scenario);
    bool isDone();
    inline Vector2 getAgentPosition(size_t i){return sim->getAgentPosition(i);}
    inline size_t getNumAgents(){return this->sim->getNumAgents();}
    ~Environment();

private:
    
    //Methods
    void setup(vector<Vector2> positions, vector<Vector2> obstacles);
    void setupScenario(size_t scenario);
    void setPrefferedVelocities(torch::Tensor actions);
    torch::Tensor calculateGlobalReward();
    torch::Tensor calculateLocalReward();
    
    //Visualization Methods
    void InitGL(void);
    

    //Parameters
    RVOSimulator * sim;
    float time= 0.0f;
    size_t n_agents;
    std::vector<RVO::Vector2> positions, goals, obstacles;
    
    
};

#endif