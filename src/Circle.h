#ifndef CIRCLE_H
#define CIRCLE_H

#pragma once

#include <RVO/RVO.h>
#include <vector>

using namespace std;
using namespace RVO;

class Circle
{
public:
    inline Circle(size_t n_agents){ this->n_agents = n_agents;}
    inline ~Circle(){}
    inline vector<Vector2> getScenarioPositions(){
        for (size_t i = 0; i < n_agents; i++)
        {
            positions_cir.push_back(200.0f *
		              RVO::Vector2(std::cos(i * 2.0f * M_PI / 250.0f),
		                           std::sin(i * 2.0f * M_PI / 250.0f)));
            
            
        }
        return positions_cir;
    }
    inline vector<Vector2> getScenarioGoals(){
        if (positions_cir.size() > 0) {

            for (auto i : positions_cir)
            {
                goals_cir.push_back(-i);
            }
            
        }
        return goals_cir;
    }

private:

    vector<Vector2> positions_cir, goals_cir;
    size_t n_agents;

};

#endif