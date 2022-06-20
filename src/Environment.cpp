#include "Environment.h"
#include "Circle.h"

Environment::Environment(size_t n_agents, float time_step, float neighbor_dists, size_t max_neig, float time_horizon,
                         float time_horizon_obst, float radius, float max_speed)
{
    
    this->n_agents = n_agents;
    this->sim = new RVOSimulator();
    this->sim->setTimeStep(0.25f);
    this->sim->setAgentDefaults(neighbor_dists, max_neig, time_horizon, time_horizon_obst, radius, max_speed);
    //this->setup(positions, obstacles);
    
    
    //std::cout << positions.size() << n_agents << std::endl;
}

void Environment::make(size_t scenario){
    this->setupScenario(scenario);
    this->setup(this->positions, this->obstacles);
}

Environment::~Environment()
{
}

void Environment::setupScenario( size_t scenario){
    if (scenario == 1){
        Circle cir_scn = Circle(this->n_agents);
        this->positions = cir_scn.getScenarioPositions();
        this->goals = cir_scn.getScenarioGoals();
        return;
    }

}

void Environment::setup(vector<Vector2> positions, vector<Vector2> obstacles){
    
    for (size_t i = 0; i < this->n_agents; i++)
    {
        // Adding agents
        this->sim->addAgent(positions[i]);
        
        //Adding obstacles
        //this->sim->addObstacle(&obstacles[i]);
    }
    

    
}

void Environment::step(torch::Tensor actions){

    this->setPrefferedVelocities(actions);
    this->sim->doStep();

    
}

void Environment::setPrefferedVelocities(torch::Tensor actions){

    float x = 0.0f, y = 0.0f;
    Vector2 v_pref_placeholder;
    for (size_t i = 0; i < this->n_agents; i++)
    {
        //Detacching tensors
        x = actions[i][0].item<float>();
        y = actions[i][0].item<float>();
        v_pref_placeholder = Vector2(x, y); // Constructing a new Vector2 with the agent i action

        if(absSq(v_pref_placeholder) > 1.0f){v_pref_placeholder = RVO::normalize(v_pref_placeholder);} // Normilizing vector

        this->sim->setAgentPrefVelocity(i, v_pref_placeholder);
    }
    
}

// Takes a random action
void Environment::sample(){

    auto actions = torch::rand({(int64_t)this->n_agents, 2}, torch::dtype(torch::kFloat32));
    this->step(actions);
}