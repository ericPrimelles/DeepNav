#include "Environment.h"
#include "Circle.h"



Environment::Environment(size_t n_agents, float time_step, float neighbor_dists, size_t max_neig, float time_horizon,
                         float time_horizon_obst, float radius, float max_speed)
{

    this->n_agents = n_agents;
    this->sim = new RVOSimulator();
    this->sim->setTimeStep(0.25f);
    this->sim->setAgentDefaults(neighbor_dists, max_neig, time_horizon, time_horizon_obst, radius, max_speed);
    this->timestep = timestep; this->neigh_dist = neigh_dist; this->max_neigh = max_neig, this->time_horizont = time_horizon;
    this->time_horizont_obst = time_horizon_obst; this->radius = radius; this->max_speed = max_speed;
    
}

void Environment::make(size_t scenario)
{
    this->setupScenario(scenario);
    this->setup(this->positions, this->obstacles);
}

Environment::~Environment()
{
    delete sim;
}

void Environment::setupScenario(size_t scenario)
{
    if (scenario == 1)
    {
        Circle cir_scn = Circle(this->n_agents);
        this->positions = cir_scn.getScenarioPositions();
        this->goals = cir_scn.getScenarioGoals();
        return;
    }
}

void Environment::setup(vector<Vector2> positions, vector<Vector2> obstacles)
{

    for (size_t i = 0; i < this->n_agents; i++)
    {
        // Adding agents
        this->sim->addAgent(positions[i]);

        // Adding obstacles
        // this->sim->addObstacle(&obstacles[i]);
    }
}

torch::Tensor Environment::step(torch::Tensor actions)
{

    this->setPrefferedVelocities(actions);
    this->sim->doStep();
    return this->calculateGlobalReward() + this->calculateLocalReward();
}

void Environment::setPrefferedVelocities(torch::Tensor actions)
{

    float x = 0.0f, y = 0.0f;
    Vector2 v_pref_placeholder;
    for (size_t i = 0; i < this->n_agents; i++)
    {
        // Detacching tensors
        x = actions[i][0].item<float>();
        y = actions[i][1].item<float>();
        v_pref_placeholder = Vector2(x, y); // Constructing a new Vector2 with the agent i action

        if (absSq(v_pref_placeholder) > 1.0f)
        {
            v_pref_placeholder = RVO::normalize(v_pref_placeholder);
        } // Normilizing vector

        this->sim->setAgentPrefVelocity(i, v_pref_placeholder);
    }
}

// Takes a random action
torch::Tensor Environment::sample()
{

    auto actions = torch::rand({(int64_t)this->n_agents, 2}, torch::dtype(torch::kFloat32));
    return this->step(actions);
}




bool Environment::isDone(){
    for (size_t i = 0; i < this->n_agents; i++)
    {
        if (RVO::absSq(this->getAgentPos(i) - goals[i]) > this->sim->getAgentRadius(i) * this->sim->getAgentRadius(i)){
            return false;
        }
        
    }
    return true;
    
}

torch::Tensor Environment::calculateGlobalReward()
{
    
    

   if(!this->isDone()) return torch::zeros((int64_t)this->getNAgents());

    return torch::full((int64_t)this->getNAgents(), 100.0f - this->sim->getGlobalTime());
}

torch::Tensor Environment::calculateLocalReward()

{
    std::vector<float> rewards(this->getNAgents());
    float r_goal = 0, r_coll_a = 0, r_coll_obs = 0, r_cong = 0;
    int64_t size = rewards.size();
    float abs = 0;
    auto calcDist = [](RVO::Vector2 x, RVO::Vector2 y) -> float

    {
        return std::sqrt(std::pow(x.x() - y.x(), 2) + std::pow(x.y() - y.y(), 2));
    };
    for (size_t i = 0; i < this->getNAgents(); i++)
    {
        r_goal = -calcDist(this->getAgentPos(i), this->goals[i]);
        abs = std::sqrt(this->sim->getAgentVelocity(i) * this->sim->getAgentVelocity(i));
        r_cong = -1 / (abs / this->sim->getAgentMaxSpeed(i));
        for (size_t j = 0; j < this->getNAgents() && j != i; j++)
        {
            if (calcDist(this->getAgentPos(i), this->getAgentPos(j)) <= 2 * this->sim->getAgentRadius(i))
                r_coll_a += -3;
        }

        for (size_t j = 0; j < this->sim->getAgentNumObstacleNeighbors(i); j++)
        {
            if (calcDist(this->getAgentPos(i),
                         this->sim->getObstacleVertex(this->sim->getAgentObstacleNeighbor(i, j))) < this->sim->getAgentRadius(i))
                r_coll_obs += -1;
        }

        rewards[i] = r_goal + r_coll_a + r_coll_obs + r_cong;
        
        //std::cout << "{ " << r_goal << ", " << r_coll_a << " ," << " ," << r_coll_obs << ", " << r_cong << "}\n";
        
        }

    return torch::from_blob(rewards.data(), {size}, torch::dtype(torch::kFloat32));
}


torch::Tensor Environment::getObservation()
{
    int64_t nAgents = this->getNAgents();
    torch::Tensor observation = torch::zeros({(int64_t) this->n_agents, 4}, torch::dtype(torch::kFloat32));
    std::vector<std::vector<float>> data;
    int64_t size;

    for (size_t i = 0; i < this->getNAgents(); i++)
    {
        data.push_back({this->getAgentPos(i).x(), this->getAgentPos(i).y(), this->sim->getAgentPrefVelocity(i).x(), this->sim->getAgentPrefVelocity(i).y()});
        size = data[i].size();
         
        observation[i] = torch::from_blob(data[i].data(), {size});
    }
    return observation;
}

// Visualization

void Environment::render(){
   
}

void Environment::reset(){
    this->sim = new RVOSimulator();
    this->sim->setAgentDefaults(this->neigh_dist, this->max_neigh, this->time_horizont, this->time_horizont_obst, radius, max_speed);
    this->setup(this->positions, this->obstacles);
    
    
    
}