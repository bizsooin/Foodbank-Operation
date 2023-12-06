#  Data-driven Approach for Optimizing Foodbank Operation

# Research Objective

Minimize the operational costs of the redistribution process and consider the appropriate level of budget for each foodbank while considering the trade-off between transportation cost and effectiveness

![image](https://github.com/bizsooin/UPDVRP_MC/assets/119101783/57c23faa-fd37-468a-91e9-68d8befe73cb)


# Model Architecture

![image](https://github.com/bizsooin/UPDVRP_MC/assets/119101783/d62e04a4-eb74-4b77-b49c-af60c070e1e6)


## Phase 1 DRL-Based UPDVRP

![image](https://github.com/bizsooin/UPDVRP_MC/assets/119101783/004ba76f-ce37-4fe1-91fe-35abb5fd71f4)

The framework of this model begins by embedding factors like the locations of food banks, pickup and delivery requests, and vehicle capacity into the State. Then, a policy network and attention mechanism are used to determine the next likely destination as a probability distribution. Once a customer is selected, they are excluded (masked) from subsequent processes and no longer chosen in the route planning phase. The route planning ends when all requests are processed or when the agent visits the depot (warehouse). Unprocessed requests are added as additional costs. This hybrid cost is used as a reward for the worker agent and is utilized for subsequent learning. The ultimate goal of this phase is to create a decision-making model that minimizes the total hybrid cost while exploring pickup and delivery routes using deep reinforcement learning.

## Phase 2 GNN based Clustering

![image](https://github.com/bizsooin/UPDVRP_MC/assets/119101783/020f0c81-0224-4852-88b5-302556be4a2f)

The process involves using a graph neural network (GNN) based on a pre-trained driver (operator) model to perform vehicle allocation tasks for customers. It involves learning the importance between the customer's location, product-specific pickup and delivery requests, and a pre-trained vehicle route search model to minimize the overall cost by optimally assigning customers to vehicles.
