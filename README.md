#  Data-driven Approach for Optimizing Foodbank Operation

# Abstract

The role of food banks is to redistribute donated goods from individuals and corporations to marginalized communities. This study presents mathematical models and algorithms to solve operational and management issues in food banks. The problem, complicated by diverse social demands and the imbalance between supply and demand, is challenging to resolve with existing algorithms. To overcome this, our study proposes a two-stage vehicle routing model (2-stage VRP) utilizing Graph Neural Network-based clustering (GNN-based Clustering) and Attention-based Deep Reinforcement Learning. The model is validated using data from domestic food banks. Ultimately, this research aims to provide efficient and effective operational strategies for food banks and a practical Decision Support System, in collaboration with Gyeonggi Sharing Food Bank, to improve the inefficient operations of domestic food banks.

![image](https://github.com/bizsooin/UPDVRP_MC/assets/119101783/57c23faa-fd37-468a-91e9-68d8befe73cb)


# Model Architecture

![image](https://github.com/bizsooin/UPDVRP_MC/assets/119101783/d62e04a4-eb74-4b77-b49c-af60c070e1e6)

The overall framework of this methodology is as follows. The problem is an Unpaired Pickup and Delivery with Vehicle Routing Problem (UPDVRP), where 'unpaired' means that the pickup and delivery pairs are not predetermined and must be considered during the route searching phase.

The process is conceived in two stages: first, assigning each vehicle to a service unit area it will cover, and second, performing deep reinforcement learning (DRL) during the delivery phase of each area. A hierarchical approach will be used to derive the optimal allocation between vehicles and customers and efficient pickup and delivery routes, to minimize the total cost.


## Phase 1 DRL-Based UPDVRP (UPDVRP_MC)

![image](https://github.com/bizsooin/UPDVRP_MC/assets/119101783/004ba76f-ce37-4fe1-91fe-35abb5fd71f4)

The framework of this model begins by embedding factors like the locations of food banks, pickup and delivery requests, and vehicle capacity into the State. Then, a policy network and attention mechanism are used to determine the next likely destination as a probability distribution. Once a customer is selected, they are excluded (masked) from subsequent processes and no longer chosen in the route planning phase. The route planning ends when all requests are processed or when the agent visits the depot (warehouse). Unprocessed requests are added as additional costs. This hybrid cost is used as a reward for the worker agent and is utilized for subsequent learning. The ultimate goal of this phase is to create a decision-making model that minimizes the total hybrid cost while exploring pickup and delivery routes using deep reinforcement learning.

## Phase 2 GNN based Clustering (Upload soon)

![image](https://github.com/bizsooin/UPDVRP_MC/assets/119101783/020f0c81-0224-4852-88b5-302556be4a2f)

The process involves using a graph neural network (GNN) based on a pre-trained driver (operator) model to perform vehicle allocation tasks for customers. It involves learning the importance between the customer's location, product-specific pickup and delivery requests, and a pre-trained vehicle route search model to minimize the overall cost by optimally assigning customers to vehicles.
