from torch.utils.data import Dataset
import torch
import os
import pickle


from problems.updvrp.state_updvrp import StateUPDVRP
from utils.beam_search import beam_search

import random


class UPDVRP(object):

    NAME = 'updvrp'  # Unpaired capacitated Vehicle Routing Problem

    VEHICLE_CAPACITY = 10.0  # (w.l.o.g. vehicle capacity is 1, demands should be scaled)

    def get_costs(dataset, pi, unfilled, center_x, beta):

        batchsize, graph_size, _ = dataset['delivery'].size()

        # Ensure all tensors are on the same device
        device = unfilled.device  # Assuming `unfilled` tensor is on your desired device
        pi = pi.to(device)
        center_x = center_x.to(device)
        if torch.is_tensor(beta):  # Check if beta is a tensor
            beta = beta.to(device)

        locations = torch.cat((dataset['depot'].unsqueeze(1), dataset['loc']), dim=1).to(device)

        d = locations.gather(1, pi[..., None].expand(*pi.size(), locations.size(-1)))

        distance_cost = (
            (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1)
            + (d[:, 0] - dataset['depot']).norm(p=2, dim=1)
            + (d[:, -1] - dataset['depot']).norm(p=2, dim=1)
        ).to(device)

        cost = beta * unfilled + distance_cost + beta * center_x

        return cost , None, beta * unfilled, distance_cost, beta * center_x



    @staticmethod
    def make_dataset(*args, **kwargs):
        return UPDVRPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateUPDVRP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = UPDVRP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


def make_instance(args):
    depot, loc, delivery, pickup, center, must, *args = args
    grid_size = 1
    if len(args) > 0:
        depot_types, customer_types, num_products, grid_size = args


    return {
        'depot': torch.tensor(depot, dtype=torch.float),
        'loc': torch.tensor(loc, dtype=torch.float),
        'delivery': torch.tensor(delivery, dtype=torch.float),
        'pickup': torch.tensor(pickup, dtype=torch.float),
        'center': torch.tensor(center, dtype=torch.float),
        'must': torch.tensor(must, dtype=torch.float)
    }



class UPDVRPDataset(Dataset):
    
    def __init__(self, filename=None, foodbank=10, num_samples=100000, distribution = None, offset=0, num_products=10):
        super(UPDVRPDataset, self).__init__()

        self.num_products = num_products
        self.foodbank = foodbank
        self.total_size = foodbank * num_products

        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.data = [make_instance(args) for args in data[offset:offset+num_samples]]
        else:
            self.data = []
            for _ in range(num_samples):
                locs = torch.FloatTensor(self.total_size, 2)
                deliveries = torch.zeros(self.total_size, self.num_products)
                pickups = torch.zeros(self.total_size, self.num_products)
                depot = torch.FloatTensor(2).uniform_(0, 1)
                centers = torch.zeros(self.total_size, 1)
                must = torch.zeros(self.total_size, 1)

                counter = 0
                for j in range(self.foodbank):
                    loc = torch.FloatTensor(2).uniform_(0, 1)
                    center = j + 1
                    for k in range(self.num_products):
                        #demand = torch.rand(1).item()
                        demand = 1
                        #while demand == 0:
                        #    demand = torch.rand(1).item()
                            
                        if random.random() > 0.5:
                            deliveries[counter][k] = demand
                        else:
                            pickups[counter][k] = demand
                        locs[counter] = loc
                        centers[counter] = center
                        counter += 1

                # Update the 'visit' column based on deliveries and pickups

                for k in range(self.num_products):
                    if deliveries[:, k].sum() >= pickups[:, k].sum():
                        must[(deliveries[:, k] != 0)] = 1
                    else:
                        must[(pickups[:, k] != 0)] = 1


                sample = {
                    'loc': locs,
                    'delivery': deliveries,
                    'pickup': pickups,
                    'depot': depot,
                    'center': centers,
                    'must': must
                }
                
                self.data.append(sample)

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]

