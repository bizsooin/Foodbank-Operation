import torch
from typing import NamedTuple

from utils.boolmask import mask_long2bool, mask_long_scatter


class StateUPDVRP(NamedTuple):
    # Fixed input
    coords: torch.Tensor  # Depot + loc
    delivery: torch.Tensor
    pickup: torch.Tensor # Pickup

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the coords and demands tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State
    unmet_demand: torch.Tensor
    unmet_pickup: torch.Tensor
    prev_a: torch.Tensor
    used_capacity: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    lengths: torch.Tensor
    cur_coord: torch.Tensor
    i: torch.Tensor 
    center: torch.Tensor 
    must: torch.Tensor
     # Keeps track of step


    VEHICLE_CAPACITY = 10.0  # Hardcoded

    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.delivery.size(-1))

    @property
    def dist(self):
        return (self.coords[:, :, None, :] - self.coords[:, None, :, :]).norm(p=2, dim=-1) # additional e-04 term (optional)

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)  # If tensor, idx all tensors by this tensor:
        return self._replace(
            ids=self.ids[key],
            prev_a=self.prev_a[key],
            used_capacity=self.used_capacity[key],
            visited_=self.visited_[key],
            lengths=self.lengths[key],
            cur_coord=self.cur_coord[key],
            unmet_demand = self.unmet_demand[key],
            unmet_pickup = self.unmet_pickup[key],
            center = self.center[key],
            must = self.must[key]
        )

    # Warning: cannot override len of NamedTuple, len should be number of fields, not batch size
    # def __len__(self):
    #     return len(self.used_capacity)

    @staticmethod
    def initialize(input, visited_dtype=torch.uint8):
        depot = input['depot']
        loc = input['loc']
        delivery = input['delivery'] # Shape: batch_size x locations x products
        pickup = input['pickup']  # Shape: batch_size x locations x products
        batch_size, n_loc, num_products = delivery.shape
        used_capacity = delivery.new_zeros(batch_size, num_products)
        center = input['center']
        must = input['must']


    # Here we're not concatenating the depot with locations in coords
    # as it seems you want to keep them separate.
        return StateUPDVRP(
            coords=torch.cat((depot[:, None, :], loc), -2),  # Excluding depot
            delivery=delivery,
            pickup=pickup,
            unmet_demand=delivery.clone(),  # Initialize unmet_demand with the full delivery amount
            unmet_pickup=pickup.clone(),  # Initialize unmet_pickup with the full pickup amount
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension
            prev_a=torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device),
            used_capacity=used_capacity,
            center = center,
            visited_=(
                # Initialize visited mask without the depot
                torch.zeros(
                batch_size, 1, n_loc + 1,    # Add depot dimension
                dtype=torch.uint8, device=loc.device
            )
            if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)  # Ceil
        ),
            lengths=torch.zeros(batch_size, 1, device=loc.device),
            cur_coord=input['depot'][:, None, :],  # Add step dimension
            i=torch.zeros(1, dtype=torch.int64, device=loc.device),  # Vector with length num_steps
            must = must
            )


    def get_final_cost(self):
    
        assert self.all_finished()

        # Compute the distance cost as before
        distance_cost = self.lengths + (self.coords[self.ids, 0, :] - self.cur_coord).norm(p=2, dim=-1)

        # Compute min_penalty for each product
        min_penalty = (self.delivery - self.pickup).sum(dim=1)  # Sum over locations for each product
        min_penalty = torch.max(min_penalty, torch.zeros_like(min_penalty))

        # Compute penalty for unmet demand
        penalty_per_unit_unmet_demand = 10
        unmet_demand_penalty = self.unmet_demand.sum(dim=1)  # Sum over locations for each product

        inevitable_penalty = (unmet_demand_penalty - min_penalty) * penalty_per_unit_unmet_demand

        # Sum over products to get total penalty
        total_inevitable_penalty = inevitable_penalty.sum(dim=-1)  # Sum over products

        #load_weight = 10

        # Add the distance cost and the penalties to get the final cost
        final_cost = distance_cost + total_inevitable_penalty

        return final_cost

    def update(self, selected):

        assert self.i.size(0) == 1, "Can only update if state represents single step"

        selected = selected[:, None]  # Add dimension for step

        prev_a = selected
        n_loc = self.delivery.size(1)  # Excludes depot

        # Calculate the lengths
        cur_coord = self.coords[self.ids, selected]

        lengths = self.lengths + (cur_coord - self.cur_coord).norm(p=2, dim=-1)

        updated_visited = self.visited_.clone()
        updated_used_capacity = self.used_capacity.clone()
        updated_unmet_demand = self.unmet_demand.clone()
        updated_unmet_pickup = self.unmet_pickup.clone()


        for i in range(prev_a.size(0)):
            if prev_a[i] == 0: #Allowing Early End
                updated_visited[i, :, :] = 1
                updated_visited[i, :, 0] = 0
                updated_used_capacity[i] = self.used_capacity[i]
                updated_unmet_demand[i] = self.unmet_demand[i]
                updated_unmet_pickup[i] = self.unmet_pickup[i]
        
            else:

                index = prev_a[i, :, None].expand(*self.visited_[i].shape)
                updated_visited[i] = self.visited_[i].scatter(-1, index, 1)

                selected_indices = torch.clamp(prev_a[i]-1, 0, n_loc - 1)

                total_delivery = self.delivery[i, selected_indices, :]
                total_pickup = self.pickup[i, selected_indices, :]

                total_deliverys = total_delivery.squeeze(0)
                total_pickups = total_pickup.squeeze(0)
                                                
                torch_id = (total_deliverys + total_pickups != 0)

                true_index_1d = torch.nonzero(torch_id)

                k = true_index_1d[0, 0].item()

                selected_delivery = self.delivery[i, selected_indices, k]

                selected_pickup = self.pickup[i, selected_indices, k]
                
                # Adjust the delivery and pickup
                adjusted_delivery = torch.min(selected_delivery, self.used_capacity[i, k])

                full_capacity = self.used_capacity.sum(dim=1)

                remaining_capacity = self.VEHICLE_CAPACITY - full_capacity[i]

                adjusted_pickup = torch.min(selected_pickup, remaining_capacity)
                # Compute the new capacity
                new_capacity = (self.used_capacity[i, k] - adjusted_delivery + adjusted_pickup)

                # Update the unmet demand and unmet pickup for the selected nodes only
                unmet_demandx = self.unmet_demand[i, :, k]
                unmet_pickupx = self.unmet_pickup[i, :, k]

                updated_dvalues = unmet_demandx.gather(0, selected_indices) - adjusted_delivery
                updated_pvalues = unmet_pickupx.gather(0, selected_indices) - adjusted_pickup

                updated_unmet_demand[i, :, k] = unmet_demandx.scatter(0, selected_indices, updated_dvalues)
                updated_unmet_pickup[i, :, k] = unmet_pickupx.scatter(0, selected_indices, updated_pvalues)

                # Update the used capacity
                updated_used_capacity[i, k] = new_capacity

        return self._replace(
            prev_a=prev_a, used_capacity=updated_used_capacity, visited_=updated_visited,
            lengths=lengths, cur_coord=cur_coord, i=self.i + 1,
            unmet_demand=updated_unmet_demand, unmet_pickup=updated_unmet_pickup
        )
    
    def all_finished(self):
        #return self.visited.all() or (self.prev_a.item() == 0 and self.i.item() > 0).all()
        return self.i.item() > 0 and (self.prev_a == 0).all()

    def get_current_node(self):
        return self.prev_a

    def get_mask(self):
    # Assuming self.delivery, self.pickup, self.used_capacity, etc. are defined elsewhere in your class

        # Determine visited locations
        if self.visited_.dtype == torch.uint8:
            visited_loc = self.visited_[:, :, 1:]
        else:
            visited_loc = mask_long2bool(self.visited_, n=self.delivery.size(-1))


        must_visit = self.must.permute(0, 2, 1)

        # Exceeds capacity condition
        total_pickup = self.pickup[self.ids, :].sum(dim=-1, keepdim=True).squeeze(-1)

        total_demand = self.delivery[self.ids, :].sum(dim=-1, keepdim=True).squeeze(-1)

        max_deliver = ((total_demand > 0) & (self.i.item() == 0))
    
        used_capacity_sum = self.used_capacity.sum(dim=1, keepdim=True).unsqueeze(-1)

        exceeds_cap_condition = (total_pickup + used_capacity_sum > self.VEHICLE_CAPACITY)

        exceeds_cap = exceeds_cap_condition


        used_capacity_expanded = self.used_capacity.unsqueeze(1)  # Shape becomes [Batch, 1, Product]

        capacity_shortfall = used_capacity_expanded - self.delivery[self.ids, :].squeeze(1)

        min_deliver_condition = capacity_shortfall < 0


        min_deliver = torch.any(min_deliver_condition, dim=-1)  # This will reduce the tensor shape to [Batch, Locations]

        min_deliver = min_deliver.unsqueeze(1)

        min_delivert = min_deliver
        
        visited_loc2 = visited_loc | max_deliver
        mask_loc = visited_loc2 | exceeds_cap
        mask_loc2 = mask_loc | min_delivert

        # Check if all required conditions are met
        condition_met = ((must_visit + visited_loc) >= 1).all(dim=-1)

        min_penalty = (self.delivery - self.pickup).sum(dim=1)  # Sum over locations for each product
        min_penalty = torch.max(min_penalty, torch.zeros_like(min_penalty))

        condition_met2 = torch.any(self.unmet_demand.sum(dim= 1) - min_penalty > 0)  

        # New condition: Check if the sum of mask_loc2 equals the size of mask_loc along the last dimension
        new_condition = mask_loc2.sum(dim=-1) == mask_loc.size(-1)

        # Update the depot mask
        # mask_depot should be 0 if new_condition is True, otherwise follow the existing logic
        mask_depot = ~(new_condition | (condition_met & ~condition_met2))

        # Return the final mask
        return torch.cat((mask_depot[:, :, None], mask_loc2), dim=-1)