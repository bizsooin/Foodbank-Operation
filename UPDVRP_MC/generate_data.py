import argparse
import os
import numpy as np
from utils.data_utils import check_extension, save_dataset

def generate_tsp_data(dataset_size, tsp_size):
    return np.random.uniform(size=(dataset_size, tsp_size, 2)).tolist()

def generate_updvrp_data(dataset_size, foodbank, num_products):

    total_size = foodbank * num_products
    data = []

    for i in range(dataset_size):
        locs = np.zeros((total_size, 2))
        deliveries = np.zeros((total_size, num_products))
        pickups = np.zeros((total_size, num_products))
        depot = np.random.rand(2)
        centers = np.zeros(total_size)
        must = np.zeros(total_size)

        counter = 0
        for j in range(foodbank):
            loc = np.random.rand(2)
            center = j + 1
            for k in range(num_products):
                #demand = np.random.rand()
                demand = 1
                #while demand == 0:
                #    demand = np.random.rand()
                if np.random.rand() > 0.5:
                    deliveries[counter][k] = demand
                else:
                    pickups[counter][k] = demand
                locs[counter] = loc
                centers[counter] = center
                counter += 1

        for k in range(num_products):
            if deliveries[:, k].sum() >= pickups[:, k].sum():
                must[(deliveries[:, k] != 0)] = 1
            else:
                must[(pickups[:, k] != 0)] = 1

        centers = centers[:, np.newaxis]
        must = must[:, np.newaxis]

        sample = (
            depot.tolist(),
            locs.tolist(),
            deliveries.tolist(),
            pickups.tolist(),
            centers.tolist(),
            must.tolist()
        )
        data.append(sample)

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", help="Filename of the dataset to create (ignores datadir)")
    parser.add_argument("--data_dir", default='data', help="Create datasets in data_dir/problem (default 'data')")
    parser.add_argument("--name", type=str, required=True, help="Name to identify dataset")
    parser.add_argument("--problem", type=str, default='updvrp',
                        help="Problem, 'tsp', or 'updvrp'"
                             " or 'all' to generate all")
    parser.add_argument('--data_distribution', type=str, default='all',
                        help="Distributions to generate for problem, default 'all'.")

    parser.add_argument("--dataset_size", type=int, default=10000, help="Size of the dataset")
    #parser.add_argument('--graph_sizes', type=int, nargs='+', default=[20, 50, 100],
    #                    help="Sizes of problem instances (default 20, 50, 100)")

    parser.add_argument("--foodbank", type=int, default=20, help="The number foodbank of for dataset")
    parser.add_argument("--num_products", type=int, default=10, help="The number of products for dataset")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument('--seed', type=int, default=1234, help="Random seed")

    opts = parser.parse_args()

    assert opts.filename is None or (len(opts.problems) == 1 and len(opts.graph_sizes) == 1), \
        "Can only specify filename when generating a single dataset"

    distributions_per_problem = {
        'tsp': [None],
        'updvrp': [None],  # Added UPDVRP problem without any specific distributions.
    }
    if opts.problem == 'all':
        problems = distributions_per_problem
    else:
        problems = {
            opts.problem:
                distributions_per_problem[opts.problem]
                if opts.data_distribution == 'all'
                else [opts.data_distribution]
        }

    for problem, distributions in problems.items():
        for distribution in distributions or [None]:

            if problem == "updvrp":  # Added a condition to handle UPDVRP
                
                datadir = os.path.join(opts.data_dir, problem)
                os.makedirs(datadir, exist_ok=True)

                if opts.filename is None:
                    filename = os.path.join(datadir, "{}_{}_{}_{}_{}_seed{}.pkl".format(
                        problem,
                        "_{}".format(distribution) if distribution is not None else "",
                        opts.foodbank, opts.num_products, opts.name, opts.seed))
                    
                else:
                    filename = check_extension(opts.filename)

                assert opts.f or not os.path.isfile(check_extension(filename)), \
                    "File already exists! Try running with -f option to overwrite."

                np.random.seed(opts.seed)    
                dataset = generate_updvrp_data(opts.dataset_size, opts.foodbank, opts.num_products)
                
                save_dataset(dataset, filename)

            else:

                for graph_size in opts.graph_sizes:

                    datadir = os.path.join(opts.data_dir, problem)
                    os.makedirs(datadir, exist_ok=True)

                    if opts.filename is None:
                        filename = os.path.join(datadir, "{}{}{}_{}_seed{}.pkl".format(
                            problem,
                            "_{}".format(distribution) if distribution is not None else "",
                            graph_size, opts.name, opts.seed))
                    else:
                        filename = check_extension(opts.filename)

                    assert opts.f or not os.path.isfile(check_extension(filename)), \
                        "File already exists! Try running with -f option to overwrite."

                    np.random.seed(opts.seed)
                    if problem == 'tsp':
                        dataset = generate_tsp_data(opts.dataset_size, graph_size)
                    else:
                        assert False, "Unknown problem: {}".format(problem)

                    print(dataset[0])

                    save_dataset(dataset, filename)
