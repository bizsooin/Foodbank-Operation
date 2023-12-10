#  DRL-based for Unpaired Pickup & Delivery VRP (UPDVRP)


Attention-based model for learning to solve the Unpaird Pickup and Delivery Vehicle Routing Problem with multiple commodities (UPDVRP-MCC) while considering effective cost and equity cost. 

Training with REINFORCE with greedy rollout baseline.

![image](https://github.com/bizsooin/UPDVRP_MC/assets/119101783/170a42f0-d52c-40c3-99d4-73da7e0c4e99)

Our graph addresses the distribution challenges of two commodities across ten food banks using a pre-trained Deep Reinforcement Learning (DRL) approach.

## Dependencies

* Python>=3.8
* NumPy
* SciPy
* [PyTorch](http://pytorch.org/)>=1.7
* tqdm
* [tensorboard_logger](https://github.com/TeamHG-Memex/tensorboard_logger)
* Matplotlib (optional, only for plotting)
* TensorFlow
* pandas (optional)

## Quick start

For training UPDVRP with Foodbank instances with 20 nodes (10 foodbank with 2 types of product) and using rollout as a REINFORCE baseline:
```bash
python run.py --foodbank 10 --num_products 2 --baseline rollout --run_name 'updvrp_10_2'
```

## Usage

### Generating data

Training data is generated on the fly. To generate validation and test data (same as used in the paper) for problems:
```bash
python generate_data.py --problem updvrp --name validation  --num_products 2 --foodbank 10 --seed 4321
python generate_data.py --problem updvrp --name test --num_products 2 --foodbank 10 --seed 1234
```

### Training

For training UPDVRP with Foodbank instances with 20 nodes (10 foodbank with 2 types of product)s and using rollout as REINFORCE baseline and using the generated validation set:
```bash
python run.py --num_products 2 --foodbank 10 --baseline rollout --run_name 'tsp20_rollout' --val_dataset data/updvrp/updvrp__10_2_validation_seed4321.pkl
```

#### Multiple GPUs
By default, training will happen *on all available GPUs*. To disable CUDA at all, add the flag `--no_cuda`. 
Set the environment variable `CUDA_VISIBLE_DEVICES` to only use specific GPUs:
```bash
CUDA_VISIBLE_DEVICES=2,3 python run.py 
```
Note that using multiple GPUs has limited efficiency for small problem sizes (up to 50 nodes).

#### Warm start
You can initialize a run using a pretrained model by using the `--load_path` option: (Working, uploaded later)
```bash
python run.py --num_products 2 --foodbank 10 --load_path pretrained/updvrp_10_2/epoch-99.pt 
```

The `--load_path` option can also be used to load an earlier run, in which case also the optimizer state will be loaded:
```bash
python run.py --num_products 2 --foodbank 10 --load_path 'outputs/updvrp_10_2/updvrp_10_2_rollout_{datetime}/epoch-0.pt'
```

The `--resume` option can be used instead of the `--load_path` option, which will try to resume the run, e.g. load additionally the baseline state, set the current epoch/step counter and set the random number generator state.

### Evaluation
To evaluate a model, you can add the `--eval-only` flag to `run.py`, or use `eval.py`, which will additionally measure the timing and save the results:
```bash
python eval.py data/updvrp/updvrp__10_2_validation_seed4321.pkl --model pretrained/updvrp_10_2/epoch-99.pt --decode_strategy greedy
```
If the epoch is not specified, by default the last one in the folder will be used.


## Acknowledgements
Special Thanks to https://github.com/zcaicaros/manager-worker-mtsptwr for helping me start with the code and insights.
