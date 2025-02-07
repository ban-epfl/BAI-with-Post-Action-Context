# Optimal Best Arm Identification with Post-Action Context

This repository includes the official codes for ["Best Arm Identification with Post-Action Contexts"](https://arxiv.org/abs/2502.03061) paper and the instructions on how to run it. 

---

## Abstract

We introduce the problem of best arm identification (BAI) with post-action context, a new BAI problem in a stochastic multi-armed bandit environment and the fixed-confidence setting. The problem addresses the scenarios in which the learner receives a *post-action context* in addition to the reward after playing each action. This post-action context provides additional information that can significantly facilitate the decision process. We analyze two different types of the post-action context: (i) *non-separator*, where the reward depends on both the action and the context, and (ii) *separator*, where the reward depends solely on the context. For both cases, we derive instance-dependent lower bounds on the sample complexity and propose algorithms that asymptotically achieve the optimal sample complexity. 
For the non-separator setting, we do so by demonstrating that the Track-and-Stop algorithm can be extended to this setting. For the separator setting, we propose a novel sampling rule called *G-tracking*, which uses the geometry of the context space to directly track the contexts rather than the actions.
Finally, our empirical results showcase the advantage of our approaches compared to the state of the art.



## Project Structure

- **`main1.py`**: Implements the Algorithms for the Non-Seperator Environment.
- **`main2.py`**: Implements the Algorithms for the Seperator Environment. 
- **Dependencies**: 
  - `utils.py`: Contains utility functions used across the project.
  - `Environments.py`: Defines the simulation environments for the Seperator and Non-Seperator problems.
  - `Instance_json.py`: Handles loading of problem instances from JSON files.
  - `output_json.py`: Manages the storage of results into JSON files.
  - `Algorithms1.py`: Implements the non-separator track and stop (NSTS) and classic track and stop (TS) algorithms for the Non-Seperator environment. Each Algorithm is implemented with a class. Each class has a Stopping Rule method and D_Tracking rule for sampling. For D_Tracking, an optimization oracle is needed which is implemented via binary search using scipy.
  - `Algorithms2.py`: Implements theseparator track and stop (STS), classic track and stop (TS), ND lazy track and stop (LTS) algorithms for the Seperator environment. Each Algorithm is implemented with a class. Each class has a Stopping Rule method and one a Tracking Method. For G_Tracking rule used in STS, an optimization oracle is needed which requires convex programming which is implemented using CVXPY.
 
For more details about the algorithms, refer to the paper.

---


## Usage

### Running `main1.py`
The script is used to test the Non Seperator algorithms. Below are the arguments you can specify:

- `--Algorithm` (default: `NSTS`): Specify the algorithm to use. Choices are `NSTS` and `TS`.
- `--instance_index` (default: 0): Index of the problem instance to load from `instances1.json`. Choices are 0 to 8.
- `--store` (default: 1): Doesn't save the results if 0 and Save results to a JSON file otherwise. 

**Example**:
```bash
python main1.py --Algorithm "NSTS" --instance_index 1 --store 0
```

### Running `main2.py`
The script is used to test the Seperator algorithms. Below are the arguments you can specify:

- `--Algorithm` (default: `STS`): Specify the algorithm to use. Choices are `STS`, `LTS`, `TSS` and `STS_C_Tracking`.
- `--instance_index` (default: 0): Index of the problem instance to load from `instances2.json`. Choices are 0 to 9.
- `--store` (default: 1): Doesn't save the results if 0 and Save results to a JSON file otherwise. 

**Example**:
```bash
python main2.py --Algorithm 'STS' --instance_index 2 --store 0
```

---

## Output

Both scripts save detailed results in JSON format in the `results/` directory if the `--store` argument is set to 1. The outputs include:

- Estimated means (`mu_hats`)
- Number of times each arm was selected (number of contexts times seen for seperator algorithm) (`N_times_seens`)
- Estimated Weights(contexts weights for seperator) (`w_s`)
- Total time steps (`T`)
- Identified best arm (`best_arm`)

Additionally, the scripts print the following information to the console:

1. The number of time steps (`T`).
2. The actual best arm.
3. The best arm identified by the algorithm.

---

## Customization

- Modify `instances1.json` and `instances2.json` to define new problem instances.

