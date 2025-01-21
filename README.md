Github repository for [Breaking Feedback Loops in Recommender Systems with Causal Inference](https://arxiv.org/pdf/2207.01616).

# Setup
Run
```
pip install reclab
```

# Usage
To reproduce the experiments in this paper add the following code to any Python script in the same folder as this repository.
```python
import experiment

experiment.run_experiments(
    recommenders={
        "als": {regularizer=0.04, eps=0.1},
        "ipw": {regularizer=0.04, eps=0.1},
    },
    env_name="ml-100k-v1",
    num_timesteps=100,
    step_size=1000,
    num_tests=10
)
experiment.run_experiments(
    recommenders={
        "als": {regularizer=0.04, eps=0.1},
        "ipw": {regularizer=0.04, eps=0.1},
    },
    env_name="beta-rank-v1",
    num_timesteps=100,
    step_size=1000,
    num_tests=10
)
```
