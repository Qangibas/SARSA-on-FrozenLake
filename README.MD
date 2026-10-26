# SARSA on FrozenLake

## Setup

Tested with Python <YOUR PYTHON VERSION> on <YOUR OS>. No GPU, internet access (after installation) or datasets are needed.

```
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies (see `requirements.txt`): `gymnasium`, `numpy`, `matplotlib`.

## Run

```
python main.py
```

Expected runtime: about <YOUR MEASURED TIME> on a laptop CPU (2 epsilon values x 5 seeds x 5000 training episodes, plus evaluation).

The script prints the results table and the learned policies, and writes:

- `results.txt`: mean and standard deviation of the success rate for each method (the numbers used in the report)
- `training_curve.png`: Figure 1, SARSA training curve (moving average over 100 episodes, mean and std over 5 seeds)
- `comparison.png`: Figure 2, greedy evaluation of SARSA vs. the random baseline

## Random seeds

- Seeds: `0, 1, 2, 3, 4` (list `SEEDS` in `main.py`), 5 independent runs.
- Training run with seed k uses environment seed k and agent seed k (`np.random.default_rng(k)`).
- Evaluation of run k uses a separate seed, `k + 100`, and the learned table Q is not changed during evaluation (greedy policy, epsilon = 0).
- The random baseline for seed k uses `np.random.default_rng(k)` and environment seed k.

## Hyperparameters

| Parameter | Value |
|---|---|
| alpha (step size) | 0.1 |
| gamma (discount) | 0.99 |
| epsilon | 0.1 and 0.3 (the parameter study) |
| training episodes per seed | 5000 |
| evaluation episodes per seed | 1000 (greedy, no learning) |
| environment | `FrozenLake-v1`, `is_slippery=True`, 4x4 map |

## Files

| File | Contents |
|---|---|
| `main.py` | Environment creation, random baseline, epsilon-greedy action choice, SARSA training, separate evaluation, experiments, results table, figures |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |
| `results.txt` | Output table  |
| `training_curve.png`, `comparison.png` | Figures  |

## References

- Gymnasium documentation, FrozenLake: https://gymnasium.farama.org/
- R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd edition.
