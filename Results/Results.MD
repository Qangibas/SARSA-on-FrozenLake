# Results

Success rate = fraction of 1000 evaluation episodes that reached the goal.
Greedy policy (epsilon = 0), no learning during evaluation.
Mean ± standard deviation over 5 independent seeds (0, 1, 2, 3, 4).

| Method | Success rate (mean ± std) |
|---|---|
| Random baseline | 0.015 ± 0.003 |
| SARSA, epsilon = 0.1 | 0.604 ± 0.188 |
| SARSA, epsilon = 0.3 | 0.473 ± 0.176 |

## Setup

- Environment: Gymnasium `FrozenLake-v1`, 4x4, `is_slippery=True`
- Training: 5000 episodes per seed
- Evaluation: 1000 episodes per seed, separate seed (seed + 100)
- alpha = 0.1, gamma = 0.99
- Parameter study: epsilon = 0.1 vs 0.3

## Learned policies (seed 4)

Actions: 0 = left, 1 = down, 2 = right, 3 = up.
Values in cells for holes and the goal (states 5, 7, 11, 12, 15) are not meaningful, because those states are terminal and Q is never updated there.

epsilon = 0.1:

```
[[0 3 0 1]
 [0 0 2 0]
 [3 1 0 0]
 [0 2 1 0]]
```

epsilon = 0.3:

```
[[0 3 0 3]
 [0 0 0 0]
 [3 1 0 0]
 [0 1 1 0]]
```