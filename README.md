# Easy 21 RL
Different RL algorithms implemented from scratch to the Easy 21 card game

This is my answer to the [Easy21 assignment](https://www.davidsilver.uk/wp-content/uploads/2020/03/Easy21-Johannes.pdf) from [David Silver's RL Course](https://www.davidsilver.uk/teaching/)

## Monte Carlo Control
$V^*(s) = max_aQ^*(s,a)$

For 10,000,000 runs:

To use: run `monte_carlo.py`

## Sarsa($\lambda$)
$With \ parameter \ values \ λ ∈ {0, 0.1, 0.2, ..., 1}$

For 10,000 episodes:


To use: run `sarsa.py`

## Sarsa($\lambda$) with Linear Function Approximation

$Binary \ feature \ vector \ φ(s, a) \ with \ 3 ∗ 6 ∗ 2 = 36 \ features $


$Dealer(s) = \{[1, 4], [4, 7], [7, 10]\} $

$Player(s) = \{[1, 6], [4, 9], [7, 12], [10, 15], [13, 18], [16, 21]\} $

$a = \{hit, stick\} $

$With \ parameter \ values \ λ ∈ \{0, 0.1, 0.2, ..., 1\} $


For 10,000 episodes:

To use: run `sarsa_linear.py`

### Dependencies
numpy, tqdm, matplotlib, pandas