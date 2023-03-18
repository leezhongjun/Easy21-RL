# Easy 21 RL
Different RL algorithms implemented from scratch to the [Easy 21 card game](https://en.wikipedia.org/wiki/Twenty-One_(banking_game))

This is based the [Easy21 assignment](https://www.davidsilver.uk/wp-content/uploads/2020/03/Easy21-Johannes.pdf) from [David Silver's RL Course](https://www.davidsilver.uk/teaching/)

## Monte Carlo Control
$V^*(s) = max_a \ Q^{\ast}(s,a)$

$\text{With } \epsilon \text{-greedy exploration strategy: }\epsilon = N_0 / (N_0 + N(s_t)) \text{, where } N_0 = 100 \text{ is a constant} $

$\text{With a time-varying scalar step-size of } \alpha_{t} = 1/N(s_t, a_t) $

For 10,000,000 episodes:

![monte-carlo](https://user-images.githubusercontent.com/80515759/225575928-74ad101c-44f3-4ec7-bf09-53396a6ca0c8.png)

To use: run `monte_carlo.py`

## Sarsa($\lambda$)
$With \ parameter \ values \ λ ∈ \\{0, 0.1, 0.2, ..., 1\\}$

$\text{With } \epsilon \text{-greedy exploration strategy: }\epsilon = N_0 / (N_0 + N(s_t)) \text{, where } N_0 = 100 \text{ is a constant} $

$\text{With a time-varying scalar step-size of } \alpha_{t} = 1/N(s_t, a_t) $

For 10,000 episodes:

![td_mse_lambda](https://user-images.githubusercontent.com/80515759/225575892-29ba212b-2949-44be-b9dd-0df9aa46313a.png)

![td_mse_ep](https://user-images.githubusercontent.com/80515759/225575912-8e56062b-b149-46aa-a4af-9e9846d47e20.png)

To use: run `sarsa.py`


## Sarsa($\lambda$) with Linear Function Approximation

$\text{Binary  feature  vector }\phi(s, a) \text{ with 3 ∗ 6 ∗ 2 = 36 features } $

$\text{Dealer(s) = } \lbrace{[1, 4], [4, 7], [7, 10]}\rbrace $

$\text{Player(s) = } \lbrace[1, 6], [4, 9], [7, 12], [10, 15], [13, 18], [16, 21]\rbrace $

$a = \lbrace{\text{hit}, \text{stick}}\rbrace $

$\text{With parameter values }\lambda \in \lbrace{0, 0.1, 0.2, ..., 1}\rbrace $


$\text{With } \epsilon \text{-greedy exploration strategy: }\epsilon = 0.05 $

$\text{With a constant step-size of } \alpha_{t} = 0.01 $

For 10,000 episodes:

![td_linear_mse_l](https://user-images.githubusercontent.com/80515759/225575827-1a2b36c1-55c4-4095-8faf-325eb7699c0d.png)

![td_linear_mse_ep](https://user-images.githubusercontent.com/80515759/225575817-71d0d368-3c47-4f83-a582-66e9968ae61d.png)

To use: run `sarsa_linear.py`


### Dependencies
numpy, tqdm, matplotlib, pandas
