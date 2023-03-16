import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from os.path import exists

from env import step, reset
from utils import mse_linear, linear_plot, to_linear
from monte_carlo import mc

# action = 0: player sticks
# action = 1: player draws a card

# state = (dealer’s first card 1–10, the player’s sum 1–21)
# terminal state = (0, 0)

def sarsa_linear():
    with open('q_star.npz', 'rb') as f:
        Q_star_s_a = np.load(f)

    episodes = 10_000
    l_ls = np.arange(0,11)/10

    l_mse_ls = []

    for l in tqdm(l_ls):
        mse_ls = []
        
        # Weights for linear func approx
        W_s = np.zeros((3, 6, 2))

        for i in tqdm(range(1, episodes + 1)):
            
            r, s = reset()
            a = random.randint(0, 1)
            S_ls = []
            E_s_a = np.zeros((11, 22, 2))

            while s != (0, 0):
                # Take action
                new_r, new_s = step(s, a)
                V_s_a_w = np.sum(W_s * to_linear(s, a))
                # Not terminated
                if new_s != (0, 0):
                    # Select new action
                    epsilon = 0.05

                    if np.random.rand() < epsilon:
                        new_a = random.randint(0, 1)
                    else:
                        new_a = np.argmax([np.sum(W_s * to_linear(new_s, 0)), np.sum(W_s * to_linear(new_s, 1))])
                
                    td_error = new_r + np.sum(W_s * to_linear(new_s, new_a)) - V_s_a_w
                
                else:
                    td_error = new_r - V_s_a_w
                
                
                E_s_a[s[0], s[1], a] += 1
                S_ls.append((s[0], s[1], a))

                for s_0, s_1, a in S_ls:
                    alpha = 0.01
                    W_s += alpha * td_error * E_s_a[s_0, s_1, a] * to_linear(s, a)
                    E_s_a[s_0, s_1, a] *= l

                s = new_s
                if s != (0, 0): 
                    a = new_a

            mse_ls.append(mse_linear(W_s, Q_star_s_a))
        
        # linear_plot(W_s)

        l_mse_ls.append(mse_linear(W_s, Q_star_s_a))

        plt.plot(range(1, episodes + 1), mse_ls, label=l)
    plt.legend(loc="upper right")
    plt.xlabel('Episode')
    plt.ylabel('Mean squared error')
    plt.title('Mean squared error per episode')
    plt.show()

    plt.plot(l_ls, l_mse_ls)
    plt.scatter(l_ls, l_mse_ls)
    plt.xlabel('Lambda')
    plt.ylabel('Mean squared error')
    plt.title('Mean squared error per lambda')
    plt.show()

if __name__ == '__main__':
    if not exists('q_star.npz'):
        print('No cached file found!\nMaking file...')
        mc(plot_graph=False, save_file=True)
    sarsa_linear()

        
    