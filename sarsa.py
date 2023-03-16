import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from os.path import exists

from env import step, reset
from utils import mse
from monte_carlo import mc

# action = 0: player sticks
# action = 1: player draws a card

# state = (dealer’s first card 1–10, the player’s sum 1–21)
# terminal state = (0, 0)

# Using lookup tables
# e.g. N_s_a[9][20][1]

def sarsa():
    with open('q_star.npz', 'rb') as f:
        Q_star_s_a = np.load(f)

    episodes = 10_000
    l_ls = np.arange(0,11)/10

    l_mse_ls = []

    for l in tqdm(l_ls):
        mse_ls = []
        N_s_a = np.zeros((11, 22, 2))
        N_s = np.zeros((11, 22))
        Q_s_a = np.zeros((11, 22, 2))

        for i in tqdm(range(1, episodes + 1)):
            
            r, s = reset()
            a = random.randint(0, 1)
            S_ls = []
            E_s_a = np.zeros((11, 22, 2))

            while s != (0, 0):
                # Take action
                new_r, new_s = step(s, a)

                # Not terminated
                if new_s != (0, 0):
                    # Select new action
                    N_s[new_s[0], new_s[1]] += 1
                    epsilon = 100 / (100 + N_s[new_s[0], new_s[1]])

                    if np.random.rand() < epsilon:
                        new_a = random.randint(0, 1)
                    else:
                        new_a = np.argmax([Q_s_a[new_s[0], new_s[1], 0], Q_s_a[new_s[0], new_s[1], 1]])
                
                    td_error = new_r +  Q_s_a[new_s[0], new_s[1], new_a] - Q_s_a[s[0], s[1], a]
                
                else:
                    td_error = new_r - Q_s_a[s[0], s[1], a]
                
                N_s_a[s[0], s[1], a] += 1
                E_s_a[s[0], s[1], a] += 1
                S_ls.append((s[0], s[1], a))

                for s_0, s_1, a in S_ls:
                    alpha = 1/N_s_a[s_0, s_1, a]
                    Q_s_a[s_0, s_1, a] += alpha * td_error * E_s_a[s_0, s_1, a]
                    E_s_a[s_0, s_1, a] *= l

                s = new_s
                if s != (0, 0): 
                    a = new_a

            mse_ls.append(mse(Q_s_a, Q_star_s_a))

        l_mse_ls.append(mse(Q_s_a, Q_star_s_a))

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
    sarsa()

        
    