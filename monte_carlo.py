import numpy as np
from tqdm import tqdm
import random
from os.path import exists

from env import step, reset
from utils import plot

# action = 0: player sticks
# action = 1: player draws a card

# state = (dealer’s first card 1–10, the player’s sum 1–21)
# terminal state = (0, 0)

# Using lookup tables
# e.g. N_s_a[9][20][1]

def mc(save_file=False, plot_graph=True):

    N_s_a = np.zeros((11, 22, 2))
    N_s = np.zeros((11, 22))
    Q_s_a = np.zeros((11, 22, 2))

    episodes = 10_000_000

    for i in tqdm(range(episodes)):
        r, s = reset()
        R_s_a = []
        while s != (0, 0):
            # Get action
            N_s[s[0], s[1]] += 1
            epsilon = 100 / (100 + N_s[s[0], s[1]])
            if np.random.rand() < epsilon:
                a = random.randint(0, 1)
            else:
                a = np.argmax([Q_s_a[s[0], s[1], 0], Q_s_a[s[0], s[1], 1]])
            new_r, new_s = step(s, a)
            R_s_a.append((s[0], s[1], a, new_r))
            s = new_s
        
        G = sum(r[-1] for r in R_s_a)
        # Since the only time a non-zero reward is achieved is the last step:
        # G == R_s_a[-1][-1]
        for s_d, s_p, a, _ in R_s_a:
            N_s_a[s_d, s_p, a] += 1
            alpha = 1/N_s_a[s_d, s_p, a]
            Q_s_a[s_d, s_p, a] += alpha * (G - Q_s_a[s_d, s_p, a])

    if save_file:
        with open('q_star.npz', 'wb') as f:
            np.save(f, Q_s_a)

    if plot_graph:
        plot(Q_s_a)


if __name__ == '__main__':
    mc(save_file=exists('q_star.npz'))