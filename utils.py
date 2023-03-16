import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

def plot(Q_s_a):
    v_star = []

    for y in range(1, 22):
        for x in range(1, 11):
            v_star.append([y, x, np.max([Q_s_a[x, y, 0], Q_s_a[x, y, 1]])])

    df = pd.DataFrame(v_star, columns=['player', 'dealer', 'value'])

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.set_xlabel('Dealer showing')
    ax.set_ylabel('Player sum')
    ax.set_zlabel('V*')
    ax.plot_trisurf(df['dealer'], df['player'], df['value'], cmap=plt.cm.viridis, linewidth=0.2)
    plt.show()

def mse(Q_s_a, Q_star_s_a):
    return np.average((Q_s_a - Q_star_s_a) ** 2)



def to_linear(s, a):
    dealer_states = [(1, 4), (4, 7), (7, 10)]
    player_states = [(1, 6), (4, 9), (7, 12), (10, 15), (13, 18), (16, 21)]

    x_s = []
    y_s = []
    
    for i, (mini, maxi) in enumerate(dealer_states):
        if s[0] >= mini and s[0] <=maxi:
            x_s.append(i)
    for i, (mini, maxi) in enumerate(player_states):
        if s[1] >= mini and s[1] <=maxi:
            y_s.append(i)

    n_s = np.zeros((3, 6, 2))

    for x in x_s:
        for y in y_s:
            n_s[x, y, a] = 1

    return n_s

def mse_linear(W_s, Q_star_s_a):
    error = []
    for y in range(1, 22):
        for x in range(1, 11):
            error.append((np.max([np.sum(W_s * to_linear((x, y), 0)), np.sum(W_s * to_linear((x, y), 1))]) - np.max([Q_star_s_a[x, y, 0], Q_star_s_a[x, y, 1]])) ** 2)

    return np.mean(error)

def linear_plot(W_s):
    v_star = []

    for y in range(1, 22):
        for x in range(1, 11):
            v_star.append([y, x, np.max([np.sum(W_s * to_linear((x, y), 0)), np.sum(W_s * to_linear((x, y), 1))])])

    df = pd.DataFrame(v_star, columns=['player', 'dealer', 'value'])

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.set_xlabel('Dealer showing')
    ax.set_ylabel('Player sum')
    ax.set_zlabel('V*')
    ax.plot_trisurf(df['dealer'], df['player'], df['value'], cmap=plt.cm.viridis, linewidth=0.2)
    plt.show()