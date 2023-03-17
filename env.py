from random import randint

# action = 0: player sticks
# action = 1: player draws a card

# state = (dealer’s first card 1–10, the player’s sum 1–21)
# terminal state = (0, 0)

# state is terminal if:
# action is stick 
# or action is draw and reward == -1

def draw():
    r = randint(1, 10)
    color = randint(1, 3)
    if color == 1:
        return - r
    else:
        return r


def step(state, action):
    '''
    Returns a sample of the next state s` and reward r
    '''
    if action == 0:
        # Player sticks
        # Dealer's turn
        dealer_value = state[0]
        while dealer_value >= 1 and dealer_value < 17:
           dealer_value += draw()

        if dealer_value > 21 or dealer_value < state[1] or dealer_value < 1:
            # Player wins
            reward = 1
        elif dealer_value > state[1]:
            # Player loses
            reward = -1
        else:
            # Player draws
            reward = 0
        return reward, (0, 0)
    
    else:
        # Player draws card
        new_state_p = state[1] + draw()

        if new_state_p > 21 or new_state_p < 1:
            # Player loses
            reward = -1
            return reward, (0, 0)
        else:
            # Game continues
            reward = 0
            return reward, (state[0], new_state_p)
    

def reset():
    state = [randint(1, 10), randint(1, 10)]
    return 0, state