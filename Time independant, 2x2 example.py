import numpy as np
import random 


states = np.array([0,1])
actions = np.array([0,1]) #0: Stay, 1: Move

N = 500
C = 5 #deterministic variable
mew = np.array([0.99, 0.01])
gamma = 0.5
phi = 500
rho_q = 0.55
rho_mew = 0.85
p = 0.01


Q = np.zeros((len(states), len(actions)))


def softmin_action(x, Q, beta):

    q_values = Q[x]
    # numerically stable softmin probabilities
    weights = np.exp(-beta * (q_values - np.min(q_values)))
    probs = weights / np.sum(weights)
    return np.random.choice(len(q_values), p=probs)


def cost_function(current_state, action, mew):
    return current_state + (C * mew[current_state])


def next_state(x, a):
    if a == 0:
        return x
    else:
        return 1-x


x = random.choice(states)

for n in range(N):

    #choose action
    a = softmin_action(x, Q, 500)

      #update mew
    mew_target = np.zeros(len(actions))
    mew_target[x] = 1

    #update mean
    mew = mew + rho_mew * (mew_target - mew)

    #observe cost
    cost = cost_function(x, a, mew)

    #apply action
    x_next = next_state(x, a)

    #update Q
    old_q = Q[x, a]
    Q[x, a] = old_q + rho_q * (
        cost + gamma * np.min(Q[x_next, :]) - old_q

        )
    
    #update state
    x = x_next

print(Q)

