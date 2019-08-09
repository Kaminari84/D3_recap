import numpy as np
from scipy.special import softmax

nan = np.nan # represents impossible actions

# Transition probabilities - shape=[s, a, s']
T = np.array([
    [[0.0, 0.0, 0.5, 0.5, 0.0], [0.0, 0.0, 0.5, 0.5, 0.0]],
    [[0.0, 0.0, 0.5, 0.5, 0.0], [0.0, 0.0, 0.5, 0.5, 0.0]],
    [[0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 1.0]],
    [[0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 1.0]],
    [[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]
])

# Rewards for transitions
R = np.array([
    [[0.0, 0.0, 10., 10., 0.0], [0.0, 0.0, -10., -10., 0.0]],   #0
    [[0.0, 0.0, -10., -10., 0.0], [0.0, 0.0, 10., 10., 0.0]],   #1
    [[0.0, 0.0, 0.0, 0.0, 10.], [0.0, 0.0, 0.0, 0.0, -10.]],    #2
    [[0.0, 0.0, 0.0, 0.0, -10.], [0.0, 0.0, 0.0, 0.0, 10.]],    #3
    [[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]      #4
])

possible_actions = [[0, 1], [0, 1], [0, 1], [0, 1], []]

# Q-learning
learning_rate0 = 0.95
learning_rate_decay = 0.05
n_iterations = 100
e_param0 = 1.0
e_param_decay = 0.01
gamma = 0.95 # the discount factor

transition_history = [] #for delayed rewards

s = np.random.choice([0,1]) # start in state 0
ET = np.full((T.shape[0], 2), 0) # explored transitions
Q = np.full((T.shape[0], 2), 0.0) # -inf for impossible actions
for state, actions in enumerate(possible_actions):
    Q[state, actions] = 0.0 # Initial value = 0.0, for all possible actions
    ET[state, actions] = 0

for iteration in range(n_iterations):
    if iteration % 1 == 0: 
        print("Iteration ", iteration)
    
    print("  Current state:", s)
    print("  Best action per state:", np.argmax(Q, axis=1))
    print("  Q-table:")
    for s_ in range(Q.shape[0]):
        for a_ in range(Q.shape[1]):
            print("    ["+str(s_)+","+str(a_)+"] -> N: ",ET[s_,a_],", R: ",Q[s_,a_])

    #Decide whether to do greedy or random
    rnd = np.random.random_sample()
    e_param = e_param0 / (1 + iteration * e_param_decay)
    print("  E-param:",e_param)
    if rnd<e_param: # be random
        print("ET:",ET[s], ", softmax:",(1.0-softmax(ET[s])))
        a = np.random.choice(possible_actions[s], p=(1.0-softmax(ET[s]))) # choose an action (randomly)
        print("  Random action:",a)
    else: # be greedy
        a = np.argmax(Q, axis=1)[s]
        print("  Greedy action:",a)

    print("   ->Distr of transitions [",s,",",a,"]: ", T[s,a])
    sp = np.random.choice(range(T.shape[0]), p=T[s, a]) # pick next state using T[s, a]
    print("  Transitioned to state:",sp)

    transition_history.append([s,a,sp])

    # Approach 1 - Just reward in end state with value depending on the actions before
    #non-terminal state
    if (sp < T.shape[0]-1):
        reward = 0 #R[s, a, sp]
    else:
        #go through transitions history and count up rewards
        reward = 0
        print("    FINAL STATE - looping through transition history")
        for t in transition_history:
            stsp_reward = R[t[0], t[1], t[2]]
            print('    S:'+str(t[0])+"->A:"+str(t[1])+"->SP:"+str(t[2])+" - R:"+str(stsp_reward))
            reward += stsp_reward
        print("    Cumulative final reward:"+str(reward))

    print("  Got reward:",reward)
    
    learning_rate = learning_rate0 / (1 + iteration * learning_rate_decay)
    print("  Learning rate:", learning_rate)
    print("  Q[s,a]:", Q[s, a])
    print("  np.max(Q[sp]):", np.max(Q[sp]))
    Q[s, a] = ((1 - learning_rate) * Q[s, a] + learning_rate * (reward + gamma*np.max(Q[sp])))
    
    K=10
    bonuses = [Q[sp, ap] + K/(1+ET[sp, ap]) for ap in range(len(Q[sp]))]
    print("Bonuses:", bonuses)
    ET[s, a] += 1

    s = sp # move to the next state
    if s == T.shape[0]-1: #terminal state
        # Approach 2 - Distribute the reward to individual actions retroactively
        transition_history = []
        print("**Reached terminal state, resetting...")
        s = np.random.choice([0,1])

    input("Press Enter...")

