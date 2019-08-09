import numpy as np

nan = np.nan # represents impossible actions

# Transition probabilities - shape=[s, a, s']
T = np.array([
    [[0.7, 0.3, 0.0], [1.0, 0.0, 0.0], [0.8, 0.2, 0.0]],
    [[0.0, 1.0, 0.0], [nan, nan, nan], [0.0, 0.0, 1.0]],
    [[nan, nan, nan], [0.8, 0.1, 0.1], [nan, nan, nan]]
])

# Rewards for transitions
R = np.array([
    [[10., 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    [[0.0, 0.0, 0.0], [nan, nan, nan], [0.0, 0.0, -50]],
    [[nan, nan, nan], [40., 0.0, 0.0], [nan, nan, nan]] 
])

possible_actions = [[0, 1, 2], [0, 2], [1]]

Q = np.full((3,3), -np.inf) # -inf for impossible actions
for state, actions in enumerate(possible_actions):
    Q[state, actions] = 0.0 # Initial value = 0.0, for all possible actions

gamma = 0.95 # the discount factor
n_iterations = 100

print("Starting...")
for iteration in range(n_iterations):
    #print("  Iteration ", iteration)
    Q_prev = Q.copy()
    #print("  Q-table:", Q)

    for s in range(3):
        #print("    Init State ",s)
        for a in possible_actions[s]:
            #print("      Action ",a)

            arr_for_sa = [ T[s, a, sp] * (R[s, a, sp] + gamma * np.max(Q_prev[sp])) for sp in range(3) ]
            #print("      Vals to all target states:", arr_for_sa)
            
            Q[s, a] = np.sum(arr_for_sa)

print(Q)
print("Optimal actions per state:", np.argmax(Q, axis=1)) 


print("---Starting actual Q-learning---")
# Q-learning
learning_rate0 = 0.95
learning_rate_decay = 0.05
n_iterations = 50000

s = 0 # start in state 0

Q = np.full((3, 3), -np.inf) # -inf for impossible actions
for state, actions in enumerate(possible_actions):
    Q[state, actions] = 0.0 # Initial value = 0.0, for all possible actions


for iteration in range(n_iterations):
    if iteration % 1000 == 999: 
        print("  Iteration ", iteration, ", lr:", learning_rate)
    a = np.random.choice(possible_actions[s]) # choose an action (randomly)
    sp = np.random.choice(range(3), p=T[s, a]) # pick next state using T[s, a]
    reward = R[s, a, sp]
    learning_rate = learning_rate0 / (1 + iteration * learning_rate_decay)
    Q[s, a] = ((1 - learning_rate) * Q[s, a] + learning_rate * (reward + gamma*np.max(Q[sp])))
    s = sp # move to the next state

print(Q)
print("Optimal actions per state:", np.argmax(Q, axis=1)) 


### E-GREEDY SIMULATION ###
print("---Doing e-greedy...---")

# Q-learning
learning_rate0 = 0.95
learning_rate_decay = 0.05
n_iterations = 100
e_param0 = 1.0
e_param_decay = 0.05

s = 0 # start in state 0

Q = np.full((3, 3), -np.inf) # -inf for impossible actions
for state, actions in enumerate(possible_actions):
    Q[state, actions] = 0.0 # Initial value = 0.0, for all possible actions

for iteration in range(n_iterations):
    if iteration % 1 == 0: 
        print("Iteration ", iteration)
    
    print("  Current state:", s)
    print("  Best action per state:", np.argmax(Q, axis=1))
    print("  Q-table:")
    for s_ in range(Q.shape[0]):
        for a_ in range(Q.shape[1]):
            print("    ["+str(s_)+","+str(a_)+"]:",Q[s_,a_])

    #Decide whether to do greedy or random
    rnd = np.random.random_sample()
    e_param = e_param0 / (1 + iteration * e_param_decay)
    print("  E-param:",e_param)
    if rnd<e_param: # be random
        a = np.random.choice(possible_actions[s]) # choose an action (randomly)
        print("  Random action:",a)
    else: # be greedy
        a = np.argmax(Q, axis=1)[s]
        print("  Greedy action:",a)

    print("   ->Distr of transitions [",s,",",a,"]: ", T[s,a])
    sp = np.random.choice(range(3), p=T[s, a]) # pick next state using T[s, a]
    print("  Transitioned to state:",sp)
    
    reward = R[s, a, sp]
    print("  Got reward:",reward)
    
    learning_rate = learning_rate0 / (1 + iteration * learning_rate_decay)
    print("  Learning rate:", learning_rate)
    Q[s, a] = ((1 - learning_rate) * Q[s, a] + learning_rate * (reward + gamma*np.max(Q[sp])))
    s = sp # move to the next state
    
    input("Press Enter...")
