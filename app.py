from flask import Flask, request, make_response, render_template, current_app, g
import json
import os
import numpy as np
from scipy.special import softmax
import pandas as pd
from datetime import datetime
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

app = Flask(__name__)
app.app_context().push()

state2idx = {}
idx2state = []

action2idx = {}
idx2action = []

# Q-learning
e_param0 = 1.0
e_param_decay = 0.05

learning_rate0 = 0.95
learning_rate_decay = 0.05

gamma = 0.95 # the discount factor



#http://kronosapiens.github.io/blog/2014/08/14/understanding-contexts-in-flask.html
#https://github.com/Kaminari84/GCloud_work_reflection_bot/blob/master/dataMgr.py
#https://botsociety.io/blog/2019/03/botsociety-dialogflow/
#https://medium.com/@jonathan_hui/rl-model-based-reinforcement-learning-3c2b6f0aa323
#https://towardsdatascience.com/model-based-reinforcement-learning-cb9e41ff1f0d

def setup_app(app):
    print("Loading the server, first init global vars...")
    #DataMgr.load_team_auths()
    #logging.info('Creating all database tables...')
    #db.create_all()
    #logging.info('Done!')

    #print("ENVIRON TEST:", os.environ['TEST_VAR'])

    with app.app_context():
        # within this block, current_app points to app.
        print("App name:",current_app.name)
        current_app.Q = np.full((1,1), -1.0)
        current_app.ET = np.full((1,1), -1)
        current_app.rET = np.full((1,1), -1)
        current_app.iteration = 0
        current_app.e_param = 1.0
        current_app.learning_rate = 1.0
        current_app.transition_history = []
        current_app.reward_history = []

    print("Start the actual server...")

setup_app(app)

#map state params into unique index
def getStateIndex(state):
    state_str = str(state['utterance_id'])+"_"+str(state['answer_id'])
    #print("Check if state registered: "+str(state_str))
    if state_str in state2idx:
        return state2idx[state_str]
    else:
        max_idx = len(idx2state)
        state2idx[state_str] = max_idx
        idx2state.append(state_str)
        return max_idx 

#map action params into unique action
def getActionIndex(action):
    action_str = str(action['action_class'])
    if action_str in action2idx:
        return action2idx[action_str]
    else:
        max_idx = len(idx2action)
        action2idx[action_str] = max_idx
        idx2action.append(action_str)
        return max_idx 

# Server paths
@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/test_layout')
def test_layout():
    return render_template('test_layout.html')

@app.route('/rl_bot')
def rl_bot():
    return render_template('rl_bot.html')

@app.route('/getEParam')
def get_e_param():
    print('returning E param:')

    with app.app_context():
        print("Current e-param:", current_app.e_param)
        json_resp = json.dumps({'status': 'OK', 
                                'message':'', 
                                'e-param': current_app.e_param})
    
    return make_response(json_resp, 200, {"content_type":"application/json"})

@app.route('/getRewardHistory')
def get_reward_history():
    print("Getting the reward history...")
    with app.app_context():
        json_resp = json.dumps({'status': 'OK', 
                                'message':'', 
                                'reward_history': current_app.reward_history})
    
    return make_response(json_resp, 200, {"content_type":"application/json"})                      

@app.route('/setEParam')
def set_e_param():
    print('setting E param:')
    e_param = request.args.get('e_param')

    with app.app_context():
        print("Old e-param:", current_app.e_param)
        current_app.e_param = e_param
        print("New e-param:", current_app.e_param)
        json_resp = json.dumps({'status': 'OK', 
                                'message':''})
    
    return make_response(json_resp, 200, {"content_type":"application/json"})

@app.route('/setReward')
def set_reward():
    print('Setting reward')

    conv_id = request.args.get('conv_id')
    dimension = request.args.get('dimension')
    reward = float(request.args.get('reward'))

    print("  Conv id"+str(conv_id))
    print("  Dimension:"+str(dimension))
    print("  Reward:"+str(reward))

    with app.app_context():
        print("  Current e-param:", current_app.e_param)

        current_app.learning_rate = learning_rate0 / (1 + current_app.iteration * learning_rate_decay)
        print("  Learning rate:", current_app.learning_rate)

        # loop through the history of transitions to this point
        print("  Looping through past transitions...")
        allGreedy = True # Keep track if all the transitions in the episode were greedy (optimal)
        for t in current_app.transition_history:
            print("    T -> ("+str('Random' if t[3] else 'Greedy')+", "+str(idx2state[t[0]])+", "+str(idx2action[t[1]])+", ", end='')
            # Not all transitions were greedy in this episode
            if t[3]:
                allGreedy = False

            next_state = t[2]
            if next_state == None:
                next_state = "END"
            else:
                next_state = idx2state[next_state]

            print(str(next_state+")"))

            # Reward component for source state
            s_reward = current_app.Q[t[0], t[1]]
            print("    Q["+str(idx2state[t[0]])+","+str(idx2action[t[1]])+"]:", s_reward)

            # Discounter reward for future states
            f_reward = 0 # reward is 0 for everything after the end state
            if next_state != "END":
                f_reward = np.max(current_app.Q[t[2]])

            print("    np.max(Q["+str(next_state)+"]):", f_reward)

            current_app.Q[t[0], t[1]] = (1 - current_app.learning_rate) * s_reward + current_app.learning_rate * (reward + gamma*f_reward)
            q_update = current_app.Q[t[0], t[1]]
            print("    Updated quality in state: "+str(q_update))

        # Reset transition history in preparation for future episode
        current_app.transition_history = []
        if allGreedy:
            print("  All GREEDY episode")
        else:
            print("  Mixed RANDOM-GREEDY episode")
        
        reward_entry = [1 if allGreedy else 0, reward]
        current_app.reward_history.append(reward_entry)

        json_resp = json.dumps({'status': 'OK', 
                                'message':''})
    
    return make_response(json_resp, 200, {"content_type":"application/json"})

@app.route('/getQTable')
def get_q_table():
    #print('returning Q table:')

    with app.app_context():
        # within this block, current_app points to app.
        #print("App name:",current_app.name)
        #print("Q:",current_app.Q)
        #print("ET:",current_app.ET)

        #print("Construct Full Q table:")
        full_q_table = {}
        for s_id in range(len(current_app.Q)):
            #print("s_id:"+str(s_id))
            state_spec = {}
            for a_id in range(len(current_app.Q[s_id])):
                eval_dims = {}
                # Ideally loop through evaluation dimensions
                eval_dims['q-value'] = current_app.Q[s_id, a_id]
                eval_dims['count'] = str(current_app.ET[s_id, a_id])
                eval_dims['r-count'] = str(current_app.rET[s_id, a_id])

                state_spec[idx2action[a_id]] = eval_dims
                print("["+str(idx2state[s_id])+", "+str(idx2action[a_id])+"]:"+str(current_app.Q[s_id,a_id]))
            full_q_table[idx2state[s_id]] = state_spec
        
        #print("Full Q table:", full_q_table)

        #for a in range(Q.shape[1]):
        #    print("\t"+str(idx2action[a]), end='')
        #print("")
        # actual table
        #for s in range(Q.shape[0]):
        #    print("Q["+str(s)+"]", end='')
        #    for a in range(Q[s].shape[0]):
        #        print("\t"+str(Q[s,a]), end=' | ')
        #    print("")

        json_resp = json.dumps({ 
            'status': 'OK', 'message':'', 
            'full-q-table': full_q_table
        })

    return make_response(json_resp, 200, {"content_type":"application/json"})

@app.route('/registerSandA')
def init_q_table():
    print("Called registration of states and actions on the server...")
    with app.app_context():
        # within this block, current_app points to app.
        print("App name:",current_app.name)
        print("Q:",current_app.Q)
        #print("ET:",current_app.ET)
        
    conv_id = request.args.get('conv_id')
    states = json.loads(request.args.get('states'))
    actions = json.loads(request.args.get('actions'))

    print("Conv id"+str(conv_id))
    print("States:"+str(states))
    print("Actions:"+str(actions))

    print("Looping throght states:")
    for state in states:
        print("Utterance ID:"+str(state['utterance_id'])+", Answer ID:"+str(state['answer_id'])+", IDX:"+str(getStateIndex(state)))

    for action_id, action_params in actions.items():
        print("Action ID:"+str(action_id)+", IDX:"+str(getActionIndex(action_params)))

    actions_in_states = [[] for s in states]
    # assigning actions in states
    for state in states:
        s_idx = getStateIndex(state)
        for action in state['actions_in_state']:
            a_idx = getActionIndex(action)
            actions_in_states[s_idx].append(a_idx)
            actions_in_states[s_idx].sort()

    print("Actions in states:"+str(actions_in_states))

    Q = np.full((len(states), len(actions)), 0.0) # -inf for impossible actions
    ET = np.full((len(states), len(actions)), 0) # -inf for impossible action
    rET = np.full((len(states), len(actions)), 0) # -inf for impossible action

    print('Q table:')
    # header for actions
    for a in range(Q.shape[1]):
        print("\t"+str(idx2action[a]), end='')
    print("")
    # actual table
    for s in range(Q.shape[0]):
        print("Q["+str(s)+"]", end='')
        for a in range(Q[s].shape[0]):
            print("\t"+str(Q[s,a]), end=' | ')
        print("")

    with app.app_context():
        current_app.Q = Q
        current_app.ET = ET
        current_app.rET = rET
        current_app.actions_in_states = actions_in_states

    json_resp = json.dumps({ 'status': 'OK', 'message':''})

    return make_response(json_resp, 200, {"content_type":"application/json"})

@app.route('/selectRLAction')
def select_rl_action():
    print("Called select RL Action on server...")
    #conv_id = request.args.get('conv_id')
    state = json.loads(request.args.get('state'))
    #actions = json.loads(request.args.get('actions'))

    #print("Conv id"+str(conv_id))
    print("State:"+str(state))
    #print("Actions:"+str(actions))

    #print("Number of actions to choose from:"+str(len(actions['reaction_list'])))
    #selected_action = np.random.choice(actions['reaction_list'])
    #action_id = actions['reaction_list'].index(selected_action)
    #print("Action ID: "+str(action_id))
    action_id = -1

    with app.app_context():
        # within this block, current_app points to app.
        print("  Iterations:",current_app.iteration)

        s_idx = getStateIndex(state)
        print("  State idx:", s_idx, ", State abbr:"+str(state['utterance_id'])+"_"+str(state['answer_id']))

        # update final state for previous transition
        if len(current_app.transition_history) > 0:
            print("There is previous unfinished transition...")
            current_app.transition_history[-1][2] = s_idx
            print("Last transition full: "+str(current_app.transition_history[-1]))

        #Decide whether to do greedy or random
        rnd = np.random.random_sample()
        e_param = e_param0 / (1 + current_app.iteration * e_param_decay)
        
        print("  E-param:",e_param)
        isRandom = False
        if rnd<e_param: # be random
            isRandom = True
            print("  rET:",current_app.rET[s_idx], ", softmax:",(1.0-softmax(current_app.rET[s_idx])))
            print("  Actions available: ", end='')
            for act_id in current_app.actions_in_states[s_idx]:
                print(str(idx2action[act_id])+"("+str(act_id)+")", end=' ')
            print("")

            action_id = np.random.choice(current_app.actions_in_states[s_idx], 
                                         p=(1.0-softmax(current_app.rET[s_idx]))) # choose an action (randomly)
            print("  Random action chosen:", idx2action[action_id]+"("+str(action_id)+")" )
        else: # be greedy
            isRandom = False
            action_id = np.argmax(current_app.Q, axis=1)[s_idx]
            print("  Greedy action chosen:", idx2action[action_id]+"("+str(action_id)+")" )

        #register this state and action in transition history, leave next state empty
        current_app.transition_history.append([s_idx, action_id, None, isRandom])

        current_app.ET[s_idx, action_id] += 1
        if isRandom:
            current_app.rET[s_idx, action_id] += 1
        current_app.iteration += 1
        current_app.e_param = e_param    

    print("Action ID: "+str(action_id), ", type:"+idx2action[action_id])

    json_resp = json.dumps({ 'status': 'OK', 'message':'', 
                             'action_class': str(idx2action[action_id]),
                             'action_source': str("Random" if isRandom else "Greedy") })

    return make_response(json_resp, 200, {"content_type":"application/json"})

@app.route('/show_aapl')
def appl_test():
    r = { 'history':{ 
        '2019-07-24': {'close': '208.67', 
                'high': '209.15', 
                'low': '207.17', 
                'open': '207.67', 
                'volume': '14991567'}, 
        '2019-07-25': {'close': '207.02', 
                'high': '209.24', 
                'low': '206.73', 
                'open': '208.89', 
                'volume': '13909562'}, 
        '2019-07-26': {'close': '207.74', 
                'high': '209.73', 
                'low': '207.14', 
                'open': '207.48',
                'volume': '17618874'} 
        }, 
    'name': 'AAPL'}

    #Load JSON
    f = open('AAPL.json', 'r')
    r = json.load(f)
    f.close()

    n = 0
    values = []
    for key in sorted(r['history'].keys(), reverse=True):
        val = r['history'][key]
        #print("Key:",key," ,Val:",val)
        values.append( { 'datetime': key, 'close': val['close'] } )
        n += 1
        if n>200:
            break

    return render_template('show_aapl.html',
        stock_prices = values
    )

@app.route('/getStockPredictions')
def get_stock_predictions():
    stock_ticket = request.args.get('stock_ticket')
    data_point = request.args.get('data_point')

    print("Stock ticket:",str(stock_ticket))
    print("Data point:", str(data_point))

    point_date = datetime.strptime(data_point, "%a %b %d %Y")

    print("Parsed date:", str(point_date.strftime("%d/%m/%y")))

    #Temp solution - replace with persistent storage
    f = open('AAPL.json', 'r')
    r = json.load(f)
    f.close()

    n = 0
    entries = {'datetime': [], 'close': []}
    for key in sorted(r['history'].keys(), reverse=True):
        val = r['history'][key]
        #print("Key:",key," ,Val:",val)
        entries['datetime'].append( key )
        entries['close'].append( float(val['close']) )
        n += 1
        if n>200:
            break

    df = pd.DataFrame(entries, columns=['datetime','close'])
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.sort_values(by=['datetime'], inplace=True, ascending=True)
    #print(df.columns)
    print(df.shape)
    #print(df.head(10))

    # Mask - only select data points from the past from this point
    mask = (df['datetime'] < data_point)
    print(df.loc[mask].shape)
    print(df.loc[mask].head(10))

    #create the training data
    n_features = 3
    features = []
    prices = []

    feature_row = []
    print("Iterating through past data...")
    for index, row in df.loc[mask].iterrows():
        print("Index:", index, ", Date:", row['datetime'], ", Close:", row['close'])
        # If enough features in history set this price as the price to predict
        if len(feature_row) >= n_features:
            prices.append(row['close'])
            features.append(feature_row)
        
        # Shift forward by one data point
        feature_row = feature_row.copy()
        feature_row.append(row['close'])
        if len(feature_row) > n_features:
            feature_row.pop(0)

    print("Features:", features)
    print("Prices:", prices)
    
    features = np.reshape(features, (len(features), n_features))
    #print(features[:,0:n_features])
   
    #Initialize different models
    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree = 2) #, gamma='scale')
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma='auto')

    #Train models
    svr_lin.fit(features[:,0:n_features], prices)
    svr_poly.fit(features[:,0:n_features], prices)
    svr_rbf.fit(features[:,0:n_features], prices)

    #Predict
    predict_lin = svr_lin.predict(features[:,0:n_features])
    predict_poly = svr_poly.predict(features[:,0:n_features])
    predict_rbf = svr_rbf.predict(features[:,0:n_features])

    #Evaluate
    print("Predicted prices - linear:",predict_lin[-5:])
    print("Predicted prices - RBF:",predict_rbf[-5:])
    print("MSE for linear - training:", mean_squared_error(prices, predict_lin))
    print("MSE for poly - training:", mean_squared_error(prices, predict_poly))
    print("MSE for RBF - training:", mean_squared_error(prices, predict_rbf))

    print("Test MSE low:",mean_squared_error([122.2, 134.5, 100.2], [122.1, 134.4, 100.2]))
    print("Test MSE high:",mean_squared_error([122.2, 134.5, 100.2], [112.1, 138.8, 105.9]))

    #Recursive prediction for k days ahead
    gen_features = features[-1,0:n_features].tolist()
    predictions = []
    print("Gen features, start:", gen_features)
    for day_i in range(30):
        #print("Gen features:", gen_features[-n_features:])
        form_features = np.reshape([gen_features[-n_features:]], (1, n_features))
        print("Iter ", day_i,", form features:", form_features[0])
        predict_rbf = svr_rbf.predict(form_features)
        pred = predict_rbf[0] #round(,2)
        gen_features.append(pred)
        predictions.append(pred)

    #Return predictions
    json_resp = json.dumps({'status': 'OK', 'message':'', 'predictions':predictions})
    
    return make_response(json_resp, 200, {"content_type":"application/json"})