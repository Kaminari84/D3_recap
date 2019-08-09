from flask import Flask, request, make_response, render_template, current_app, g
import json
import os
import numpy as np

app = Flask(__name__)
app.app_context().push()

state2idx = {}
idx2state = []

action2idx = {}
idx2action = []

#http://kronosapiens.github.io/blog/2014/08/14/understanding-contexts-in-flask.html
#https://github.com/Kaminari84/GCloud_work_reflection_bot/blob/master/dataMgr.py
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

    print("Start the actual server...")

setup_app(app)

#map state params into unique index
def getStateIndex(state):
    state_str = str(state['utterance_id'])+"_"+str(state['answer_id'])
    print("Check if state registered: "+str(state_str))
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

@app.route('/getQTable')
def get_q_table():
    print('returning Q table:')

    with app.app_context():
        # within this block, current_app points to app.
        print("App name:",current_app.name)
        print("Q:",current_app.Q)

        #for a in range(Q.shape[1]):
        #    print("\t"+str(idx2action[a]), end='')
        #print("")
        # actual table
        #for s in range(Q.shape[0]):
        #    print("Q["+str(s)+"]", end='')
        #    for a in range(Q[s].shape[0]):
        #        print("\t"+str(Q[s,a]), end=' | ')
        #    print("")

        json_resp = json.dumps({ 'status': 'OK', 'message':'', 'q-table': current_app.Q.tolist()})#ctx.g.Q.tolist()})

    return make_response(json_resp, 200, {"content_type":"application/json"})

@app.route('/registerSandA')
def init_q_table():
    print("Called registration of states and actions on the server...")
    with app.app_context():
        # within this block, current_app points to app.
        print("App name:",current_app.name)
        print("Q:",current_app.Q)
        
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

    Q = np.full((len(states), len(actions)), 0.0) # -inf for impossible actions
    #for state, actions in enumerate(possible_actions):
    #    Q[state, actions] = 0.0 # Initial value = 0.0, for all possible actions

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
    #ctx.push()

    with app.app_context():
        current_app.Q = Q

    json_resp = json.dumps({ 'status': 'OK', 'message':''})

    return make_response(json_resp, 200, {"content_type":"application/json"})

@app.route('/selectRLAction')
def select_rl_action():
    print("Called select RL Action on server...")
    conv_id = request.args.get('conv_id')
    state = json.loads(request.args.get('state'))
    actions = json.loads(request.args.get('actions'))

    print("Conv id"+str(conv_id))
    print("State:"+str(state))
    print("Actions:"+str(actions))

    print("Number of actions to choose from:"+str(len(actions['reaction_list'])))
    selected_action = np.random.choice(actions['reaction_list'])
    action_id = actions['reaction_list'].index(selected_action)

    

    json_resp = json.dumps({ 'status': 'OK', 'message':'', 'action_id': action_id })

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
        if n>1000:
            break

    return render_template('show_aapl.html',
        stock_prices = values
    )