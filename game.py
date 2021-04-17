import numpy as np
import time

def game_over(state):
    max_in_a_row = get_lanes(state).sum(axis=-1).max()
    min_in_a_row = get_lanes(state).sum(axis=-1).min()
    if max_in_a_row == 3:
        print('x won')
        return 1
    if min_in_a_row == -3:
        print('o won')
        return 2
    if not 0 in state:
        print('tie game')
        return 3
    return 0
    
def get_lanes(state):
    lanes = np.zeros((8,3))
    lanes[:3] = state
    lanes[3:6] = state.transpose()
    lanes[6] = state.diagonal()
    lanes[7] = np.diagonal(np.fliplr(state))
    return lanes

def play(model, state):
    possible_states = None
    for i in range(3):
        for j in range(3):
            # print(state[i,j])
            if not state[i,j] == 0: continue
            # print('found one')
            
            state[i,j] = 1
            if possible_states is None:
                possible_states = state.copy().reshape((1,3,3))
            else:
                possible_states = np.concatenate([possible_states, state.reshape((1,3,3))])
            state[i,j] = 0
            # print(possible_states)
    # print('found {} possible_states'.format(possible_states.shape[0]))
    # print(possible_states)
    possible_states = possible_states.reshape((-1,3,3))
    values = model.predict(possible_states)
    # print(values)
    return possible_states[values.argmax()]

class Model():
    def __init__(self, W=None, b=None):
        if W == None:
            self.W = np.random.normal(size=(4))
        else:
            if not W.shape == (4): raise Exception('incorrect size')
            self.W = W.copy()
        if b == None:
            self.b = 0
        else:
            self.b = float(b)
        print(self.W)
        
    def predict(self, possible_states):
        size = possible_states.shape[0]
        y = np.zeros(size)
        for i in range(size):
            x = self.get_features(possible_states[i])
            y[i] = self.W.dot(x) + self.b
        return y
        
    def get_features(self, state):
        lanes = get_lanes(state)
        features = np.zeros((4))
        for lane in lanes:
            ours = 0
            opps = 0
            for v in lane:
                if v == 1: ours += 1
                if v == -1: opps += 1
            if ours == 2: features[0] += 1
            if ours == 3: features[1] += 1
            if opps == 2: features[2] += 1
            if opps == 3: features[3] += 1
        return features
    
    def fit(self, train_states, ytrain, lr=0.01):
        print('training model')
        for i in range(train_states.shape[0]):
            x = self.get_features(train_states[i])
            diff = ytrain[i] - (self.W.dot(x) + self.b)
            self.b += lr * diff
            for j in range(self.W.shape[0]):
                self.W[j] += lr * diff * x[j]
            print(self.W)
            print(self.b)
        print('done training')
        

def critic(trace, model):
    flip = -1
    xtrain = np.zeros((trace.shape[0]-1,3,3))
    ytrain = np.zeros((trace.shape[0]-1))
    for i in range(trace.shape[0]-1):
        flip *= -1
        xtrain[i] = flip*trace[i]
        
        value = None
        type = game_over(flip*trace[i+1])
        if type == 0:
            value = model.predict(flip*trace[i+1].reshape(1,3,3))
        elif type == 1:
            # you won
            value = 100
        elif type == 2:
            # you lost fool
            value = -100
        elif type == 3:
            # you tied
            value = -10
        ytrain[i] = value
    return xtrain, ytrain
    

if __name__ == '__main__':
    #optimal model after 50k games
    # model = Model(np.array([2.76e-13,5.66e-03,-5.499e+01,-2.284e-01]), 99.9)
    model = Model()
    model_frozen = Model()
    
    won = 0.0
    total = 0.0
    ties = 0.0
    while True:
        # time.sleep(5)
        total += 1
        state = np.zeros((3,3))
        print(state)
        flip = -1
        
        trace = state.copy().reshape(1,3,3)
        while True:
            state = play(model, state)
            trace = np.concatenate([trace,state.reshape(1,3,3)])
            print(state)
            
            type = game_over(state)
            if type:
                print('game over' +str(type))
                if type == 1:
                    won += 1
                elif type == 3:
                    ties += 1
                break
                
            # flip *= -1
            state = flip*play(model_frozen, flip*state)
            trace = np.concatenate([trace,state.reshape(1,3,3)])
            
            print(state)
            # time.sleep(25)
            
            type = game_over(state)
            if type:
                print('game over' +str(type))
                if type == 1:
                    won += 1
                elif type == 3:
                    ties += 1
                break
        
        print('final state')
        print(state)  
        xtrain, ytrain = critic(trace, model)
        model.fit(xtrain, ytrain)
        print(won)
        print(ties)
        print(total)
        print(float(won)/total)
        if total%500 == 0:
            time.sleep(2)