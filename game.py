import numpy as np

def game_over(state):
    max_in_a_row = np.absolute(get_lanes(state).sum(axis=-1)).max()
    if max_in_a_row == 3:
        return True
    else:
        return False
    
def get_lanes(state):
    lanes = np.zeros((8,3))
    lanes[:3] = state
    lanes[3:6] = state.transpose()
    lanes[6] = state.diagonal()
    lanes[7] = np.diagonal(np.fliplr(state))
    return lanes

def play(model, state):
    max_value = -100
    possible_states = None
    for i in range(3):
        for j in range(3):
            print(state[i,j])
            if not state[i,j] == 0: continue
            # print('found one')
            
            state[i,j] = 1
            if possible_states is None:
                possible_states = state.copy().reshape((1,3,3))
            else:
                possible_states = np.concatenate([possible_states, state.reshape((1,3,3))])
            state[i,j] = 0
            print(possible_states)
    print('found {} possible_states'.format(possible_states.shape[0]))
    print(possible_states)
    possible_states = possible_states.reshape((-1,3,3))
    values = model.predict(possible_states)
    print(values)
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
    
    def fit(x, y, lr=0.01):
        print('training model')
        

if __name__ == 'main':
    model = Model()
    state = np.zeros((3,3))
    print(state)
    flip = 1

    while True:
        flip *= 1
        state = flip*play(model, flip*state)
        if game_over(state):
            break