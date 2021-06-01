import numpy as np
import time
import matplotlib.pyplot as plt

# returns an integer representing game status, nonzero if game is over
def game_over(state):
    # tallies move count for each playable lane
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

# chops 3x3 game board up into eight lanes    
def get_lanes(state):
    lanes = np.zeros((8,3))
    lanes[:3] = state # rows
    lanes[3:6] = state.transpose() # columns
    lanes[6] = state.diagonal() # positive diagonal
    lanes[7] = np.diagonal(np.fliplr(state)) # negative diagonal
    return lanes

# finds possible moves given board state and picks highest value one according to model
def play(model, state):
    possible_states = None
    
    # loop over all nine board spots
    for i in range(3):
        for j in range(3):
            # skip played on spots
            if not state[i,j] == 0: continue
            
            # hypothetically play the empty spot
            state[i,j] = 1
            if possible_states is None:
                possible_states = state.copy().reshape((1,3,3))
            else:
                possible_states = np.concatenate([possible_states, state.reshape((1,3,3))])
            state[i,j] = 0
    
    # shuffle possible moves and then value them
    np.random.shuffle(possible_states)
    possible_states = possible_states.reshape((-1,3,3))
    values = model.predict(possible_states)
    
    # select most optimal move
    return possible_states[values.argmax()]

# linear model of board value
class Model():
    def __init__(self, W=None, b=None):
        if W is None:
            self.W = np.random.normal(size=(4))
        else:
            if not W.shape == (4,): raise Exception('incorrect shape')
            self.W = W.copy()
        if b is None:
            self.b = 0
        else:
            self.b = float(b)
        print(self.W)
        
    def predict(self, possible_states):
        # initialize prediction array
        size = possible_states.shape[0]
        y = np.zeros(size)
        
        # fill with predictions from linear model
        for i in range(size):
            x = self.get_features(possible_states[i])
            y[i] = self.W.dot(x) + self.b
        return y
    
    # model class decides how to perform feature extraction from the board
    def get_features(self, state):
        lanes = get_lanes(state)
        features = np.zeros((4))
        
        for lane in lanes:
            ours = 0
            opps = 0
            
            # count plays in lane
            for v in lane:
                if v == 1: ours += 1
                if v == -1: opps += 1
                
            # we have two in a row, three in a row
            if ours == 2: features[0] += 1
            if ours == 3: features[1] += 1
            
            # they have two in a row, three in a row
            if opps == 2: features[2] += 1
            if opps == 3: features[3] += 1
        return features

    # gradient descent on squared error
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
        
# generates training data given a matrix of chronological game states
def critic(trace, model):
    # flip is used to train on opponents data
    flip = -1
    
    xtrain = np.zeros((trace.shape[0]-1,3,3))
    ytrain = np.zeros((trace.shape[0]-1))
    for i in range(trace.shape[0]-1):
        # makes all states look like x's perspective
        flip *= -1        
        xtrain[i] = flip*trace[i]
        
        value = None
        type = game_over(flip*trace[i+1])
        if type == 0:
            # training value is set to our current prediction of NEXT state
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
    
# this main is rather botched together but it's mostly PoC for me
if __name__ == '__main__':
    # create two models to play against each other
    # model = Model(np.array([2.76e-13,5.66e-03,-5.499e+01,-2.284e-01]), 99.9) # optimal model after 50k games
    model = Model()
    model_frozen = Model()
    
    # metrics
    won = 0.0
    total = 0.0
    ties = 0.0
    history = []
    
    # train until terminating condition
    while True:
        # time.sleep(5)
        total += 1
        state = np.zeros((3,3))
        print(state)
        flip = -1
        
        trace = state.copy().reshape(1,3,3)
        while True:
            # x plays
            state = play(model, state)
            trace = np.concatenate([trace,state.reshape(1,3,3)])
            print(state)
            
            # check game over
            type = game_over(state)
            if type:
                print('game over' +str(type))
                if type == 1:
                    won += 1
                elif type == 3:
                    ties += 1
                break
                
            # o plays
            state = flip*play(model_frozen, flip*state)
            trace = np.concatenate([trace,state.reshape(1,3,3)])
            print(state)
            # time.sleep(25)
            
            # check game over
            type = game_over(state)
            if type:
                print('game over' +str(type))
                if type == 1:
                    won += 1
                elif type == 3:
                    ties += 1
                break
        
        # the game has ended
        print('final state')
        print(state)  
        
        # train model on this game
        xtrain, ytrain = critic(trace, model)
        model.fit(xtrain, ytrain)
        
        # handle metrics
        win_rate = float(won)/total
        print('Total games you have won: {}'.format(won))
        print('Total ties: {}'.format(ties))
        print('Total games you have played: {}'.format(total))
        print('Win rate: {}'.format(win_rate))
        history.append(win_rate)
        
        # terminating condition
        if win_rate > .8 and total > 3000:
            break
        
        # if total%500 == 0:
            # time.sleep(2)
    
    plt.title('Win Rate Per Iteration')
    plt.plot(history)
    plt.show()