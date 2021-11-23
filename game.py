import numpy as np
import time
import matplotlib.pyplot as plt
from reprint import output
import enum

class Moves():
    EMPTY = 0
    X = 1
    O = -1

class BoardStatus():
    INPROGRESS = 0
    XWIN = 1
    OWIN = 2
    TIE = 3

class Board():
    def __init__(self):
        self.reset_board()

    # chops 3x3 game board up into eight lanes    
    def get_lanes(state):
        lanes = np.zeros((8,3))
        lanes[:3] = state # rows
        lanes[3:6] = state.transpose() # columns
        lanes[6] = state.diagonal() # positive diagonal
        lanes[7] = np.diagonal(np.fliplr(state)) # negative diagonal

        return lanes

    def reset_board(self):
        self.state = np.zeros((3,3))
        self.trace = self.state.copy().reshape(1,3,3)
    
    def _get_board_status(self):
        return Board.get_board_status(self.state)

    def get_board_status(state):
        # tallies move count for each playable lane
        max_in_a_row = Board.get_lanes(state).sum(axis=-1).max()
        min_in_a_row = Board.get_lanes(state).sum(axis=-1).min()
    
        if max_in_a_row == 3*Moves.X:
            return BoardStatus.XWIN
        if min_in_a_row == 3*Moves.O:
            return BoardStatus.OWIN
        if not Moves.EMPTY in state:
            return BoardStatus.TIE
        return BoardStatus.INPROGRESS

    # finds possible moves given board state and picks highest value one according to model
    def play_turn(self, model):
        possible_states = None
    
        # loop over all nine board spots
        for i in range(3):
            for j in range(3):

                # skip played on spots
                if not self.state[i,j] == Moves.EMPTY: continue
                
                # hypothetically play the empty spot
                self.state[i,j] = model.player_id
                if possible_states is None:
                    possible_states = self.state.copy().reshape((1,3,3))
                else:
                    possible_states = np.concatenate([possible_states, self.state.reshape((1,3,3))])
                
                # undo the hypothetical move
                self.state[i,j] = Moves.EMPTY
        
        # shuffle possible moves and then value them
        np.random.shuffle(possible_states)
        possible_states = possible_states.reshape((-1,3,3))
        values = model.predict(possible_states)
        
        # select most optimal move and make it the current state
        self.state = possible_states[values.argmax()]

        # append state to trace
        self.trace = np.concatenate([self.trace,self.state.reshape(1,3,3)])

        # return the status
        return self._get_board_status()
    
    # outputs array of human characters from sate
    def human_readable(self):
        arr = [
            [' ',' ',' '],
            [' ',' ',' '],
            [' ',' ',' '],
        ]
        for i in range(3):
            for j in range(3):
                if self.state[i,j] == Moves.X: arr[i][j] = 'X'
                if self.state[i,j] == Moves.O: arr[i][j] = 'O'
                if self.state[i,j] == Moves.EMPTY: arr[i][j] = '-'
        return arr


# linear model of board value
class Model():
    def __init__(self, player_id, W=None, b=None):
        
        # used to denote X or O
        self.player_id = player_id
        
        # model weights
        if W is None:
            self.W = np.zeros((4))
        else:
            if not W.shape == (4,): raise Exception('incorrect shape')
            self.W = W.copy()
        
        # model bias
        if b is None:
            self.b = 0
        else:
            self.b = float(b)
        
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
        lanes = Board.get_lanes(state)
        features = np.zeros((4))
        
        for lane in lanes:
            ours = 0
            opps = 0
            
            # count plays in lane
            for v in lane:
                if v == self.player_id:
                    ours += 1
                elif not v == Moves.EMPTY:
                    opps += 1
                
            # we have two in a lane, three in a row
            if ours == 2: features[0] += 1
            if ours == 3: features[1] += 1
            
            # they have two in a lane, three in a row
            if opps == 2: features[2] += 1
            if opps == 3: features[3] += 1
        
        return features

    # gradient descent on squared error
    def fit(self, train_states, ytrain, lr=0.01):
        # print('training model')
        for i in range(train_states.shape[0]):
            x = self.get_features(train_states[i])
            diff = ytrain[i] - (self.W.dot(x) + self.b)
            self.b += lr * diff
            for j in range(self.W.shape[0]):
                self.W[j] += lr * diff * x[j]
            # print(self.W)
            # print(self.b)
        # print('done training')

    # generates training data given a matrix of chronological game states
    def generate_dataset(self, trace):
        # discard the initial state of all 0s
        trace = trace[1:]

        # ascertain who played first
        first_player = None
        start_idx = None
        if Moves.X in trace[0]:
            first_player = Moves.X
            start_idx = 0
        elif Moves.O in trace[0]:
            first_player = Moves.O
            start_idx = 1
        else:
            raise('Couldn\'t find first player')

        # will later hold the training dataset
        xtrain = None
        ytrain_list = []

        # loop through the game trace
        for i in range(start_idx, trace.shape[0], 2):
            
            value = None
            board_status = None
            if i == trace.shape[0]-1:
                board_status = Board.get_board_status(trace[i])
            else:
                board_status = Board.get_board_status(trace[i+1])
            if board_status == BoardStatus.INPROGRESS:
                # training value is set to our current prediction of NEXT state
                value = self.predict(trace[i+1].reshape(1,3,3))[0]
            elif board_status == BoardStatus.XWIN:
                # you won
                value = 100
            elif board_status == BoardStatus.OWIN:
                # you lost fool
                value = -100
            elif board_status == BoardStatus.TIE:
                # you tied
                value = -10
            
            # store the values
            if xtrain is None:
                xtrain = trace[i].copy().reshape(1,3,3)
            else:
                xtrain = np.concatenate([xtrain, trace[i].reshape(1,3,3)])
            ytrain_list.append(value)

        return xtrain, np.array(ytrain_list)
    
# this main is rather botched together but it's mostly PoC for me
if __name__ == '__main__':
    # USER PARAMS - TERMINATING CONDITIONS both must be met
    win_rate_stop = 0.8
    game_num_stop = 3000
    slow_mode = False

    # create two models to play against each other
    # model = Model(np.array([2.76e-13,5.66e-03,-5.499e+01,-2.284e-01]), 99.9) # optimal model after 50k games
    model = Model(player_id=Moves.X)
    model_frozen = Model(player_id=Moves.O)
    board = Board()

    # TODO Model.generate_datset() and some other functions break if the LEARNING model is not set to 'X'! Fix this later.
    
    # training metrics
    learner_won = 0.0
    learner_tied = 0.0
    total_games = 0.0
    history = []
    
    with output(output_type='list', initial_len=10) as output_lines:
        # play and train until user-defined terminating condition
        while True:

            # new game, tend to metrics and reset board
            total_games += 1
            board.reset_board()
            output_lines[9] = 'Program status: PLAYING'

            # output models' states
            output_lines[0] = 'Model X params:\t\t{:.2f} {:.2f} {:.2f} {:.2f} {:.2f} \tLEARNER'.format(model.W[0],model.W[1],model.W[2],model.W[3],model.b)
            output_lines[1] = 'Model O params:\t\t{:.2f} {:.2f} {:.2f} {:.2f} {:.2f} \tFROZEN'.format(model_frozen.W[0],model_frozen.W[1],model_frozen.W[2],model_frozen.W[3],model_frozen.b)
            
            # play one game
            board_status = BoardStatus.INPROGRESS
            turn_num = np.random.randint(0,2) # x or o can start, allows all possible training cases
            while board_status == BoardStatus.INPROGRESS:
                # models take turns playing
                if turn_num % 2 == 0:
                    board_status = board.play_turn(model)
                else:
                    board_status = board.play_turn(model_frozen)

                # get human readable 3x3 game array
                human_state = board.human_readable()
                
                # update output
                output_lines[2] = '{} {} {}'.format(human_state[0][0],human_state[0][1],human_state[0][2])
                output_lines[3] = '{} {} {}'.format(human_state[1][0],human_state[1][1],human_state[1][2])
                output_lines[4] = '{} {} {}'.format(human_state[2][0],human_state[2][1],human_state[2][2])

                # inc the counter
                turn_num += 1

                # if slow mode is on, slight delay
                if slow_mode:
                    time.sleep(0.5)

            # the game has ended, update metrics and output accordingly
            if board_status == BoardStatus.XWIN:
                learner_won += 1
            elif board_status == BoardStatus.TIE:
                learner_tied += 1
            output_lines[5] = 'LEARNER Wins:\t{}'.format(learner_won)
            output_lines[6] = 'LEARNER Losses:\t{}'.format(total_games-learner_won-learner_tied)
            output_lines[7] = 'LEARNER Ties:\t{}'.format(learner_tied)
            output_lines[8] = 'Win Rate:\t{:.2f}'.format(float(learner_won)/total_games)
            output_lines[9] = 'Program status: TRAINING'


            # train model on this game
            xtrain, ytrain = model.generate_dataset(board.trace)
            model.fit(xtrain, ytrain)
            
            # handle metrics
            win_rate = float(learner_won)/total_games

            history.append(win_rate)
            
            # terminating condition
            if win_rate > win_rate_stop and total_games > game_num_stop:
                break

            # if slow mode is on, slight delay
            if slow_mode:
                time.sleep(0.5)
        
        plt.title('Win Rate Per Iteration')
        plt.plot(history)
        plt.show()