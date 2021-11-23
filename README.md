# tic-tac-toe-machine-learning
A full-circle machine learning system for tic tac toe.  

Task: to win tic tac toe  
Performance: win ratio  
Experience: learning model plays against a fixed-weight model  

## Dependencies
reprint
numpy
matplotlib
## Game Representation
- the board is a 3x3 numpy matrix
- `0` denotes an empty space
- `1` marks x's moves
- `-1` marks o's moves

## Model, Loss, and Optimization
- the model is a linear approximation of a **value function** for board states
- four features are extracted from a given board state
	1. number of x's 2 in a row
	2. number of x's 3 in a row
	3. number of o's 2 in a row
	4. number of o's 3 in a row
- all model objects play from x's perspective, so you must flip the incoming and outgoing board states for the model that plays as "o"
- loss function (J) is squared error
- weights and biases are updated by performing gradient descent on J

## To Do
- Make the output overwrite-able, so that you can watch the model's weights and game state change and get overwritten
- implement weight regularization, model tends to converge to ridiculously large weights 
- human_readable() to turn board into x's and o's instead of 1's and -1's
- reset frozen_model's weights on every new game
- encapsulate main loop in a Game class with methods
	- get_trace()
	- play_until_over()
	- print_metrics()
