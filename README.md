This Python project is an implementation of reinforcement learning, using Q-learning method with Numpy and OpenAI Gym environment.
Prerequisites:
Matplotlib(for visualisations)
Numpy(for all the computation)
Gym(the environment)

Introduction:
In this implementation the goal is to get to know Reinforcement Learning and Q-learning basics.

Goal:
MountainCar-V0
A car is on a one-dimensional track, positioned between two "mountains". The goal is to drive up the mountain on the right; however, the car's engine is not strong enough to scale the mountain in a single pass. Therefore, the only way to succeed is to drive back and forth to build up momentum.
The reward is 0 if we reach the goal, -1 if we don't.
mountaincarpic


Method:
We are using Q-Table.
Q-Table is a simple lookup table where we calculate the maximum expected future rewards for action at each state. This will guide us to the best action at each state.
In the Q-Table, the columns are the actions and the rows are the states. We update Q with the best Q value after each iteration.

We use epsilon to set some Exploration/Exploitation rate, this helps the model to try new things instaed of always doing the same once it found ONE way to get rewarded.
For now, our epsilon is decaying with every iteration.

Adjustable hyperparameters:
LEARNING_RATE:
The learning rate(or step size) determines to what extent newly acquired information overrides old information
DISCOUNT:
The discount factor determines the importance of future rewards.
EPISODES:
One Episode is one sequence of: states, actions and rewards
DISCRETE_OS_SIZE:
We reduced the states to discrete size to reduce time to train. This is the binsize for each state.

epsilon:
Exploration vs Exploitation

RENDER_EVERY:
How often to render the environment
STAT_EVERY:
How often capture stats

For more details or to try, see the .py file.

Results:

