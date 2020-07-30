# Q-Learning implementation

#### This Python project is an implementation of reinforcement learning, using Q-learning method with Numpy and OpenAI Gym environment.
**Prerequisites:**
1. **Matplotlib**(for visualisations)
2. **Numpy**(for all the computation)
3. **Gym**(the environment)

The versions I'm using are available in the **requirements.txt**, but in theory most versions should suffice.

**Introduction:**
In this implementation the goal is to get to know Reinforcement Learning and Q-learning basics.

**Goal:**
*MountainCar-V0*
A car is on a one-dimensional track, positioned between two "mountains". The goal is to drive up the mountain on the right; however, the car's engine is not strong enough to scale the mountain in a single pass. Therefore, the only way to succeed is to drive back and forth to build up momentum.
The **reward is 0 if we reach the goal, -1 if we don't.**

![Mountaincar](/images/mountaincar.jpg)


**Method:**
We are using *Q-Table*.
Q-Table is a simple lookup table where we calculate the maximum expected future rewards for action at each state. This will guide us to the best action at each state.
In the Q-Table, the *columns are the actions and the rows are the states*. We update Q with the best Q value after each iteration.
![Qtable](/images/q-table.gif)


Adjustable hyperparameters:
**LEARNING_RATE:**
The learning rate(or step size) determines to what extent newly acquired information overrides old information
**DISCOUNT:**
The discount factor determines the importance of future rewards.
**EPISODES:**
One Episode is one sequence of: states, actions and rewards
**DISCRETE_OS_SIZE:**
We reduced the states to discrete size to reduce time to train. This is the binsize for each state.

**epsilon:**
to set some Exploration/Exploitation rate, this helps the model to try new things instaed of always doing the same once it found ONE way to get rewarded.
Our epsilon is decaying with every iteration.

**RENDER_EVERY:**
How often to render the environment, rendering takes much more time than the actual training and its unnecessary to render every single run.
**STAT_EVERY:**
How often capture stats.



## The end result in a single picture:
![Figure](/images/Figure_1.png)

