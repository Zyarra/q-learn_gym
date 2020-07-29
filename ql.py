# Import required libraries
import gym
import numpy as np
import matplotlib.pyplot as plt

'''The OpenAI GYM environment's MountainCar's goal is to reach the flag with the cartpole
For each action we receice an observation from the environment(New Position, Reward, Finished)'''
env = gym.make('MountainCar-v0')


EPISODES = 50000
'''One Episode is one sequence of: states, actions and rewards, which ends with terminal state
For example, playing an entire game can be considered as one episode, the terminal state being reached when one player loses/wins/draws'''

LEARNING_RATE = 0.6
START_LR_DECAY = 1
END_LR_DECAY = EPISODES*0.8
LR_DECAY = LEARNING_RATE / (END_LR_DECAY - START_LR_DECAY)
'''The learning rate(or step size) determines to what extent newly acquired information overrides old information
A factor of 0 makes the agent learn nothing (exclusively exploiting prior knowledge),
A factor of 1 makes the agent consider only the most recent information (ignoring prior knowledge to explore possibilities)
In fully deterministic environments, a learning rate of 1 is optimal.
'''

DISCOUNT = 0.9
'''The discount factor determines the importance of future rewards.
A factor of 0 will make the agent "myopic" (or short-sighted) by only considering current rewards,
While a factor approaching 1 will make it strive for a long-term high reward.

The strength with which we encourage a sampled action is the weighted sum of all rewards afterwards,
but later rewards are usually exponentially less important.'''

RENDER_EVERY = 5000
# As rendering takes very long time its beneficial to only display every X round.
STAT_EVERY = 100

# print(env.observation_space.high)
# print(env.observation_space.low)
'''The observation space is continuous, converting it to the same shape array with lower size.
for faster training. DISCRETE_OS_SIZE is just the number of groups'''
DISCRETE_OS_SIZE = [100] * len(env.observation_space.high)  # This creates a list [40, 40], the number is arbitrary
DISCRETE_OS_WIN_SIZE = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE
'''This is how much to increment the range by for each bucket
Now let's create a function that converts the states to discrete'''


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / DISCRETE_OS_WIN_SIZE
    # We use this tuple to look up the Q values per actions
    return tuple(discrete_state.astype(np.int))


# Create Q table with random uniform distribution, with values between -2 and 0(we have 3 actions)
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

'''Exploration vs exploitation:
Lower epsilon is lower chance to explore new things and higher chance to follow the route that is working better
Its decaying because we want lower randomness towards the end.
As the agent learns it moves from exploration to exploitation.
'''
epsilon = 0.6
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES*0.5
EPSILON_DECAY_VALUE = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# To keep track of the rewards
ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': [], 'acts': []}

# The training loop:
for episode in range(EPISODES):
    episode_rewards = 0
    # Render every Xth episode
    if episode % RENDER_EVERY == 0:
        render = False
    else:
        render = False
    discrete_state = get_discrete_state(env.reset())
    done = False
    while not done:
        # Use Q table or explore:
        if np.random.random() > epsilon:
            # Get the ACTION index with the highest Q value:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)  # pick random move due Epsylon
        # Get state, rewards and 'win state'
        state, reward, done, _ = env.step(action)
        episode_rewards += reward
        new_discrete_state = get_discrete_state(state)
        if render:
            env.render()
        # If the 'win' is not achieved, update the Q value:
        if not done:
            # Max possible Q value for current state
            max_future_q = np.max(q_table[new_discrete_state])
            # Current Q value(current state and performed action)
            current_q = q_table[discrete_state + (action,)]
            # Formula for new Q value
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            # Update the table with the new Q value
            q_table[discrete_state + (action,)] = new_q
        # If the simulation ends update Q value with 0(highest value)
        elif state[0] >= env.goal_position:
            # np.save(f'qtable_E{episode}.npy', q_table)
            print('\r', end='')
            print(f'Episode: {episode}, avg: {avg_reward}, min: {min(ep_rewards[-RENDER_EVERY:])}, max: {max(ep_rewards[-RENDER_EVERY:])}, epsilon: {epsilon:>1.2f}, LR: {LEARNING_RATE:>1.2f}, === > REACHED OBJECTIVE', flush = True, end='')

            q_table[discrete_state + (action,)] = 0
        # Reset state
        discrete_state = new_discrete_state
    ep_rewards.append(episode_rewards)

    # Decay epsilon/LR
    if END_EPSILON_DECAYING*0.85 >= episode >= START_EPSILON_DECAYING:
        epsilon -= EPSILON_DECAY_VALUE
    if END_LR_DECAY*0.85 >= episode >= START_LR_DECAY:
        LEARNING_RATE -= LR_DECAY

    # Append Episode, average min and max rewards to the dict every xth episode:
    if not episode % STAT_EVERY:
        avg_reward = sum(ep_rewards[-STAT_EVERY:]) / STAT_EVERY
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(avg_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-STAT_EVERY:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-STAT_EVERY:]))

        print('\r', end='')
        print(
            f'Episode: {episode}, avg: {avg_reward}, min: {min(ep_rewards[-RENDER_EVERY:])}, max: {max(ep_rewards[-RENDER_EVERY:])}, epsilon: {epsilon:>1.2f}, LR: {LEARNING_RATE:>1.2f}',
            flush=True, end='')

env.close()
# Plot average, min and max rewards per episode
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label='avg')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label='min')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label='max')

plt.legend(loc=4)
plt.grid(True)
plt.show()
