import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time

style.use('ggplot')

EPISODES = 50000
GRID_SIZE = 20
MOVE_PENALTY = 1
ENEMY_PENALTY = 200
ITEM_REWARD = 50

epsilon = 0.6
EPS_DECAY = 0.99
RENDER_EVERY = 3000
# To use Q table from earlier trainings file
start_q_table = None

LEARNING_RATE = 0.7
DISCOUNT = 0.95

PLAYER_N = 1
ITEM_N = 2
ENEMY_N = 3
colors = {1: (255, 175, 0),
          2: (0, 255, 0),
          3: (0, 0, 255)}


class Blob:
    def __init__(self):
        self.x = np.random.randint(0, GRID_SIZE)
        self.y = np.random.randint(0, GRID_SIZE)

    def __str__(self):
        return f'{self.x, self.y}'

    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)

    def action(self, choice):
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)
        elif choice == 4:
            self.move(x=1, y=0)
        elif choice == 5:
            self.move(x=-1, y=0)
        elif choice == 6:
            self.move(x=0, y=1)
        elif choice == 7:
            self.move(x=0, y=-1)

    def move(self, x=False, y=False):
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x
        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        if self.x < 0:
            self.x = 0
        elif self.x > GRID_SIZE - 1:
            self.x = GRID_SIZE - 1

        if self.y < 0:
            self.y = 0
        elif self.y > GRID_SIZE - 1:
            self.y = GRID_SIZE - 1


if start_q_table is None:
    q_table = {}
    for x1 in range(-GRID_SIZE + 1, GRID_SIZE):
        for y1 in range(-GRID_SIZE + 1, GRID_SIZE):
            for x2 in range(-GRID_SIZE + 1, GRID_SIZE):
                for y2 in range(-GRID_SIZE + 1, GRID_SIZE):
                    q_table[((x1, y1), (x2, y2))] = [np.random.uniform(-10, -5) for i in range(8)]

else:
    with open(start_q_table, 'rb') as f:
        q_table = pickle.load(f)

episode_rewards = []

for episode in range(EPISODES):

    player = Blob()
    item = Blob()
    enemy = Blob()
    if episode % RENDER_EVERY == 0:
        print(f'on #{episode}, epsilon {epsilon}')
        if len(episode_rewards) > 1:
            print(f'{RENDER_EVERY} ep mean {np.mean(episode_rewards[-RENDER_EVERY:])}')
        show = True
    else:
        show = False

    episode_reward = 0
    for i in range(200):
        obs = (player - item, player - enemy)
        if np.random.random() > epsilon:
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0, 8)
        player.action(action)
        # enemy.move()
        # item.move()

        if player.x == enemy.x and player.y == enemy.y:
            reward = -ENEMY_PENALTY
        elif player.x == item.x and player.y == item.y:
            reward = ITEM_REWARD
        else:
            reward = -MOVE_PENALTY

        new_obs = (player - item, player - enemy)
        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][action]

        if reward == ITEM_REWARD:
            new_q = ITEM_REWARD
        # elif reward == ENEMY_PENALTY:
        #     new_q = ENEMY_PENALTY
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        q_table[obs][action] = new_q

        if show:
            env = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=np.uint8)
            env[item.x][item.y] = colors[ITEM_N]
            env[player.x][player.y] = colors[PLAYER_N]
            env[enemy.x][enemy.y] = colors[ENEMY_N]
            img = Image.fromarray(env, 'RGB')
            img = img.resize((300, 300))
            cv2.imshow("image", np.array(img))
            if reward == ITEM_REWARD or reward == -ENEMY_PENALTY:
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        episode_reward += reward
        if reward == ITEM_REWARD or reward == -ENEMY_PENALTY:
            break
        episode_rewards.append(episode_reward)
        epsilon *= EPS_DECAY

moving_avg = np.convolve(episode_rewards, np.ones((RENDER_EVERY,)) / RENDER_EVERY, mode='valid')

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f'Reward {RENDER_EVERY}')
plt.xlabel(f'Episode #')
plt.show()

with open(f"q-table_{int(time.time())}.pickle", 'wb') as f:
    pickle.dump(q_table, f)
