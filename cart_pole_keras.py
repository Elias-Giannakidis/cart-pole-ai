import tensorflow as tf
from tensorflow import keras
import gym
import math
import numpy as np
import random
from collections import deque

env_name = "CartPole-v1"
env = gym.make(env_name)

n_episodes = 1000
n_win_ticks = 300
max_env_steps = None

gamma = 1.0
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
alpha = 0.01
alpha_decay = 0.01

batch_size = 64
#monitor = False   meaby USELESS
quet = False  #### That parameters is only for prints
### or render. If it is True, the program will does't show enything

memory = deque(maxlen = 100000)

SHOW_EVERY = 20

if max_env_steps is not None: env.max_episode_steps = max_env_steps

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

model = Sequential()
model.add(Dense(24, input_dim = 4, activation = 'relu'))
model.add(Dense(48, activation = 'relu'))
model.add(Dense(2, activation = 'relu'))
model.compile(loss = 'mse', optimizer = Adam(lr = alpha, decay = alpha_decay))


def remember(state, action, reward, next_state, done):
    #### Save the models???
    memory.append((state, action, reward, next_state, done))


def choose_action(state, epsilon):
    ### choose an action according to the output.
    ### if predict is low, make random choice else choose the predict
    return env.action_space.sample() if np.random.random() <= epsilon else np.argmax(model.predict(state))


def get_epsilon(t):
    ### calculate the next epsilon according an equation
    return max(epsilon_min, min(epsilon, 1.0 - math.log10((t + 1) * epsilon_decay)))


def preprocess_state(state):
    ### risize the state to fit with neural network
    return np.reshape(state, [1, 4])


def replay(batch_size, epsilon):
    ##### I Don't know????????
    x_batch, y_batch = [], []
    ### From the memory take a amount of batch size for proseccing
    minibatch = random.sample(memory, min(len(memory), batch_size))

    for state, action, reward, next_state, done in minibatch:
        y_target = model.predict(state)
        y_target[0][action] = reward if done else reward + gamma * np.max(model.predict(next_state)[0])
        x_batch.append(state[0])
        y_batch.append(y_target[0])

    model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay


def run():
    ### Here run the model!!!!!

    scores = deque(maxlen=100)
    ### scores memory possibl!!!

    for e in range(n_episodes):

        ###start the episode, so reset the enviroment
        state = preprocess_state(env.reset())
        done = False

        i = 0  ### zero the i at the start!!

        while not done:
            # env.render()

            # STEP1 choose an action according, the state and action_equation,
            ### in this step we use the get epsilon equation olso,
            action = choose_action(state, get_epsilon(e))

            # STEP2 we get the new state of real enviroment according to the choosen action
            next_state, reward, done, _ = env.step(action)
            ### Here we just resize the state matrix to fit in the model
            next_state = preprocess_state(next_state)

            # STEP3 save the situation
            remember(state, action, reward, next_state, done)
            ## and restart the loop
            state = next_state
            ## ingrece the i in each loop!
            i += 1

            # We are outside of the loop
        scores.append(i)  ## add the numper of loop "i"
        mean_score = np.mean(scores)  ### get the average!!

        # STEP4 some if???
        if mean_score >= n_win_ticks and e >= 100:
            ### Test if the pole stay straight for long time
            if not quet:
                # env.render()
                print('Run {} episodes. Solved after {} trails'.format(e, e - 100))
            return e - 100

        if e % SHOW_EVERY == 0 and not quet:
            ## Take a image every 20 cycles
            # env.render()
            print('[Episode {}] - Mean survival time over last 100 episodes was {} ticks.'.format(e, mean_score))

        # STEP5 is the replay function
        replay(batch_size, get_epsilon(e))

    if not quet:
        print('Did not solve after {} episodes'.format(e))

    return e
### quet is a boolean parameter to able or disable  the print and render functions.


run()

env.close()
model.save('saved.model')