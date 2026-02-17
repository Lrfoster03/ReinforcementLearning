import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import importlib.util
import random
import gymnasium as gym
import numpy as np
from collections import deque
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.optimizers import RMSprop

MODEL_PATH = "models/cartpole-dqn.keras"
LEGACY_MODEL_PATH = "models/cartpole-dqn.h5"

def compile_dqn_model(model):
    model.compile(
        loss="mean_squared_error",
        optimizer=RMSprop(learning_rate=0.00025, rho=0.95, epsilon=0.01),
        metrics=['accuracy']
    )

def OurModel(input_shape, action_space):
    X_input = Input(shape=input_shape)

    # 'Dense' is the basic form of a neural network layer
    # Input layer with a state size(4), and Hidden layer with 1024 nodes. 
    X = Dense(1024, activation="relu", kernel_initializer='he_uniform')(X_input)

    # Hidden layer with 512 nodes
    X = Dense(512, activation="relu", kernel_initializer='he_uniform')(X)

    # Hidden layer with 256 nodes
    X = Dense(256, activation="relu", kernel_initializer='he_uniform')(X)

    # Hidden layer with 64 nodes
    X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)

    # Output layer with # of actions: 2 nodes (left, or right)
    X = Dense(int(action_space), activation="linear", kernel_initializer='he_uniform')(X)

    model = Model(inputs=X_input, outputs=X, name='Cartpole_DQN_model')
    compile_dqn_model(model)

    model.summary()
    return model

class DQNAgent:
    def __init__(self):
        self.render_enabled = importlib.util.find_spec("pygame") is not None
        render_mode = "human" if self.render_enabled else None
        self.env = gym.make('CartPole-v1', render_mode=render_mode)
        if not self.render_enabled:
            print("Rendering disabled: install pygame to enable CartPole window output.")
        # by default, CartPole-v1 has max episode steps = 500
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.EPISODES = 1000
        self.memory = deque(maxlen=2000)

        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0 # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999
        self.batch_size = 64
        self.train_start = 1000

        # create main model
        self.model = OurModel(input_shape=(self.state_size,), action_space = self.action_size)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model.predict(state))
        
    def replay(self):
        if len(self.memory) < self.train_start:
            return
        # Randomly sample a minibatch from the memory
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))

        state = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros((self.batch_size, self.state_size))
        action, reward, done = [], [], []

        # do this before prediction
        # for speedup, this could be done on the tensor level
        # but easier to understand using a loop

        for i in range(self.batch_size):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        # do batch prediction to save speed
        target = self.model.predict(state)
        target_next = self.model.predict(next_state)

        for i in range(self.batch_size):
            # correction on the Q value for the action used
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # Standard - DQN
                # DQN chooses the max q value among next actions
                # selection and evaluation of action is on the target Q Network
                # Q_max = max_a' Q_target(s', a')
                target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))

        # Train the Neural Network with batches
        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)
    
    def load(self, name):
        # Avoid legacy training-config deserialization from old .h5 files.
        self.model = load_model(name, compile=False)
        compile_dqn_model(self.model)

    def save(self, name):
        self.model.save(name)
    
    def run(self):
        try:
            for e in range(self.EPISODES):
                state, _ = self.env.reset()
                state = np.reshape(state, [1, self.state_size])
                done = False
                i = 0
                while not done:
                    if self.render_enabled:
                        self.env.render()
                    action = self.act(state)
                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated
                    next_state = np.reshape(next_state, [1, self.state_size])
                    if not done or i == self.env.spec.max_episode_steps - 1:
                        reward = reward
                    else: 
                        reward = -100
                    self.remember(state, action, reward, next_state, done)
                    state = next_state
                    i += 1
                    if done:
                        print("episode: {}/{}, score: {}, e: {:.2}".format(e, self.EPISODES, i, self.epsilon))
                        if i == 500:
                            print(f"Saving trained model as {MODEL_PATH}")
                            self.save(MODEL_PATH)
                            return
                    self.replay()
        finally:
            self.env.close()

    def test(self):
        model_path = MODEL_PATH if os.path.exists(MODEL_PATH) else LEGACY_MODEL_PATH
        self.load(model_path)
        try:
            for e in range(self.EPISODES):
                state, _ = self.env.reset()
                state = np.reshape(state, [1, self.state_size])
                done = False
                i = 0
                while not done:
                    if self.render_enabled:
                        self.env.render()
                    action = np.argmax(self.model.predict(state))
                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated
                    state = np.reshape(next_state, [1, self.state_size])
                    i += 1
                    if done:
                        print("episode: {}/{}, score: {}".format(e, self.EPISODES, i))
                        break
        finally:
            self.env.close()

if __name__ == "__main__":
    agent = DQNAgent()
    # agent.run()
    agent.test()
