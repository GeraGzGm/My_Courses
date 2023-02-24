import matplotlib.pyplot as plt
from alive_progress import alive_bar
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from collections import deque
import numpy as np
import random
import time
import cv2


plt.ion()


class DQN():
    Steps = 5_000
    Plays = 1_000
    Epsilon = 1
    Min_Epsilon = 0.05
    Epsilon_decay = 0.9999
    Gamma = 0.9
    Memory_size = 5_000
    Min_observations = 2_500
    Num_observations = 0
    Target_Net_update = 1_500
    Minibatch = 32

    def __init__(self, Load_model=False):
        self.env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
        self.env = FrameStack(self.env, num_stack=4)

        self.observation_shape = (4, 84, 84)
        self.action_space = self.env.action_space.n

        self.Q_network = self.__dqn_model()

        if Load_model:
            self.Q_network.load_weights('./weights/model_weight')

        self.Target_network = self.__dqn_model()
        self.__set_Target_network()

        self.memory = deque(maxlen=self.Memory_size)

    def __dqn_model(self, verbose=False):
        model = Sequential([
            Conv2D(filters=32, kernel_size=(8, 8), strides=(4, 4),
                   activation='relu', input_shape=self.observation_shape+(1,)),
            Conv2D(filters=64, kernel_size=(8, 8),
                   strides=(2, 2), activation='relu'),
            Conv2D(filters=64, kernel_size=(4, 4), activation='relu'),
            Conv2D(filters=64, kernel_size=(4, 4), activation='relu'),
            Flatten(),
            Dense(units=256, activation='relu'),
            Dense(units=self.action_space, activation='linear'),
        ])

        model.compile(optimizer="adam", loss='mse', metrics=['accuracy'])

        if verbose:
            print(model.summary())
        return model

    def __set_Target_network(self):
        self.Target_network.set_weights(self.Q_network.get_weights())

    def image_processing(self, state: np.array) -> np.array:
        new_state = np.zeros(self.observation_shape)

        for i in range(4):
            img = state[i]
            cropped_img = img[30:195, :, :]
            bgr2gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)/255
            resized_img = cv2.resize(bgr2gray_img, (84, 84))

            new_state[i, :, :] = resized_img
        return np.array([new_state])

    def select_action(self, state) -> int:
        """
        With probability \epsilon select a random action

        Returns
        -------
        int: Action.
        -------
        """

        if random.random() < self.Epsilon or len(self.memory) > self.Min_observations:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q_network.predict(state, verbose=0))

    def Experience_replay(self):
        states = []
        targets = []

        if len(self.memory) > self.Min_observations:
            minibatch = random.sample(self.memory, self.Minibatch)
            
            for state, action, reward, new_state, done in minibatch:
                target = self.Q_network.predict(state, verbose = 0)
                
                if done:
                    target[0][action] = reward
                else:
                    t = self.Target_network.predict(new_state, verbose = 0)
                    target[0][action] = reward + self.Gamma * np.amax(t)
            
                targets.append(target)
                states.append(state)
            
            states = np.array(states).reshape((self.Minibatch, 4, 84, 84, 1))
            targets = np.array(targets)
            
            self.Q_network.fit(states, targets,
                               batch_size=(self.Minibatch),epochs=1, verbose=0)
            
            del targets,states

    def training(self):
        rewards = []

        fig, ax = plt.subplots(1)
       
        
        
        with alive_bar(total=self.Plays) as bar:
            for episode in range(self.Plays):

                observation, _ = self.env.reset()
                observation = self.image_processing(observation)

                score = 0

                for step in range(self.Steps):
                    action = self.select_action(observation)

                    new_observation, reward, done, _, _ = self.env.step(action)
                    new_observation = self.image_processing(new_observation)

                    self.memory.append(
                        [observation, action, reward, new_observation, done])

                    self.Experience_replay()

                    self.Epsilon = max(
                        self.Epsilon * self.Epsilon_decay, self.Min_Epsilon)

                    observation = new_observation
                    score += reward
                    self.Num_observations += 1
                    if self.Num_observations % self.Target_Net_update == 0:
                        self.Q_network.save_weights('./weights/model_weight')
                        self.__set_Target_network()
                        bar.text = f"SAVED!  Step: {step}, Score: {score}, Epsilon: {self.Epsilon}"
                        self.Num_observations = 0
                    else:
                        bar.text = f"Step: {step}, Score: {score}, Epsilon: {self.Epsilon}, Total Steps: {self.Num_observations}"

                    if done:
                        break
                bar()
                rewards.append(score)
                
                ax.bar(np.arange(0, episode+1, 1),rewards)
                ax.set_title("Score: {}, Epsilon: {}".format(score,self.Epsilon))
                plt.pause(0.01)
                #print("score: {}".format(score))
                plt.show()
        
        self.env.close()

    def Visualize_policy(self):

        fig, axis = plt.subplots(1)
        plt.show()

        self.Q_network.load_weights('./weights/model_weight')

        observation, _ = self.env.reset()
        observation = self.image_processing(observation)

        while True:
            axis.imshow(self.env.render())
            plt.pause(0.1)

            # np.argmax(self.Q_network.predict(observation, verbose = 0))
            action = self.env.action_space.sample()
            new_observation, reward, done, _, _ = self.env.step(action)
            new_observation = self.image_processing(new_observation)

            print(done)

            if done:
                break


a = DQN(Load_model=False)
a.training()
# a.Visualize_policy()
