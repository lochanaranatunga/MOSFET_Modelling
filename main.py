import pandas as pd
import numpy as np
import tensorflow as tf
from gym import Env
from gym.spaces import Discrete, Box
import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from MOS_Env import MOS
from keras.models import load_model

#loading the training model

def my_loss_fn(y, y_pred):
    squared_difference = tf.square(y - y_pred)
    return tf.reduce_mean(squared_difference, axis=-1)  # Note the `axis=-1
model = load_model('Trained_DNN_model', custom_objects={'my_loss_fn':my_loss_fn})

model.summary()

#loading the environment for RL

env = MOS()

episodes = 10
for episode in range(1, episodes+1):
    env.reset()
    done = False
    score = 0 
    
    while not done:
        
        action_vds=env.action_space.sample()
        action_vgs=env.action_space.sample()
        action_W=env.action_space.sample()
        action_vto=env.action_space.sample()
        action_TOX=env.action_space.sample()
        action_L=env.action_space.sample()
        vds, vgs, W, vto, TOX, L, reward, done, info = env.step(action_vds,action_vgs,action_W,action_vto,action_TOX,action_L)
        x=np.array([vds,vgs,W,vto,TOX,L])
        x=np.transpose(x)
        ids_pred = model.predict(x)
        score+=reward
    print('Episode:{} Score:{} Ids{}'.format(episode, score,ids_pred))
    
vds = env.observation_space_vds.shape
vgs = env.observation_space_vgs.shape
W = env.observation_space_W.shape
vto = env.observation_space_vto.shape
TOX = env.observation_space_TOX.shape
L = env.observation_space_L.shape
actions = env.action_space.n

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

def build_model(states, actions):
    model = Sequential()    
    model.add(Dense(24, activation='relu', input_shape=states))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model

Q_model_vds=build_model(vds,actions)
Q_model_vgs=build_model(vgs,actions)
Q_model_W=build_model(W,actions)
Q_model_vto=build_model(vto,actions)
Q_model_TOX=build_model(TOX,actions)
Q_model_L=build_model(L,actions)

def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                  nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn

#del model
dqn = build_agent(Q_model_vds, actions)