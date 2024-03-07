import tensorflow as tf
import numpy as np
from gym import Env
from gym.spaces import Discrete, Box
from keras.models import load_model

vdsmax=30
ids_target= 1e-7
def my_loss_fn(y, y_pred):
    squared_difference = tf.square(y - y_pred)
    return tf.reduce_mean(squared_difference, axis=-1)  # Note the `axis=-1
model = load_model('Trained_DNN_model', custom_objects={'my_loss_fn':my_loss_fn})

class MOS:
    def __init__(self):
        
        # Actions we can take, increase, hold, decrease
        self.action_space = Discrete(3)
        
        # Observation ranges
        self.observation_space_vds = Box(low=np.array([0]), high=np.array([vdsmax]))
        self.observation_space_vgs = Box(low=np.array([0]), high=np.array([10]))
        self.observation_space_vto = Box(low=np.array([0]), high=np.array([5]))
        self.observation_space_W = Box(low=np.array([0.05]), high=np.array([0.4]))
        self.observation_space_L = Box(low=np.array([5e-7]), high=np.array([3e-6]))
        self.observation_space_TOX = Box(low=np.array([5e-7]), high=np.array([3e-6]))
        
        # Initail state values
        self.vgs = (((self.observation_space_vgs.high)+(self.observation_space_vgs.low))/2)
        self.vto = (((self.observation_space_vto.high)+(self.observation_space_vto.low))/2)
        if self.vgs>self.vto:
            self.vgs=self.vgs
            self.vto=self.vto
        else:
            self.vgs=self.observation_space_vgs.high
        self.vds = (((((self.vgs)-(self.vto)))+(self.observation_space_vds.high))/2)
        self.W= (((self.observation_space_W.high)+(self.observation_space_W.low))/2)
        self.L=(((self.observation_space_L.high)+(self.observation_space_L.low))/2)
        self.TOX=(((self.observation_space_TOX.high)+(self.observation_space_TOX.low))/2)
        self.ids_targer= ids_target
        #current calculation
        x=np.array([self.vds,self.vgs,self.W,self.vto,self.TOX,self.L])
        x=np.transpose(x)
        self.ids_pred = model.predict(x)

        # iterations
        self.iter = 100
        
    def step(self, action_vds,action_vgs,action_W,action_vto,action_TOX,action_L):
        # Apply action
        # 0 -1 = -1 
        # 1 -1 = 0 
        # 2 -1 = 1 
        self.vds += action_vds -1 
        self.vgs += action_vgs -1 
        self.W += action_W -1 
        self.vto += action_vto -1 
        self.TOX += action_TOX -1 
        self.L += action_L -1 
        
        # Reduce iteration by 1
        self.iter -= 1 
        
        # Calculate reward
        self.error_percent= abs( ((self.ids_pred - self.ids_targer)/self.ids_targer))
        if self.error_percent >=0 and self.error_percent <=0.1: 
            reward=5
        elif self.error_percent >=0.1 and self.error_percent <=0.2:
            reward= 4
        elif self.error_percent >=0.2 and self.error_percent <=0.3:
            reward= 3
        elif self.error_percent >=0.3 and self.error_percent <=0.4:
            reward= 2
        elif self.error_percent >=0.4 and self.error_percent <=0.5:
            reward =1
        elif self.error_percent >=0.5 and self.error_percent <=0.6:
            reward =-1
        elif self.error_percent >=0.6 and self.error_percent <=0.7:
            reward= -2
        elif self.error_percent >=0.7 and self.error_percent <=0.8:
            reward= -3
        elif self.error_percent >=0.8 and self.error_percent <=0.9:
            reward =-4
        elif self.error_percent >=0.9 and self.error_percent <=0.1:
            reward= -5
        else: 
            reward = 0 
        
        # Check if iter is done
        if self.iter <= 0: 
            done = True
        else:
            done = False
        

        info = {}
        
        # Return step information
        return self.vds, self.vgs, self.W, self.vto, self.TOX, self.L, reward, done, info

    def reset(self):
        self.observation_space_vds = Box(low=np.array([0]), high=np.array([vdsmax]))
        self.observation_space_vgs = Box(low=np.array([0]), high=np.array([10]))
        self.observation_space_vto = Box(low=np.array([0]), high=np.array([5]))
        self.observation_space_W = Box(low=np.array([0.05]), high=np.array([0.4]))
        self.observation_space_L = Box(low=np.array([5e-7]), high=np.array([3e-6]))
        self.observation_space_TOX = Box(low=np.array([5e-7]), high=np.array([3e-6]))
        
        # Initail state values
        self.vgs = (((self.observation_space_vgs.high)+(self.observation_space_vgs.low))/2)
        self.vto = (((self.observation_space_vto.high)+(self.observation_space_vto.low))/2)
        if self.vgs>self.vto:
            self.vgs=self.vgs
            self.vto=self.vto
        else:
            self.vgs=self.observation_space_vgs.high
        self.vds = (((((self.vgs)-(self.vto)))+(self.observation_space_vds.high))/2)
        self.W= (((self.observation_space_W.high)+(self.observation_space_W.low))/2)
        self.L=(((self.observation_space_L.high)+(self.observation_space_L.low))/2)
        self.TOX=(((self.observation_space_TOX.high)+(self.observation_space_TOX.low))/2)
        self.ids_targer= ids_target
        
        return self.vds, self.vgs, self.W, self.vto, self.TOX, self.L
