# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 14:02:59 2018

@author: Robert
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


class LakeLoadEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.pc = 2.4   #half-saturation constant for P
        self.r = 0.34   #recycling contant
        self.s = 3.3    #transfer to mud constant
        self.h = 0.19   #flushing constant
        self.b = 0.022  #removal-from-mud constant
        self.q = 8      #shape constant for recycling
        self.var = 0.01 #noise
        #constants -- see table 1
        self.alpha = 1  #benefit per unit loading
        self.beta1 = 0  #loss of amenity
        self.beta2 = 0.065  #loss of amenity
        #see eq 13
        
        self.pThresh = 7
        self.mThresh = 200
        #the max these values can achieve in our model
        
        
        self.action_space = spaces.Discrete(12)
        # 12 actions - action 0 is do not add P, action 11 is add 12 units of P
        
        l = np.array([0,0])
        h = np.array([self.pThresh, self.mThresh])
        self.observation_space = spaces.Box(l, h )
        #observation space is (P,M)
        
        self.seed()
        self.viewer = None
        self.state = None
        
    def e(self):
        return math.exp(-self.s-self.h)
    
    
    def g(self):
        return (self.s+self.h-1+ self.e())/(self.s+self.h)*self.s/(self.s+self.h)
    
    def f(self,P):
        return P**self.q/ (self.pc**self.q + P**self.q)
    #some helper functions
    

    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        P, M = state
        
        L = action * 12/11
        z = self.np_random.normal(0, math.sqrt(self.var) )
        N = math.exp(z- 0.5*self.var)
        e = self.e()
        f = self.f(P)
        g = self.g()
        
        Pnext = e*P + (1-e)/(self.s+self.h) *(L*N + self.r*M*f)
        Mnext = M*(1-self.b) + (1-e)*self.s/(self.s+self.h)*P + g*L*N+ (g-1)*self.r*M*f
        #see eq 6-10
        
        self.state = (Pnext, Mnext)
        done =  (Pnext >= 7 or Mnext >= 200)

        if not done:
            reward = self.alpha* math.exp(z)*L - self.beta1*P - self.beta2 * P**2
        elif self.steps_beyond_done is None:
            # OOb
            self.steps_beyond_done = 0
            reward = -100
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=np.array([0,0]), high=np.array([5,150]), size=(2,))
        #pick random new state
        self.steps_beyond_done = None
        return np.array(self.state)

    def start_at_state(self,P,M):
        if (0<=P<=7)==False or (0<=M<=200)==False:
            print(P,M)
            print("Invalid values, please enter a value between 0 and 7, and M between 1 and 150. Start state will be randomized")
            self.state = self.np_random.uniform(low=np.array([0, 0]), high=np.array([7, 200]), size=(2,))
        else:
            self.state= np.array([P,M])
        self.steps_beyond_done = None
        return np.array(self.state)




    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400


        Xscale = screen_width/self.pThresh
        Yscale = screen_height/self.mThresh

        
        dotL = 5
        dotH=5

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -dotL/2, dotL/2, dotH/2, -dotH/2
            dot = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
           # ax = rendering.Image("render.png", 600, 400)
            self.dottrans = rendering.Transform()
            dot.add_attr(self.dottrans)
          #  self.axtrans = rendering.Transform()
           # ax.add_attr(self.axtrans)
            self.viewer.add_geom(dot)
         #   self.viewer.add_geom(ax)
            

        if self.state is None: return None

        x = self.state
        Xpos = x[0] * Xscale # MIDDLE OF CART
        Ypos = x[1] * Yscale
        self.dottrans.set_translation(Xpos, Ypos)
        #self.axtrans.set_translation(screen_width/2, screen_height/2)
        
        
        
        return self.viewer.render(return_rgb_array = False)

    def close(self):
        if self.viewer: self.viewer.close()
