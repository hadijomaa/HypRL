#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 07:46:27 2019

@author: hsjomaa
"""

import numpy as np
import os
import gym
import envs.spaces as spaces
import pandas as pd
class ALEInterface(object):
    def __init__(self,config):
        hyper_parameter_size,max_sequence_length = config['obs_space']
        DATASET_PATH = config['path']
        train = os.listdir(os.path.join(DATASET_PATH,'train'))
        test  = os.listdir(os.path.join(DATASET_PATH,'test'))
        metadata        = {}
        self.N_f =N_f= config['metafeatures']
        for idx,file in enumerate(train):
            metadata[idx] = {'rewards' : np.asarray(pd.read_csv(os.path.join(DATASET_PATH,'train',file),delimiter=' ',header=None))[:,:1],
                             'features': np.asarray(pd.read_csv(os.path.join(DATASET_PATH,'train',file),delimiter=' ',header=None))[0,-N_f:].reshape(1,-1),
                             'name':file}
        Lambda              = np.asarray(pd.read_csv(os.path.join(DATASET_PATH,'train',metadata[0]['name']),delimiter=' ',header=None))[:,1:-N_f]        
        self._used_lives    = 1
        self.metatest = {}
        for idx,file in enumerate(test):
            self.metatest[idx] = {'rewards' : np.asarray(pd.read_csv(os.path.join(DATASET_PATH,'test',file),delimiter=' ',header=None))[:,:1],
                             'features': np.asarray(pd.read_csv(os.path.join(DATASET_PATH,'test',file),delimiter=' ',header=None))[0,-N_f:].reshape(1,-1),
                             'name':file}
            
        def getScreenDims():
            return hyper_parameter_size,max_sequence_length
        
        def getMinimalActionSet():
            return [_ for _ in range(len(metadata[0]['rewards']))]

        def _transition_fx(action):
            _lambda = Lambda[action].reshape(1,-1)
            self._ep_len +=1
            return _lambda
            
        def act(action,dataset_id):
            self._lambda           = _transition_fx(action)            
            self._current_action = action
            return metadata[dataset_id]['rewards'][action][0]

        def getScreenRGB2():
            return self._lambda
    
        def lives():
            return self._used_lives
        
        def reset_game():
            self._ep_len              = 1
            self._ep_act              = []
        
        def episode_length():
            return self._ep_len
        
        def game_over():
            terminal = self._ep_len%(max_sequence_length-1)==0 or self._current_action in self._ep_act
            self._ep_act.append(self._current_action)
            return terminal
        
        def reset_lives():
            self._used_lives = 1
            
        self.getScreenDims       = getScreenDims
        self.getMinimalActionSet = getMinimalActionSet
        self.act                 = act
        self.getScreenRGB2       = getScreenRGB2
        self.lives               = lives
        self.game_over           = game_over
        self.episode_length               = episode_length
        self.reset_game          = reset_game
        self.reset_lives         = reset_lives
        self.metadata            = metadata
        reset_game()
        
class nnMetaEnv(gym.Env):

    def __init__(self, configs):

        assert configs['obs_type'] in ('multi_variate')

        self.game_path = configs['path']
        if not os.path.exists(self.game_path):
            raise IOError('You asked for nn Meta Dataset but path %s does not exist'%(self.game_path))
        self._obs_type = configs['obs_type']
        self.ale = ALEInterface(configs)
        self.viewer = None

        self.seed()

        (hyper_parameter_size, max_sequence_length) = self.ale.getScreenDims()

        self._action_set = self.ale.getMinimalActionSet()
        self.action_space = spaces.Discrete(len(self._action_set))

        (hyper_parameter_size,max_sequence_length) = self.ale.getScreenDims()
        if self._obs_type == 'multi_variate':
            self.observation_space = spaces.Box(low=0, high=255, shape=(max_sequence_length, hyper_parameter_size), dtype=np.float)

    def seed(self, seed=None):
        np.random.seed(seed)

    def step(self, a,dataset_id):
        action = self._action_set[a]
        reward = self.ale.act(action,dataset_id)
        ob     = self._get_obs()
        return ob, reward, self.ale.game_over(), {"ale.lives": self.ale.lives()}

    def _get_ep_len(self):
        return self.ale.episode_length()
    
    def _get_multi_variate(self):
        return self.ale.getScreenRGB2()

    def switch_datasets(self):
        return self.ale.reset_lives()
    @property
    def _n_actions(self):
        return len(self._action_set)

    def _get_obs(self):
        if self._obs_type == 'multi_variate':
            img = self._get_multi_variate()
        return img

    def reset(self):
        self.ale.reset_game()
        return None

    def render(self, mode='human'):
        img = self._get_multi_variate()
        return img

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def get_action_meanings(self):
        file                = np.random.choice([_ for _ in os.listdir(self.game_path)])
        Lambda              = np.asarray(pd.read_csv(os.path.join(self.game_path,file),delimiter=' ',header=None))[:,1:-self.ale.N_f]
        ACTION_MEANING = {}
        for i in range(Lambda.shape[0]):
            ACTION_MEANING[i] = Lambda[i]        
        return [ACTION_MEANING[i] for i in self._action_set]

