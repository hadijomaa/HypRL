#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 19:22:02 2019

@author: hsjomaa
"""

class AgentConfig(object):
  scale = 20
  num_datasets = 40
  max_step = scale*100000*num_datasets
  memory_size = scale*100*num_datasets

  discount = 0.99
  target_q_update_step = scale*1
  learning_rate = 1e-3

  ep_end = 0.05
  ep_start = 0.95
  ep_end_t = memory_size

  train_frequency = 5
  learn_start = scale*3*num_datasets
  num_hidden = 64
  ei = False
  switch = 1
  seed = 0
  static = False
  zsplit = 0
