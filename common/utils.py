#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 07:57:18 2018

@author: hsjomaa
"""
import numpy as np
import inspect

def class_vars(obj):
  return {k:v for k, v in inspect.getmembers(obj)
      if not k.startswith('__') and not callable(k)}
  
def model_dir(config):
    try:
      _attrs = config.__dict__['__flags']
    except:
      _attrs = class_vars(config)
    try:
        model_dir = config._dataset+'/'
    except Exception:
        model_dir=''
    for k in sorted(_attrs.keys()):
      if not k.startswith('_') and k not in ['display']:
        v=_attrs[k];model_dir += "%s-%s/" % (k, ",".join([str(i) for i in v])
            if type(v) == list else v)
    return '/home/hsjomaa/reinforcement-learning/adaptive-results/logs/'+model_dir
    
def as_scalar(x):
    if isinstance(x, np.ndarray):
        assert x.size == 1
        return x[0]
    elif np.isscalar(x):
        return x
    else:
        raise ValueError('expected scalar, got %s'%x)