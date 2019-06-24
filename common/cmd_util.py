#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 11:01:54 2019

@author: hsjomaa
"""
from envs import make
from mpi4py import MPI
from bench import Monitor
import os
import logger
import numpy as np
import random

def make_meta_env(env_id, seed):
    """
    Create a wrapped, monitored gym.Env for MuJoCo.
    """
    rank = MPI.COMM_WORLD.Get_rank()
    set_global_seeds(seed + 10000 * rank)
    env = make(env_id)
    env = Monitor(env, os.path.join(logger.get_dir(), str(rank)),allow_early_resets=True)
    return env


def nn_meta_arg_parser():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)    
    parser.add_argument('--num-timesteps', help='Number of Epochs', type=int,default=500)
    parser.add_argument('--nbatch', help='Batch Size', type=int,default=32)
    parser.add_argument('--env_id', help='dataset name',default='nnMeta-v0')
    parser.add_argument('--seed', help='Seed', type=int,default=0)
    return parser

def meta_arg_parser():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)    
    parser.add_argument('--num-timesteps', help='Number of Epochs', type=int,default=500)
    parser.add_argument('--nbatch', help='Batch Size', type=int,default=32)

    parser.add_argument('--num_hidden_ddpg', help='number of hidden unit', type=int,default=35)

    parser.add_argument('--nb_train_steps', help='Number of train steps', type=int,default=1)
    parser.add_argument('--lr_actor', help='Actor Learning Rate', type=float,default=6e-5)
    parser.add_argument('--lr_critic', help='Critic Learning Rate', type=float,default=6e-5)
    
    parser.add_argument('--noise_type', type=str, default='none')  # choices are adaptive-param_xx, ou_xx, normal_xx, none
    parser.add_argument('--env_id', help='dataset name',default='abalone-v4')
    parser.add_argument('--seed', help='Seed', type=int,default=0)
    return parser


def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)