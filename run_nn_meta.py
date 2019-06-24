#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 20:19:57 2019

@author: hsjomaa
"""

from agent import deepq
from common.misc_util import set_global_seeds
import argparse
import logger
from common.cmd_util import  make_meta_env
import tensorflow as tf
from common.utils import model_dir
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env_id', help='environment ID', default='nnMeta-v40')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--prioritized', type=int, default=0)
    parser.add_argument('--prioritized-replay-alpha', type=float, default=0.6)
    parser.add_argument('--dueling', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(10e7))
    parser.add_argument('--checkpoint-freq', type=int, default=10000)
    parser.add_argument('--train_freq', type=int, default=1)
    parser.add_argument('--learning_starts', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--target_network_update_freq', type=int, default=10)
    parser.add_argument('--cell', type=int, default=32)
    parser.add_argument('--nhidden', type=int, default=128)
    parser.add_argument('--ei', default='True')
    
    args                  = parser.parse_args()
    args.buffer_size      = args.learning_starts
    if args.env_id.startswith('nn'):
        N_t = 14; N_f = 16
    elif args.env_id.startswith('svm'):
        N_t = 7; N_f = 3
    checkpoint_path = model_dir(args)
    logger.configure(checkpoint_path)
    set_global_seeds(args.seed)
    env = make_meta_env(args.env_id,seed=args.seed)
    model = deepq.models.lstm_to_mlp(
        cell=(args.cell,N_t,N_f),
        aktiv = tf.nn.tanh,
        hiddens=[args.nhidden],
        max_length = env.observation_space.shape[0],
        dueling=bool(args.dueling),
    )
    
    deepq.learn(
        env,
        q_func=model,
        lr=args.lr,
        max_timesteps=args.num_timesteps,
        buffer_size=args.buffer_size,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=args.train_freq,
        learning_starts=args.learning_starts,
        target_network_update_freq=args.target_network_update_freq,
        gamma=0.99,
        prioritized_replay=bool(args.prioritized),
        prioritized_replay_alpha=args.prioritized_replay_alpha,
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_path=checkpoint_path.replace('/logs/','/checkpoints/'),
        ei=eval(args.ei),
        N_t=N_t
    )
    
    env.close()


if __name__ == '__main__':
    main()
