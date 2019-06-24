import os
import tempfile

import tensorflow as tf
import zipfile
import cloudpickle
import numpy as np

import common.tf_util as U
from common.tf_util import load_state, save_state
import logger
from common.schedules import LinearSchedule

from agent import deepq
from agent.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from agent.deepq.utils import ObservationInput
import copy

class ActWrapper(object):
    def __init__(self, act, act_params):
        self._act = act
        self._act_params = act_params

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            model_data, act_params = cloudpickle.load(f)
        act = deepq.build_act(**act_params)
        sess = tf.Session()
        sess.__enter__()
        with tempfile.TemporaryDirectory() as td:
            arc_path = os.path.join(td, "packed.zip")
            with open(arc_path, "wb") as f:
                f.write(model_data)

            zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
            load_state(os.path.join(td, "model"))

        return ActWrapper(act, act_params)

    def __call__(self, *args, **kwargs):
        return self._act(*args, **kwargs)

    def save(self, path=None):
        """Save model to a pickle located at `path`"""
        if path is None:
            path = os.path.join(logger.get_dir(), "model.pkl")

        with tempfile.TemporaryDirectory() as td:
            save_state(os.path.join(td, "model"))
            arc_name = os.path.join(td, "packed.zip")
            with zipfile.ZipFile(arc_name, 'w') as zipf:
                for root, dirs, files in os.walk(td):
                    for fname in files:
                        file_path = os.path.join(root, fname)
                        if file_path != arc_name:
                            zipf.write(file_path, os.path.relpath(file_path, td))
            with open(arc_name, "rb") as f:
                model_data = f.read()
        with open(path, "wb") as f:
            cloudpickle.dump((model_data, self._act_params), f)


def load(path):
    """Load act function that was returned by learn function.

    Parameters
    ----------
    path: str
        path to the act function pickle

    Returns
    -------
    act: ActWrapper
        function that takes a batch of observations
        and returns actions.
    """
    return ActWrapper.load(path)

def learn(env,
          q_func,
          max_lives=1,
          lr=5e-4,
          max_timesteps=100000,
          buffer_size=50000,
          exploration_fraction=0.1,
          exploration_final_eps=0.02,
          train_freq=1,
          batch_size=32,
          print_freq=100,
          checkpoint_freq=10000,
          checkpoint_path=None,
          learning_starts=1000,
          gamma=1.0,
          target_network_update_freq=500,
          prioritized_replay=False,
          prioritized_replay_alpha=0.6,
          prioritized_replay_beta0=0.4,
          prioritized_replay_beta_iters=None,
          prioritized_replay_eps=1e-6,
          param_noise=False,
          ei = False,
          N_t = 14,
          callback=None):
    """Train a deepq model.

    Parameters
    -------
    env: gym.Env
        environment to train on
    q_func: (tf.Variable, int, str, bool) -> tf.Variable
        the model that takes the following inputs:
            observation_in: object
                the output of observation placeholder
            num_actions: int
                number of actions
            scope: str
            reuse: bool
                should be passed to outer variable scope
        and returns a tensor of shape (batch_size, num_actions) with values of every action.
    lr: float
        learning rate for adam optimizer
    max_timesteps: int
        number of env steps to optimizer for
    buffer_size: int
        size of the replay buffer
    exploration_fraction: float
        fraction of entire training period over which the exploration rate is annealed
    exploration_final_eps: float
        final value of random action probability
    train_freq: int
        update the model every `train_freq` steps.
        set to None to disable printing
    batch_size: int
        size of a batched sampled from replay buffer for training
    print_freq: int
        how often to print out training progress
        set to None to disable printing
    checkpoint_freq: int
        how often to save the model. This is so that the best version is restored
        at the end of the training. If you do not wish to restore the best version at
        the end of the training set this variable to None.
    learning_starts: int
        how many steps of the model to collect transitions for before learning starts
    gamma: float
        discount factor
    target_network_update_freq: int
        update the target network every `target_network_update_freq` steps.
    prioritized_replay: True
        if True prioritized replay buffer will be used.
    prioritized_replay_alpha: float
        alpha parameter for prioritized replay buffer
    prioritized_replay_beta0: float
        initial value of beta for prioritized replay buffer
    prioritized_replay_beta_iters: int
        number of iterations over which beta will be annealed from initial value
        to 1.0. If set to None equals to max_timesteps.
    prioritized_replay_eps: float
        epsilon to add to the TD errors when updating priorities.
    callback: (locals, globals) -> None
        function called at every steps with state of the algorithm.
        If callback returns true training stops.

    Returns
    -------
    act: ActWrapper
        Wrapper over act function. Adds ability to save it and load it.
        See header of baselines/deepq/categorical.py for details on the act function.
    """
    # Create all the functions necessary to train the model

    sess = tf.Session()
    sess.__enter__()

    # capture the shape outside the closure so that the env object is not serialized
    # by cloudpickle when serializing make_obs_ph

    def make_obs_ph(name):
        return ObservationInput(env.observation_space, name=name)

    act, train, update_target, debug = deepq.build_train(
        make_obs_ph=make_obs_ph,
        q_func=q_func,
        num_actions=env.action_space.n,
        optimizer=tf.train.RMSPropOptimizer(learning_rate=lr,momentum=0.95, epsilon=0.01),
        gamma=gamma,
        grad_norm_clipping=10,
        param_noise=param_noise,
        session=sess
    )

    act_params = {
        'make_obs_ph': make_obs_ph,
        'q_func': q_func,
        'num_actions': env.action_space.n,
    }

    act = ActWrapper(act, act_params)

    # Create the replay buffer
    if prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
        if prioritized_replay_beta_iters is None:
            prioritized_replay_beta_iters = max_timesteps
        beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                       initial_p=prioritized_replay_beta0,
                                       final_p=1.0)
    else:
        replay_buffer = ReplayBuffer(buffer_size)
        beta_schedule = None
    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * max_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)

    # Initialize the parameters and copy them to the target network.
    U.initialize()
    update_target()

    episode_rewards = [0.0]
    episode_ei = [0.0]
    episode_lengths = [1]
    errors          = []
    q_t_train       = []
    saved_mean_reward = None
    
    dataset_idx = 0
    dataset_ctr = np.zeros(shape=(len(env.env.ale.metadata)))
    dataset_ctr[dataset_idx] +=1
    state      = env.reset()
    obs        = np.zeros(shape=env.observation_space.shape)
    obs[0,:]   = np.append(np.repeat(np.NaN,repeats=N_t),env.env.ale.metadata[dataset_idx]['features']).reshape(1,-1)
    reset   = True
    prev_r  = 0
    with tempfile.TemporaryDirectory() as td:
        td = checkpoint_path or td

        model_file = os.path.join(td, "model")
        model_saved = False
        if tf.train.latest_checkpoint(td) is not None:
            load_state(model_file)
            logger.log('Loaded model from {}'.format(model_file))
            model_saved = True

        for t in range(max_timesteps):
            if callback is not None:
                if callback(locals(), globals()):
                    break
            kwargs = {}
            if not param_noise:
                update_eps = exploration.value(t)
                update_param_noise_threshold = 0.
            else:
                update_eps = 0.
                update_param_noise_threshold = -np.log(1. - exploration.value(t) + exploration.value(t) / float(env.action_space.n))
                kwargs['reset'] = reset
                kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                kwargs['update_param_noise_scale'] = True
            seq = env.env._get_ep_len()
            action = act(np.array(obs)[None],[seq], update_eps=update_eps, **kwargs)[0]
            env_action = action
            reset = False
            new_state, rew, done, _ = env.step(env_action,dataset_id=dataset_idx)
            new_obs = copy.copy(obs)
            new_obs[env.env._get_ep_len(),:] = np.append(new_state,np.append(rew,env.env.ale.metadata[dataset_idx]['features'])).reshape(1,-1)
            replay_buffer.add(obs, action, np.maximum(0,rew-prev_r) if ei else rew, new_obs, float(done),seq=seq)
            obs = new_obs

            episode_rewards[-1] += rew
            episode_ei[-1]      += np.maximum(0,rew-prev_r)
            prev_r = copy.copy(rew)
            if done:
                episode_lengths[-1]=env.env._get_ep_len()
                state      = env.reset()
                obs        = np.zeros(shape=env.observation_space.shape)
                obs[0,:]   = np.append(np.repeat(np.NaN,repeats=N_t),env.env.ale.metadata[dataset_idx]['features']).reshape(1,-1)
                env.env.ale._used_lives +=1
                if env.env.ale._used_lives % max_lives == 0:
                    dataset_idx +=1
                    dataset_idx = dataset_idx%len(env.env.ale.metadata)
                    dataset_ctr[dataset_idx] +=1
                episode_lengths.append(1)
                episode_rewards.append(0.0)
                episode_ei.append(0.0)
                reset = True
                prev_r = 0

            if t > learning_starts and t % train_freq == 0:
                if prioritized_replay:
                    experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(t))
                    if len(experience)==5:
                        (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                    else:
                        (obses_t, actions, rewards, obses_tp1, dones,seqes, weights, batch_idxes) = experience
                else:
                    experience = replay_buffer.sample(batch_size)
                    if len(experience)==5:
                        obses_t, actions, rewards, obses_tp1, dones = experience
                    else:
                        obses_t, actions, rewards, obses_tp1, dones,seqes = experience
                        
                    weights, batch_idxes = np.ones_like(rewards), None
                td_errors,q_t = train(obses_t, seqes,actions, rewards, obses_tp1,seqes+1, dones, weights)
                errors.append(td_errors)
                q_t_train.append(np.mean(q_t))
                if prioritized_replay:
                    new_priorities = np.abs(td_errors) + prioritized_replay_eps
                    replay_buffer.update_priorities(batch_idxes, new_priorities)

            if t > learning_starts and t % target_network_update_freq == 0:
                update_target()

            mean_100ep_length = round(np.mean(episode_lengths[-101:-1]), 4)
            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 4)
            mean_100ep_reward_running = round(np.mean(episode_rewards[-201:-100]), 4)
            mean_100ep_ei     = round(np.mean(episode_ei[-101:-1]), 4)
            num_episodes      = len(episode_rewards)
            assert(num_episodes==len(episode_lengths)),(num_episodes,len(episode_lengths))
            if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
                logger.record_tabular("q_t", round(np.mean(q_t_train), 4))
                logger.record_tabular("td_erros", round(np.mean(errors), 4))
                logger.record_tabular("steps", t)
                logger.record_tabular("most_used_dataset", np.argmax(dataset_ctr))
                logger.record_tabular("number_of_used", np.max(dataset_ctr))
                logger.record_tabular("episodes", num_episodes)
                logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                logger.record_tabular("mean 100 episode ei", mean_100ep_ei)
                logger.record_tabular("mean 100 episode length", mean_100ep_length)
                logger.record_tabular("lives", env.env.ale._used_lives )
                logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                logger.dump_tabular()
                errors = []
                q_t_train = []
                dataset_ctr = np.zeros(shape=(len(env.env.ale.metadata)))
                dataset_ctr[dataset_idx] +=1                
            if (checkpoint_freq is not None and t > learning_starts and
                    num_episodes > 100 and t % checkpoint_freq == 0):
                if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
                    if print_freq is not None:
                        logger.log("Saving model due to mean reward increase: {} -> {}".format(
                                   saved_mean_reward, mean_100ep_reward))
                    save_state(model_file)
                    model_saved = True
                    saved_mean_reward = mean_100ep_reward
                if mean_100ep_reward  > mean_100ep_reward_running:
                    save_state(model_file+'-running')
                    if print_freq is not None:
                        logger.log("Saving model due to running mean reward increase: {} -> {}".format(
                                   mean_100ep_reward_running, mean_100ep_reward))
                    
        if model_saved:
            if print_freq is not None:
                logger.log("Restored model with mean reward: {}".format(saved_mean_reward))
            load_state(model_file)
    save_state(model_file+'-final')
    return act


def play(env,
          q_func,
          max_lives=1,
          lr=5e-4,
          gamma=1.0,
          max_timesteps=100000,
          checkpoint_path=None,
          N_t=14,
          param_noise=False,):
    # Create all the functions necessary to train the model

    sess = tf.Session()
    sess.__enter__()

    # capture the shape outside the closure so that the env object is not serialized
    # by cloudpickle when serializing make_obs_ph

    def make_obs_ph(name):
        return ObservationInput(env.observation_space, name=name)

    act, train, update_target, debug = deepq.build_train(
        make_obs_ph=make_obs_ph,
        q_func=q_func,
        num_actions=env.action_space.n,
        optimizer=tf.train.RMSPropOptimizer(learning_rate=lr,momentum=0.95, epsilon=0.01),
        gamma=gamma,
        grad_norm_clipping=10,
        param_noise=param_noise,
        session=sess
    )

    act_params = {
        'make_obs_ph': make_obs_ph,
        'q_func': q_func,
        'num_actions': env.action_space.n,
    }

    act = ActWrapper(act, act_params)
    U.initialize()
    update_target()

    episode_rewards = [0.0]
    
    model_file = os.path.join(checkpoint_path, "model")
    load_state(model_file)
    logger.log('Loaded model from {}'.format(model_file))
    
    seq_list = []
    rew_list = []
    for dataset_idx in range(len(env.env.ale.metatest)):
        env.reset()
        obs        = np.zeros(shape=env.observation_space.shape)
        obs[0,:]   = np.append(np.repeat(np.NaN,repeats=N_t),env.env.ale.metatest[dataset_idx]['features']).reshape(1,-1)        
        print(env.env.ale.metatest[dataset_idx]['name'])
        rewards = env.env.ale.metatest[dataset_idx]['rewards']
        tmp_r = []
        for t in range(0,max_timesteps):
            update_eps = 0.
            seq         = env.env._get_ep_len()
            print(seq)
            seq_list.append(seq)
            action      = act(np.array(obs)[None],[seq], update_eps=update_eps)[0]
            env_action = action
            new_state, _, _, _ = env.step(env_action,dataset_id=dataset_idx)
            rew = rewards[action]
            tmp_r.append(rew)
            new_obs = copy.copy(obs)
            try:
                new_obs[env.env._get_ep_len(),:] = np.append(new_state,np.append(rew,env.env.ale.metatest[dataset_idx]['features'])).reshape(1,-1)
                obs = new_obs
                episode_rewards[-1] += rew
            except Exception: pass
        rew_list.append(tmp_r)

    return rew_list