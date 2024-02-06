from stable_baselines3 import PPO
from supply_chain_env import SupplyChainEnv
import numpy as np
from config import Config

import logging
import os

def step_decay_schedule(total_timesteps, initial_value, decay_steps, decay_rate):
    """step decay learning rate schedule

    Args:
        total_timesteps (float): initial learning rate
        initial_value (float): initial learning rate
        decay_steps (int): number of steps between each decay
        decay_rate (float): factor by which to decay the learning rate

    Returns:
        func (function): a function that takes the current timestep and returns
                         the learning rate according to the schedule
    """    

    def func(t):
        """takes the current timestep and returns the learning rate

        Args:
            t (float): fraction of the training remaining (goes from 1 to 0)

        Returns:
            learning rate (float): updated learning rate by remaining training
        """ 
        lr = initial_value * (decay_rate ** (((1 - t) *total_timesteps) // decay_steps))
        return lr
    return func

def linear_schedule(initial_value):
    """linear decay learning rate schedule

    Args:
        initial_value (float): initial learning rate
    
    Returns:
        func (function): a function that takes the current timestep and returns
                         the learning rate according to the schedule
    """    
    def schedule(t):
        """takes the current timestep and returns the learning rate

        Args:
            t (float): fraction of the training remaining (goes from 1 to 0)

        Returns:
            learning rate (float): updated learning rate by remaining training
        """        
        return initial_value * t
    return schedule

def exponential_schedule(initial_value):
    """exponentiall decay learning rate schedule

    Args:
        initial_value (float): initial learning rate
    
    Returns:
        func (function): a function that takes the current timestep and returns
                         the learning rate according to the schedule
    """
    def schedule(t):
        """takes the current timestep and returns the learning rate

        Args:
            t (float): fraction of the training remaining (goes from 1 to 0)

        Returns:
            learning rate (float): updated learning rate by remaining training
        """
        return initial_value * (t**2)
    return schedule


def train_agent(env, total_timesteps = 10000000, name = 'ppo_supplychain'):
    """ppo agent training function. stores trained agent in runtime

    Args:
        env (SupplyChainEnv): instance of the training environment
        total_timesteps (int, optional): total number of training timesteps. Defaults to 10000000.
        name (str, optional): name used for storing the model. Defaults to 'ppo_supplychain'.

    Returns:
        model: returns trained ppo agent
    """    
    # instantiate the ppo agent
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard_logs/", 
                gae_lambda=0.95)
    model.lr_schedule = linear_schedule((3e-4)*0.75)
    # train the agent
    model.learn(total_timesteps=total_timesteps, tb_log_name="ppo_run")
    # save the trained agent
    model.save(os.path.join("runtime", name))

    return model

def evaluate_agent(env, model, num_episodes=100):
    """ppo agent evaluation function

    Args:
        env (SupplyChainEnv): instance of the evaluation environment
        model (PPO): trained PPO agent
        num_episodes (int, optional): number of episodes to evaluate agent on. Defaults to 100.
    Returns:
        total_reward, total_cost (list(float), list(float)): total reward and total cost for each episodes
    """    
    env.set_policy('DRL') #! ensure environment is set for drl
    total_rewards = []
    total_costs = []
    # for all episodes ... 
    for _ in range(num_episodes): 
        # ... reset environment and variables
        obs, _ = env.reset() 
        done = False
        total_reward = 0
        total_cost = 0
        # ... perfrom training episode ...
        while not done:
            action, _ = model.predict(obs, deterministic=False) # ... predict actions
            obs, reward, done, truncated, info = env.step(action) # ... take step 
            if info['step'] >= Config().EVALUATION_START: # ... if start time is over ... 
                total_reward += reward # ... add reward
                total_cost += np.sum(info['cost']) # ... add cost
        # ... add total reward and total cost to output arrays
        total_rewards.append(total_reward)
        total_costs.append(total_cost)
    return total_rewards, total_costs

def train_agent_checkpoint(env, total_timesteps = 10000000, update_interval = 500000, num_episodes = 100, 
                           early_stopping_threshold = 2000): 
    """ppo agent training function storing checkpoints to ensure continous improvement. 
       stores best trained agent in runtime

    Args:
        env (SupplyChainEnv): instance of the training environment
        total_timesteps (int, optional): total number of training timesteps. Defaults to 10000000.
        update_interval (int, optional): number of training steps before checkpoint. Defaults to 500000.
        num_episodes (int, optional): number of episodes to evaluate agent on. Defaults to 100.
        early_stopping_threshold (int, optional): allowed performance impact before reloading checkpoint. 
                                                  Defaults to 2000.
    """    
    # instantiate the ppo agent
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard_logs/", gae_lambda=0.90)
    # initialize best reward and performance metrics 
    best_reward = -float("inf")
    performance_metrics = []
    # calculate total number of checkpoints in training 
    total_updates = total_timesteps // update_interval
    # for every checkpoint ... 
    for update in range(total_updates):
        # ... train the agent
        model.learn(total_timesteps=update_interval, reset_num_timesteps=False, tb_log_name = "ppo_run_1")
        # ... evaluate current agent performance 
        total_reward = evaluate_agent(SupplyChainEnv(Config()), model, num_episodes=num_episodes)
        avg_reward = sum(total_reward) / num_episodes # ... calculate average reward
        performance_metrics.append(avg_reward) 
        logging.info(f"Update {update} finished with average reward: {avg_reward}")

        # ... checkpointing: save if agent is the best so far
        if avg_reward > best_reward:
            logging.info(f"Improvement: performance increased from {best_reward} to {avg_reward}")
            best_reward = avg_reward # ... update best reward
            model.save(os.path.join("runtime", "ppo_supplychain_best")) # ... store model in runtime

        # ... checkpoint condition: if performance drops below threshold and checkpoint exists
        if len(performance_metrics) > 1 and (-avg_reward) - (-performance_metrics[-2]) > early_stopping_threshold:
            logging.warn(f"Early stopping: performance dropped from {performance_metrics[-2]} to {avg_reward}")
            # ... load the previously best model
            model = PPO.load(os.path.join("runtime", "ppo_supplychain_best"), env=env) 
            
def retrain_agent(env, total_timesteps = 10000000, model = 'ppo_supplychain_v042', log = 'ppo_run_v042'): 
    """retrains an already trained ppo agent stored in runtime. stores the retrained agent in runtime

    Args:
        env (SupplyChainEnv): instance of the training environment
        total_timesteps (int, optional): total number of training timesteps. Defaults to 10000000.
        model (str, optional): name of the ppo model to retrain. Defaults to 'ppo_supplychain_v042'.
        log (str, optional): name of the tensorboard log to append retrainining data to. Defaults to 'ppo_run_v042'.

    Returns:
        model: returns retrained ppo agent
    """    
    # load the ppo model
    model = PPO.load(os.path.join("runtime", model), env=env) 
    # make adjustments
    # model.gae_lambda = 0.5 # adjust general advantage estimate lambda    
    # model.lr_schedule = linear_schedule(3e-4) # adjust learning rate
    # perform additional training 
    model.learn(total_timesteps = total_timesteps, 
                reset_num_timesteps = False, 
                tb_log_name = log)
    
    # save the retrained agent to runtime
    model.save(os.path.join("runtime", "ppo_supplychain"))
    # return agent
    return model