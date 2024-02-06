import logging
import numpy as np 
from config import Config
from supply_chain_env import SupplyChainEnv
from tqdm import tqdm

def run_base_stock_policy(env,  num_episodes = 100):
    """run the base stock policy in the given environment

    Args:
        env (SupplyChainEnv): instance of the evaluation environment
        num_episodes (int, optional): number of episodes to evaluate agent on. Defaults to 100.

    Returns:
        total_reward, total_cost (list(float), list(float)): total reward and total cost for each episodes
    """    
    env.set_policy('BS') #! ensure environment is set for bs
    total_rewards = []
    total_costs = []
    # for all episodes ... 
    for episode in range(num_episodes):
        # ... reset environment and variables
        env.reset()
        done = False
        total_reward = 0
        total_cost = 0
        # ... perfrom training episode ...  
        while not done:
            _, reward, done, _, info = env.step(None) # ... take step, NOTE: environment chooses base stock action
            if info['step'] >= Config().EVALUATION_START: # ... if start time is over ... 
                total_reward += reward # ... add reward
                total_cost += np.sum(info['cost']) # ... add cost
        # ... add total reward and total cost to output arrays
        total_rewards.append(total_reward)
        total_costs.append(total_cost)
    return total_rewards, total_costs

def run_sterman_policy(env,  num_episodes=100):
    """run the sterman policy in the given environment

    Args:
        env (SupplyChainEnv): instance of the evaluation environment
        num_episodes (int, optional): number of episodes to evaluate agent on. Defaults to 100.
        desired_OO (_type_, optional): _description_. Defaults to None.
        alpha (int, optional): _description_. Defaults to 1.
        beta (int, optional): _description_. Defaults to 1.

    Returns:
        total_reward, total_cost (list(float), list(float)): total reward and total cost for each episodes
    """    
    env.set_policy('STRM') #! ensure environment is set for sterman
    total_rewards = []
    total_costs = []
    # for all episodes ... 
    for episode in range(num_episodes):
         # ... reset environment and variables
        env.reset()
        done = False
        total_reward = 0
        total_cost = 0
        # ... perfrom training episode ...  
        while not done:
            _, reward, done, _, info = env.step(None) # ... take step, NOTE: environment chooses sterman action
            if info['step'] >= Config().EVALUATION_START: # ... if start time is over ... 
                total_reward += reward # ... add reward
                total_cost += np.sum(info['cost']) # ... add cost
        # ... add total reward and total cost to output arrays
        total_rewards.append(total_reward)
        total_costs.append(total_cost)
    return total_rewards, total_costs


def find_optimal_base_stock_levels(base_stock_range, num_episodes = 50, num_trials = 1000):
    """finds approximate optimal base stock levels using simulation

    Args:
        base_stock_range (_type_): range of base stock levels to test
        num_episodes (int, optional): number of episodes to run each simulation. Defaults to 50.
        num_trials (int, optional): mumber of different base stock levels to try. Defaults to 1000.

    Returns:
        optimal_base_stock_levels (list(int)): optimal base stock levels
    """    
    # initialize best reward and performance metrics 
    best_score = -float('inf')
    optimal_base_stock_levels = None
    # create environment
    conf_bs = Config()
    env = SupplyChainEnv(conf_bs, seed=None)
    env.set_policy('BS') #! ensure environment is set for bs
    # for number of random combinations to try ...
    for _ in tqdm(range(num_trials)):
        # ... randomly select and set base stock levels within the given range
        base_stock_levels = np.random.randint(base_stock_range[0], base_stock_range[1], size=3)
        env.config.BASE_STOCK_LEVELS = base_stock_levels
        # ... evaluate current base stock levels performance 
        _, total_cost = run_base_stock_policy(env, num_episodes)
        average_reward = sum(total_cost) / (num_episodes-Config().EVALUATION_START)

        # ... update the best score and levels
        if average_reward > best_score:
            logging.info(f"New optimal base-stock levels {base_stock_levels} with average cost: {average_reward}")
            best_score = average_reward
            optimal_base_stock_levels = base_stock_levels

    return optimal_base_stock_levels