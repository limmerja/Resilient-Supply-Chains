from supply_chain_env import SupplyChainEnv
from config import Config

from agent import train_agent, evaluate_agent, retrain_agent, train_agent_checkpoint
from fixed_policies import find_optimal_base_stock_levels, run_base_stock_policy, run_sterman_policy

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO

import logging
import os

def main():
    num_episodes = 100  # Number of episodes to run
    seed = None # Set other than none for deterministic environment
    
    logging.basicConfig(level=logging.INFO,  filemode='w', filename= os.path.join('logs', 'app.log'),
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.info("Starting new Run")
    
    # #* Prerequisites
    env = SupplyChainEnv(Config(), seed=seed)
    vec_env_conf = Config()
    vec_env_conf.BS = False
    vec_env_conf.STRM = False
    vec_env_conf.DRL = True
    vec_env = make_vec_env(lambda: SupplyChainEnv(vec_env_conf), n_envs=1)

    # #* Agent
    
    # #! Train the agent
    # model = train_agent(vec_env, total_timesteps = 5000000)
    
    # model = retrain_agent(vec_env, model = 'ppo_supplychain_v042', log = 'ppo_run_v042')

    # #! Evaluate the trained agent
    # model = PPO.load(os.path.join('runtime', Config().EVALUATION_MODEL))
    # agent_results, agent_costs = evaluate_agent(env, model, num_episodes)
    # print("Proximal Policy Optimization Results:", sum(agent_results)/ (num_episodes-Config().EVALUATION_START), sum(agent_costs)/ (num_episodes-Config().EVALUATION_START))
    
    
    # #* Fixed Policies

    # #! Find approximate optimal base stock levels
    # #Define the range for base stock levels to test
    # base_stock_range = (5, 100)  # Example range
    # optimal_levels = find_optimal_base_stock_levels(base_stock_range, num_episodes=num_episodes, num_trials=1000)
    # print("Optimal Base Stock Levels:", optimal_levels)
    
    #! Run the base stock policy and get the results
    # base_stock_results, base_stock_costs = run_base_stock_policy(env, num_episodes = num_episodes)
    # print("Base Stock Policy Results:", sum(base_stock_results) / (num_episodes-Config().EVALUATION_START), sum(base_stock_costs) / (num_episodes-Config().EVALUATION_START))

    # #! Run the Sterman policy and get the results
    # sterman_results, sterman_costs = run_sterman_policy(env, num_episodes = num_episodes)
    # print("Sterman Policy Results:", sum(sterman_results)/ (num_episodes-Config().EVALUATION_START),  sum(sterman_costs)/ (num_episodes-Config().EVALUATION_START))

if __name__ == "__main__":
    main()
