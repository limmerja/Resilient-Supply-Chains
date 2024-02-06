from config import Config
from supply_chain_env import SupplyChainEnv
from agent import evaluate_agent
from fixed_policies import run_base_stock_policy, run_sterman_policy

from stable_baselines3 import PPO 

import numpy as np 
import pandas as pd
import os.path as path

    
def calculate_cost_multiplier_dependency(type, range_min = 0, range_max = 5.1, step = 0.1, num_episodes = 100): 
    x = []
    y = []
    z = []
    conf = Config()
    conf.DISRUPTION_PROBABILITY_TOTAL = 1.0
    for i in np.arange (range_min, range_max, step): 
        for j in np.arange (range_min, range_max, step): 
            x.append(i)
            y.append(j)
            conf.INVENTORY_COST_MULTIPLIER = [i, i, i]
            conf.SHORTAGE_COST_MULTIPLIER = [j, j, j]
            env = SupplyChainEnv(conf, seed=conf.EVALUATION_SEED)
            if type == 'PPO': 
                model = PPO.load(path.join("runtime", conf.EVALUATION_MODEL), env=env) 
                _, cost = evaluate_agent(env, model, num_episodes)
            elif type == 'BS': 
                _, cost = run_base_stock_policy(env, num_episodes=num_episodes)
            elif type == 'STRM': 
                _, cost = run_sterman_policy(env, num_episodes)
            else: 
                return -1
            z.append(sum(cost)/ (num_episodes - conf.EVALUATION_START))
            
    cost_multipliers = pd.DataFrame({'x': x, 'z': z, 'y': y})
    cost_multipliers.to_csv(path.join('data', 'sensitivity', f'cost_multipliers_{type}.csv'))
    
    return 1


def calculate_distribution_sensitivity(demand_distributions, num_episodes = 100): 
    with open(path.join('svg', 'tables', 'distributions.txt'), 'w') as f: 
        for i in demand_distributions.keys(): 
            for j in range(len(demand_distributions[i]['distribution'])):  
                conf = Config()
                if i == 'normal': 
                    conf.DEMAND_MEAN = demand_distributions[i]['distribution'][j][0]
                    conf.DEMAND_STD_DEV = demand_distributions[i]['distribution'][j][1]
                    conf.BASE_STOCK_LEVELS = demand_distributions[i]['bs_levels'][j]
                    f.write('$\\mathcal{N}(' + str(demand_distributions[i]['distribution'][j][0]) + ', \\,' + str(demand_distributions[i]['distribution'][j][1]) + '^{2})$\n\t')
                elif i == 'uniform': 
                    conf.DEMAND_NORMAL = False
                    conf.DEMAND_MIN = demand_distributions[i]['distribution'][j][0]
                    conf.DEMAND_MAX = demand_distributions[i]['distribution'][j][0]
                    conf.BASE_STOCK_LEVELS = demand_distributions[i]['bs_levels'][j]
                    f.write('$\\mathcal{U}(' + str(demand_distributions[i]['distribution'][j][0]) + ', \\,' + str(demand_distributions[i]['distribution'][j][1]) + ')$\n\t')
                else: 
                    continue
                    
                
                env = SupplyChainEnv(conf, seed=conf.EVALUATION_SEED)
                model = PPO.load(path.join("runtime", conf.EVALUATION_MODEL), env=env) 
                
                _, base_stock_costs = run_base_stock_policy(env, num_episodes = num_episodes)
                f.write(' & ' + str(round(sum(base_stock_costs) / (num_episodes-conf.EVALUATION_START), 2)))
                
                _, sterman_costs = run_sterman_policy(env, num_episodes = num_episodes)
                f.write(' & ' + str(round(sum(sterman_costs) / (num_episodes-conf.EVALUATION_START), 2)))
                
                _, agent_costs = evaluate_agent(env, model, num_episodes)
                f.write(' & ' + str(round(sum(agent_costs) / (num_episodes-conf.EVALUATION_START), 2)) + '\\\\\n')
                
                
def calculate_sensitivity(min, max, attribute, norm = False, norm_base = 0, num_episodes = 100): 
    x = []
    BS_y = []
    STRM_y = []
    PPO_y = []
    conf = Config()
    for i in range(min, max): 
        x.append(i)
        setattr(conf, attribute, i)
        env = SupplyChainEnv(conf, seed=conf.EVALUATION_SEED)
        model = PPO.load(path.join("runtime", conf.EVALUATION_MODEL), env=env) 
        
        _, base_stock_costs = run_base_stock_policy(env, num_episodes = num_episodes)
        BS_y.append(sum(base_stock_costs)/ (num_episodes-conf.EVALUATION_START))
        
        
        _, strm_costs = run_sterman_policy(env, num_episodes = num_episodes)
        STRM_y.append(sum(strm_costs)/ (num_episodes-conf.EVALUATION_START))
        
        
        _, agent_costs = evaluate_agent(env, model, num_episodes)
        PPO_y.append(sum(agent_costs)/ (num_episodes-conf.EVALUATION_START))
        
    length = pd.DataFrame({'x': x, 'BS': BS_y, 'STRM': STRM_y, 'PPO': PPO_y})
    if norm: 
        length['y_norm'] = length.y * (norm_base / length.x)
    length.to_csv(path.join('data', 'sensitivity', f'{attribute}.csv'))