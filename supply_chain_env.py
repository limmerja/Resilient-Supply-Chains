import gymnasium as gym
import numpy as np
import math as m
from config import Config
import logging
from tools.tools import welford_finalize, welford_update
import os

class SupplyChainEnv(gym.Env):
    """
    custom environment for the supply chain simulation
    """
    def __init__(self, config: Config, seed = None, deterministic_reset = False):
        super(SupplyChainEnv, self).__init__() # initialize environment with base environment

        self.training = False #! important if standardization is used: False for evaluation, True for training
        self.normalized = False #! indicate if standardization is used
        #* adjust action space 
        max_order_size = Config.MAX_ORDER_SIZE  
        min_order_size = Config.MIN_ORDER_SIZE
        self.action_space = gym.spaces.Box(low=min_order_size, high=max_order_size, shape=(3,), dtype=np.int64)
        # define action with normalized range (optional)
        # self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

         #* adjust observation space
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4* (config.NUM_ECHELONS) + config.m,), dtype=np.int64)
        #* standardization parameters 
        if self.normalized: 
            if self.training:
                self.state_running_mean = np.zeros(4* (config.NUM_ECHELONS) + config.m)
                self.state_running_M = np.zeros(4* (config.NUM_ECHELONS) + config.m)
                self.state_running_std = np.zeros(4* (config.NUM_ECHELONS) + config.m)
            else: 
                self.state_running_mean = np.array([22.80929546, 24.06172302, 24.41980907,  3.61247216,  4.93367836,
                                                    5.34089304, 19.1968233 , 19.12804466, 19.07891603, -3.60528807,
                                                    -1.85017843, -1.30461791,  9.74911987,  9.70159472,  9.65418378,
                                                    9.60686438,  9.55976299])
                self.state_running_std = np.array([10.27101022,  5.06907593,  3.7749136 , 14.28181031,  9.79042443,
                                                8.4319734 ,  9.25932749,  8.92900632,  8.55142546,  8.80122052,
                                                4.74617088,  3.31245221,  2.6025185 ,  2.68807297,  2.77015969,
                                                2.8490797 ,  2.92495296])
        self.total_steps = 0

        #* randoms 
        self.env_seed = seed # if seed = None equals no seed 
        self.deterministic_reset = deterministic_reset
        np.random.seed(self.env_seed) # seed the standard numpy generator
        self.np_random = None 
        self.seed(seed=self.env_seed)  # create and seed the random generator

        #* config vars 
        self.config = config
        self.b_c = self.config.SHORTAGE_COST_MULTIPLIER # backorder costs
        self.h_c = self.config.INVENTORY_COST_MULTIPLIER # inventory costs 
        self.h_r = self.config.BUFFER_STOCK_REWARD # holding costs
        self.tlt_min = self.config.MATERIAL_LEAD_TIME_MIN # max material lead times 
        self.tlt_max = self.config.MATERIAL_LEAD_TIME_MAX # max material lead times 
        
        #* global disruption vars 
        self.is_in_disruption = False
        self.disruption_duration = 0
        self.cooldown_period = 0
        
        self.reset()
        
    def seed(self, seed=None):
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        
    def random_number(self, a, b):
        return self.np_random.integers(a, b+1)

    def reset(self, **kwargs):
        """reset the environment to an initial state

        Returns:
            state, info (list, dict): state variable 
        """        
        # for normalization training store running parameters after each episode
        if self.normalized and self.training: 
            with open(os.path.join('runtime', 'standardization_parameters_v.txt'), mode = 'w') as f: 
                f.write(repr(self.state_running_mean))
                f.write('\n---------------------------\n')
                f.write(repr(self.state_running_std))
        #* reset environment vector start inventory
        self.start_inventory = np.zeros((8, 3), dtype=int) # start inventory 
        self.s = np.array(self.config.BASE_STOCK_LEVELS) # desired inventory (base stock levels)
        self.start_inventory[0] = np.array(self.s) # inventory levels = base stock levels
        
        # seed randoms for deterministic resets 
        if self.deterministic_reset:
            self.seed(seed=self.env_seed)
            
        #* reset current environment counters
        self.current_step = 0
        self.disruption_counter = 0
        self.bs_orders = np.zeros((4, ))
        
        # randomly choose if disruptive environment 
        self.disruptive = self.np_random.random() < self.config.DISRUPTION_PROBABILITY_TOTAL
        # store customer demand history 
        self.D = np.zeros(self.config.MAX_STEPS)
        # update state
        self.update_state()
        
        # Return the initial state and an empty info dictionary
        return self.state, {}
    
    def step(self, action = None):
        """take a step in the environment

        Args:
            action (List[int], optional): Actions to take, BS or STRM calculated in place. Defaults to None.

        Returns:
            state: current state
            reward: step reward
            done: indicates if episode is done
            truncated: indicates if episode is truncated
            info: dictionary with addtional evaluation data
        """        
        n = self.current_step # get current step
        
        # initialize placeholders 
        transportation_lead_times = np.array([0, 0, 0])
        incoming_orders = np.array([0, 0, 0, 0]) 
        outgoing_shipments = np.array([0, 0, 0])

        # extract incomming shipments and inventory levels
        incoming_shipments = [self.start_inventory[1][i] for i in range(3)]
        inventory_level = [self.start_inventory[0][i] for i in range(3)]
        arr_size = 8
        
        #! simulate disruption 
        self.handle_disruption()

        #* update shipments and orders in the supply chain
        for i in range(3):
            for j in range(arr_size):
                if j == 0:
                    # shipments arrive
                    self.start_inventory[j][i] += incoming_shipments[i]
                else:
                    # all shipments in the supply chain are updated by 1 time period
                    if j < arr_size - 1:
                        self.start_inventory[j][i] = self.start_inventory[j + 1][i]
                if j == arr_size - 1:
                    self.start_inventory[j][i] = 0

        #* process orders and shipments
        for i in range(3):
            if i == 0:
                # sample external customer demand
                incoming_orders[i] = self.simulate_demand()
                self.D[n] = incoming_orders[i]


            if inventory_level[i] > 0:
                if incoming_orders[i] < incoming_shipments[i] + inventory_level[i]:     # [cf. CASE 1 in the paper of preil and krapp 2022]
                    outgoing_shipments[i] = incoming_orders[i]
                else:                                                                   # [cf. CASE 2 in the paper of preil and krapp 2022]
                    outgoing_shipments[i] = inventory_level[i] + incoming_shipments[i]  
            else:
                if incoming_orders[i] < incoming_shipments[i] + inventory_level[i]:     # [cf. CASE 3 in the paper of preil and krapp 2022], NOTE [-(inventory_level)] is positive! 
                    outgoing_shipments[i] = incoming_orders[i] - inventory_level[i]
                else:                                                                   # [cf. CASE 4 in the paper of preil and krapp 2022]
                    outgoing_shipments[i] = incoming_shipments[i]

            #* PPO order policy 
            if self.config.DRL: 
                if self.current_step > 0: 
                    action[i] = action[i] + self.D[self.current_step-1]
                incoming_orders[i + 1] = max(action[i], 0)
            #* BS order policy 
            elif self.config.BS:                                            
                incoming_orders[i + 1] = self.bs_orders[i] 
                self.bs_orders[i] = incoming_orders[i]
            #* STRM order policy 
            elif self.config.STRM: 
                alpha = 1 
                beta = 1
                IL = self.start_inventory[0][i]
                OO = sum([self.start_inventory[j][i] for j in range(1, len(self.start_inventory))])
                desired_IL = self.config.BASE_STOCK_LEVELS[i]
                desired_OO = 0
                demand_forecast = self._calculate_demand_average()
                incoming_orders[i + 1] = max(demand_forecast + alpha * ((desired_IL + beta * desired_OO) - IL - beta * OO), 0)


        #* updating the inventory levels, i.e. subtracting the demand(s) in the current time period
        for i in range(3):
            self.start_inventory[0][i] -= incoming_orders[i]

        #* shipments are placed in transit
        for i in range(3):
            if self.tlt_min[i] == self.tlt_max[i]: 
                transportation_lead_times[i] = self.tlt_min[i]  
            else: 
                transportation_lead_times[i] = self.random_number(self.tlt_min[i], self.tlt_max[i])

            if i <= 1:
                # shipment that is placed in transit by agent i+1 for agent i. It will arrive with a delay of transportation_lead_times[i]
                self.start_inventory[transportation_lead_times[i]][i] += outgoing_shipments[i + 1]
            else:
                # shipment that is placed in transit by the external source for the most usptream agent, the supplier. It will arrive with a delay of transportation_lead_times[i]
                self.start_inventory[transportation_lead_times[i]][i] += incoming_orders[i + 1]
        
        #* calculate reward                   
        cost = self.calc_cost()
        reward = self.calc_reward()
        
        #* update state and info 
        self.update_state()
        info = {
            'inventory_levels': self.start_inventory[0].copy(), 
            'inventory_position': self._calculate_inventory_position(),
            'orders_placed': incoming_orders[1:].copy(),
            'arriving_shipment': incoming_shipments[i],
            'on_order_inventory': [sum([self.start_inventory[j][i] for j in range(1, len(self.start_inventory))]) for i in range(3)],
            'demand': self.D[n], 
            'shortages': np.minimum(self.start_inventory[0].copy(), 0),
            'overstock': np.maximum(self.start_inventory[0].copy(), 0),
            'cost': cost,
            'reward': reward,
            'disruption': self.is_in_disruption, 
            'step': self.current_step
        }
        
        #* increase counters 
        self.current_step += 1
        self.total_steps += 1
        done = self.is_done()
        truncated = self.is_truncated()
        
        return self.state, np.sum(reward), done, truncated, info

    def calc_cost(self):
        """calculate the cost (negative) based on the current state and ection

        Returns:
            cost (float): current cost
        """        
        total_cost = np.zeros(3)
        for i in range(3):
            inventory = self.start_inventory[0][i]
            #* holding cost
            if inventory > 0:
                total_cost[i] += inventory * self.h_c[i]
            #* shortage cost
            else:
                total_cost[i] -= inventory * self.b_c[i]  # inventory is negative for backorders
        return -total_cost
    
    def calc_reward(self):
        """ calculate the reward (negative) based on the current state and action

        Returns:
            reward (float): current reward
        """               
        n = self.current_step
        e = self.config.NUM_ECHELONS
        total_reward = np.zeros(3)
        for i in range(3):
            inventory = self.start_inventory[0][i]
            #* holding cost
            if inventory > 0:
                total_reward[i] += inventory * self.h_c[i]
                #* buffer stock reward 
                if self.is_in_disruption:
                    if inventory > self.D[n]: # if available inventor exceeds incomming customer demand
                        total_reward[i] -= self.h_r[i] * (inventory - self.D[n]) # add reward
            #* shortage cost
            else:
                # update shortage cost multiplier for disruption
                shortage_cost = (self.disruption_peak + 1.0) * self.b_c[i] if self.is_in_disruption else self.b_c[i]
                total_reward[i] -= inventory * shortage_cost  # inventory is negative for backorders
        return -total_reward
    
    def update_state(self):
        """
        update the state based on the current environment parameters
        """                
        e = self.config.NUM_ECHELONS
        n = self.current_step
        t = self.total_steps
        m = self.config.m
        
        state = np.zeros(4*e + self.config.m, dtype=np.int32)
        
        #* add inventory position
        state[:e] = self._calculate_inventory_position()
        #* add inventory level
        state[e:2*e] = self.start_inventory[0].copy()
        #* add on order inventroy 
        state[2*e:3*e] = [sum([self.start_inventory[j][i] for j in range(1, len(self.start_inventory))]) for i in range(3)]
        #* add backorders
        state[3*e:4*e] = np.minimum(self.start_inventory[0].copy(), 0)
        #* add demand history
        if n < m: # if not enough observations only add available
            state[4*e:4*e+n] = self.D[:n]
        else: # else add all
            state[4*e:] = self.D[n-m:n]
        
        #* standardization 
        if self.normalized:
            means = self.state_running_mean.copy() # get running values
            stds = self.state_running_std.copy()
            # check if values are recalculated (training)
            if self.training: 
                M = self.state_running_M.copy() # get m falue
                # for every element of state ... 
                for index, s in enumerate(state):
                    # ... compute updated aggreagate for the new value
                    aggregate = welford_update((t, means[index], M[index]), s)
                    M[index] = aggregate[2] # ... update m2
                    means[index], stds[index] = welford_finalize(aggregate) # ... update mean, stdv
                    # ... standardize the new state value
                    if stds[index] != 0: 
                        state[index] = (s - means[index]) / stds[index]
                    else: 
                        state[index] = (s - means[index])
                # set updated running values
                self.state_running_mean = means
                self.state_running_M = M 
                self.state_running_std = stds
            else: 
                # for non training (testing) just normalize the state values by already computed mean
                for index, s in enumerate(state):
                    if stds[index] != 0: 
                        state[index] = (s - means[index]) / stds[index]
                    else: 
                        state[index] = (s - means[index]) 
        
        #* set updated state 
        self.state = state.copy()
    
    
    
    def simulate_demand(self):
        """function to simulate customer demand based on distribution and current disruption state

        Returns:
            cusomter demand (int): external customer demand at retailer echelon
        """    
        # set disruption multiplier based on disruptive state    
        if self.is_in_disruption:
            disruption_multiplier = 1 + (self.disruption_peak * self.simulate_disruption_demand()) 
            
        else: 
            disruption_multiplier = 1.0 
        # check underlying demand distribution and adjust according to paper
        if self.config.DEMAND_NORMAL: 
            return max(self.np_random.normal(self.config.DEMAND_MEAN * disruption_multiplier, self.config.DEMAND_STD_DEV), 0) 
        else: 
            demand_max_new = m.ceil(self.config.DEMAND_MAX * disruption_multiplier)
            demand_max_delta =  demand_max_new - self.config.DEMAND_MAX
            return self.np_random.uniform(self.config.DEMAND_MIN + demand_max_delta, demand_max_new) 

    def simulate_disruption_demand(self): 
        """model disrutption demand demand multiplier 

        Returns:
            DM (float): disruption demand multiplier based on current step of disruption
        """        
        # check if disruption is still building up (before reaching reflection)  
        if (self.total_disruption_duration - self.left_disruption_duration) < (self.config.DISRUPTION_INFLECTION * self.total_disruption_duration): 
            return (self.total_disruption_duration - self.left_disruption_duration) / (self.config.DISRUPTION_INFLECTION * self.total_disruption_duration)
        else: 
            return self.left_disruption_duration / (self.total_disruption_duration - (self.config.DISRUPTION_INFLECTION * self.total_disruption_duration))

    def handle_disruption(self):
        """
        handle disruption behaviour: count down ongoing disruptions, start new disruptions
        """        
        if self.is_in_disruption:
            # count down the disruption duration
            self.left_disruption_duration -= 1
            if self.left_disruption_duration <= 0:
                self.is_in_disruption = False
        elif (self.config.DISRUPTION_START_MIN <= self.current_step) and (self.current_step <= self.config.DISRUPTION_START_MAX) and (self.disruption_counter < self.config.MAX_DISRUPTION_COUNT):
            # randomly check if a new disruption should start
            if self.np_random.random() < self.config.DISRUPTION_PROBABILITY:
                # start a new disruption if not in cooldown
                self.is_in_disruption = True
                self.disruption_counter = self.disruption_counter + 1
                # randomly choose max disruption multiplier
                self.disruption_peak = self.np_random.uniform(self.config.DISRUPTION_PEAK_MIN, self.config.DISRUPTION_PEAK_MAX)
                # randomly choose disruption duration
                self.total_disruption_duration = float(self.np_random.uniform(self.config.MIN_DISRUPTION_DURATION, self.config.MAX_DISRUPTION_DURATION))
                self.left_disruption_duration = self.total_disruption_duration

    def is_done(self):
        """
        determine if the episode is over or needs to be aborted
        """
        # End after a fixed number of time steps
        if self.current_step >= self.config.MAX_STEPS:
            return True
        return False

    def is_truncated(self):
        """
        determine if the episode is over
        """
        # End after a fixed number of time steps
        if self.current_step >= self.config.MAX_STEPS:
            return True
        return False

        
    def _calculate_inventory_position(self):
        """get current state of the system: inventory position at each echelon except supplier

        Returns:
            IP (List[int]): inventory position for each echelon
        """        
        n = self.current_step
        e = self.config.NUM_ECHELONS
        IP = np.array([0, 0, 0])
        start_inventory = self.start_inventory.copy()
        for i in range(3): 
            IL = start_inventory[0][i] # inventory level
            OO = sum([start_inventory[j][i] for j in range(1, len(start_inventory))]) # outstanding orders 
            IP[i] = IL + OO # inventory position
            
        return IP
    
    def _calculate_demand_average(self):
        """calculate the running average of the demand history

        Returns:
            demand_history (float): running average of demand (or simulated demand if no average exists)
        """        
        n = self.current_step
        m = self.config.m
        D = self.D.copy()
        
        if n >= m: 
            # enough evaluation periods
            return D[n-m:n].mean() # use last m demands
        elif n == 0: 
            # no evaluation period
            return self.simulate_demand() # simulate random demand
        else: 
            # not all evaluation periods 
            return D[:n].mean() # use available
        
    def set_policy(self, policy):
        """set environment step policy

        Args:
            policy (str): Policy to use (BS, STRM, DRL or PPO)
        """        
        if policy == 'BS': # BS policy
            self.config.BS = True
            self.config.STRM = False
            self.config.DRL = False 
        elif policy == 'STRM': # STRM policy
            self.config.BS = False
            self.config.STRM = True
            self.config.DRL = False 
        elif policy in ['DRL', 'PPO']: # RSPPO policy
            self.config.BS = False
            self.config.STRM = False
            self.config.DRL = True 