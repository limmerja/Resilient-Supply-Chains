class Config:
    NUM_ECHELONS = 3 # number of echelons in sc environment 
    DRL = False
    BS = False
    STRM = False
    
    #* external customer demand distribution
    DEMAND_NORMAL = True # indicates normal distribution of customer demand
    DEMAND_MEAN = 10 # mean of customer demand distribution (normal)
    DEMAND_STD_DEV = 2 # standard deviation of customer demand distribution (normal)
    DEMAND_MIN = 0 # minimum of customer demand distribution (uniform)
    DEMAND_MAX = 2 + 1 # maximum of customer demand distribution (uniform)
    
    #* lead times
    MATERIAL_LEAD_TIME_MIN = [1, 2, 3] # [retailer, distributor, manufacturer]
    MATERIAL_LEAD_TIME_MAX = [3, 4, 5] # [retailer, distributor, manufacturer]
    
    #* ppo action space
    MAX_ORDER_SIZE = 40 # maximum action
    MIN_ORDER_SIZE = -25 # minimum aciton 
    
    #* optimal base stock levels
    BASE_STOCK_LEVELS = [35, 47, 72]
    
    #* cost multipliers 
    INVENTORY_COST_MULTIPLIER = [8, 4, 1] # [retailer, distributor, manufacturer]
    SHORTAGE_COST_MULTIPLIER = [24, 12, 3] # [retailer, distributor, manufacturer]
    BUFFER_STOCK_REWARD = [1, 2, 4] # [retailer, distributor, manufacturer]
    
    #* episode settings 
    MAX_STEPS = 200  # maximum steps in an episode
    m = 5  # historical demand window
    
    #* disruption settings
    MAX_DISRUPTION_COUNT = 1 # maximum number of disruptions per episode
    DISRUPTION_PROBABILITY_TOTAL = 0.5 # probability of episode to be disrupted
    DISRUPTION_PROBABILITY = 0.1 # probability of disruption at each step
    DISRUPTION_START_MIN = MAX_STEPS * 0.40 # earliest disruption start date
    DISRUPTION_START_MAX =  MAX_STEPS * 0.60 # latest disruption start date
    DISRUPTION_PEAK_MAX = 1.0 # maximum disruption multiplier peak, NOTE: 1.0 equals 100% increase
    DISRUPTION_PEAK_MIN = 0.6 # maximum disruption multiplier peak, NOTE: 1.0 equals 100% increase
    DISRUPTION_INFLECTION = 0.3 # time of peak reach
    MIN_DISRUPTION_DURATION = 15  # Minimum number of steps for a disruption
    MAX_DISRUPTION_DURATION = 20  # Maximum number of steps for a disruption
    
    #* evaluation settings 
    EVALUATION_START = 5 # start time of evaluation period (allows sc to get in running state)
    EVALUATION_SEED = 1 # evaluation seed 
    EVALUATION_MODEL = 'ppo_supplychain_v042' # ppo evaluation model name (must be stored in runtime)
    
    #* ppo training settings
    INITIAL_LR = 3e-4 # Initial Learning Rate 3e-4 PPO Basic
    DECAY_STEPS = 3000000  # Decay every x  steps
    DECAY_RATE = 0.50  # Reduce to half the previous value
