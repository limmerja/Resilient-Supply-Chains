import datetime
import os
from typing import Tuple, List
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import pandas as pd 
from os import path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def welford_update(existing_aggregate, new_value):
    """for a new value new_value, compute the new count, new mean, the new M2.

    Args:
        existing_aggregate (tuple): tuple of count, mean, m2
            count: aggregates the number of samples seen so far
            mean: accumulates the mean of the entire dataset
            m2 : aggregates the squared distance from the mean
        new_value (float): new observation

    Returns:
        (count, mean, M2) (tuple): new count, new mean, the new m2 for the new value
    """    
    (count, mean, M2) = existing_aggregate
    count += 1 # update count 
    delta = new_value - mean
    mean += delta / count # update mean 
    delta2 = new_value - mean
    M2 += delta * delta2 # update m2
    return (count, mean, M2)

def welford_finalize(existing_aggregate):
    """retrieve the mean and standard deviation from an aggregate

    Args:
        existing_aggregate (tuple): tuple of count, mean, m2
            count: aggregates the number of samples seen so far
            mean: accumulates the mean of the entire dataset
            m2 : aggregates the squared distance from the mean

    Returns:
        (mean, std) (tuple): mean and std deviation from the current aggregate
    """    
    (count, mean, M2) = existing_aggregate
    if count < 2:
        # not enoug historic aggregates for std calculation
        return (mean, 0)
    else:
        (mean, std) = (mean, (M2 / count)**0.5) # extract mean, compute std
        return (mean, std)

def set_size(width, fraction=1.0, subplots=(1, 1)):
    """set figure dimensions to avoid scaling in LaTeX

    Args:
        width (float or string):  document width in points, or string of predined document type
        fraction (float, optional): fraction of the width which you wish the figure to occupy. Defaults to 1.0.
        subplots (tuple, optional): the number of rows and columns of subplots. Defaults to (1, 1).

    Returns:
        tuple:  dimensions of figure in inches
    """    
    if width == 'thesis': # adjusted for thesis latex format
        width_pt = 446.70827
    else:
        assert isinstance(width, float)
        width_pt = width
    # width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # convert from pt to inches
    inches_per_pt = 1 / 72.27
    # golden ratio to set aesthetic figure height
    golden_ratio = (5**.5 - 1) / 2
    # figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)

def save_fig(fig, name=('_{date:%Y-%m-%d_%H:%M:%S}'.format(date=datetime.datetime.now()))): 
    fig.savefig(os.path.join('svg', name+'.pdf'), format='pdf', bbox_inches='tight')

def evaluate_drl_policy(policy, env):
    env.set_policy('DRL')
    obs, _ = env.reset()  # Reset the environment which should now behave deterministically
    done = False
    info_list = []  # List to store info dicts on each step

    while not done:
        action, _states = policy.predict(obs, deterministic=False)  # Get action from policy
        obs, reward, done, truncated, info = env.step(action)  # Take action in the environment
        info_list.append(info)  # Store info

    return info_list

def evaluate_bs_policy(env):
    env.set_policy('BS')
    obs, _ = env.reset()  # Reset the environment which should now behave deterministically
    done = False
    info_list = []  # List to store info dicts on each step

    while not done:
        obs, reward, done, truncated, info = env.step(None)  # Take action in the environment
        info_list.append(info)  # Store info

    return info_list

def evaluate_strm_policy(env, desired_OO = None, alpha = 1, beta = 1):
    env.set_policy('STRM') 
    obs, _ = env.reset()  # Reset the environment which should now behave deterministically
    done = False
    info_list = []  # List to store info dicts on each step

    while not done:
        #action = env.sterman_action(desired_OO = desired_OO, alpha = alpha, beta = beta)  # Get action from policy
        obs, reward, done, truncated, info = env.step(None)  # Take action in the environment
        info_list.append(info)  # Store info

    return info_list

def extract_info_dict(info_list, category): 
    # Extract the demands and inventory levels from the info_list
    if category in ['demand', 'step', 'disruption']: 
        return [info[category] for info in info_list]
    else: 
        warehouse = [info[category][0] for info in info_list] 
        distributor = [info[category][1] for info in info_list]
        retailer = [info[category][2] for info in info_list]
        return [warehouse, distributor, retailer]

def read_tensorboard(folder: str, base: str) -> pd.DataFrame:
    df = pd.DataFrame(columns=['step', 'value'])
    training_iterations = []

    file_df = pd.read_csv(path.join('data', folder, f"{base}.csv")).drop(columns=['Wall time'])
    file_df.columns = ['step', 'value']
    training_iterations.append(file_df.loc[len(file_df)-1, 'step'])

    df = pd.concat([df, file_df], axis=0)
    df.reset_index(drop = True, inplace=True)    

    return df
    
def read_tensorboard_multiple(folder: str, base: str, data: List[str]) -> Tuple[pd.DataFrame, List[int]]:
    
    df = pd.DataFrame(columns=['step', 'value'])
    training_iterations = []

    for file in data: 
        file_df = pd.read_csv(path.join('data', folder, f"{base}{file}.csv")).drop(columns=['Wall time'])
        file_df.columns = ['step', 'value']
        training_iterations.append(file_df.loc[len(file_df)-1, 'step'])

        df = pd.concat([df, file_df], axis=0)

    df.reset_index(drop = True, inplace=True)    

    return df, training_iterations[:-1]

def extract_disruption_duration(info): 
    disruption_info = extract_info_dict(info, 'disruption')
    disruptions = []
    current = False
    start = -99

    for index, disruption in enumerate(disruption_info): 
        if (not current) and disruption: 
            current = True
            start = index 
        elif current and (not disruption): 
            disruptions.append((start, index))
            current = False 
            start = -99
            
    if start != -99: 
        disruptions.append((start, len(disruption_info)))
    
    return disruptions

def extract_info_list(bs, strm, rsppo, type): 
    base_stock = extract_info_dict(bs, type)
    sterman = extract_info_dict(strm, type)
    rsppo = extract_info_dict(rsppo, type)
    return [base_stock, sterman, rsppo]
    # return [base_stock, rsppo]

def double_row_plot(info_list, disruptions, plot_data, savename = 'test', width = 'thesis'):
    labels = [r'Manufacturer', r'Warehouse', r'Retailer']
    info_labels = [r'BS', r'STRM', r'RS-PPO']
    # info_labels = [r'BS', r'RS-PPO']
    y_label = plot_data['y_labels']
    handles = []
    rows = len(info_list)
    cols = len(info_list[0][0])
    print(cols)

    fig, ax = plt.subplots(rows, cols, figsize=set_size(width))

    # Plot
    for row in range(rows): 
        infos = info_list[row]
        for i in range(cols): 
            for disruption in disruptions: 
                ax[row][i].axvspan(disruption[0], disruption[1], facecolor='k', alpha=0.2)
            for j in range(len(info_list[0])):
                handle, = ax[row][i].plot(range(len(infos[j][i])), infos[j][i], label = info_labels[j], linewidth = 0.5)
                ax[row][i].set_xlim(left = 0, right = len(infos[j][i])-1)
                if row == 0 and i == 0: 
                    handles.append(handle)
            
            
            
            if row == 1: 
                ax[row][i].set_xlabel(r'Time $t$')
                ax[row][i].set_ylim(plot_data['y_min_1'], plot_data['y_max_1'])
            if row == 0: 
                ax[row][i].set_xticklabels([])
                ax[row][i].title.set_text(labels[i])
                ax[row][i].set_ylim(plot_data['y_min_0'], plot_data['y_max_0'])
            if i == 0: 
                ax[row][i].set_ylabel(y_label[row])
            if i > 0: 
                ax[row][i].set_yticklabels([])
    
    disruption = mpatches.Patch(color='k', alpha= 0.1)
    handles.append(disruption)
    info_labels.append('Disruption')
                
    fig.legend(handles=handles, labels=info_labels, 
            loc='lower center', bbox_to_anchor=(0.542,-0.04), ncol=4)
        
    fig.tight_layout()

    save_fig(fig, savename)
    
def extract_evaluation_csv(info):
    step = extract_info_dict(info, 'step')
    demand = extract_info_dict(info, 'demand')
    arriving_shipment = extract_info_dict(info, 'arriving_shipment')
    ip = extract_info_dict(info, 'inventory_position')
    il = extract_info_dict(info, 'inventory_levels')
    op = extract_info_dict(info, 'orders_placed')
    bo = extract_info_dict(info, 'shortages')

    return pd.DataFrame({'step': step, 'demand': demand, 'disruption': extract_info_dict(info, 'disruption'), 
                'ip_1': ip[0], 'ip_2': ip[1], 'ip_3': ip[2], 
                'il_1': il[0], 'il_2': il[1], 'il_3': il[2], 
                'op_1': op[0], 'op_2': op[1], 'op_3': op[2], 
                'as_1': arriving_shipment[0], 'as_2': arriving_shipment[1], 'as_3': arriving_shipment[2], 
                'bo_1': bo[0], 'bo_1': bo[1], 'bo_3': bo[2]})
    

def extract_resilience_data(info_dict, echelon = -99):
    res = {'step': extract_info_dict(info_dict, 'step'), 
            'disruption': extract_info_dict(info_dict, 'disruption'), 
            'cost': extract_info_dict(info_dict, 'cost')}
    if echelon in [0, 1, 2]: 
        res['cost'] = res['cost'][echelon]
    else: 
        zipped_list = zip(res['cost'][0], res['cost'][1], res['cost'][2])
        res['cost'] = [sum(item) for item in zipped_list]
    df = pd.DataFrame(res)
    disruption_df = df.loc[df.disruption]

    start = disruption_df.index[0]
    end = disruption_df.index[len(disruption_df)-1] + 1

    t_0 = start
    t_e = -99
    t_d = disruption_df[['cost']].idxmin()[0]
    t_s = -99
    t_f = end
    t_max = end + 10

    t_f_updated = False
    for index in disruption_df.index.to_list(): 
        current_cost = df.at[index, 'cost']
        old_cost = df.at[index-1, 'cost']
        if t_e == -99 and current_cost < (old_cost-150): 
            t_e = index - 1
        elif t_s == -99 and index >= t_d and current_cost > old_cost: 
            t_s = index - 1 
        if t_e != -99 and index >= t_d and current_cost > (df.at[t_e, 'cost']-50) and not t_f_updated:
            t_f_updated = True
            t_f = index
    
    if t_s == -99: 
        t_s = t_d
    if t_e == -99: 
        t_e = t_0
     
    d_disruption = end - start
    d_absorb = t_d - t_e
    d_endure = t_s - t_d 
    d_recovery = t_f - t_s
        

    performance_impact = df.at[t_s, 'cost'] - df.at[t_e, 'cost']

    x = performance_impact / df.at[t_e, 'cost']

    summary = ((t_max - t_0) - ((x * d_recovery) / 2)) / (t_max - t_0)

    failure_rate = (df.at[t_e, 'cost'] - df.at[t_d, 'cost']) / (t_d - t_e)
    recovery_rate = (df.at[t_f, 'cost'] - df.at[t_s, 'cost']) / (t_f - t_s)

    output = {'t_0': t_0, 
            't_e': t_e, 
            't_d': t_d, 
            't_s': t_s, 
            't_f': t_f, 
            't_max': t_max, 
            'd_disruption': d_disruption, 
            'd_absorp': d_absorb, 
            'd_endure': d_endure, 
            'd_recovery': d_recovery, 
            'performance_impact': performance_impact, 
            'summary': summary,
            'failure_rate': failure_rate, 
            'recovery_rate': recovery_rate}

    return output, df

def create_resilience_table(res_info_0, res_df_0, res_info_1, res_df_1, name): 
    output_print = {'t_0': '$t_0$', 
                    't_e': '$t_e$', 
                    't_d': '$t_d$', 
                    't_s': '$t_s$', 
                    't_f': '$t_f$', 
                    't_max': '$t_{max}$', 
                    'd_disruption': '$d(disruption)$', 
                    'd_absorp': '$d(absorb)$', 
                    'd_endure': '$d(endure)$', 
                    'd_recovery': '$d(recovery)$', 
                    'performance_impact': '$performance \\; impact$', 
                    'summary': '$summary$',
                    'failure_rate': '$failure \\; rate$', 
                    'recovery_rate': '$recovery \\; rate$'}
    used_keys = ['performance_impact', 'summary', 'failure_rate', 'recovery_rate']
    additional_keys = {'t_0': '$p(t_0)$', 
                       't_s': '$p(t_s)$',
                       't_f': '$p(t_f)$', 
                       't_max': '$p(t_{max})$'}
    


    with open(path.join('svg', 'tables', ('resilience_' + name + '.txt')), 'w') as f:
        for key in additional_keys.keys():
            dif = round(res_df_1.loc[res_info_1[key], 'cost']*1.0 - res_df_0.loc[res_info_0[key], 'cost']*1.0, 1)
            arrow = r'$\nearrow$' if dif > 0 else r'$\searrow$'
            f.write(additional_keys[key] + '\n\t')
            f.write(' & ' + str(round(res_df_0.loc[res_info_0[key], 'cost']*1.0, 1)))
            f.write(' & ' + str(round(res_df_1.loc[res_info_1[key], 'cost']*1.0, 1)))
            f.write(' & ' + str(dif)+ arrow +'\\\\\n')
        for key in used_keys: 
            dif = round(res_info_1[key]*1.0 - res_info_0[key]*1.0, 1)
            arrow = r'$\nearrow$' if dif > 0 else r'$\searrow$'
            f.write(output_print[key] + '\n\t')
            f.write(' & ' + str(round(res_info_0[key]*1.0, 1)))
            f.write(' & ' + str(round(res_info_1[key]*1.0, 1)))
            f.write(' & ' + str(dif) + arrow + '\\\\\n')
    
def create_resilience_plot(res_df, res_info, ax, disruptions, plot_data):
    plot_df = res_df.loc[res_info['t_0']-10 : res_info['t_max']+10, : ]

    # fig, ax = plt.subplots(1, 1, figsize=set_size(width))

    ax.plot(plot_df.step, plot_df.cost)
    ax.axvspan(disruptions[0][0], disruptions[0][1], facecolor='k', alpha=0.1)
    ax.set_xlim(left = res_info['t_0']-10, right = res_info['t_max']+10)
    ax.set_ylim(plot_data['y_min'], plot_data['y_max'])

    ax.plot([res_info['t_e'], res_info['t_s']], [res_df.at[res_info['t_e'], 'cost'], res_df.at[res_info['t_s'], 'cost']])
    ax.plot([res_info['t_s'], res_info['t_f']], [res_df.at[res_info['t_s'], 'cost'], res_df.at[res_info['t_f'], 'cost']])

    x = res_info['t_e']
    y = res_df.at[res_info['t_e'], 'cost']
    # ax.axvline(x=x, color = 'k', linestyle = '--')
    ax.axhline(y=y, color = 'k', linestyle = '--')
    # ax.text(x=res_info['t_max']+10.5, y=y, s = f'$p(t_0)={y}$')
    ax.plot(x, y, "x:k")

    x = res_info['t_s']
    y = res_df.at[res_info['t_s'], 'cost']
    # ax.axvline(x=x, color = 'k', linestyle = '--')
    ax.axhline(y=y, color = 'k', linestyle = '--')
    # ax.text(x=res_info['t_max']+10.5, y=y-0.5, s = f'$p(t_s)={y}$')
    ax.plot(x, y, "x:k")

    x = res_info['t_f']
    y = res_df.at[res_info['t_f'], 'cost']
    # ax.axvline(x=x, color = 'k', linestyle = '--')
    ax.axhline(y=y, color = 'k', linestyle = '--')
    # ax.text(x=res_info['t_max']+10.5, y=y-1.5, s = f'$p(t_f)={y}$')
    ax.plot(x, y, "x:k")

    # x = res_info['t_max']
    # y = res_df.at[res_info['t_max'], 'cost']
    # # ax.axvline(x=x, color = 'k', linestyle = '--')
    # # ax.axhline(y=y, color = 'k', linestyle = '--')
    # # ax.text(x=res_info['t_max']+10.5, y=y-0.5, s = r'$p(t_{max})$'+f'$={y}$')
    # ax.plot(x, y, "x:k")

    #save_fig(fig, path.join('resilience', f'{name}_resilience'))
    
def create_resilience_plot_complete_deprecated(info_dict, disruptions, plot_data, name, width = 'thesis'):
    size = set_size(width) 
    size = (size[0], size[1] * 1.5)
    
    fig, axs = plt.subplots(4, 1, figsize=size)

    titles = ['Complete SC', 'Manufacturer', 'Warehouse', 'Retailer']

    for i, ax in enumerate(axs): 
        res_info, res_df = extract_resilience_data(info_dict, i-1)
        create_resilience_plot(res_df, res_info, ax, disruptions, plot_data)
        ax.title.set_text(titles[i])
        ax.set_ylabel(r'Cost $c$')
        if i < 3: 
            ax.set_xticklabels([])
        if i >= 3: 
            ax.set_xlabel(r'Time $t$')

    # manually define a new patch 
    cost = Line2D([0], [0], label='Cost', color='blue')
    failure = Line2D([0], [0], label='Failure Rate', color='red')
    recovery = Line2D([0], [0], label='Recovery Rate', color='green')
    disruption = mpatches.Patch(color='k', alpha= 0.1, label='Disruption')


    handles = [cost, failure, recovery, disruption]

    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.542,-0.04), ncol=4)

    fig.tight_layout()

    save_fig(fig, path.join('resilience', f'{name}_resilience'))
    
def create_resilience_plot_complete(info_dict, disruptions, plot_data, name, width = 'thesis'):
    fig = plt.figure(figsize=set_size(width))

    gs = GridSpec(2, 3, figure=fig)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[-1,0])
    ax3 = fig.add_subplot(gs[-1,-2])
    ax4 = fig.add_subplot(gs[-1,-1])
    axs = [ax1, ax2, ax3, ax4]

    titles = ['Complete SC', 'Manufacturer', 'Warehouse', 'Retailer']

    for i, ax in enumerate(axs): 
        res_info, res_df = extract_resilience_data(info_dict, i-1)
        if i < 1: 
            create_resilience_plot(res_df, res_info, ax, disruptions, plot_data)
        else: 
            create_resilience_plot(res_df, res_info, ax, disruptions, {'y_min': plot_data['y_min_1'], 'y_max': plot_data['y_max_1']})
        ax.title.set_text(titles[i])
        if i < 2: 
            ax.set_ylabel(r'Cost $c$')
        if i >= 2: 
            ax.set_yticklabels([])
        ax.set_xlabel(r'Time $t$')

    # manually define a new patch 
    cost = Line2D([0], [0], label='Cost', color='blue')
    failure = Line2D([0], [0], label='Failure Rate', color='red')
    recovery = Line2D([0], [0], label='Recovery Rate', color='green')
    disruption = mpatches.Patch(color='k', alpha= 0.1, label='Disruption')


    handles = [cost, failure, recovery, disruption]

    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.542,-0.04), ncol=4)

    fig.tight_layout()

    save_fig(fig, path.join('resilience', f'{name}_resilience'))

