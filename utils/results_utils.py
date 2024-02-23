from tqdm import tqdm
import ast
import pandas as pd
import numpy as np
from scipy import stats
import shelve
import glob
import torch
import random
import os, sys
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(SCRIPT_DIR)
sys.path.append('./libs_m/ct/')
from src.data.pkpd.pkpd_simulation import MAX_VALUE as EQ_4_MAX_VALUE

method_name_map = {     'sindy': 'A-SINDy',
                        'wsindy': 'A-WSINDy',
                        'te-cde': 'TE-CDE',
                        'insite': r'\bf INSITE',
                        'crn': 'CRN',
                        'msm': 'MSM',
                        'gnet': 'G-Net',
                        'rmsn': 'RMSN',
                        'ct': 'CT',
                        'edct': 'EDCT'}


dataset_name_ordering = {'EQ_4_A': 0,
                        'EQ_4_B': 1,
                        'EQ_4_C': 2,
                        'EQ_4_D': 3,
                        'EQ_5_A': 4,
                        'EQ_5_B': 5,
                        'EQ_5_C': 6,
                        'EQ_5_D': 7}

method_name_ordering = {
                    'msm': 0,
                    'rmsn': 1,
                    'crn': 2,
                    'gnet': 3,
                    'te-cde': 4,
                    'ct': 5,
                    'edct': 6,
                    'sindy': 7,
                    'wsindy': 8,
                    'insite': 9,
                    }

STEP_AHEAD_NAME_MAP = {'encoder_test_rmse_orig': 1,
                        'decoder_test_rmse_2-step': 2,
                        'decoder_test_rmse_3-step': 3,
                        'decoder_test_rmse_4-step': 4,
                        'decoder_test_rmse_5-step': 5,
                        'decoder_test_rmse_6-step': 6}

def load_metrics_from_tensorboard_file(log_path, metrics=["charts/episodic_return"]):
    """
    log_path: path to the tensorboard log file
    metrics: list of metrics to load: Possible metrics: ["charts/episodic_return", "charts/episodic_length", "charts/epsilon", "losses/td_loss", "losses/q_values", "charts/SPS"]
    """
    from tensorflow.python.summary.summary_iterator import summary_iterator
    data = {}
    for metric in metrics:
        data[metric] = []
    # for summary in tqdm(summary_iterator(log_path)):
    for summary in summary_iterator(log_path):
        if summary.HasField('summary'):
            if summary.summary.value[0].tag in metrics:
                data[summary.summary.value[0].tag].append((summary.step, summary.summary.value[0].simple_value))
    return data
    
def file_path_from_parent_directory(parent_dir):
    files = glob.glob(parent_dir + '/*')
    return files[-1]

def moving_average(x, N):
    return np.convolve(x, np.ones(N)/N, mode='valid')

def ci(data, confidence=0.95, axis=0):
    # https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
    a = 1.0 * np.array(data)
    n = a.shape[axis]
    m, se = np.mean(a, axis=axis), stats.sem(a, axis=axis)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return h

def configure_plotting_sn_params(sn, SCALE, HEIGHT_SCALE, use_autolayout=True):
    pd.set_option('mode.chained_assignment', None)
    sn.set(rc={'figure.figsize': (SCALE, int(HEIGHT_SCALE * SCALE)), 'figure.autolayout': use_autolayout, 'text.usetex': True, 
    'text.latex.preamble': '\n'.join([
       r'\usepackage{siunitx}',   # i need upright \micro symbols, but you need...
       r'\sisetup{detect-all}',   # ...this to force siunitx to actually use your fonts
       r'\usepackage{helvet}',    # set the normal font here
       r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
       r'\sansmath'               # <- tricky! -- gotta actually tell tex to use!
                    ])  
    })
    sn.set(font_scale=2.0)
    sn.set_style('white', {'font.family':'serif',
                            'font.serif':'Times New Roman',
                            "pdf.fonttype": 42,
                            "ps.fonttype": 42,
                            "font.size": 14})
    sn.color_palette("colorblind")
    return sn

def load_df(path, remove_extra_columns=True, load_from_cache=False):
    if load_from_cache:
        try:
            with shelve.open("logs") as db:
                df = db[path]
        except KeyError:
            df = df_from_log(path, remove_extra_columns=remove_extra_columns)
            with shelve.open("logs") as db:
                db[path] = df
    else:
        df = df_from_log(path, remove_extra_columns=remove_extra_columns)
    return df

def df_from_log(path, remove_extra_columns=True, load_tensorboard_data=True):
    with open(path) as f:
        lines = f.readlines()
    pd_l = []
    for line in tqdm(lines):
        if '[Exp evaluation complete] {' in line:
            result_dict = line.split('[Policy evaluation complete] ')[1].strip()
            result_dict = result_dict.replace('nan', '\'nan\'')
            result_dict = result_dict.replace('array', '')
            result_dict = ast.literal_eval(result_dict)
            # try:
            if load_tensorboard_data:
                if 'run_name' in result_dict:
                    run_name = result_dict['run_name']
                    log_path = file_path_from_parent_directory(f'./runs/{run_name}')
                    tensorboard_data = load_metrics_from_tensorboard_file(log_path)
                    result_dict['episodic_return_all'] = tensorboard_data['charts/episodic_return']
                    # result_dict['episodic_length_all'] = tensorboard_data['charts/episodic_length']
            pd_l.append(result_dict)
            # except:
            #     pass
    dfm = pd.DataFrame(pd_l)
    if remove_extra_columns:
        columns_to_remove_if_exist = ['costs_std_stats', 'planner', 'observed_times', 'observed_times_diff', 'costs_std_median', 's', 'a', 'r', 'cost_std_plot', 'ri', 'telem_file_path']
        current_columns = list(dfm.columns)
        columns_to_drop = set(columns_to_remove_if_exist) & set(current_columns)
        columns_to_drop = list(columns_to_drop)
        dfm = dfm.drop(columns=columns_to_drop)
    else:
        columns_to_np_arrays_if_exist = ['observed_times', 'observed_times_diff', 's', 'a', 'r', 'cost_std_plot', 'ri']
        current_columns = list(dfm.columns)
        columns_to_np_arrays = set(columns_to_np_arrays_if_exist) & set(current_columns)
        columns_to_np_arrays = list(columns_to_np_arrays)
        dfm[columns_to_np_arrays] = dfm[columns_to_np_arrays].applymap(np.array)
    # numeric_columns = ['roll_outs',
    #                     'time_steps',
    #                     'episode_elapsed_time',
    #                     'episode_elapsed_time_per_it',
    #                     'dt_sim',
    #                     'dt_plan',
    #                     'total_reward',
    #                     'state_reward',
    #                     'state_reward_std',
    #                     'observation_reward',
    #                     'observations_taken',
    #                     'observing_var_threshold',
    #                     'observing_cost',
    #                     'observation_noise',
    #                     'seed']
    # dfm[numeric_columns] = dfm[numeric_columns].apply(pd.to_numeric, errors='coerce')
    # dfm['name'] = dfm.model_name + '+' + dfm.sampling_policy
    return dfm

def normalize_means(df):
    df_means = df.groupby(['env_name', 'policy', 'network_specific']).agg(np.mean).reset_index()
    for env_name in df_means.env_name.unique():
        pass
        df_means_env = df_means[df_means.env_name == env_name]
        random_row = df_means_env[df_means_env.sampling_policy == 'random'].iloc[0]
        best_row = df_means_env[df_means_env.sampling_policy == 'continuous_planning'].iloc[0]

        df.loc[df.env_name==env_name, 'total_reward'] = ((df[df.env_name == env_name].total_reward - random_row.total_reward) / (best_row.total_reward - random_row.total_reward)) * 100.0
        df.loc[df.env_name==env_name, 'state_reward'] = ((df[df.env_name == env_name].state_reward - random_row.state_reward) / (best_row.state_reward - random_row.state_reward)) * 100.0
    return df

def remove_unneeded_columns(df):
    columns_to_remove_if_exist = ['errored', 'costs_std_stats', 'planner', 'observed_times', 'observed_times_diff', 'costs_std_median', 's', 'a', 'r', 'cost_std_plot', 'ri', 'telem_file_path']
    current_columns = list(df.columns)
    columns_to_drop = set(columns_to_remove_if_exist) & set(current_columns)
    columns_to_drop = list(columns_to_drop)
    df = df.drop(columns=columns_to_drop)
    return df

def compute_norm_metrics(df):
    cancer_norm = 1150
    single_eq_norm = 764
    dataset_name_norm_map = {'eq_1': single_eq_norm,
                            'eq_2': single_eq_norm,
                            'eq_3': single_eq_norm,
                            'eq_4': single_eq_norm,
                            'eq_5': cancer_norm,
                            'eq_6': cancer_norm,
                            'eq_7': cancer_norm,
                            'eq_8': cancer_norm,
                            'eq_9': cancer_norm}
    
    for dataset_name in df.dataset_name.unique():
        norm = dataset_name_norm_map[dataset_name]
        df.loc[df.dataset_name == dataset_name, 'test_rmse'] = df[df.dataset_name == dataset_name].test_rmse / norm
    return df

def generate_main_results_table_paper_format(df_results, wandb=None, use_95_ci=True, return_all_next_step_head_n_tables=True):
    # Process seeds here
    df_results = remove_unneeded_columns(df_results)
    if use_95_ci:
        df_out = df_results.groupby(['dataset_name', 'method_name']).agg([np.mean, ci]).reset_index()
        error_metric = 'ci'
    else:
        df_out = df_results.groupby(['dataset_name', 'method_name', 'model_name']).agg([np.mean, np.std]).reset_index()
        error_metric = 'std'

    # sf = 3
    sf = 2
    # EQ_4_NAME = r"Eq. \ref{eq:one-compartment-pkpd}"

    EQ_4_NAME = r"{\bf\cref{eq:one-compartment-pkpd}"
    EQ_5_NAME = r"{\bf\cref{eq:tumor}"
    dataset_name_map = {'EQ_4_A': f"{EQ_4_NAME}.A" + r"}",
                        'EQ_4_B': f"{EQ_4_NAME}.B" + r"}",
                        'EQ_4_C': f"{EQ_4_NAME}.C" + r"}",
                        'EQ_4_D': f"{EQ_4_NAME}.D" + r"}",
                        'EQ_5_A': f"{EQ_5_NAME}.A" + r"}",
                        'EQ_5_B': f"{EQ_5_NAME}.B" + r"}",
                        'EQ_5_C': f"{EQ_5_NAME}.C" + r"}",
                        'EQ_5_D': f"{EQ_5_NAME}.D" + r"}",
                        'cancer_sim': 'Cancer PKPD'}


    
                            #  'discrete_planning': 'Discrete Planning',
                            #  'continuous_planning': 'Continuous Planning',
                            #  'active_observing_control': r'\bf Active Sampling Control',
                            #  'random': 'Random'}
    if not return_all_next_step_head_n_tables:
        raise NotImplementedError
    
    next_step_ahead_metrics = [col for col in df_results.columns if 'decoder_test_rmse' in col]
    next_step_ahead_results = {}
    for next_step_ahead_metric in next_step_ahead_metrics:                                
        # Indivdual sorting
        # df_out = df_out.sort_values(by=['method_name'], key=lambda x: x.map(method_name_ordering))
        # df_out = df_out.sort_values(by=['dataset_name'], key=lambda x: x.map(dataset_name_ordering))

        # Grouped sorting
        df_out['dataset_name_order'] = df_out['dataset_name'].map(dataset_name_ordering)
        df_out['method_name_order'] = df_out['method_name'].map(method_name_ordering)
        df_out = df_out.sort_values(by=['dataset_name_order', 'method_name_order'])
        df_out = df_out.drop(columns=['dataset_name_order', 'method_name_order'])
        # df_out = df_out.sort_values(by=['dataset_name', 'method_name'], key=lambda k: (k[0].map(dataset_name_ordering), k[1].map(method_name_ordering)))
        table_lines = []
        line = r'\begin{tabularx}{\textwidth}{cr | *{' + f'{df_out.dataset_name.nunique()}' + r'}{X}}'
        table_lines.append(line)
        table_lines.append(r'\toprule')
        # table_lines.append(''.join([r'&  \multicolumn{3}{c|}{' + dataset_name_map[dataset_name] + '}' for dataset_name in df_out.dataset_name.unique()]) + r'\\')
        # table_lines.append(r'Method ' + r'& $\mathcal{U}$ & $\mathcal{R}$ & $\mathcal{O}$' * df_out.dataset_name.nunique() + r'\\' )
        table_lines.append(r'&{\bf Method}' + r'&' + '&'.join([dataset_name_map[dn] for dn in df_out.dataset_name.unique()]) + r'\\' )
        table_lines.append(r'\midrule')
        table_lines.append(r'\multirow{5}{*}{\rotatebox{90}{\bf LTE}}')
        set_next_row = False
        for method_name in df_out.method_name.unique():
            if method_name in ['sindy', 'wsindy', 'insite'] and not set_next_row:
                table_lines.append(r'\midrule')
                table_lines.append(r'\multirow{3}{*}{\rotatebox{90}{\bf ODE-D}}')
                set_next_row = True
            if method_name == 'insite':
                line = r'& \CC{black!5} INSITE'
            else:
                line = r'&' + method_name_map[method_name]
            for dataset_name in df_out.dataset_name.unique():
                row = df_out[(df_out.method_name == method_name) & (df_out.dataset_name == dataset_name)]
                if row[next_step_ahead_metric]['mean'].size == 0:
                    line += r'& NA'
                    continue
                mean = custom_format(row[next_step_ahead_metric]['mean'].iloc[0])
                error = custom_format(row[next_step_ahead_metric][error_metric].iloc[0])
                if method_name == 'insite':
                    line += r'& \CC{black!5} {\bf ' + mean + r'} ' + r'{\footnotesize $\pm$' + error + r'}'
                    # line += r'& \textbf{' + mean + r'$\pm$' + error + r'}'
                else:
                    line += r'&' + mean + r'{\footnotesize $\pm$' + error + r'}'
            line += r'\\'
            table_lines.append(line)
        table_lines.append(r'\bottomrule')
        table_lines.append(r'\end{tabularx}')
        table = '\n'.join(table_lines)
        next_step_ahead_results[next_step_ahead_metric] = table
        print('')
        print(f'Latex Table:: {next_step_ahead_metric}')
        print(table)
        print('')
    return next_step_ahead_results



def generate_main_results_table(df_results, wandb=None, use_95_ci=True, return_all_next_step_head_n_tables=True):
    # Process seeds here
    df_results = remove_unneeded_columns(df_results)
    if use_95_ci:
        df_out = df_results.groupby(['dataset_name', 'method_name']).agg([np.mean, ci]).reset_index()
        error_metric = 'ci'
    else:
        df_out = df_results.groupby(['dataset_name', 'method_name']).agg([np.mean, np.std]).reset_index()
        error_metric = 'std'

    # sf = 3
    sf = 2
    EQ_4_NAME = r"Eq. \ref{eq:one-compartment-pkpd}"
    EQ_5_NAME = r"Eq. \ref{eq:tumor}"
    dataset_name_map = {'EQ_4_A': f"{EQ_4_NAME}.A",
                        'EQ_4_B': f"{EQ_4_NAME}.B",
                        'EQ_4_C': f"{EQ_4_NAME}.C",
                        'EQ_4_D': f"{EQ_4_NAME}.D",
                        'EQ_5_A': f"{EQ_5_NAME}.A",
                        'EQ_5_B': f"{EQ_5_NAME}.B",
                        'EQ_5_C': f"{EQ_5_NAME}.C",
                        'EQ_5_D': f"{EQ_5_NAME}.D",
                        'cancer_sim': 'Cancer PKPD'}


    
                            #  'discrete_planning': 'Discrete Planning',
                            #  'continuous_planning': 'Continuous Planning',
                            #  'active_observing_control': r'\bf Active Sampling Control',
                            #  'random': 'Random'}
    if not return_all_next_step_head_n_tables:
        raise NotImplementedError
    
    next_step_ahead_metrics = [col for col in df_results.columns if 'decoder_test_rmse' in col]
    next_step_ahead_results = {}
    for next_step_ahead_metric in next_step_ahead_metrics:                                
        # Indivdual sorting
        # df_out = df_out.sort_values(by=['method_name'], key=lambda x: x.map(method_name_ordering))
        # df_out = df_out.sort_values(by=['dataset_name'], key=lambda x: x.map(dataset_name_ordering))

        # Grouped sorting
        df_out['dataset_name_order'] = df_out['dataset_name'].map(dataset_name_ordering)
        df_out['method_name_order'] = df_out['method_name'].map(method_name_ordering)
        df_out = df_out.sort_values(by=['dataset_name_order', 'method_name_order'])
        df_out = df_out.drop(columns=['dataset_name_order', 'method_name_order'])
        # df_out = df_out.sort_values(by=['dataset_name', 'method_name'], key=lambda k: (k[0].map(dataset_name_ordering), k[1].map(method_name_ordering)))
        table_lines = []
        line = r'\begin{tabular}{@{}l' + 'c' * df_out.dataset_name.nunique() + '}'
        table_lines.append(line)
        table_lines.append(r'\toprule')
        # table_lines.append(''.join([r'&  \multicolumn{3}{c|}{' + dataset_name_map[dataset_name] + '}' for dataset_name in df_out.dataset_name.unique()]) + r'\\')
        # table_lines.append(r'Method ' + r'& $\mathcal{U}$ & $\mathcal{R}$ & $\mathcal{O}$' * df_out.dataset_name.nunique() + r'\\' )
        table_lines.append(r'Method ' + r'&' + '&'.join([dataset_name_map[dn] for dn in df_out.dataset_name.unique()]) + r'\\' )
        table_lines.append(r'\midrule')
        for method_name in df_out.method_name.unique():
            line = method_name_map[method_name]
            for dataset_name in df_out.dataset_name.unique():
                row = df_out[(df_out.method_name == method_name) & (df_out.dataset_name == dataset_name)]
                if row[next_step_ahead_metric]['mean'].size == 0:
                    line += r'& NA'
                    continue
                mean = custom_format(row[next_step_ahead_metric]['mean'].iloc[0])
                error = custom_format(row[next_step_ahead_metric][error_metric].iloc[0])
                if method_name == 'insite':
                    line += r'& \textbf{' + mean + r'$\pm$' + error + r'}'
                else:
                    line += r'&' + mean + r'$\pm$' + error
            line += r'\\'
            table_lines.append(line)
        table_lines.append(r'\bottomrule')
        table_lines.append(r'\end{tabular}')
        table = '\n'.join(table_lines)
        next_step_ahead_results[next_step_ahead_metric] = table
        print('')
        print(f'Latex Table:: {next_step_ahead_metric}')
        print(table)
        print('')
    return next_step_ahead_results

def custom_format(number, threshold=1e-2):
    if abs(number) < threshold:
        if number == 0:
            return '0.00'
        else:
            return f"{number:.2e}"
    else:
        return f"{number:.2f}"

def generate_n_step_graph(df_results, wandb=None, use_95_ci=True):
    # Process seeds here
    df_results = remove_unneeded_columns(df_results)
    if use_95_ci:
        df_out = df_results.groupby(['dataset_name', 'method_name', 'gamma']).agg([np.mean, ci]).reset_index()
        error_metric = 'ci'
    else:
        df_out = df_results.groupby(['dataset_name', 'method_name', 'gamma']).agg([np.mean, np.std]).reset_index()
        error_metric = 'std'

    import matplotlib.pyplot as plt
    from matplotlib import cm
    import pandas as pd
    import seaborn as sn
    # plt.rcParams["font.family"] = "Times New Roman"
    SCALE = 8
    HEIGHT_SCALE = 0.5
    LEGEND_Y_CORD = 0.5  # * (HEIGHT_SCALE / 2.0)
    SUBPLOT_ADJUST = 1 / HEIGHT_SCALE  # -(0.05 + LEGEND_Y_CORD)
    LEGEND_X_CORD = 0.45
    PLOT_FROM_CACHE = False
    PLOT_SAFTEY_MARGIN = 1.25
    MODEL_NAME_MAP = {}
    sn = configure_plotting_sn_params(sn, SCALE, HEIGHT_SCALE)
    # plt.gcf().subplots_adjust(bottom=(1-1/HEIGHT_SCALE), left=0.15, top=0.99)
    plt.gcf().subplots_adjust(bottom=0.40, left=0.2) #, top=0.95)
    method_name_map = {'sindy': 'SINDY', 'te-cde': 'TE-CDE'}
                            #  'discrete_planning': 'Discrete Planning',
                            #  'continuous_planning': 'Continuous Planning',
                            #  'active_observing_control': r'\bf Active Sampling Control',
                            #  'random': 'Random'}

    y_metric = 'test_rmse'

    for dataset_name in df_out.dataset_name.unique():
        for method_name in df_out.method_name.unique():
            df = df_out[df_out.dataset_name == dataset_name]
            x = [1]
            # x = df[df.method_name == method_name]['gamma']
            y_mean = df[df.method_name == method_name][y_metric]['mean'].iloc[0]
            y_std = df[df.method_name == method_name][y_metric][error_metric].iloc[0]
            plt.plot(x, y_mean, '--o', label=method_name_map[method_name])
            plt.fill_between(x,y_mean - y_std,y_mean + y_std,alpha=0.25)

        # cp_y_mean = df_t[df_t.sampling_policy == 'continuous_planning'][y_metric]['mean'].iloc[0]
        # cp_y_mean = np.ones_like(y_mean) * cp_y_mean
        # cp_y_std = df_t[df_t.sampling_policy == 'continuous_planning'][y_metric][error_metric].iloc[0]
        # cp_y_std = np.ones_like(y_mean) * cp_y_std
        # plt.plot(x,cp_y_mean,'--o',label=sampling_policy_map['continuous_planning'])
        # plt.fill_between(x,cp_y_mean - cp_y_std,cp_y_mean + cp_y_std,alpha=0.25)
        plt.xlabel(r'$n$-step')
        plt.ylabel(r'Normalized RMSE')
        plt.yscale('log')
        # plt.xscale('log')
        # plt.axvline(x=threshold_we_used, color='r')

        # plt.legend(loc="lower center", bbox_to_anchor=(
        #             LEGEND_X_CORD, LEGEND_Y_CORD), ncol=1, fancybox=True, shadow=True)
        plt.legend(loc="upper right", ncol=1, fancybox=True, shadow=True)
        plt.tight_layout()                    
        plt.savefig(f'./results/n_step_{dataset_name}.png')
        plt.savefig(f'./results/n_step_{dataset_name}.pdf')
        print(f'./results/n_step_{dataset_name}.png')
        plt.clf()


def generate_n_step_graph(df_results, wandb=None, use_95_ci=True):
    df_results = df_results.drop(columns=['global_equation_string', 'fine_tuned', 'method'])
    # Process seeds here
    df_results = remove_unneeded_columns(df_results)
    if use_95_ci:
        df_out = df_results.groupby(['dataset_name', 'method_name', 'domain_conf']).agg([np.mean, ci]).reset_index()
        error_metric = 'ci'
    else:
        df_out = df_results.groupby(['dataset_name', 'method_name', 'domain_conf']).agg([np.mean, np.std]).reset_index()
        error_metric = 'std'

    # Grouped sorting
    df_out['dataset_name_order'] = df_out['dataset_name'].map(dataset_name_ordering)
    df_out['method_name_order'] = df_out['method_name'].map(method_name_ordering)
    df_out = df_out.sort_values(by=['dataset_name_order', 'method_name_order'])
    df_out = df_out.drop(columns=['dataset_name_order', 'method_name_order'])

    import matplotlib.pyplot as plt
    from matplotlib import cm
    import pandas as pd
    import seaborn as sn
    # plt.rcParams["font.family"] = "Times New Roman"
    SCALE = 10
    HEIGHT_SCALE =0.8
    LEGEND_Y_CORD = -1.2  # * (HEIGHT_SCALE / 2.0)
    SUBPLOT_ADJUST = 1 / HEIGHT_SCALE  # -(0.05 + LEGEND_Y_CORD)
    LEGEND_X_CORD = 0.45
    PLOT_FROM_CACHE = False
    PLOT_SAFTEY_MARGIN = 1.25
    MODEL_NAME_MAP = {}
    sn = configure_plotting_sn_params(sn, SCALE, HEIGHT_SCALE)
    # plt.gcf().subplots_adjust(bottom=(1-1/HEIGHT_SCALE), left=0.15, top=0.99)
    
    y_metric = 'test_rmse'

    # Calculate global y-axis limits
    global_min_y = np.inf
    global_max_y = -np.inf

    for dataset_name in df_out.dataset_name.unique():
        for decoder_test_step in list(set(l[0] for l in list(df_out.columns) if 'decoder_test_rmse_' in l[0])):
            for method_name in df_out.method_name.unique():
                df = df_out[df_out.dataset_name == dataset_name]
                df = df[df.method_name == method_name]
                y_mean = df[decoder_test_step]['mean']
                y_std = df[decoder_test_step][error_metric]
                global_min_y = min(global_min_y, (y_mean - y_std).min())
                global_max_y = max(global_max_y, (y_mean + y_std).max())

    global_min_y = 0.5
    global_max_y = 10

    steps = list(set(l[0] for l in list(df_out.columns) if 'decoder_test_rmse_' in l[0]))
    steps.append('encoder_test_rmse_orig')
    steps.sort(key=lambda x: STEP_AHEAD_NAME_MAP[x])

    # n-step graph first
    dataset_name = 'EQ_4_D'
    # for dataset_name in df_out.dataset_name.unique():
    plt.figure()
    plt.gcf().subplots_adjust(bottom=0.40, left=0.2) #, top=0.95)
    data_dict = {}
    # res_l = []
    # std_l = []
    for method_name in df_out.method_name.unique():
        x, y_mean, y_std = [], [], []
        data_dict[method_name] = {'res': [], 'std': []}
        for decoder_test_step in steps:
            x.append(STEP_AHEAD_NAME_MAP[decoder_test_step])
            df = df_out[df_out.dataset_name == dataset_name]
            df = df[df.method_name == method_name] 
            y_mean.append(df[decoder_test_step]['mean'].to_numpy()[0])
            y_std.append(df[decoder_test_step][error_metric].to_numpy()[0])
        x = np.array(x)
        y_mean = np.array(y_mean)
        y_std = np.array(y_std)
        plt.plot(x, y_mean, '--o', label=method_name_map[method_name])
        plt.fill_between(x,y_mean - y_std,y_mean + y_std, alpha=0.25)
        # plt.ylim(bottom=0)
        data_dict[method_name]['res'].append([(xi, yi) for xi, yi in zip(x, y_mean)])
        data_dict[method_name]['std'].append([(y_meani - y_stdi,y_meani + y_stdi) for y_meani, y_stdi in zip(y_mean, y_std)])

    # cp_y_mean = df_t[df_t.sampling_policy == 'continuous_planning'][y_metric]['mean'].iloc[0]
    # cp_y_mean = np.ones_like(y_mean) * cp_y_mean
    # cp_y_std = df_t[df_t.sampling_policy == 'continuous_planning'][y_metric][error_metric].iloc[0]
    # cp_y_std = np.ones_like(y_mean) * cp_y_std
    # plt.plot(x,cp_y_mean,'--o',label=sampling_policy_map['continuous_planning'])
    # plt.fill_between(x,cp_y_mean - cp_y_std,cp_y_mean + cp_y_std,alpha=0.25)
    plt.xlabel(r'$\tau$-step ahead prediction')
    plt.ylabel(r'RMSE (normalized)')
    plt.xticks(x)
    plt.yscale('log')
    # plt.ylim(global_min_y, global_max_y)
    # plt.xscale('log')
    # plt.axvline(x=threshold_we_used, color='r')

    plt.legend(loc="lower center", bbox_to_anchor=(
            LEGEND_X_CORD, LEGEND_Y_CORD), ncol=2, fancybox=True, shadow=True)                 
    plt.savefig(f'./results/domain_conf_{dataset_name}_n-step-ahead.png')
    plt.savefig(f'./results/domain_conf_{dataset_name}_n-step-ahead.pdf')
    print(f'./results/domain_conf_{dataset_name}_n-step-ahead.png')
    plt.clf()
    plt.close()
    print('')
    print(decoder_test_step)
    print(data_dict)
    # print(std_l)
    print('')

def generate_overlap_graph(df_results, wandb=None, use_95_ci=True):
    df_results = df_results.drop(columns=['global_equation_string', 'fine_tuned', 'method'])
    # Process seeds here
    df_results = remove_unneeded_columns(df_results)
    if use_95_ci:
        df_out = df_results.groupby(['dataset_name', 'method_name', 'domain_conf']).agg([np.mean, ci]).reset_index()
        error_metric = 'ci'
    else:
        df_out = df_results.groupby(['dataset_name', 'method_name', 'domain_conf']).agg([np.mean, np.std]).reset_index()
        error_metric = 'std'

    # Grouped sorting
    df_out['dataset_name_order'] = df_out['dataset_name'].map(dataset_name_ordering)
    df_out['method_name_order'] = df_out['method_name'].map(method_name_ordering)
    df_out = df_out.sort_values(by=['dataset_name_order', 'method_name_order'])
    df_out = df_out.drop(columns=['dataset_name_order', 'method_name_order'])

    import matplotlib.pyplot as plt
    from matplotlib import cm
    import pandas as pd
    import seaborn as sn
    # plt.rcParams["font.family"] = "Times New Roman"
    SCALE = 10
    HEIGHT_SCALE =0.8
    LEGEND_Y_CORD = -1.2  # * (HEIGHT_SCALE / 2.0)
    SUBPLOT_ADJUST = 1 / HEIGHT_SCALE  # -(0.05 + LEGEND_Y_CORD)
    LEGEND_X_CORD = 0.45
    PLOT_FROM_CACHE = False
    PLOT_SAFTEY_MARGIN = 1.25
    MODEL_NAME_MAP = {}
    sn = configure_plotting_sn_params(sn, SCALE, HEIGHT_SCALE)
    # plt.gcf().subplots_adjust(bottom=(1-1/HEIGHT_SCALE), left=0.15, top=0.99)
    
    y_metric = 'test_rmse'

    # Calculate global y-axis limits
    global_min_y = np.inf
    global_max_y = -np.inf

    for dataset_name in df_out.dataset_name.unique():
        for decoder_test_step in list(set(l[0] for l in list(df_out.columns) if 'decoder_test_rmse_' in l[0])):
            for method_name in df_out.method_name.unique():
                df = df_out[df_out.dataset_name == dataset_name]
                df = df[df.method_name == method_name]
                y_mean = df[decoder_test_step]['mean']
                y_std = df[decoder_test_step][error_metric]
                global_min_y = min(global_min_y, (y_mean - y_std).min())
                global_max_y = max(global_max_y, (y_mean + y_std).max())

    global_min_y = 0.5
    global_max_y = 10

    steps = list(set(l[0] for l in list(df_out.columns) if 'decoder_test_rmse_' in l[0]))
    steps.append('encoder_test_rmse_orig')
    steps.sort(key=lambda x: STEP_AHEAD_NAME_MAP[x])

    # n-step graph first
    domain_conf_to_plot_for_n_step_graph = 2
    for dataset_name in df_out.dataset_name.unique():
        plt.figure()
        plt.gcf().subplots_adjust(bottom=0.40, left=0.2) #, top=0.95)
        data_dict = {}
        # res_l = []
        # std_l = []
        for method_name in df_out.method_name.unique():
            x, y_mean, y_std = [], [], []
            data_dict[method_name] = {'res': [], 'std': []}
            for decoder_test_step in steps:
                x.append(STEP_AHEAD_NAME_MAP[decoder_test_step])
                df = df_out[df_out.dataset_name == dataset_name]
                df = df[df.method_name == method_name] 
                x_d = df['domain_conf']
                d_idx = np.where(np.array(x_d == 2))[0]
                y_mean.append(df[decoder_test_step]['mean'].to_numpy()[d_idx][0])
                y_std.append(df[decoder_test_step][error_metric].to_numpy()[d_idx][0])
            x = np.array(x)
            y_mean = np.array(y_mean)
            y_std = np.array(y_std)
            plt.plot(x, y_mean, '--o', label=method_name_map[method_name])
            plt.fill_between(x,y_mean - y_std,y_mean + y_std, alpha=0.25)
            # plt.ylim(bottom=0)
            data_dict[method_name]['res'].append([(xi, yi) for xi, yi in zip(x, y_mean)])
            data_dict[method_name]['std'].append([(y_meani - y_stdi,y_meani + y_stdi) for y_meani, y_stdi in zip(y_mean, y_std)])

        # cp_y_mean = df_t[df_t.sampling_policy == 'continuous_planning'][y_metric]['mean'].iloc[0]
        # cp_y_mean = np.ones_like(y_mean) * cp_y_mean
        # cp_y_std = df_t[df_t.sampling_policy == 'continuous_planning'][y_metric][error_metric].iloc[0]
        # cp_y_std = np.ones_like(y_mean) * cp_y_std
        # plt.plot(x,cp_y_mean,'--o',label=sampling_policy_map['continuous_planning'])
        # plt.fill_between(x,cp_y_mean - cp_y_std,cp_y_mean + cp_y_std,alpha=0.25)
        plt.xlabel(r'$\tau$-step ahead prediction')
        plt.ylabel(r'RMSE (normalized)')
        plt.xticks(x)
        plt.yscale('log')
        # plt.ylim(global_min_y, global_max_y)
        # plt.xscale('log')
        # plt.axvline(x=threshold_we_used, color='r')

        plt.legend(loc="lower center", bbox_to_anchor=(
                LEGEND_X_CORD, LEGEND_Y_CORD), ncol=2, fancybox=True, shadow=True)                 
        plt.savefig(f'./results/domain_conf_{dataset_name}_n-step-ahead.png')
        plt.savefig(f'./results/domain_conf_{dataset_name}_n-step-ahead.pdf')
        print(f'./results/domain_conf_{dataset_name}_n-step-ahead.png')
        plt.clf()
        plt.close()
        print('')
        print(decoder_test_step)
        print(data_dict)
        # print(std_l)
        print('')


    for dataset_name in df_out.dataset_name.unique():
        for decoder_test_step in steps:
            plt.figure()
            plt.gcf().subplots_adjust(bottom=0.40, left=0.2) #, top=0.95)
            data_dict = {}
            # res_l = []
            # std_l = []

            for method_name in df_out.method_name.unique():
                data_dict[method_name] = {'res': [], 'std': []}
                df = df_out[df_out.dataset_name == dataset_name]
                df = df[df.method_name == method_name] 
                x = df['domain_conf']
                y_mean = df[decoder_test_step]['mean']
                y_std = df[decoder_test_step][error_metric]
                plt.plot(x, y_mean, '--o', label=method_name_map[method_name])
                plt.fill_between(x,y_mean - y_std,y_mean + y_std, alpha=0.25)
                # plt.ylim(bottom=0)
                data_dict[method_name]['res'].append([(xi, yi) for xi, yi in zip(x, y_mean)])
                data_dict[method_name]['std'].append([(y_meani - y_stdi,y_meani + y_stdi) for y_meani, y_stdi in zip(y_mean, y_std)])

            # cp_y_mean = df_t[df_t.sampling_policy == 'continuous_planning'][y_metric]['mean'].iloc[0]
            # cp_y_mean = np.ones_like(y_mean) * cp_y_mean
            # cp_y_std = df_t[df_t.sampling_policy == 'continuous_planning'][y_metric][error_metric].iloc[0]
            # cp_y_std = np.ones_like(y_mean) * cp_y_std
            # plt.plot(x,cp_y_mean,'--o',label=sampling_policy_map['continuous_planning'])
            # plt.fill_between(x,cp_y_mean - cp_y_std,cp_y_mean + cp_y_std,alpha=0.25)
            plt.xlabel(r'Degree of time-dependent confounding $\gamma$')
            plt.ylabel(r'RMSE (normalized)')
            plt.xticks(x.values)
            plt.yscale('log')
            # plt.ylim(global_min_y, global_max_y)
            # plt.xscale('log')
            # plt.axvline(x=threshold_we_used, color='r')

            plt.legend(loc="lower center", bbox_to_anchor=(
                    LEGEND_X_CORD, LEGEND_Y_CORD), ncol=2, fancybox=True, shadow=True)                 
            plt.savefig(f'./results/domain_conf_{dataset_name}_{decoder_test_step}.png')
            plt.savefig(f'./results/domain_conf_{dataset_name}_{decoder_test_step}.pdf')
            print(f'./results/domain_conf_{dataset_name}_{decoder_test_step}.png')
            plt.clf()
            plt.close()
            print('')
            print(decoder_test_step)
            print(data_dict)
            # print(std_l)
            print('')


def plot_threshold_plots(df, use_95_ci=True):
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import pandas as pd
    import seaborn as sn
    # plt.rcParams["font.family"] = "Times New Roman"
    SCALE = 13
    HEIGHT_SCALE =0.8
    LEGEND_Y_CORD = -1.2  # * (HEIGHT_SCALE / 2.0)
    SUBPLOT_ADJUST = 1 / HEIGHT_SCALE  # -(0.05 + LEGEND_Y_CORD)
    LEGEND_X_CORD = 0.45
    PLOT_FROM_CACHE = False
    PLOT_SAFTEY_MARGIN = 1.25
    MODEL_NAME_MAP = {}

    if use_95_ci:
        error_metric = 'ci'
    else:
        error_metric = 'std'
    sn = configure_plotting_sn_params(sn, SCALE, HEIGHT_SCALE)
    # plt.gcf().subplots_adjust(bottom=0.40, left=0.2, top=0.95)

    method_name_map = {'discrete_monitoring': 'Discrete Monitoring',
                             'discrete_planning': 'Discrete Planning',
                             'continuous_planning': r'Continuous Planning $\mathcal{O}=13$',
                             'active_observing_control': r'Active Sampling Control',
                             'random': 'Random'}


    thresholds_used = {'oderl-cancer': 6.760299902695876,
                        'oderl-pendulum': 0.012269268,
                        'oderl-acrobot': 0.08927406,
                        'oderl-cartpole': 0.029934801}

    print('')
    x_metric = 'observing_var_threshold' # 'observing_var_threshold'
    plots_total = 3
    for env_name in df.env_name.unique():
        threshold_we_used = thresholds_used[env_name]
        df_t = df[df.env_name==env_name]
        ax = plt.subplot(plots_total, 1, 1)
        y_metric = 'total_reward'
        x = df_t[df_t.sampling_policy == 'active_observing_control'][x_metric]#['mean']
        y_mean = df_t[df_t.sampling_policy == 'active_observing_control'][y_metric]['mean']
        y_std = df_t[df_t.sampling_policy == 'active_observing_control'][y_metric][error_metric]
        plt.plot(x,y_mean,'--o',label=sampling_policy_map['active_observing_control'])
        plt.fill_between(x,y_mean - y_std,y_mean + y_std,alpha=0.25)
        cp_y_mean = df_t[df_t.sampling_policy == 'continuous_planning'][y_metric]['mean'].iloc[0]
        cp_y_mean = np.ones_like(y_mean) * cp_y_mean
        cp_y_std = df_t[df_t.sampling_policy == 'continuous_planning'][y_metric][error_metric].iloc[0]
        cp_y_std = np.ones_like(y_mean) * cp_y_std
        plt.plot(x,cp_y_mean,'--o',label=sampling_policy_map['continuous_planning'])
        plt.fill_between(x,cp_y_mean - cp_y_std,cp_y_mean + cp_y_std,alpha=0.25)
        plt.ylabel(r'$\mathcal{U}$')
        plt.axvline(x=threshold_we_used, color='r')

        ax = plt.subplot(plots_total, 1, 2, sharex=ax)
        y_metric = 'state_reward'
        x = df_t[df_t.sampling_policy == 'active_observing_control'][x_metric]#['mean']
        y_mean = df_t[df_t.sampling_policy == 'active_observing_control'][y_metric]['mean']
        y_std = df_t[df_t.sampling_policy == 'active_observing_control'][y_metric][error_metric]
        plt.plot(x,y_mean,'--o',label=sampling_policy_map['active_observing_control'])
        plt.fill_between(x,y_mean - y_std,y_mean + y_std,alpha=0.25)
        cp_y_mean = df_t[df_t.sampling_policy == 'continuous_planning'][y_metric]['mean'].iloc[0]
        cp_y_mean = np.ones_like(y_mean) * cp_y_mean
        cp_y_std = df_t[df_t.sampling_policy == 'continuous_planning'][y_metric][error_metric].iloc[0]
        cp_y_std = np.ones_like(y_mean) * cp_y_std
        plt.plot(x,cp_y_mean,'--o',label=sampling_policy_map['continuous_planning'])
        plt.fill_between(x,cp_y_mean - cp_y_std,cp_y_mean + cp_y_std,alpha=0.25)
        plt.ylabel(r'$\mathcal{R}$')
        plt.axvline(x=threshold_we_used, color='r')
        # ax2 = ax.twinx()
        ax = plt.subplot(plots_total, 1, 3, sharex=ax)
        y_metric = 'observation_reward'
        x = df_t[df_t.sampling_policy == 'active_observing_control'][x_metric]#['mean']
        y_mean = df_t[df_t.sampling_policy == 'active_observing_control'][y_metric]['mean']
        y_std = df_t[df_t.sampling_policy == 'active_observing_control'][y_metric][error_metric]
        plt.plot(x,y_mean,'--o',label=sampling_policy_map['active_observing_control'])
        plt.fill_between(x,y_mean - y_std,y_mean + y_std,alpha=0.25)
        cp_y_mean = df_t[df_t.sampling_policy == 'continuous_planning'][y_metric]['mean'].iloc[0]
        cp_y_mean = np.ones_like(y_mean) * cp_y_mean
        cp_y_std = df_t[df_t.sampling_policy == 'continuous_planning'][y_metric][error_metric].iloc[0]
        cp_y_std = np.ones_like(y_mean) * cp_y_std
        plt.plot(x,cp_y_mean,'--o',label=sampling_policy_map['continuous_planning'])
        plt.fill_between(x,cp_y_mean - cp_y_std,cp_y_mean + cp_y_std,alpha=0.25)
        plt.ylabel(r'$-\mathcal{C}$')
        plt.xlabel(r'Threshold $\tau$')
        plt.axvline(x=threshold_we_used, color='r')

        plt.legend(loc="lower center", bbox_to_anchor=(
                    LEGEND_X_CORD, LEGEND_Y_CORD), ncol=1, fancybox=True, shadow=True)
        # plt.tight_layout()                    
        plt.savefig(f'./plots/threshold_{env_name}.png')
        plt.savefig(f'./plots/threshold_{env_name}.pdf')
        print(f'./plots/threshold_{env_name}.png')
        plt.clf()
    print('')


# # https://stackoverflow.com/questions/42281844/what-is-the-mathematics-behind-the-smoothing-parameter-in-tensorboards-scalar#_=_
# def smooth(scalars, weight):  # Weight between 0 and 1
#     last = scalars[0]  # First value in the plot (first timestep)
#     smoothed = list()
#     for point in scalars:
#         smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
#         smoothed.append(smoothed_val)  # Save it
#         last = smoothed_val  # Anchor the last smoothed value

#     return smoothed


def load_df(path, remove_extra_columns=True):
    with open(path) as f:
        lines = f.readlines()
    pd_l = []
    for line in tqdm(lines):
        if '[Exp evaluation complete] {' in line:
            result_dict = line.split('[Exp evaluation complete] ')[1].strip()
            result_dict = result_dict.replace('nan', '\'nan\'')
            result_dict = result_dict.replace('array', '')
            result_dict = ast.literal_eval(result_dict)
            pd_l.append(result_dict)
            # except:
            #     pass
    dfm = pd.DataFrame(pd_l)
    if remove_extra_columns:
        columns_to_remove_if_exist = ['costs_std_stats', 'planner', 'observed_times', 'observed_times_diff', 'costs_std_median', 's', 'a', 'r', 'cost_std_plot', 'ri', 'telem_file_path']
        current_columns = list(dfm.columns)
        columns_to_drop = set(columns_to_remove_if_exist) & set(current_columns)
        columns_to_drop = list(columns_to_drop)
        dfm = dfm.drop(columns=columns_to_drop)
    else:
        columns_to_np_arrays_if_exist = ['observed_times', 'observed_times_diff', 's', 'a', 'r', 'cost_std_plot', 'ri']
        current_columns = list(dfm.columns)
        columns_to_np_arrays = set(columns_to_np_arrays_if_exist) & set(current_columns)
        columns_to_np_arrays = list(columns_to_np_arrays)
        dfm[columns_to_np_arrays] = dfm[columns_to_np_arrays].applymap(np.array)
    return dfm

def extract_state_rewards(df):
    dd = {}
    for _, row in df.iterrows():
        k, v = row['observations_taken'], row['state_reward']
        if k in dd:
            dd[k].append(v)
        else:
            dd[k] = [v]
    return dd

def smooth(scalars: np.ndarray, weight: float) -> np.ndarray:
    """
    EMA implementation according to
    https://github.com/tensorflow/tensorboard/blob/34877f15153e1a2087316b9952c931807a122aa7/tensorboard/components/vz_line_chart2/line-chart.ts#L699
    """
    last = 0
    smoothed = []
    num_acc = 0
    for next_val in scalars:
        last = last * weight + (1 - weight) * next_val
        num_acc += 1
        # de-bias
        debias_weight = 1
        if weight != 1:
            debias_weight = 1 - np.power(weight, num_acc)
        smoothed_val = last / debias_weight
        smoothed.append(smoothed_val)

    return np.array(smoothed)

def seed_all(seed=None):
    """
    Set the torch, numpy, and random module seeds based on the seed
    specified in config. If there is no seed or it is None, a time-based
    seed is used instead and is written to config.
    """
    # Default uses current time in milliseconds, modulo 1e9
    if seed is None:
        seed = round(time() * 1000) % int(1e9)

    # Set the seeds using the shifted seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)