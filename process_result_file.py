import pandas as pd
import numpy as np
import torch
import random
from utils.results_utils import normalize_means, generate_main_results_table, df_from_log, ci, moving_average, configure_plotting_sn_params, load_df, compute_norm_metrics, generate_overlap_graph, seed_all, generate_n_step_graph, generate_main_results_table_paper_format, generate_n_step_graph
from time import time
import shelve
from enum import Enum
seed_all(0)
class Experiment(Enum):
    MAIN_TABLE = 1
    INSIGHT_CONFOUNDING = 2
    INSIGHT_NEXT_STEP_PREDICTION = 3
    INSIGHT_NOISE = 4
    INSIGHT_LESS_SAMPLES = 5

experiment = Experiment.MAIN_TABLE

if experiment == Experiment.MAIN_TABLE:
    # Main results table
    LOG_PATH = './results/2_main_table/final_with_insite.txt'
    df = load_df(LOG_PATH)
    tables = generate_main_results_table_paper_format(df)
    print('')
    print(tables)
elif experiment == Experiment.INSIGHT_CONFOUNDING:
    LOG_PATH = './logs/run_ct-20230516-123859_insite-sindy-wsindy-crn-msm-gnet-ct-rmsn_EQ_4_D_1_5-runs_log.txt'
    df = load_df(LOG_PATH)
    generate_overlap_graph(df)
elif experiment == Experiment.INSIGHT_NEXT_STEP_PREDICTION:
    LOG_PATH = './results/2_main_table/final_with_insite.txt'
    df = load_df(LOG_PATH)
    generate_n_step_graph(df)
print('fin.')