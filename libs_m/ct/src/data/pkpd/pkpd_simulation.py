# -*- coding: utf-8 -*-
"""
PKPD simulation model

Notes:
- Simulation time taken to be in days
"""

import logging
import jax
import jax
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random, vmap, jit, lax
# from jax.experimental.ode import odeint
from jax import device_put
from jax.lax import stop_gradient, scan
from functools import partial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import truncnorm  # we need to sample from truncated normal distributions
from enum import IntEnum
from time import time
from pysindy import SINDy
from pysindy.feature_library import PolynomialLibrary
from pysindy.differentiation import FiniteDifference, SmoothedFiniteDifference
from scipy.signal import savgol_filter
from torch import multiprocessing
from sympy import sympify
import sympy2jax
import sys
from jax.scipy.optimize import minimize

sns.set()

from src.data.pkpd.utils import odeint, LSQIntialMask, convert_sindy_model_to_sympyjax_model, convert_sindy_model_to_sympyjax_model_core, convert_sindy_model_to_sympy_model, process_dataset_into_de_format_old_format, STEPS_FOR_DT, MAX_SEQUENCE_LENGTH, HMAX, STANDARD_DT, MAX_TIME_HORIZON, MAX_VALUE, debug_scan, debug_vmap, smoother_kws
# from utils import odeint, LSQIntialMask, convert_sindy_model_to_sympyjax_model, convert_sindy_model_to_sympyjax_model_core, convert_sindy_model_to_sympy_model, process_dataset_into_de_format_old_format, STEPS_FOR_DT, MAX_SEQUENCE_LENGTH, HMAX, STANDARD_DT, MAX_TIME_HORIZON, MAX_VALUE, debug_scan, debug_vmap, smoother_kws
integrator_keywords = {}
OBSERVATION_NOISE = 0.01
# OBSERVATION_NOISE = 1.0
RECOVERY_MULTIPLIER = 5.8 * 10 ** (8 + 3)  # cells per cm^3
sindy_threshold = 0.02
sindy_alpha = 0.5
smooth_input_data = False
# CONF_SCALE = 25

class Equation(IntEnum):
    EQ_4_A = 1
    EQ_4_B = 2
    EQ_4_C = 3
    EQ_4_D = 4
    EQ_5_A = 5
    EQ_5_B = 6
    EQ_5_C = 7
    EQ_5_D = 8
    EQ_4_M = 9 # Multi-modal test dataset

class CfSeqMode(IntEnum):
    SLIDING_TREATMENT = 1
    RANDOM_TRAJECTORIES = 2

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Simulation Functions

def dy_dt(y, t, treatment, hidden_C_0, hidden_C_1):
    return jax.lax.cond(treatment == 0,
                        lambda _: - hidden_C_0 * y,
                        lambda _: - hidden_C_1 * y,
                        operand=None)
dy_dt = jax.tree_util.Partial(dy_dt)

def generate_params(num_patients, conf_coeff, window_size, lag, key: jnp.ndarray, equation: Equation):
    """
    Get original patient-specific simulation parameters, and add extra ones to control confounding

    :param num_patients: Number of patients to simulate
    :param conf_coeff: Bias on action policy for treatment assignments
    :return: dict of parameters
    """

    basic_params = get_standard_params(num_patients, equation, key)
    # Parameters controlling sigmoid application probabilities

    basic_params['sigmoid_intercept'] = MAX_VALUE / 2.0
    basic_params['sigmoid_gamma'] = conf_coeff / MAX_VALUE

    basic_params['window_size'] = window_size
    basic_params['lag'] = lag

    return basic_params

def get_standard_params(num_patients, equation: Equation, key: jnp.ndarray, sample_random_theta_values = False):  # additional params
    """
    Simulation parameters

    :param num_patients: Number of patients to simulate
    :return: simulation_parameters: Initial volumes + Static variables (e.g. response to treatment); randomly shuffled
    """
    SCALE = 0.5

    if 'EQ_4' in equation.name:
        # Observed static covariates for the patients
        sigma = 0.2 * SCALE
        sigma_0 = 0.1 * SCALE
        sigma_1 = 0.1 * SCALE
        theta_upper = 10.0 * SCALE
        theta_lower = 1.0 * SCALE
        c_0_mean = 1.0 * SCALE
        # c_1_mean = 3.0
        c_1_mean = 1.0 * SCALE

        # Action 0
        key, subkey = random.split(key)
        c_0 = jax.random.normal(subkey, shape=(num_patients, )) * sigma_0 + c_0_mean

        # Action 1
        key, subkey = random.split(key)
        c_1 = jax.random.normal(subkey, shape=(num_patients, )) * sigma_1 + c_1_mean
        
        # V at bottom
        v = 1.0
        C_0 = c_0
        C_1 = c_1
    if equation.name == 'EQ_4_C' or equation.name == 'EQ_4_D':
        # Parameters linear dependence
        # Action 0
        if sample_random_theta_values:
            key, subkey = random.split(key)
            theta_0_0 = random.uniform(subkey, minval=theta_lower, maxval=theta_upper)
            key, subkey = random.split(key)
            theta_0_1 = random.uniform(subkey, minval=theta_lower, maxval=theta_upper)
        else:
            theta_0_0 = 1.0
            theta_0_1 = 0.1 * SCALE
        C_0 = theta_0_0 * c_0 + theta_0_1

        # Action 1
        if sample_random_theta_values:
            key, subkey = random.split(key)
            theta_1_0 = random.uniform(subkey, minval=theta_lower, maxval=theta_upper)
            key, subkey = random.split(key)
            theta_1_1 = random.uniform(subkey, minval=theta_lower, maxval=theta_upper)
        else:
            theta_1_0 = 1.0
            theta_1_1 = 0.3 * SCALE
        C_1 = theta_1_0 * c_1 + theta_1_1

        if equation.name == 'EQ_4_D':
            sigma_C_params = 0.5 * SCALE

            key, subkey = random.split(key)
            C_0 = jax.random.normal(subkey) * sigma_C_params + C_0
            key, subkey = random.split(key)
            C_1 = jax.random.normal(subkey) * sigma_C_params + C_1
    elif equation.name == 'EQ_4_M':
        key, subkey = random.split(key)
        means = jax.random.choice(subkey, jnp.array([0.1, 0.3]) * SCALE, shape=(num_patients,))
        C_0 = c_0 + means
        key, subkey = random.split(key)
        means = jax.random.choice(subkey, jnp.array([0.1, 0.3]) * SCALE, shape=(num_patients,))
        C_1 = c_1 + means
    elif equation.name == 'EQ_5_A':
        raise NotImplementedError
    elif equation.name == 'EQ_5_B':
        raise NotImplementedError
        # pass
    elif equation.name == 'EQ_5_C':
        raise NotImplementedError
        # pass
    elif equation.name == 'EQ_5_D':
        raise NotImplementedError
        # pass

    C_0 = C_0 / v
    C_1 = C_1 / v
    
    key, subkey = random.split(key)
    initial_volumes = random.uniform(subkey, shape=(num_patients,), minval=1.0, maxval=MAX_VALUE)

    output_holder = {
                     'initial_volumes': stop_gradient(initial_volumes),
                     'hidden_C_0': stop_gradient(C_0),
                     'hidden_C_1': stop_gradient(C_1),
                     'observed_static_c_0': stop_gradient(c_0),
                     'observed_static_c_1': stop_gradient(c_1),
                     }
    # np.random.exponential(expected_treatment_delay, num_patients),

    # Randomise output params
    logging.info("Randomising outputs")
    idx = jnp.arange(num_patients)
    key, subkey = random.split(key)
    idx = random.permutation(subkey, idx, independent=True)
    output_params = {}
    for k in output_holder:
        output_params[k] = output_holder[k][idx]

    output_params.update({'observation_noise': OBSERVATION_NOISE})
    return output_params

def simulate_factual(simulation_params, seq_length, key: jnp.ndarray, equation: Equation, assigned_actions=None):
    """
    Simulation of factual patient trajectories (for train and validation subset)

    :param simulation_params: Parameters of the simulation
    :param seq_length: Maximum trajectory length
    :param assigned_actions: Fixed non-random treatment assignment policy, if None - standard biased random assignment is applied
    :return: simulated data dict
    """
    t0 = time()
    dt = MAX_TIME_HORIZON / seq_length

    # Unpack simulation parameters
    observation_noise = simulation_params['observation_noise']
    initial_volumes = simulation_params['initial_volumes']
    hidden_C_0 = simulation_params['hidden_C_0']
    hidden_C_1 = simulation_params['hidden_C_1']
    observed_static_c_0 = simulation_params['observed_static_c_0']
    observed_static_c_1 = simulation_params['observed_static_c_1']
    window_size = simulation_params['window_size']
    lag = simulation_params['lag']

    # Coefficients for treatment assignment probabilities
    sigmoid_intercept = simulation_params['sigmoid_intercept']
    sigmoid_gamma = simulation_params['sigmoid_gamma']

    num_patients = initial_volumes.shape[0]

    key, subkey = random.split(key)
    recovery_rvs = jax.random.uniform(subkey, shape=(num_patients, seq_length), minval=0.0, maxval=1.0)
    key, subkey = random.split(key)
    treatment_application_rvs = jax.random.uniform(subkey, shape=(num_patients,), minval=0.0, maxval=1.0)

    def recovery_fn(cancer_volumes, recovery_cond):
        first_true_idx = jnp.argmax(recovery_cond)
        sequence_length = first_true_idx + 1
        mask = jnp.arange(cancer_volumes.shape[0]) < first_true_idx
        updated_cancer_volumes =  cancer_volumes * mask
        return updated_cancer_volumes, sequence_length

    def death_fn(cancer_volumes, death_cond):
        first_true_idx = jnp.argmax(death_cond)
        sequence_length = first_true_idx + 1
        mask = jnp.arange(cancer_volumes.shape[0]) >= first_true_idx
        updated_cancer_volumes =  cancer_volumes * (1 - mask) + mask * MAX_VALUE
        return updated_cancer_volumes, sequence_length

    @jit
    def simulate_patient(initial_volume, hidden_C_0, hidden_C_1, treatment_application_rv, recovery_rv, sigmoid_gamma, sigmoid_intercept):
        sequence_length = seq_length - 1
        treatment_prob = (1.0 / (1.0 + jnp.exp(-sigmoid_gamma * (initial_volume - sigmoid_intercept))))
        treatment = stop_gradient(jax.lax.cond(treatment_application_rv < treatment_prob,
                              lambda _: 1,
                              lambda _: 0,
                              operand=None))

        t = stop_gradient(jnp.arange(0, MAX_TIME_HORIZON, dt).astype(jnp.float64))
        cancer_volumes = odeint(dy_dt, initial_volume, t, treatment, hidden_C_0, hidden_C_1, hmax=HMAX, **integrator_keywords) # Non batchable

        recovery_cond = recovery_rv < jnp.exp(-cancer_volumes * RECOVERY_MULTIPLIER)
        cancer_volumes, sequence_length = jax.lax.cond(jnp.any(recovery_cond), lambda _: recovery_fn(cancer_volumes, recovery_cond), lambda _: (cancer_volumes, sequence_length), operand=None)

        death_cond = cancer_volumes > MAX_VALUE
        cancer_volumes, sequence_length = jax.lax.cond(jnp.any(death_cond), lambda _: death_fn(cancer_volumes, death_cond), lambda _: (cancer_volumes, sequence_length), operand=None)

        return cancer_volumes, treatment * jnp.ones(shape=(seq_length-1)), sequence_length

    cancer_volume, treatment_action, sequence_lengths = vmap(simulate_patient, in_axes=(0, 0, 0, 0, 0, None, None))(initial_volumes, hidden_C_0, hidden_C_1, treatment_application_rvs, recovery_rvs, sigmoid_gamma, sigmoid_intercept)

    # Check overlap here
    # print(f'Initival_volume statistics: {jnp.mean(initial_volumes)}, {jnp.std(initial_volumes)} | max {jnp.max(initial_volumes)}, min {jnp.min(initial_volumes)}')
    # initial_volumes_assinged_treatment_0 = initial_volumes[treatment_action[:,0] == 0]
    # initial_volumes_assinged_treatment_1 = initial_volumes[treatment_action[:,0] == 1]
    # print(f'Initival_volume statistics for treatment 0: {jnp.mean(initial_volumes_assinged_treatment_0)}, {jnp.std(initial_volumes_assinged_treatment_0)} | max {jnp.max(initial_volumes_assinged_treatment_0)}, min {jnp.min(initial_volumes_assinged_treatment_0)}')
    # print(f'Initival_volume statistics for treatment 1: {jnp.mean(initial_volumes_assinged_treatment_1)}, {jnp.std(initial_volumes_assinged_treatment_1)} | max {jnp.max(initial_volumes_assinged_treatment_1)}, min {jnp.min(initial_volumes_assinged_treatment_1)}')
    # print('')

    # # Debug purposes
    # cancer_volume, treatment_action, sequence_lengths = [], [], []
    # for i in tqdm(range(num_patients)):
    #     cancer_volume_i, treatment_action_i, sequence_lengths_i = simulate_patient(initial_volumes[i], hidden_C_0[i], hidden_C_1[i], treatment_application_rvs[i], recovery_rvs[i], sigmoid_gamma, sigmoid_intercept)
    #     cancer_volume.append(cancer_volume_i), treatment_action.append(treatment_action_i), sequence_lengths.append(sequence_lengths_i)
    # cancer_volume, treatment_action, sequence_lengths = jnp.stack(cancer_volume), jnp.stack(treatment_action_i), jnp.stack(sequence_lengths_i)

    if equation.name.split('_')[-1] in ['B', 'C', 'D']:
        key, subkey = random.split(key)
        cancer_volume = cancer_volume + observation_noise * random.normal(subkey, shape=cancer_volume.shape)

    # Pad if too short
    # if cancer_volume.shape[1] < seq_length:
        # cancer_volume = jnp.concatenate((cancer_volume, jnp.zeros((cancer_volume.shape[0],1))), axis=1)
    treatment_action = jnp.concatenate((treatment_action, jnp.zeros((treatment_action.shape[0],1))), axis=1)

    assert cancer_volume.shape[1] == seq_length, 'Cancer volume shape is not correct'
    assert treatment_action.shape[1] == seq_length, 'Treatment action shape is not correct'
    print('factual simulated took {:.2f} seconds'.format(time() - t0))
    outputs = {'cancer_volume': np.asarray(cancer_volume),
               'treatment_application': np.asarray(treatment_action),
               'sequence_lengths': np.asarray(sequence_lengths),
               'observed_static_c_0': np.asarray(observed_static_c_0),
               'observed_static_c_1': np.asarray(observed_static_c_1),
               }
    
    assert not jnp.any(jnp.isnan(cancer_volume)), 'Cancer volume contains NaN'
    return outputs

def pad_and_stack(list_of_lists):
    max_len = max([len(sub_list) for sub_list in list_of_lists])
    padded_lists = [jnp.pad(jnp.array(sub_list), (0, max_len - len(sub_list)), mode='constant') for sub_list in list_of_lists]
    stacked_matrix = jnp.stack(padded_lists)
    return stacked_matrix

def filter_patient(cancer_trajectory, action_trajectory, sequence_length, recovery_rv):
    recovery_cond = recovery_rv < jnp.exp(-cancer_trajectory * RECOVERY_MULTIPLIER)
    cancer_trajectory, sequence_length_out = jax.lax.cond(jnp.any(recovery_cond), lambda _: recovery_fn(cancer_trajectory, recovery_cond), lambda _: (cancer_trajectory, sequence_length), operand=None)
    sequence_length = jax.lax.min(sequence_length_out, sequence_length)

    death_cond = cancer_trajectory > MAX_VALUE
    cancer_trajectory, sequence_length_out = jax.lax.cond(jnp.any(death_cond), lambda _: death_fn(cancer_trajectory, death_cond), lambda _: (cancer_trajectory, sequence_length), operand=None)
    sequence_length = jax.lax.min(sequence_length_out, sequence_length)
    return cancer_trajectory, action_trajectory, sequence_length

def recovery_fn(cancer_volumes, recovery_cond):
    first_true_idx = jnp.argmax(recovery_cond)
    sequence_length = first_true_idx + 1
    mask = jnp.arange(cancer_volumes.shape[0]) < first_true_idx
    updated_cancer_volumes =  cancer_volumes * mask
    return updated_cancer_volumes, sequence_length

def death_fn(cancer_volumes, death_cond):
    first_true_idx = jnp.argmax(death_cond)
    sequence_length = first_true_idx + 1
    mask = jnp.arange(cancer_volumes.shape[0]) >= first_true_idx
    updated_cancer_volumes =  cancer_volumes * (1 - mask) + mask * MAX_VALUE
    return updated_cancer_volumes, sequence_length

def scan_fn_simulate_counterfactual_1_step(carry, t_tuple):
    cancer_volume, treatment, hidden_C_0, hidden_C_1 = carry
    counterfactual_treatment = 1 - treatment
    # jax.debug.print("counterfactual_treatment {counterfactual_treatment} ", counterfactual_treatment=counterfactual_treatment)
    # jax.debug.print("cancer_volume {cancer_volume} ", cancer_volume=cancer_volume)
    # jax.debug.print("=====================================")
    counterfactual_cancer_volume = odeint(dy_dt, cancer_volume, t_tuple, counterfactual_treatment, hidden_C_0, hidden_C_1, hmax=HMAX, **integrator_keywords)[1]

    cancer_volume = odeint(dy_dt, cancer_volume, t_tuple, treatment, hidden_C_0, hidden_C_1, hmax=HMAX, **integrator_keywords)[1]
    return (cancer_volume, treatment, hidden_C_0, hidden_C_1), (cancer_volume, treatment, counterfactual_cancer_volume, counterfactual_treatment)

def simulate_counterfactual_1_step(simulation_params, seq_length, key: jnp.ndarray, equation: Equation):
    """
    Simulation of test trajectories to asses all one-step ahead counterfactuals
    :param simulation_params: Parameters of the simulation
    :param seq_length: Maximum trajectory length (number of factual time-steps)
    :return: simulated data dict with number of rows equal to num_patients * seq_length * num_treatments
    """
    t0 = time()
    dt = MAX_TIME_HORIZON / seq_length

    # Unpack simulation parameters
    observation_noise = simulation_params['observation_noise']
    initial_volumes = simulation_params['initial_volumes']
    hidden_C_0 = simulation_params['hidden_C_0']
    hidden_C_1 = simulation_params['hidden_C_1']
    observed_static_c_0 = simulation_params['observed_static_c_0']
    observed_static_c_1 = simulation_params['observed_static_c_1']
    window_size = simulation_params['window_size']
    lag = simulation_params['lag']

    # Coefficients for treatment assignment probabilities
    sigmoid_intercept = simulation_params['sigmoid_intercept']
    sigmoid_gamma = simulation_params['sigmoid_gamma']

    num_patients = initial_volumes.shape[0]
    num_treatments = 2
    # num_test_points = num_patients * seq_length * num_treatments

    key, subkey = random.split(key)
    recovery_rvs = jax.random.uniform(subkey, shape=(num_patients, seq_length-1), minval=0.0, maxval=1.0)
    key, subkey = random.split(key)
    treatment_application_rvs = jax.random.uniform(subkey, shape=(num_patients,), minval=0.0, maxval=1.0)

    # @jit
    def simulate_patient(initial_volume, hidden_C_0, hidden_C_1, treatment_application_rv, recovery_rv, sigmoid_gamma, sigmoid_intercept):
        # sequence_length = seq_length
        treatment_prob = (1.0 / (1.0 + jnp.exp(-sigmoid_gamma * (initial_volume - sigmoid_intercept))))
        treatment = jax.lax.cond(treatment_application_rv < treatment_prob,
                              lambda _: 1,
                              lambda _: 0,
                              operand=None)
        
        t = jnp.arange(0, MAX_TIME_HORIZON, dt).astype(jnp.float64)
        t_tuples = jnp.array([(t[i-1], t[i]) for i in range(1, t.shape[0])])
        carry_init = (initial_volume, treatment, hidden_C_0, hidden_C_1)
        _, ys = scan(scan_fn_simulate_counterfactual_1_step, carry_init, t_tuples)
        cancer_volumes, treatments, counterfactual_cancer_volumes, counterfactual_treatments = ys

        cancer_volumes = jnp.concatenate([initial_volume[None, ...], cancer_volumes], axis=0)
        # treatments = jnp.concatenate([treatment[None, ...], treatments], axis=0)

        total_cancer_trajectories = []
        total_action_trajectories = []
        total_sequence_lengths = []
        for i, (counterfactual_cancer_volume, counterfactual_treatment) in enumerate(zip(counterfactual_cancer_volumes, counterfactual_treatments)):
            # Factual 
            total_cancer_trajectories.append(cancer_volumes[:i+2])
            total_action_trajectories.append(treatments[:i+1])
            total_sequence_lengths.append(i+1)

            # Counterfactual
            total_cancer_trajectories.append(jax.lax.concatenate((cancer_volumes[:i+1], counterfactual_cancer_volume.reshape(1,)), dimension=0))
            total_action_trajectories.append(jax.lax.concatenate((treatments[:i],counterfactual_treatment.reshape(1,)), dimension=0))
            total_sequence_lengths.append(i+1)

        total_cancer_trajectories = pad_and_stack(total_cancer_trajectories)
        total_action_trajectories = pad_and_stack(total_action_trajectories)
        total_sequence_lengths = jnp.array(total_sequence_lengths)        

        # Reason for commented out line below: We want to skip filtering trajectories---to match same as cancer simulation
        # total_cancer_trajectories, total_action_trajectories, total_sequence_lengths = vmap(filter_patient, in_axes=(0, 0, 0, None))(total_cancer_trajectories, total_action_trajectories, total_sequence_lengths, recovery_rv)

        # # Debug purposes
        # total_cancer_trajectories_l, total_action_trajectories_l, total_sequence_lengths_l = [], [], []
        # for i in tqdm(range(total_cancer_trajectories.shape[0])):
        #     cancer_volume_i, treatment_action_i, sequence_lengths_i = filter_patient(total_cancer_trajectories[i], total_action_trajectories[i], total_sequence_lengths[i], recovery_rv)
        #     total_cancer_trajectories_l.append(cancer_volume_i), total_action_trajectories_l.append(treatment_action_i), total_sequence_lengths_l.append(sequence_lengths_i)
        # total_cancer_trajectories, total_action_trajectories, total_sequence_lengths = jnp.stack(total_cancer_trajectories_l), jnp.stack(total_action_trajectories_l), jnp.stack(total_sequence_lengths_l)
        return total_cancer_trajectories, total_action_trajectories, total_sequence_lengths

    cancer_volume, treatment_action, sequence_lengths = jit(vmap(simulate_patient, in_axes=(0, 0, 0, 0, 0, None, None)))(initial_volumes, hidden_C_0, hidden_C_1, treatment_application_rvs, recovery_rvs, sigmoid_gamma, sigmoid_intercept)
    # cancer_volume, treatment_action, sequence_lengths = debug_vmap(simulate_patient, in_axes=(0, 0, 0, 0, 0, None, None), args=(initial_volumes, hidden_C_0, hidden_C_1, treatment_application_rvs, recovery_rvs, sigmoid_gamma, sigmoid_intercept))

    # # Debug purposes
    # cancer_volume, treatment_action, sequence_lengths = [], [], []
    # for i in tqdm(range(num_patients)):
    #     cancer_volume_i, treatment_action_i, sequence_lengths_i = simulate_patient(initial_volumes[i], hidden_C_0[i], hidden_C_1[i], treatment_application_rvs[i], recovery_rvs[i], sigmoid_gamma, sigmoid_intercept)
    #     cancer_volume.append(cancer_volume_i), treatment_action.append(treatment_action_i), sequence_lengths.append(sequence_lengths_i)
    # cancer_volume, treatment_action, sequence_lengths = jnp.stack(cancer_volume), jnp.stack(treatment_action_i), jnp.stack(sequence_lengths_i)

    if equation.name.split('_')[-1] in ['B', 'C', 'D']:
        key, subkey = random.split(key)
        cancer_volume = cancer_volume + observation_noise * random.normal(subkey, shape=cancer_volume.shape)

    print('1-step ahead counterfactuals simulated took {:.2f} seconds'.format(time() - t0))
    observed_static_c_0 = jnp.repeat(observed_static_c_0, cancer_volume.shape[1])
    observed_static_c_1 = jnp.repeat(observed_static_c_1, cancer_volume.shape[1])
    cancer_volume = jnp.reshape(cancer_volume, (-1, cancer_volume.shape[2]))
    treatment_action = jnp.reshape(treatment_action, (-1, treatment_action.shape[2]))
    sequence_lengths = jnp.reshape(sequence_lengths, (-1))
    treatment_action = jnp.concatenate((treatment_action, jnp.zeros((treatment_action.shape[0],1))), axis=1)
    assert cancer_volume.shape[1] == seq_length, 'Cancer volume shape is not correct'
    assert treatment_action.shape[1] == seq_length, 'Treatment action shape is not correct'

    # # Pad if too short
    # if cancer_volume.shape[1] < seq_length:
    #     cancer_volume = jnp.concatenate((cancer_volume, jnp.zeros((cancer_volume.shape[0],1))), axis=1)
    #     treatment_action = jnp.concatenate((treatment_action, jnp.zeros((treatment_action.shape[0],2))), axis=1)

    outputs = {'cancer_volume': np.asarray(cancer_volume),
               'treatment_application': np.asarray(treatment_action),
               'sequence_lengths': np.asarray(sequence_lengths),
               'observed_static_c_0': np.asarray(observed_static_c_0),
               'observed_static_c_1': np.asarray(observed_static_c_1),
               }
    
    assert not jnp.any(jnp.isnan(cancer_volume)), 'Cancer volume contains NaN'

    print("Call to simulate counterfactuals data")
    return outputs

# @partial(jit, static_argnums=(2,))
def cf_seq_sliding_treatment(t_tuple, cancer_volume, projection_horizon, dt, hidden_C_0, hidden_C_1):
    t_start, t_end = t_tuple[0], t_tuple[1]
    treatment_options = jnp.concatenate((jnp.eye(projection_horizon).astype(jnp.int64), 1 - jnp.eye(projection_horizon).astype(jnp.int64)), axis=0)
    counterfactual_cancer_volume_l_all, counterfactual_treatment_l_all = [], []
    for treatment_plan in treatment_options:
        counterfactual_cancer_volume_l, counterfactual_treatment_l = [], []
        counterfactual_cancer_volume = cancer_volume
        current_t_start, current_t_end = t_start, t_end
        for j in range(projection_horizon):
            counterfactual_cancer_volume = odeint(dy_dt, counterfactual_cancer_volume, jnp.array([current_t_start, current_t_end]), treatment_plan[j], hidden_C_0, hidden_C_1, hmax=HMAX, **integrator_keywords)[1]
            current_t_start, current_t_end = current_t_start + dt, current_t_end + dt
            counterfactual_cancer_volume_l.append(counterfactual_cancer_volume), counterfactual_treatment_l.append(treatment_plan[j])
        counterfactual_cancer_volume_l_all.append(counterfactual_cancer_volume_l), counterfactual_treatment_l_all.append(counterfactual_treatment_l)
    return jnp.array(counterfactual_cancer_volume_l_all), jnp.array(counterfactual_treatment_l_all)
    
def cf_seq_random_trajectories(t_tuple, cancer_volume, projection_horizon_arange, dt, hidden_C_0, hidden_C_1, key):
    t_start, t_end = t_tuple[0], t_tuple[1]
    key, subkey = random.split(key)
    counterfactual_treatments = random.randint(subkey, shape=(projection_horizon_arange.shape[0] * 2, projection_horizon_arange.shape[0],), minval=0, maxval=2)
    counterfactual_cancer_volume_l_all, counterfactual_treatment_l_all = [], []
    for treatment_plan in counterfactual_treatments:
        counterfactual_cancer_volume_l, counterfactual_treatment_l = [], []
        counterfactual_cancer_volume = cancer_volume
        current_t_start, current_t_end = t_start, t_end
        for j in projection_horizon_arange:
            counterfactual_cancer_volume = odeint(dy_dt, counterfactual_cancer_volume, jnp.array([t_start, t_end]),treatment_plan[j], hidden_C_0, hidden_C_1, hmax=HMAX, **integrator_keywords)[1]
            current_t_start, current_t_end = current_t_start + dt, current_t_end + dt
            counterfactual_cancer_volume_l.append(counterfactual_cancer_volume), counterfactual_treatment_l.append(treatment_plan[j])
        counterfactual_cancer_volume_l_all.append(counterfactual_cancer_volume_l), counterfactual_treatment_l_all.append(counterfactual_treatment_l)
    return jnp.array(counterfactual_cancer_volume_l_all), jnp.array(counterfactual_treatment_l_all)
    
def scan_fn_simulate_counterfactuals_treatment_seq(carry, t_tuple, cf_seq_mode, projection_horizon, dt):
    cancer_volume, treatment, hidden_C_0, hidden_C_1, key  = carry
    key, subkey = random.split(key)
    counterfactual_cancer_volume_l, counterfactual_treatment_l = jax.lax.cond(cf_seq_mode == CfSeqMode.SLIDING_TREATMENT, lambda _: cf_seq_sliding_treatment(t_tuple, cancer_volume, projection_horizon, dt, hidden_C_0, hidden_C_1), lambda _: cf_seq_random_trajectories(t_tuple, cancer_volume, jnp.arange(projection_horizon), dt, hidden_C_0, hidden_C_1, subkey), operand=None)
    # if cf_seq_mode == CfSeqMode.SLIDING_TREATMENT:
    #     counterfactual_cancer_volume_l, counterfactual_treatment_l = cf_seq_sliding_treatment(t_tuple, cancer_volume, projection_horizon, dt, hidden_C_0, hidden_C_1)
    # else:
    #     counterfactual_cancer_volume_l, counterfactual_treatment_l = cf_seq_random_trajectories(t_tuple, cancer_volume, jnp.arange(projection_horizon), dt, hidden_C_0, hidden_C_1, key)
    cancer_volume = odeint(dy_dt, cancer_volume, t_tuple, treatment, hidden_C_0, hidden_C_1, hmax=HMAX, **integrator_keywords)[1]
    return (cancer_volume, treatment, hidden_C_0, hidden_C_1, key), (cancer_volume, treatment, counterfactual_cancer_volume_l, counterfactual_treatment_l)

def simulate_counterfactuals_treatment_seq(simulation_params, seq_length, projection_horizon,  key: jnp.ndarray, equation: Equation, cf_seq_mode='sliding_treatment'):
    """
    Simulation of test trajectories to asses a subset of multiple-step ahead counterfactuals
    :param simulation_params: Parameters of the simulation
    :param seq_length: Maximum trajectory length (number of factual time-steps)
    :param cf_seq_mode: Counterfactual sequence setting: sliding_treatment / random_trajectories
    :return: simulated data dict with number of rows equal to num_patients * seq_length * 2 * projection_horizon
    """

    # ToDo: Fix this; same way for the 1-step ahead counterfactuals

    # 'sliding_treatment' : Given on treatment, choose opposite, for n-steps!
    # 'random_trajectories': Random; select a random treatment, after n-periods.
    # 
    assert cf_seq_mode in ['sliding_treatment', 'random_trajectories']
    if cf_seq_mode == 'sliding_treatment':
        cf_seq_mode = CfSeqMode.SLIDING_TREATMENT
    elif cf_seq_mode == 'random_trajectories':
        cf_seq_mode = CfSeqMode.RANDOM_TRAJECTORIES
    t0 = time()
    dt = MAX_TIME_HORIZON / seq_length
    t = jnp.arange(0, seq_length + 1).astype(jnp.float64) * dt

    # Unpack simulation parameters
    observation_noise = simulation_params['observation_noise']
    initial_volumes = simulation_params['initial_volumes']
    hidden_C_0 = simulation_params['hidden_C_0']
    hidden_C_1 = simulation_params['hidden_C_1']
    observed_static_c_0 = simulation_params['observed_static_c_0']
    observed_static_c_1 = simulation_params['observed_static_c_1']
    window_size = simulation_params['window_size']
    lag = simulation_params['lag']

    # Coefficients for treatment assignment probabilities
    sigmoid_intercept = simulation_params['sigmoid_intercept']
    sigmoid_gamma = simulation_params['sigmoid_gamma']

    num_patients = initial_volumes.shape[0]

    key, subkey = random.split(key)
    recovery_rvs = jax.random.uniform(subkey, shape=(num_patients, seq_length + projection_horizon -1), minval=0.0, maxval=1.0)
    key, subkey = random.split(key)
    treatment_application_rvs = jax.random.uniform(subkey, shape=(num_patients,), minval=0.0, maxval=1.0)

    scan_fn = partial(scan_fn_simulate_counterfactuals_treatment_seq, cf_seq_mode=cf_seq_mode, projection_horizon=projection_horizon, dt=dt)

    # @jit
    def simulate_patient(initial_volume, hidden_C_0, hidden_C_1, treatment_application_rv, recovery_rv, key, sigmoid_gamma, sigmoid_intercept):
        # (0, 0, 0, 0, 0, 0, None, None, None, None)
        # sequence_length = seq_length
        treatment_prob = (1.0 / (1.0 + jnp.exp(-sigmoid_gamma * (initial_volume - sigmoid_intercept))))
        treatment = jax.lax.cond(treatment_application_rv < treatment_prob,
                              lambda _: 1,
                              lambda _: 0,
                              operand=None)
        t_tuples = jnp.array([(t[i-1], t[i]) for i in range(2, t.shape[0])])

        # Matching the simulation of the cancer simulation
        cancer_volume_second = odeint(dy_dt, initial_volume, jnp.array([t[0], t[1]]), treatment, hidden_C_0, hidden_C_1, hmax=HMAX, **integrator_keywords)[1]

        # volumes = jnp.array([initial_volume, cancer_volume_second])
        # treatments = jnp.array([treatment])
        carry_init = (cancer_volume_second, treatment, hidden_C_0, hidden_C_1, key)

        # @jit


        _, ys = scan(scan_fn, carry_init, t_tuples)
        # _, ys = debug_scan(scan_fn, carry_init, t_tuples)
        cancer_volumes, treatments, counterfactual_cancer_volume_ls, counterfactual_treatment_ls = ys

        cancer_volumes = jnp.concatenate([initial_volume[None, ...], cancer_volume_second[None, ...], cancer_volumes], axis=0)
        treatments = jnp.concatenate([treatment[None, ...], treatments], axis=0)

        total_cancer_trajectories = []
        total_action_trajectories = []
        total_sequence_lengths = []
        for i, (counterfactual_cancer_volume_l, counterfactual_treatment_l) in enumerate(zip(counterfactual_cancer_volume_ls, counterfactual_treatment_ls)):
            for counterfactual_cancer_volumes, counterfactual_treatments in zip(counterfactual_cancer_volume_l, counterfactual_treatment_l):
                total_cancer_trajectories.append(jax.lax.concatenate((cancer_volumes[:i+2], counterfactual_cancer_volumes), dimension=0))
                total_action_trajectories.append(jax.lax.concatenate((treatments[:i+1], counterfactual_treatments), dimension=0))
                total_sequence_lengths.append(i + 1 + projection_horizon)

        total_cancer_trajectories = pad_and_stack(total_cancer_trajectories)
        total_action_trajectories = pad_and_stack(total_action_trajectories)
        total_sequence_lengths = jnp.array(total_sequence_lengths)     

        # Reason for commented out line below: We want to skip filtering trajectories
        # total_cancer_trajectories, total_action_trajectories, total_sequence_lengths = vmap(filter_patient, in_axes=(0, 0, 0, None))(total_cancer_trajectories, total_action_trajectories, total_sequence_lengths, recovery_rv)

        return total_cancer_trajectories, total_action_trajectories, total_sequence_lengths

    #     # # Debug purposes
    #     # total_cancer_trajectories_l, total_action_trajectories_l, total_sequence_lengths_l = [], [], []
    #     # for i in tqdm(range(total_cancer_trajectories.shape[0])):
    #     #     cancer_volume_i, treatment_action_i, sequence_lengths_i = filter_patient(total_cancer_trajectories[i], total_action_trajectories[i], total_sequence_lengths[i], recovery_rv)
    #     #     total_cancer_trajectories_l.append(cancer_volume_i), total_action_trajectories_l.append(treatment_action_i), total_sequence_lengths_l.append(sequence_lengths_i)
    #     # total_cancer_trajectories, total_action_trajectories, total_sequence_lengths = jnp.stack(total_cancer_trajectories_l), jnp.stack(total_action_trajectories_l), jnp.stack(total_sequence_lengths_l)
    #     return total_cancer_trajectories, total_action_trajectories, total_sequence_lengths

    key, *subkeys = random.split(key, num_patients + 1)
    subkeys = jnp.array(subkeys)

    cancer_volume, treatment_action, sequence_lengths = jit(vmap(simulate_patient, in_axes=(0, 0, 0, 0, 0, 0, None, None)))(initial_volumes, hidden_C_0, hidden_C_1, treatment_application_rvs, recovery_rvs, subkeys, sigmoid_gamma, sigmoid_intercept)
    # debug_vmap(simulate_patient, in_axes=(0, 0, 0, 0, 0, 0, None, None, None), args=(initial_volumes, hidden_C_0, hidden_C_1, treatment_application_rvs, recovery_rvs, subkeys, sigmoid_gamma, sigmoid_intercept, cf_seq_mode))
    #1, 0 : initial_volumes
    #2, 0 : hidden_C_0
    #3, 0 : hidden_C_1
    #4, 0 : treatment_application_rvs
    #5, 0 : recovery_rvs
    #6, 0 : subkeys
    #7, None : sigmoid_gamma
    #8, None : sigmoid_intercept
    #9, None : cf_seq_mode


    # # Debug purposes
    # cancer_volume, treatment_action, sequence_lengths = [], [], []
    # for i in tqdm(range(num_patients)):
    #     cancer_volume_i, treatment_action_i, sequence_lengths_i = simulate_patient(initial_volumes[i], hidden_C_0[i], hidden_C_1[i], treatment_application_rvs[i], recovery_rvs[i], subkeys[i], sigmoid_gamma, sigmoid_intercept, cf_seq_mode)
    #     cancer_volume.append(cancer_volume_i), treatment_action.append(treatment_action_i), sequence_lengths.append(sequence_lengths_i)
    # cancer_volume, treatment_action, sequence_lengths = jnp.stack(cancer_volume), jnp.stack(treatment_action_i), jnp.stack(sequence_lengths_i)

    if equation.name.split('_')[-1] in ['B', 'C', 'D']:
        key, subkey = random.split(key)
        cancer_volume = cancer_volume + observation_noise * random.normal(subkey, shape=cancer_volume.shape)

    print('{}-step ahead counterfactuals simulated took {:.2f} seconds'.format(projection_horizon, time() - t0))
    observed_static_c_0 = jnp.repeat(observed_static_c_0, cancer_volume.shape[1])
    observed_static_c_1 = jnp.repeat(observed_static_c_1, cancer_volume.shape[1])
    cancer_volume = jnp.reshape(cancer_volume, (-1, cancer_volume.shape[2]))
    treatment_action = jnp.reshape(treatment_action, (-1, treatment_action.shape[2]))
    sequence_lengths = jnp.reshape(sequence_lengths, (-1))

    # Pad if too short
    # if treatment_action.shape[1] < (seq_length + projection_horizon):
    treatment_action = jnp.concatenate((treatment_action, jnp.zeros((treatment_action.shape[0],1))), axis=1)

    assert cancer_volume.shape[1] == (seq_length + projection_horizon), 'Cancer volume shape is not correct'
    assert treatment_action.shape[1] == (seq_length + projection_horizon), 'Treatment action shape is not correct'

    outputs = {'cancer_volume': np.asarray(cancer_volume),
               'treatment_application': np.asarray(treatment_action),
               'sequence_lengths': np.asarray(sequence_lengths),
               'observed_static_c_0': np.asarray(observed_static_c_0),
               'observed_static_c_1': np.asarray(observed_static_c_1),
               }
    
    assert not jnp.any(jnp.isnan(cancer_volume)), 'Cancer volume contains NaN'

    print("Call to simulate counterfactuals data")
    return outputs


def get_scaling_params(sim):
    real_idx = ['cancer_volume']

    # df = pd.DataFrame({k: sim[k] for k in real_idx})
    means = {}
    stds = {}
    seq_lengths = sim['sequence_lengths']
    for k in real_idx:
        active_values = []
        for i in range(seq_lengths.shape[0]):
            end = int(seq_lengths[i])
            active_values += list(sim[k][i, :end])

        means[k] = np.mean(active_values)
        stds[k] = np.std(active_values)

    # Add means for static variables`
    means['observed_static_c_0'] = np.mean(sim['observed_static_c_0'])
    stds['observed_static_c_0'] = np.std(sim['observed_static_c_0'])

    means['observed_static_c_1'] = np.mean(sim['observed_static_c_1'])
    stds['observed_static_c_1'] = np.std(sim['observed_static_c_1'])

    return pd.Series(means), pd.Series(stds)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plotting Functions


def plot_treatments(data: dict, patient: int):
    df = pd.DataFrame({'N(t)': data['cancer_volume'][patient],
                       'C(t)': data['chemo_dosage'][patient],
                       'd(t)': data['radio_dosage'][patient],
                       })
    df = df[['N(t)', "C(t)", "d(t)"]]
    df.plot(secondary_y=['C(t)', 'd(t)'])
    plt.xlabel("$t$")
    plt.show()
    plt.savefig(f'treatments_{patient}.png')


def plot_sigmoid_function(data: dict):
    """
    Simple plots to visualise probabilities of treatment assignments

    :return:
    """

    # Profile of treatment application sigmoid
    for coeff in [i for i in range(11)]:
        tumour_death_threshold = calc_volume(13)
        assigned_beta = coeff / tumour_death_threshold
        assigned_interp = tumour_death_threshold / 2
        idx = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        volumes = idx * tumour_death_threshold

        def sigmoid_fxn(volume, beta, intercept):
            return (1.0 / (1.0 + np.exp(-beta * (volume - intercept))))

        data[coeff] = pd.Series(sigmoid_fxn(volumes, assigned_beta, assigned_interp), index=idx)

    df = pd.DataFrame(data)
    df.plot()
    plt.show()
    plt.savefig(f'sigmoid.png')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Data Processing Functions

def check_factual_data_with_oracle(data, dy_dt, smooth=False, joint=False):
    cancer_volume = jnp.array(data['cancer_volume']).astype(jnp.float64)
    treatment_action = jnp.array(data['treatment_application']).astype(jnp.int64)
    sequence_lengths = jnp.array(data['sequence_lengths']).astype(jnp.int64)
    observed_static_c_0 = jnp.array(data['observed_static_c_0']).astype(jnp.float64)
    observed_static_c_1 = jnp.array(data['observed_static_c_1']).astype(jnp.float64)

    # print(cancer_volume[0][-2])
    # print(cancer_volume[0][0])

    if smooth:
        smoother_kws_ = smoother_kws.copy()
        smoother_kws_.update({'axis': 1}) 
        cancer_volume = jnp.array(savgol_filter(cancer_volume, **smoother_kws))
    
    dt = MAX_TIME_HORIZON / seq_length
    sequence_length_max = sequence_lengths.max()
    t = stop_gradient(jnp.arange(0,sequence_length_max) * dt)

    @jit
    def simulate_cancer_volume(initial_volume, treatment, t, observed_static_c_0, observed_static_c_1):
        treatment = treatment.mean().astype(jnp.int64)
        return odeint(dy_dt, initial_volume, t, treatment, observed_static_c_0, observed_static_c_1, hmax=HMAX, **integrator_keywords)

    cancer_volumes_recreate = vmap(simulate_cancer_volume, in_axes=(0,0,None,0,0))(cancer_volume[:,0], treatment_action[:,:sequence_length_max], t, observed_static_c_0, observed_static_c_1)

    # debug_vmap(simulate_cancer_volume, in_axes=(0,0,None,0,0), args=(cancer_volume[:,0], treatment_action[:,:sequence_length_max], t, observed_static_c_0, observed_static_c_1))

    def compute_batched_mse(true, pred, sequence_length):
        # return jnp.mean(jnp.power(true[:sequence_length-1] - pred[:sequence_length-1], 2))
        return jnp.mean(jnp.power(jax.lax.dynamic_slice(true, (0,), (seq_length - 2,)) - jax.lax.dynamic_slice(pred, (0,), (seq_length - 2,)), 2))

    true_mse = vmap(compute_batched_mse, in_axes=(0,0,0))(cancer_volume, cancer_volumes_recreate, sequence_lengths)
    # true_mse = debug_vmap(compute_batched_mse, in_axes=(0,0,0), args=(cancer_volume, cancer_volumes_recreate, sequence_lengths))

    print('[check_factual_data_with_oracle]: MSE error: ', true_mse.mean())
    return true_mse.mean()

def determine_equation_coeffs(args, global_coefs=None, dt=None):
    i, action, X, U = args
    coefs = SINDy(optimizer=LSQIntialMask(threshold=sindy_threshold, alpha=sindy_alpha, max_iter=100, ridge_kw={'tol': 1e-6}, initial_guess=global_coefs),  differentiation_method=FiniteDifference(is_uniform=True, order=2)).fit(X, u=U, t=dt).coefficients()
    # SINDy(optimizer=LSQIntialMask(threshold=sindy_threshold, alpha=.001, max_iter=100, ridge_kw={'tol': 1e-6, 'solver': 'lbfgs', 'positive': True}, initial_guess=global_coefs),  differentiation_method=SmoothedFiniteDifference(smoother_kws=smoother_kws, is_uniform=True, order=4)).fit(X, u=U, t=dt).coefficients()
    if np.abs(coefs).sum() > 10:
        # Handle numerical overflow issues of ill-conditioned matrices
        coefs = SINDy(optimizer=LSQIntialMask(threshold=sindy_threshold, alpha=sindy_alpha, max_iter=100, ridge_kw={'tol': 1e-6}, initial_guess=global_coefs),  differentiation_method=FiniteDifference(is_uniform=True, order=2)).fit(X, u=U, t=dt, unbias=False).coefficients()
        # print('Error coefs ', coefs)
        return (i, action, coefs)
        # return (None, None)
    else:
        return (i, action, coefs)
    
def determine_individualized_equation_coefs(args, global_coefs=None, dt=None):
    i, action, X, U = args
    model = SINDy(optimizer=LSQIntialMask(threshold=sindy_threshold, alpha=sindy_alpha, max_iter=100, ridge_kw={'tol': 1e-6}, initial_guess=global_coefs),  differentiation_method=SmoothedFiniteDifference(smoother_kws=smoother_kws, is_uniform=True, order=4), feature_library=PolynomialLibrary(degree=2, interaction_only=True)).fit(X, u=U, t=dt)
    coefs = model.coefficients()
    if np.abs(coefs).sum() > 10:
        # Handle numerical overflow issues of ill-conditioned matrices
        model = SINDy(optimizer=LSQIntialMask(threshold=sindy_threshold, alpha=sindy_alpha, max_iter=100, ridge_kw={'tol': 1e-6}, initial_guess=global_coefs),  differentiation_method=SmoothedFiniteDifference(smoother_kws=smoother_kws, is_uniform=True, order=4), feature_library=PolynomialLibrary(degree=2, interaction_only=True)).fit(X, u=U, t=dt, unbias=False)
        coefs = model.coefficients()
    expr, str_exp = convert_sindy_model_to_sympy_model(model)
    return (i, action, expr, str_exp)

def predict_and_evalauate_individualised_trajectories(args, global_model=None, dt=None, sequence_length_max=None):
    device = jax.devices("cpu")[0]
    global_coefs = global_model.coefficients()

    i, action, X, U = args
    seq_length = MAX_SEQUENCE_LENGTH
    model = SINDy(optimizer=LSQIntialMask(threshold=sindy_threshold, alpha=sindy_alpha, max_iter=100, ridge_kw={'tol': 1e-6}, initial_guess=global_coefs),  differentiation_method=SmoothedFiniteDifference(smoother_kws=smoother_kws, is_uniform=True, order=4), feature_library=PolynomialLibrary(degree=2, interaction_only=True)).fit(X, u=U, t=dt)
    coefs = model.coefficients()
    if np.abs(coefs).sum() > 10:
        # Handle numerical overflow issues of ill-conditioned matrices
        model = SINDy(optimizer=LSQIntialMask(threshold=sindy_threshold, alpha=sindy_alpha, max_iter=100, ridge_kw={'tol': 1e-6}, initial_guess=global_coefs),  differentiation_method=SmoothedFiniteDifference(smoother_kws=smoother_kws, is_uniform=True, order=4), feature_library=PolynomialLibrary(degree=2, interaction_only=True)).fit(X, u=U, t=dt, unbias=False)
        coefs = model.coefficients()
    expr, str_exp = convert_sindy_model_to_sympy_model(model, quantize=False)
    mod = sympy2jax.SymbolicModule([expr])
    def pred_dy_dt(y, t, observed_C_0, observed_C_1):
        return mod(x0=y, u0=observed_C_0, u1=observed_C_1)[0]
    pred_dy_dt = jax.tree_util.Partial(pred_dy_dt)

    dt = MAX_TIME_HORIZON / seq_length
    t = np.arange(0,sequence_length_max)[:-1] * dt

    # Evaluate on data
    @partial(jit, device=device)
    def simulate_cancer_volume(initial_volume, t, observed_static_c_0, observed_static_c_1):
        return odeint(pred_dy_dt, initial_volume, t, observed_static_c_0, observed_static_c_1, hmax=HMAX, **integrator_keywords)
    

    cancer_volumes_recreate = simulate_cancer_volume(X[0], t, U[0,0], U[0,1])
    cancer_volumes_recreate = np.asarray(cancer_volumes_recreate)
    # cancer_volumes_recreate = vmap(simulate_cancer_volume, in_axes=(0,None,0,0))(X[0], t, U[:,0], U[:,1])

    mse = jnp.mean(jnp.power(cancer_volumes_recreate - X, 2))
    print('MSE error: ', mse)
    mse_val = mse.item()
    return mse_val

def predict_and_evalauate_individualised_trajectories_using_gradient_refinement(args, coefs=None, feature_library_names=None, feature_names=None, dt=None, sequence_length_max=None):
    device = jax.devices("cpu")[0]
    i, action, X, U = args
    seq_length = MAX_SEQUENCE_LENGTH
    # Have model passed in as global coefs
    non_zero_indexes = jnp.nonzero(np.abs(coefs)>1e-3)[0]
    reduced_coefs = coefs[non_zero_indexes]
    feature_library_names = [fn.replace(' ', '*') for fn in feature_library_names]
    fln_l = []
    for fln in feature_library_names:
        for i in range(len(feature_names)):
            fln = fln.replace(f'x{i}', feature_names[i])
        fln_l.append(fln)
    feature_library_names = fln_l
    bases_funcs = [sympy2jax.SymbolicModule(sympify(fl)) for fl in feature_library_names]
    def wrapper_bf(bf):
        def bf_wrapper(x0, u0, u1):
            return jnp.asarray(bf(x0=x0, u0=u0, u1=u1), dtype=jnp.float64).reshape()
        return bf_wrapper
    bases_funcs = [wrapper_bf(bf) for bf in bases_funcs]
    ## Debug Pre-compile checks of shapes etc
    # bases_funcs_check = [bf(X[0], U[0,0], U[0,1]) for bf in bases_funcs]
    # print('Basis functions check: ', bases_funcs_check)
    def dy_dt_func(y, t, u, reduced_coefs):
        result = 0
        for i, non_zero_idx in enumerate(non_zero_indexes):
            base_func_res = lax.switch(non_zero_idx, bases_funcs, y, u[0], u[1])
            result += reduced_coefs[i] * base_func_res
        return jnp.sum(result)
    dy_dt_func = jax.tree_util.Partial(dy_dt_func)
    dt = MAX_TIME_HORIZON / seq_length
    t = jnp.arange(0, MAX_TIME_HORIZON, dt).astype(jnp.float64)[:X.shape[0]]
    X_pred = jit(odeint)(dy_dt_func, X[0], t, U[0], reduced_coefs, hmax=HMAX, **integrator_keywords)
    mse = jnp.mean(jnp.power(X_pred - X, 2))
    # print('Starting mse: ', mse)

    @partial(jit, device=device)
    def fun(reduced_coefs):
        y_pred = odeint(dy_dt_func, X[0], t, U[0], reduced_coefs, hmax=HMAX, **integrator_keywords)
        return jnp.mean(jnp.power(X - y_pred, 2))
    t0 = time()
    res = minimize(fun, reduced_coefs, method='BFGS', tol=1e-6)
    # print('Time taken: ', time() - t0)
    reduced_coefs_updated = res.x
    # @jit
    # def test_jit(reduced_coefs):
    #     res = minimize(fun, reduced_coefs, method='BFGS', tol=1e-6)
    #     return res.x
    # reduced_coefs_updated = test_jit(reduced_coefs)
    X_pred = jit(odeint)(dy_dt_func, X[0], t, U[0], reduced_coefs_updated, hmax=HMAX, **integrator_keywords)
    mse = jnp.mean(jnp.power(X_pred - X, 2))
    # print('Optimized mse: ', mse)
    mse_val = mse.item()
    return mse_val

def train_and_evaluate_sindy(training_data, validation_data, test_data, joint=False):
    from pysindy import SINDy
    from pysindy.optimizers import STLSQ, SR3, SSR, FROLS
    from pysindy.differentiation import FiniteDifference, SmoothedFiniteDifference, SpectralDerivative
    from utils import LSQIntialMask
    sequence_length_max = jnp.array(training_data['sequence_lengths']).astype(jnp.int64).max()


    # Train model on training data
    if not joint:
        individualized_equation_arg_list, X_0, U_0, X_1, U_1, dt = process_dataset_into_de_format_old_format(training_data, sequence_lengths_offset=1, smooth=smooth_input_data, joint=joint)
    else:
        individualized_equation_arg_list, X, U, dt = process_dataset_into_de_format_old_format(training_data, sequence_lengths_offset=1, smooth=smooth_input_data, joint=joint)

    if not joint:
        model_0 = SINDy(optimizer=STLSQ(threshold=sindy_threshold, alpha=sindy_alpha, max_iter=100, ridge_kw={'tol': 1e-6}),  differentiation_method=SmoothedFiniteDifference(smoother_kws=smoother_kws, is_uniform=True, order=4), feature_library=PolynomialLibrary(degree=2, interaction_only=True)).fit(X_0, u=U_0, t=dt, multiple_trajectories=True)
        model_1 = SINDy(optimizer=STLSQ(threshold=sindy_threshold, alpha=sindy_alpha, max_iter=100, ridge_kw={'tol': 1e-6}),  differentiation_method=SmoothedFiniteDifference(smoother_kws=smoother_kws, is_uniform=True, order=4), feature_library=PolynomialLibrary(degree=2, interaction_only=True)).fit(X_1, u=U_1, t=dt, multiple_trajectories=True)
        mod_0, str_0 = convert_sindy_model_to_sympyjax_model(model_0, quantize=False)
        mod_1, str_1 = convert_sindy_model_to_sympyjax_model(model_1, quantize=False)
        joint_coefs = (model_0.coefficients()[0] + model_1.coefficients()[0]) / 2.0
        feature_library_names = model_0.feature_library.get_feature_names().copy() # Same for both models
        feature_names = model_0.feature_names # Same for both models
        print(f'[Model Raw]: Treatment 0: x_dot = {str_0} \t | Treatment 1: x_dot = {str_1}')
    else:
        model_0 = SINDy(optimizer=STLSQ(threshold=sindy_threshold, alpha=sindy_alpha, max_iter=100, ridge_kw={'tol': 1e-6}),  differentiation_method=SmoothedFiniteDifference(smoother_kws=smoother_kws, is_uniform=True, order=4), feature_library=PolynomialLibrary(degree=2, interaction_only=True)).fit(X, u=U, t=dt, multiple_trajectories=True)
        mod_0, str_0 = convert_sindy_model_to_sympyjax_model(model_0, quantize=False)
        joint_coefs = model_0.coefficients()[0]
        feature_library_names = model_0.feature_library.get_feature_names().copy() # Same for both models
        feature_names = model_0.feature_names # Same for both models
        print(f'[Model Raw]: Joint Model: x_dot = {str_0}')

    # mod_0, str_0 = convert_sindy_model_to_sympyjax_model(model_0, quantize=True, quantize_round_to=2)
    # mod_1, str_1 = convert_sindy_model_to_sympyjax_model(model_1, quantize=True, quantize_round_to=2)
    # print(f'[Model Quantized]: Treatment 0: x_dot = {str_0} \t | Treatment 1: x_dot = {str_1}')

    if not joint:
        def pred_dy_dt(y, t, treatment, observed_C_0, observed_C_1):
            return jax.lax.cond(treatment == 0,
                                lambda _: mod_0(x0=y, u0=observed_C_0, u1=observed_C_1)[0],
                                lambda _: mod_1(x0=y, u0=observed_C_0, u1=observed_C_1)[0],
                                operand=None)
    else:
        def pred_dy_dt(y, t, treatment, observed_C_0, observed_C_1):
            return mod_0(x0=y, u0=treatment, u1=observed_C_0, u2=observed_C_1)[0]

    pred_dy_dt = jax.tree_util.Partial(pred_dy_dt)
    train_mse = check_factual_data_with_oracle(training_data, pred_dy_dt, smooth=smooth_input_data, joint=joint)
    val_mse = check_factual_data_with_oracle(validation_data, pred_dy_dt, smooth=smooth_input_data, joint=joint)
    test_mse = check_factual_data_with_oracle(test_data, pred_dy_dt, smooth=smooth_input_data, joint=joint)

    # raise NotImplementedError
    # Test if can process individual trajectories in parallel with Jax
    individualized_equation_arg_list, X_0, U_0, X_1, U_1, dt = process_dataset_into_de_format(validation_data, sequence_lengths_offset=1, smooth=smooth_input_data)
    # pool_outer = multiprocessing.Pool(multiprocessing.cpu_count())

    DEBUG = True
    # determine_equation_coeffs_partial = partial(predict_and_evalauate_individualised_trajectories, global_model=model_0, dt=dt, sequence_length_max=sequence_length_max)
    determine_equation_coeffs_partial = partial(predict_and_evalauate_individualised_trajectories_using_gradient_refinement, coefs=joint_coefs, feature_library_names=feature_library_names, feature_names=feature_names, dt=dt, sequence_length_max=sequence_length_max)
    if DEBUG:
        results = []
        for args in tqdm(individualized_equation_arg_list):
            results.append(determine_equation_coeffs_partial(args))
    else:
        pool_outer = multiprocessing.Pool(multiprocessing.cpu_count())
        results = list(tqdm(pool_outer.imap(determine_equation_coeffs_partial, individualized_equation_arg_list), total=len(individualized_equation_arg_list)))
        pool_outer.close()

    mse = np.array(results).mean()
    print('Validation MSE: ', mse)

    individualized_equation_arg_list, X_0, U_0, X_1, U_1, dt = process_dataset_into_de_format(test_data, sequence_lengths_offset=1, smooth=smooth_input_data)

    determine_equation_coeffs_partial = partial(predict_and_evalauate_individualised_trajectories_using_gradient_refinement,coefs=joint_coefs, feature_library_names=feature_library_names, feature_names=feature_names, dt=dt, sequence_length_max=sequence_length_max)
    if DEBUG:
        results = []
        for args in tqdm(individualized_equation_arg_list):
            results.append(determine_equation_coeffs_partial(args))
    else:
        pool_outer = multiprocessing.Pool(multiprocessing.cpu_count())
        results = list(tqdm(pool_outer.imap(determine_equation_coeffs_partial, individualized_equation_arg_list), total=len(individualized_equation_arg_list)))
        pool_outer.close()

    mse = np.array(results).mean()
    print('Test MSE: ', mse)

# def train_and_evaluate_sindy_ours_seq(training_data, validation_data, seq_test_data):
#     from pysindy import SINDy
#     from pysindy.optimizers import STLSQ, SR3, SSR, FROLS
#     from pysindy.differentiation import FiniteDifference, SmoothedFiniteDifference, SpectralDerivative
#     from utils import LSQIntialMask
#     sequence_length_max = jnp.array(training_data['sequence_lengths']).astype(jnp.int64).max()

#     # Train model on training data
#     individualized_equation_arg_list, X_0, U_0, X_1, U_1, dt = process_dataset_into_de_format(training_data, sequence_lengths_offset=1, smooth=smooth_input_data)

#     model_0 = SINDy(optimizer=STLSQ(threshold=sindy_threshold, alpha=sindy_alpha, max_iter=100, ridge_kw={'tol': 1e-6}),  differentiation_method=SmoothedFiniteDifference(smoother_kws=smoother_kws, is_uniform=True, order=4), feature_library=PolynomialLibrary(degree=2, interaction_only=True)).fit(X_0, u=U_0, t=dt, multiple_trajectories=True)

#     model_1 = SINDy(optimizer=STLSQ(threshold=sindy_threshold, alpha=sindy_alpha, max_iter=100, ridge_kw={'tol': 1e-6}),  differentiation_method=SmoothedFiniteDifference(smoother_kws=smoother_kws, is_uniform=True, order=4), feature_library=PolynomialLibrary(degree=2, interaction_only=True)).fit(X_1, u=U_1, t=dt, multiple_trajectories=True)

#     mod_0, str_0 = convert_sindy_model_to_sympyjax_model(model_0, quantize=False)
#     mod_1, str_1 = convert_sindy_model_to_sympyjax_model(model_1, quantize=False)

#     joint_coefs = (model_0.coefficients()[0] + model_1.coefficients()[0]) / 2.0
#     feature_library_names = model_0.feature_library.get_feature_names().copy() # Same for both models
#     feature_names = model_0.feature_names # Same for both models

#     print(f'[Model]: Treatment 0: x_dot = {str_0} \t | Treatment 1: x_dot = {str_1}')

#     def pred_dy_dt(y, t, treatment, observed_C_0, observed_C_1):
#         return jax.lax.cond(treatment == 0,
#                             lambda _: mod_0(x0=y, u0=observed_C_0, u1=observed_C_1)[0],
#                             lambda _: mod_1(x0=y, u0=observed_C_0, u1=observed_C_1)[0],
#                             operand=None)
#     pred_dy_dt = jax.tree_util.Partial(pred_dy_dt)
#     train_mse = check_factual_data_with_oracle(training_data, pred_dy_dt, smooth=smooth_input_data)
#     val_mse = check_factual_data_with_oracle(validation_data, pred_dy_dt, smooth=smooth_input_data)
#     test_mse = check_data_counterfactuals_with_oracle(seq_test_data, pred_dy_dt, smooth=smooth_input_data)

#     # Test if can process individual trajectories in parallel with Jax
#     individualized_equation_arg_list, X_0, U_0, X_1, U_1, dt = process_dataset_into_de_format(validation_data, sequence_lengths_offset=1, smooth=smooth_input_data)
#     # pool_outer = multiprocessing.Pool(multiprocessing.cpu_count())

#     DEBUG = False
#     # determine_equation_coeffs_partial = partial(predict_and_evalauate_individualised_trajectories, global_model=model_0, dt=dt, sequence_length_max=sequence_length_max)
#     determine_equation_coeffs_partial = partial(predict_and_evalauate_individualised_trajectories_using_gradient_refinement, coefs=joint_coefs, feature_library_names=feature_library_names, feature_names=feature_names, dt=dt, sequence_length_max=sequence_length_max)
#     if DEBUG:
#         results = []
#         for args in tqdm(individualized_equation_arg_list):
#             results.append(determine_equation_coeffs_partial(args))
#     else:
#         pool_outer = multiprocessing.Pool(multiprocessing.cpu_count())
#         results = list(tqdm(pool_outer.imap(determine_equation_coeffs_partial, individualized_equation_arg_list), total=len(individualized_equation_arg_list)))
#         pool_outer.close()

#     mse = np.array(results).mean()
#     print('Validation MSE: ', mse)

#     individualized_equation_arg_list, X_0, U_0, X_1, U_1, dt = process_dataset_into_de_format(seq_test_data, sequence_lengths_offset=1, smooth=smooth_input_data)

#     determine_equation_coeffs_partial = partial(predict_and_evalauate_individualised_trajectories_using_gradient_refinement,coefs=joint_coefs, feature_library_names=feature_library_names, feature_names=feature_names, dt=dt, sequence_length_max=sequence_length_max)
#     if DEBUG:
#         results = []
#         for args in tqdm(individualized_equation_arg_list):
#             results.append(determine_equation_coeffs_partial(args))
#     else:
#         pool_outer = multiprocessing.Pool(multiprocessing.cpu_count())
#         results = list(tqdm(pool_outer.imap(determine_equation_coeffs_partial, individualized_equation_arg_list), total=len(individualized_equation_arg_list)))
#         pool_outer.close()

#     mse = np.array(results).mean()
#     print('Test MSE: ', mse)

def check_data_counterfactuals_with_oracle(data, dy_dt, smooth=False):
    cancer_volume = jnp.array(data['cancer_volume']).astype(jnp.float64)
    treatment_action = jnp.array(data['treatment_application']).astype(jnp.int64)
    sequence_lengths = jnp.array(data['sequence_lengths']).astype(jnp.int64)
    observed_static_c_0 = jnp.array(data['observed_static_c_0']).astype(jnp.float64)
    observed_static_c_1 = jnp.array(data['observed_static_c_1']).astype(jnp.float64)
    
    dt = MAX_TIME_HORIZON / seq_length
    t = jnp.arange(0, cancer_volume.shape[1]).astype(jnp.float64) * dt

    if smooth:
        smoother_kws_ = smoother_kws.copy()
        smoother_kws_.update({'axis': 1}) 
        cancer_volume = jnp.array(savgol_filter(cancer_volume, **smoother_kws))

    @jit
    def simulate_cancer_volume(initial_volume, treatment, t, observed_static_c_0, observed_static_c_1):
        # print('initial_volume ', initial_volume.shape)
        # print('treatment ', treatment.shape)
        # print('t ', t.shape)
        # print('observed_static_c_0 ', observed_static_c_0.shape)
        # print('observed_static_c_1 ', observed_static_c_1.shape)
        
        iter_tuples = jnp.array([(t[i-1], t[i], treatment[i-1]) for i in range(1, t.shape[0])])
        carry_init = (initial_volume, observed_static_c_0, observed_static_c_1)

        def scan_fn(carry, iter_tuple):
            cancer_volume, observed_static_c_0, observed_static_c_1 = carry
            t_start, t_end, treatment = iter_tuple
            t_tuple = jnp.array([t_start, t_end])
            # print('treatment ', treatment)
            # jax.debug.print("treatment {treatment} ", treatment=treatment)
            # jax.debug.print("cancer_volume {cancer_volume} ", cancer_volume=cancer_volume)
            # jax.debug.print("=====================================")
            cancer_volume = odeint(dy_dt, cancer_volume, t_tuple, treatment, observed_static_c_0, observed_static_c_1, hmax=HMAX, **integrator_keywords)[1]
            return (cancer_volume, observed_static_c_0, observed_static_c_1), cancer_volume
        
        _, predicted_volumes = scan(scan_fn, carry_init, iter_tuples)
        predicted_volumes = jnp.concatenate([initial_volume[None, ...], predicted_volumes], axis=0)
        return predicted_volumes

    cancer_volumes_recreate = vmap(simulate_cancer_volume, in_axes=(0,0,None,0,0))(cancer_volume[:,0], treatment_action, t, observed_static_c_0, observed_static_c_1)
    # cancer_volumes_recreate = debug_vmap(simulate_cancer_volume, in_axes=(0,0,None,0,0), args=(cancer_volume[:,0], treatment_action[:,:sequence_length_max], t, observed_static_c_0, observed_static_c_1))

    # # @partial(jit, static_argnums=(2,))
    # def compute_batched_mse(true, pred, sequence_length):
    #     # return jnp.mean(jnp.power(true[:sequence_length-1] - pred[:sequence_length-1], 2))
    #     # jax.debug.print("sequence_length {sequence_length} ", sequence_length=sequence_length)
    #     return jnp.mean(jnp.power(jax.lax.dynamic_slice(true, (0,), (sequence_length,)) - jax.lax.dynamic_slice(pred, (0,), (sequence_length,)), 2))

    # true_mse = vmap(compute_batched_mse, in_axes=(0,0,0))(cancer_volume, cancer_volumes_recreate, sequence_lengths)
    # # true_mse = debug_vmap(compute_batched_mse, in_axes=(0,0,0), args=(cancer_volume, cancer_volumes_recreate, sequence_lengths))

    # @partial(jit, static_argnums=(2,))
    def compute_batched_mse(true, pred, sequence_length):
        return jnp.mean(jnp.power(jax.lax.dynamic_slice(true, (0,), (sequence_length + 1,)) - jax.lax.dynamic_slice(pred, (0,), (sequence_length + 1,)), 2))

    # @jit
    def batched_mse_for_lengths(true, pred, sequence_lengths):
        results = []
        for i in range(true.shape[0]): # Can speed this up with a scan, not sure why vmap failed?!
            results.append(compute_batched_mse(true[i], pred[i], sequence_lengths[i].item()))
        return jnp.array(results)

    true_mse = batched_mse_for_lengths(cancer_volume, cancer_volumes_recreate, sequence_lengths)

    print('[check_data_counterfactuals_with_oracle]: MSE error: ', true_mse.mean())
    print('')


if __name__ == "__main__":
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    multiprocessing.set_start_method('spawn')

    SEED = 50
    # key = jax.random.PRNGKey(100)
    key = jax.random.PRNGKey(SEED)
    device = jax.devices()[0]
    key = device_put(key, device)
    
    seq_length = MAX_SEQUENCE_LENGTH  # about half a year
    window_size = 15
    lag = 0  # lag of treatment assignment
    # num_patients = 10_000
    num_patients = 1000
    # num_patients = 10
    for conf_coeff in [50]:
    # for conf_coeff in [0, 1, 10, 100, 1000, 10000]:
        for equation in Equation:
            print('Now running: ', equation.name)
            # if True:
            # if 'EQ_4' in equation.name:
            if equation.name == 'EQ_4_D':
            # if equation.name == 'EQ_4_M':
            # if equation.name.split('_')[-1] in ['B', 'C', 'D']:
            # if equation.name.split('_')[-1] in ['C', 'D']:
            # if equation.name.split('_')[-1] in ['D']:
            # if equation.name.split('_')[-1] in ['D']:
            # # with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
                key, subkey = random.split(key)
                params = generate_params(num_patients, conf_coeff=conf_coeff, window_size=window_size, lag=lag, equation=equation, key=subkey)
                key, subkey = random.split(key)
                training_data = simulate_factual(params, seq_length, equation=equation, key=subkey)
                check_factual_data_with_oracle(training_data, dy_dt)        

                key, subkey = random.split(key)
                params = generate_params(int(num_patients / 10), conf_coeff=conf_coeff, window_size=window_size, lag=lag, equation=equation, key=subkey)
                key, subkey = random.split(key)
                validation_data = simulate_factual(params, seq_length, equation=equation, key=subkey)
                check_factual_data_with_oracle(validation_data, dy_dt)

                key, subkey = random.split(key)
                params = generate_params(int(num_patients / 10), conf_coeff=conf_coeff, window_size=window_size, lag=lag, equation=equation, key=subkey)
                key, subkey = random.split(key)
                test_data_factuals = simulate_factual(params, seq_length, equation=equation, key=subkey)
                check_factual_data_with_oracle(test_data_factuals, dy_dt)

                # # Test SINDY and our method
                # train_and_evaluate_sindy(training_data, validation_data, test_data_factuals, joint=True)

                key, subkey = random.split(key)
                test_data_counterfactuals = simulate_counterfactual_1_step(params, seq_length, equation=equation, key=subkey)
                # check_data_counterfactuals_with_oracle(test_data_counterfactuals, dy_dt)

                key, subkey = random.split(key)
                params = generate_params(int(num_patients / 10), conf_coeff=conf_coeff, window_size=window_size, lag=lag, equation=equation, key=subkey)
                params['window_size'] = window_size
                key, subkey = random.split(key)
                test_data_seq = simulate_counterfactuals_treatment_seq(params, seq_length, 5, equation=equation, key=subkey)
                # check_data_counterfactuals_with_oracle(test_data_seq, dy_dt)
                # train_and_evaluate_sindy_ours_seq(training_data, validation_data, test_data_seq)
                test_data_seq

                # Want to test MSE or exact equation on data. As worried that sequence length biases the ODE discovery method, as values get clipped to 0---violating ODEs.

                # # Plot patient
                # plot_treatments(training_data, 572)
                # plot_sigmoid_function({})
                # print('')

                # patient = 572
                # df = pd.DataFrame({'N(t)': training_data['cancer_volume'][patient],
                #                 'C(t)': training_data['chemo_dosage'][patient],
                #                 'd(t)': training_data['radio_dosage'][patient],
                #                 })
                # df = df[['N(t)', "C(t)", "d(t)"]]
                # df.plot(secondary_y=['C(t)', 'd(t)'])
                # plt.xlabel("$t$")
                # plt.show()
                # plt.savefig(f'treatments_{patient}.png')


                # Idea; JIT, make everything functional, feed the treatment and static covariates to the ODE as arguments.