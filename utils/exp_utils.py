import time
import random
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf
from scipy.integrate import solve_ivp
from functools import partial
import pysindy as ps
from collections import deque
from tqdm import tqdm
import shelve

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

def config_to_dict(config):
    """
    Convert a config object to a dictionary.
    """
    return OmegaConf.to_container(config, resolve=True)

def dict_to_config(d):
    """
    Convert a dictionary to a config object.
    """
    return OmegaConf.create(d)

def load_dataset(dataset_name, seed, config, load_from_cache=False, force_recache=False, gamma=1.0):
    """
    Load the dataset specified in config.
    """
    path = f'{dataset_name}_{config.run.train_samples}_{config.run.val_samples}_{config.run.test_samples}_{config.run.step_actions}_{gamma}'
    if force_recache:
        df = load_dataset_internal(dataset_name, seed, config, gamma)
        with shelve.open("datasets") as db:
            db[path] = df
            return df
    elif load_from_cache:
        try:
            with shelve.open("datasets") as db:
                return db[path]
        except KeyError:
            df = load_dataset_internal(dataset_name, seed, config, gamma)
            with shelve.open("datasets") as db:
                db[path] = df
                return df
    else:
        return load_dataset_internal(dataset_name, seed, config, gamma)


def load_dataset_internal(dataset_name, seed, config, gamma=1.0):
    """
    Load the dataset specified in config.
    """
    print(f'Loading dataset {dataset_name} with seed {seed}')
    if dataset_name == 'eq_1':
        return load_eq_single_pkpd_updated(seed, config, gamma=gamma)
    elif dataset_name == 'eq_2':
        return load_eq_single_pkpd_updated(seed, config, gamma=gamma, obs_noise=config.run.obs_noise)
    elif dataset_name == 'eq_3':
        return load_eq_single_pkpd_updated(seed, config, gamma=gamma, bsv_noise=config.run.bsv_noise)
    elif dataset_name == 'eq_4':
        return load_eq_single_pkpd_updated(seed, config, gamma=gamma, bsv_noise=config.run.bsv_noise, fractional_weight=True)
    if dataset_name == 'eq_5':
        return load_eq_double_pkpd_updated(seed, config, gamma=gamma)
    elif dataset_name == 'eq_6':
        return load_eq_double_pkpd_updated(seed, config, gamma=gamma, obs_noise=config.run.obs_noise)
    elif dataset_name == 'eq_7':
        return load_eq_double_pkpd_updated(seed, config, gamma=gamma, bsv_noise=config.run.bsv_noise)
    elif dataset_name == 'eq_8':
        return load_eq_double_pkpd_updated(seed, config, gamma=gamma, bsv_noise=config.run.bsv_noise, fractional_weight=True)
    else:
        raise NotImplementedError

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_action(y_average, gamma=1.0, max_covariate=1, action_dim=1):
    """Get action with time dependent confounding"""
    p_u = sigmoid(gamma * (1 / max_covariate) * (y_average - max_covariate/2.0))
    u_val = np.random.binomial(1, p=p_u, size=(1,action_dim))
    # chemo_prob = 1.0 / (1.0+ np.exp(-(chemo_coeff / d_max)* (cancer_metric_used - (d_max / 2.0))))
    # Discrepancy arises, due to the maximum needs to be capped, therefore the action can then have equal probaility within the range of both.
    return u_val

def solve_ivp_euler_sim(f, t_span, y0, t_eval, args=(), action_mean_window=15, max_covariate=15, gamma=1.0, step_actions=30.0, action_dim=1):
    """Solving ODE with Euler method"""
    step_actions, action_mean_window = int(step_actions), int(action_mean_window)
    # y0 = np.array(y0)
    t0, tf = t_span
    dt = t_eval[1] - t_eval[0]
    t = t0
    y = y0
    y_sol = [y0]
    action_buffer = deque(maxlen=step_actions)
    action_buffer.extend(get_action(np.mean(np.stack(y_sol[-action_mean_window:])[:,0]), gamma=gamma, max_covariate=max_covariate, action_dim=action_dim).repeat(step_actions, axis=0))
    u = action_buffer.popleft()
    u_sol = [u]
    t_sol = [t]
    for _ in range(t_eval.shape[0] - 1):
        y = y + dt * np.squeeze(f(t, y, u))
        if len(action_buffer) == 0:
            action_buffer.extend(get_action(np.mean(np.stack(y_sol[-action_mean_window:])[:,0]), gamma=gamma, max_covariate=max_covariate, action_dim=action_dim).repeat(step_actions, axis=0))
        u = action_buffer.popleft()
        y_sol.append(y)
        u_sol.append(u)
        t += dt
        t_sol.append(t)
    return np.array(y_sol), np.array(u_sol), np.array(t_sol)

def sample_patients(eq_skeleton, get_equation_with_sampled_parameters, eq_sample_random_state, t_train, sample_size=100, observation_noise_std=0.0, gamma=1.0, step_actions=1.0, integrator_keywords={}, bsv_noise=0.0, bsv_std=0.0, fractional_weight=False, action_dim=1, max_covariate=15):
    t_train_span = (t_train[0], t_train[-1])
    dt = t_train[1] - t_train[0]
    states, actions = [], []

    for x0 in tqdm(eq_sample_random_state(sample_size)):
    # for x0 in tqdm(np.random.randint(0, 10, size=sample_size)):
        eq = get_equation_with_sampled_parameters(eq_skeleton, bsv_noise, fractional_weight)
        def eq_u(t, x, u_fun):
            return eq(t, x, u_fun(t, x))
        x, u, t = solve_ivp_euler_sim(eq, t_train_span, x0, t_eval=t_train, gamma=gamma, step_actions=step_actions, max_covariate=max_covariate, action_dim=action_dim, **integrator_keywords)
        def u_fun(t, x):
            if np.where(t_train > t)[0].size == 0:
                return u[-1].reshape(1,-1)
            else:
                t_idx = int(np.where(t_train > t)[0][0] - 1)
                return u[t_idx].reshape(1,-1)
        x_h = solve_ivp(eq_u, t_train_span, x0, t_eval=t_train, args=(u_fun,), max_step=dt/10.0, **integrator_keywords).y.T
        if observation_noise_std > 0.0:
            x_h += np.random.normal(0.0, observation_noise_std, size=x_h.shape)
        states.append(x_h)
        actions.append(u)
    return np.stack(states), np.array(actions)


def load_eq_single_pkpd_updated(seed, config, obs_noise=0.0, bsv_noise=0.0, fractional_weight=False, gamma=1.0):
    multiplier_time = 1
    t_f  = int(10 * multiplier_time)
    total_time_steps = int(60 * multiplier_time)
    t_train = np.linspace(0, t_f, total_time_steps)
    # Only one action
    action_dim = 1
    max_covariate = 15

    def eq_skeleton(t, x, u, c_1, v, c_0):
        return [
            x[0] * (u * (c_0 / v - c_1 / v) - c_0 / v),
        ]

    def get_equation_with_sampled_parameters(eq_skeleton, bsv_noise=0.0, fractional_weight=False):
        C_1 = 1
        V = 1
        C_0 = -1

        bsv_std = 1.0
        w_0_mean = 1.0
        if bsv_noise > 0.0 and not fractional_weight:
            c_1 = C_1 + np.random.normal(0, bsv_noise) * bsv_std
            c_0 = C_0 + np.random.normal(0, bsv_noise) * bsv_std
            v = V + np.random.normal(0, bsv_noise) * bsv_std
        elif bsv_noise > 0.0 and fractional_weight:
            c_1 = C_1 * (np.random.normal(w_0_mean, bsv_noise) / w_0_mean)**(bsv_std)
            c_0 = C_0 * (np.random.normal(w_0_mean, bsv_noise) / w_0_mean)**(bsv_std)
            v = V * (np.random.normal(w_0_mean, bsv_noise) / w_0_mean)**(bsv_std)
        else:
            c_1 = C_1
            c_0 = C_0
            v = V
        eq = partial(eq_skeleton, c_1=c_1, v=v, c_0=c_0)
        return eq

    def eq_sample_random_state(sample_size):
        # return np.random.randint(0, 10, size=sample_size)
        return np.random.uniform(low=0, high=10.0, size=(sample_size, 1))

    states_train, actions_train = sample_patients(eq_skeleton, get_equation_with_sampled_parameters, eq_sample_random_state, t_train, sample_size=config.run.train_samples, observation_noise_std=obs_noise, gamma=gamma, step_actions=config.run.step_actions, integrator_keywords={}, bsv_noise=bsv_noise, fractional_weight=fractional_weight, max_covariate=max_covariate)
    states_val, actions_val = sample_patients(eq_skeleton, get_equation_with_sampled_parameters, eq_sample_random_state, t_train, sample_size=config.run.val_samples, observation_noise_std=obs_noise, gamma=0.0, step_actions=config.run.step_actions, integrator_keywords={}, bsv_noise=bsv_noise, fractional_weight=fractional_weight, max_covariate=max_covariate)
    states_test, actions_test = sample_patients(eq_skeleton, get_equation_with_sampled_parameters, eq_sample_random_state, t_train, sample_size=config.run.val_samples, observation_noise_std=obs_noise, gamma=0.0, step_actions=config.run.step_actions, integrator_keywords={}, bsv_noise=bsv_noise, fractional_weight=fractional_weight, max_covariate=max_covariate)
    metadata = {
        'x_dim': states_train.shape[2],
        'action_dim': action_dim,
        'action_type': 'binary',
        't': t_train,
        'total_timesteps': t_train.shape[0],
    }
    train = {'x': states_train, 'a': actions_train, 'y': states_train}
    val = {'x': states_val, 'a': actions_val, 'y': states_val}
    test = {'x': states_test, 'a': actions_test, 'y': states_test}
    return train, val, test, metadata


def load_eq_double_pkpd_updated(seed, config, obs_noise=0.0, bsv_noise=0.0, fractional_weight=False, gamma=1.0):
    multiplier_time = 1
    t_f  = int(10 * multiplier_time)
    total_time_steps = int(60 * multiplier_time)
    t_train = np.linspace(0, t_f, total_time_steps)
    # Two actions
    action_dim = 2

    alpha_r = 0.0398
    max_chemo_drug = 5.0
    max_radio = 2.0

    max_covariate = calc_volume(13)

    def eq_skeleton(t,
                    x,
                    u,
                    rho=7e-5 + 7.23e-3 * 2,
                    K=calc_volume(30),
                    beta_c = 0.028,
                    alpha_r = alpha_r,
                    beta_r=alpha_r /10.0,
                    e_noise=0.0):
        u = np.squeeze(u)
        ca, ra = u[0], u[1]
        v, c = x[0], x[1]

        e = e_noise

        ca = np.clip(ca, a_min=0, a_max=max_chemo_drug)
        ra = np.clip(ra, a_min=0, a_max=max_radio)
        v = np.clip(v, a_min=0, a_max=None)

        dc_dt = - c / 2 + ca
        dv_dt = (-np.log(v) * 0.05) * v
        dv_dt = 0 if v <= 0 else dv_dt
        dv_dt = np.nan_to_num(dv_dt, nan=0.0, neginf=0.0)

        return [
            dv_dt,
            dc_dt
        ]

    def get_equation_with_sampled_parameters(eq_skeleton, bsv_noise=0.0, fractional_weight=False):
        # rho_m = 7e-5 + 7.23e-3 * 2
        rho_m = 7e-5
        K_m = calc_volume(30)
        beta_c_m = 0.028
        alpha_r_m = 0.0398
        # beta_r_m = alpha_r /10.0

        bsv_std = 1.0
        w_0_mean = 1.0
        if bsv_noise > 0.0 and not fractional_weight:
            rho = rho_m + np.random.normal(0, 7.23e-3)
            beta_c = beta_c_m + np.random.normal(0, 0.0007)
            alpha_r = alpha_r_m + np.random.normal(0, 0.168)
        elif bsv_noise > 0.0 and fractional_weight:
            rho = rho_m * (np.random.normal(w_0_mean, 7.23e-3) / w_0_mean)**(bsv_std)
            beta_c = beta_c_m * (np.random.normal(w_0_mean, 0.0007) / w_0_mean)**(bsv_std)
            alpha_r = alpha_r_m * (np.random.normal(w_0_mean, 0.168) / w_0_mean)**(bsv_std)
        else:
            rho = rho_m
            beta_c = beta_c_m
            alpha_r = alpha_r_m
        beta_r = alpha_r / 10.0
        eq = partial(eq_skeleton, rho=rho, K=K_m, beta_c = beta_c, alpha_r = alpha_r, beta_r=beta_r)
        return eq

    def eq_sample_random_state(sample_size):
        dim_0 = np.random.uniform(low=calc_volume(13)*0.80, high=calc_volume(13)*0.99, size=(sample_size,1))
        dim_1 = np.zeros_like(dim_0)
        return np.concatenate((dim_0,dim_1),axis=1)

    states_train, actions_train = sample_patients(eq_skeleton, get_equation_with_sampled_parameters, eq_sample_random_state, t_train, sample_size=config.run.train_samples, observation_noise_std=obs_noise, gamma=gamma, step_actions=config.run.step_actions, integrator_keywords={}, bsv_noise=bsv_noise, fractional_weight=fractional_weight, action_dim=action_dim, max_covariate=max_covariate)
    states_val, actions_val = sample_patients(eq_skeleton, get_equation_with_sampled_parameters, eq_sample_random_state, t_train, sample_size=config.run.val_samples, observation_noise_std=obs_noise, gamma=0.0, step_actions=config.run.step_actions, integrator_keywords={}, bsv_noise=bsv_noise, fractional_weight=fractional_weight, action_dim=action_dim, max_covariate=max_covariate)
    states_test, actions_test = sample_patients(eq_skeleton, get_equation_with_sampled_parameters, eq_sample_random_state, t_train, sample_size=config.run.val_samples, observation_noise_std=obs_noise, gamma=0.0, step_actions=config.run.step_actions, integrator_keywords={}, bsv_noise=bsv_noise, fractional_weight=fractional_weight, action_dim=action_dim, max_covariate=max_covariate)
    metadata = {
        'x_dim': states_train.shape[2],
        'action_dim': action_dim,
        'action_type': 'binary',
        't': t_train,
        'total_timesteps': t_train.shape[0],
    }
    train = {'x': states_train, 'a': actions_train, 'y': states_train[:,:,:1]}
    val = {'x': states_val, 'a': actions_val, 'y': states_val[:,:,:1]}
    test = {'x': states_test, 'a': actions_test, 'y': states_test[:,:,:1]}
    return train, val, test, metadata

def calc_volume(diameter):
    return 4.0 / 3.0 * np.pi * (diameter / 2.0) ** 3.0
