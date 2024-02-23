import hydra
from omegaconf import DictConfig, OmegaConf
from torch import multiprocessing
import os
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={multiprocessing.cpu_count()//2}'
import jax
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)

import numpy as np
import random
from collections import defaultdict
import time

import os
import random
import time
import traceback
import pandas as pd

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from functools import partial
from copy import deepcopy
from enum import Enum

from utils.logging_utils import create_logger_in_process, generate_log_file_path
from utils.exp_utils import seed_all, config_to_dict, dict_to_config
from utils.results_utils import normalize_means, generate_main_results_table

class Experiment(Enum):
    MAIN_TABLE = 1
    INSIGHT_CONFOUNDING = 2
    ABLATION_ONE_ODE = 3
    ABLATION_MORE_COMPLEX_BASIS_FUNCTIONS = 4
    INSIGHT_RECOVER_PARAMETRIC_DIST = 5
    INSIGHT_NOISE = 6
    INSIGHT_LESS_SAMPLES = 7


@hydra.main(version_base=None, config_path="config", config_name="config.yaml")
def run(config: DictConfig) -> None:
    log_path = generate_log_file_path(__file__, log_folder=config.setup.log_dir, config=config)
    logger = create_logger_in_process(log_path)
    config.run.log_path = log_path
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if config.setup.cuda else "cpu"
    config.run.device = str(device)
    if config.setup.debug_mode:
        config.setup.multi_process_results = False    
    if config.setup.multi_process_results:
        if ('sindy' in config.setup.ct_methods_to_evaluate or 'insite' in config.setup.ct_methods_to_evaluate):
            logger.info(f'[WARNING] In DEBUG MODE -- Disabling multiprocessing')
            config.setup.multi_process_results = False
        else:
            multiprocessing.set_start_method('spawn')
            config.setup.wandb.track = False
    if config.setup.wandb.track:
        import wandb
        wandb.init(
            project=config.setup.wandb.project,
            config=config_to_dict,
        )
    else:
        wandb = None
    seed_all(0)
    logger.info(f'Starting run \t | See log at : {log_path}')
    if config.setup.flush_mode:
        logger.info(f'[WARNING] In FLUSH MODE -- TEST RUN ONLY')
        config.run.epochs = 1
        config.setup.seed_start = 0
        config.setup.seed_runs = 1
        # config.setup.load_from_cache = True
        # config.setup.force_recache = False
        config.gnet.mcsamples = 2
        config.run.train_samples = 1000
        config.run.val_samples = 10
        config.run.test_samples = 10
    logger.info(f'[Main Config] {config}')
    main(config, wandb, logger)
    if config.setup.wandb.track:
        wandb.finish()
    logger.info('Run over. Fin.')
    logger.info(f'[Log found at] {log_path}')


def main(config, wandb, logger):
    if config.setup.multi_process_results:
        pool_outer = multiprocessing.Pool(config.setup.multi_process_cores)
    args_for_runs = []
    t0 = time.perf_counter()
    experiment = Experiment[config.setup.experiment]
    if experiment == Experiment.MAIN_TABLE or experiment == Experiment.ABLATION_ONE_ODE or experiment == Experiment.ABLATION_MORE_COMPLEX_BASIS_FUNCTIONS or experiment == Experiment.INSIGHT_RECOVER_PARAMETRIC_DIST:
        for seed in range(config.setup.seed_start, config.setup.seed_runs + config.setup.seed_start):
            for dataset_name in config.setup.ct_datasets_to_evaluate:
                for method_name in config.setup.ct_methods_to_evaluate:
                    if dataset_name == 'cancer_sim' and method_name == 'wsindy':
                        continue
                    elif 'EQ_5' in dataset_name and method_name == 'wsindy':
                        continue
                    args_for_runs.append((dataset_name, method_name, seed, config.run.domain_conf))
    elif experiment == Experiment.INSIGHT_CONFOUNDING:
        dataset_name = 'EQ_4_D'
        for seed in range(config.setup.seed_start, config.setup.seed_runs + config.setup.seed_start):
            for domain_conf in config.setup.domain_confs:
                for method_name in config.setup.ct_methods_to_evaluate:
                    if dataset_name == 'cancer_sim' and method_name == 'wsindy':
                        continue
                    elif 'EQ_5' in dataset_name and method_name == 'wsindy':
                        continue
                    args_for_runs.append((dataset_name, method_name, seed, domain_conf))
    evaluate_policy_single = partial(run_exp_wrapper_outer, config=config, wandb=wandb)
    results = []
    if not config.setup.multi_process_results:
        for args_for_run in args_for_runs:
                result = evaluate_policy_single(args_for_run)
                printable_result = {k : v.tolist() if isinstance(v, np.ndarray) else v for k,v in result.items()}
                logger.info(f'[Exp evaluation complete] {printable_result}')
                results.append(result)
    else:
        for i, result in tqdm(enumerate(pool_outer.imap_unordered(evaluate_policy_single, args_for_runs)), total=len(args_for_runs), smoothing=0):
                printable_result = {k : v.tolist() if isinstance(v, np.ndarray) else v for k,v in result.items()}
                logger.info(f'[Exp evaluation complete] {printable_result}')
                results.append(result)
    time_taken = time.perf_counter() - t0
    logger.info(f'Time taken for all runs: {time_taken}s\t| {time_taken/60.0} minutes')
    if config.setup.multi_process_results:
        pool_outer.close()
    df_results = pd.DataFrame(results)
    tables = generate_main_results_table(df_results)
    logger.info(f'Tables: {tables}')
    print('')
    # print(table_str)
    print('fin.')

def run_exp_wrapper(args, logger, **kwargs):
    (dataset_name, method_name, seed, domain_conf) = args
    seed_all(seed)
    config = kwargs['config']
    config = dict_to_config(deepcopy(OmegaConf.to_container(config, resolve=True)))
    kwargs['config'] = config
    result = run_exp_ct(dataset_name=dataset_name,
                        method_name=method_name,
                        seed=seed,
                        domain_conf=domain_conf,
                        logger=logger,
                        **kwargs)
    result['errored'] = False
    return result

def run_exp_wrapper_outer(args, **kwargs):
    (dataset_name, method_name, seed, domain_conf) = args
    config = kwargs['config']
    logger = create_logger_in_process(config.run.log_path)
    logger.info(f'[Now evaluating exp] {args}')
    if config.setup.debug_mode:
        result = run_exp_wrapper(args, logger, **kwargs)
    else:
        try:
            result = run_exp_wrapper(args, logger, **kwargs)
        except Exception as e:
            logger.exception(f'[Error] {e}')
            logger.info(f"[Failed evaluating exp] {args}\t| error={e}")
            traceback.print_exc()
            result = {'errored': True}
            print('')            
    result.update({'dataset_name': dataset_name, 'seed': seed, 'method_name': method_name, 'domain_conf': domain_conf})
    return result

def run_exp_ct(dataset_name,
            method_name,
            seed,
            domain_conf,
            logger,
            config={}, 
            wandb=None):
    logger.info(f'Running {dataset_name} {method_name} {seed} | domain_conf={domain_conf}')
    domain_conf = int(domain_conf)
    from hydra import compose, initialize
    from omegaconf import OmegaConf

    sindy_threshold = [v for k, v in config.sindy.dataset_params.sindy_threshold.items() if k in dataset_name]
    assert len(sindy_threshold) == 1, 'Must only specify one sindy threshold'
    sindy_threshold = sindy_threshold[0]

    lam = [v for k, v in config.sindy.dataset_params.lam.items() if k in dataset_name]
    assert len(lam) == 1, 'Must only specify one lam'
    lam = lam[0]

    HYPER_PARAMETER_TUNE = False
    experiment = Experiment[config.setup.experiment]

    t00 = time.perf_counter()
    overrides = [f"+backbone={method_name}", f"exp.seed={seed}", f"exp.max_epochs={config.run.epochs}", f"dataset.num_patients.train={config.run.train_samples}", f"dataset.num_patients.val={config.run.val_samples}", f"dataset.num_patients.test={config.run.test_samples}", f"force_recache={config.setup.force_recache}", f"load_from_cache={config.setup.load_from_cache}", f"dataset.coeff={domain_conf}"]
    if experiment == Experiment.ABLATION_ONE_ODE:
        if method_name in ['sindy', 'insite', 'wsindy']:
            # Need best comparison for joint model
            overrides.extend(['model.joint_model=true', "dataset.treatment_mode=multilabel"])
    else:
        if method_name in ['sindy', 'insite', 'wsindy']:
            overrides.extend(["dataset.treatment_mode=multiclass"])
    if experiment == Experiment.ABLATION_MORE_COMPLEX_BASIS_FUNCTIONS:
        if method_name in ['sindy', 'insite', 'wsindy']:
            # Need best comparison for joint model
            overrides.extend(['model.ablation_more_complex_basis_functions=true'])
    if experiment == Experiment.INSIGHT_RECOVER_PARAMETRIC_DIST:
        if method_name in ['sindy', 'insite', 'wsindy']:
            # Need best comparison for joint model
            overrides.extend(['model.insight_recover_parametric_dist=true'])
    if dataset_name == 'cancer_sim' or 'EQ_5' in dataset_name:
        if dataset_name == 'cancer_sim':
            overrides.extend([f"+dataset={dataset_name}"])
        elif 'EQ_5' in dataset_name:
            overrides.extend(["+dataset=continuous", f"dataset.equation_str={dataset_name}"])
        if method_name == 'msm':
            overrides.extend([f"+backbone/benchmark_hparams=ct", "dataset.treatment_mode=multilabel"])
        elif method_name == 'rmsn':
            overrides.append(f"+backbone/benchmark_hparams=rmsn")
        elif method_name == 'crn':
            # overrides.extend([f"+backbone/benchmark_hparams=crn", "+backbone/crn_hparams=cancer_sim_hparams_grid"])
            overrides.extend([f"+backbone/benchmark_hparams=crn"])
        elif method_name == 'gnet':
            overrides.extend([f"+backbone/benchmark_hparams=gnet", f'model.g_net.mc_samples={config.gnet.mcsamples}'])
        elif method_name == 'ct':
            overrides.append(f"+backbone/benchmark_hparams=ct")
        elif method_name == 'edct':
            overrides.append(f"+backbone/benchmark_hparams=edct")
        elif method_name == 'sindy':
            overrides.extend([f"+backbone/benchmark_hparams=ct",  f"model.dataset_name={dataset_name}", f"model.sindy_threshold={sindy_threshold}", f"model.sindy_alpha={config.sindy.sindy_alpha}", f"model.lam={lam}"])
        elif method_name == 'insite':
            # overrides.extend([f"+backbone/benchmark_hparams=ct", "+backbone/insite_hparams=insite_hparams_grid", f"model.dataset_name={dataset_name}"])
            overrides.extend([f"+backbone/benchmark_hparams=ct", f"model.dataset_name={dataset_name}", f"model.sindy_threshold={sindy_threshold}", f"model.sindy_alpha={config.sindy.sindy_alpha}", f"model.lam={lam}"])
            if HYPER_PARAMETER_TUNE:
                overrides.append("+backbone/insite_hparams=insite_hparams_grid")
        elif method_name == 'wsindy':
            overrides.extend([f"+backbone/benchmark_hparams=ct",  f"model.dataset_name={dataset_name}", f"model.sindy_threshold={sindy_threshold}", f"model.sindy_alpha={config.sindy.sindy_alpha}", f"model.lam={lam}"])
    elif 'EQ_4' in dataset_name:
        overrides.extend([f"+dataset=pkpd_sim", f"dataset.equation_str={dataset_name}"])
        if method_name == 'msm':
            overrides.append(f"+backbone/benchmark_hparams=ct")
        elif method_name == 'rmsn':
            overrides.append(f"+backbone/benchmark_hparams=rmsn")
        elif method_name == 'crn':
            overrides.append(f"+backbone/benchmark_hparams=crn")
        elif method_name == 'gnet':
            overrides.extend([f"+backbone/benchmark_hparams=gnet", f'model.g_net.mc_samples={config.gnet.mcsamples}'])
        elif method_name == 'ct':
            overrides.append(f"+backbone/benchmark_hparams=ct")
        elif method_name == 'edct':
            overrides.append(f"+backbone/benchmark_hparams=edct")
        elif method_name == 'sindy':
            overrides.extend([f"+backbone/benchmark_hparams=ct",  f"model.dataset_name={dataset_name}", f"model.sindy_threshold={sindy_threshold}", f"model.sindy_alpha={config.sindy.sindy_alpha}", f"model.lam={lam}"])
        elif method_name == 'insite':
            overrides.extend([f"+backbone/benchmark_hparams=ct",  f"model.dataset_name={dataset_name}", f"model.sindy_threshold={sindy_threshold}", f"model.sindy_alpha={config.sindy.sindy_alpha}", f"model.lam={lam}"])
            if HYPER_PARAMETER_TUNE:
                overrides.append("+backbone/insite_hparams=insite_hparams_grid")
        elif method_name == 'wsindy':
            overrides.extend([f"+backbone/benchmark_hparams=ct",  f"model.dataset_name={dataset_name}", f"model.sindy_threshold={sindy_threshold}", f"model.sindy_alpha={config.sindy.sindy_alpha}", f"model.lam={lam}"])
    if not np.any(['dataset.treatment_mode' in o for o in overrides]):
        overrides.append(f"dataset.treatment_mode={config.setup.treatment_mode}")

    def run(config, logger):
        cfg = compose(config_name=f'ct_config', overrides=overrides)
        logger.info(f'[CT Sub Config] {cfg}')
        if method_name == 'msm':
            from libs_m.ct.runnables.train_msm import main
            result = main(cfg)
        elif method_name == 'rmsn':
            from libs_m.ct.runnables.train_rmsn import main
            result = main(cfg)
        elif method_name == 'crn':
            from libs_m.ct.runnables.train_enc_dec import main
            result = main(cfg)
        elif method_name == 'gnet':
            from libs_m.ct.runnables.train_gnet import main
            result = main(cfg)
        elif method_name == 'ct':
            from libs_m.ct.runnables.train_multi import main
            result = main(cfg)
        elif method_name == 'edct':
            from libs_m.ct.runnables.train_enc_dec import main
            result = main(cfg)
        elif method_name == 'sindy':
            from libs_m.ct.runnables.train_sindy import main
            result = main(cfg, dataset_name=dataset_name)
        elif method_name == 'insite':
            from libs_m.ct.runnables.train_sindy import main
            result = main(cfg, dataset_name=dataset_name)
        elif method_name == 'wsindy':
            from libs_m.ct.runnables.train_sindy import main
            result = main(cfg, dataset_name=dataset_name)
        else:
            raise NotImplementedError
        return result

    if not config.setup.multi_process_results:
        result = run(config, logger)
    else:
        with initialize(config_path="config"):
            result = run(config, logger)

    seconds_taken = time.perf_counter() - t00
    result.update({'method': method_name, 'seed': seed, 'seconds_taken': seconds_taken})
    return result

if __name__ == "__main__":
    run()
