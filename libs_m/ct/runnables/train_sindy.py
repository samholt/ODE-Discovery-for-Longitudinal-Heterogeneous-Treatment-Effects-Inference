import logging
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from pytorch_lightning.utilities.seed import seed_everything

import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from src.models.utils import FilteringMlFlowLogger
from src.models.sindy import SINDY
from .run_utils import get_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch.set_default_dtype(torch.double)


# @hydra.main(config_name=f'config.yaml', config_path='../config/')
def main(args: DictConfig, dataset_name=''):
    """
    Training / evaluation script for SINDY
    Args:
        args: arguments of run as DictConfig

    Returns: dict with results (one and nultiple-step-ahead RMSEs)
    """

    results = {}

    # Non-strict access to fields
    OmegaConf.set_struct(args, False)
    OmegaConf.register_new_resolver("sum", lambda x, y: x + y, replace=True)
    logger.info('\n' + OmegaConf.to_yaml(args, resolve=True))

    # Initialisation of data to calculate dim_outcomes, dim_treatments, dim_vitals and dim_static_features
    seed_everything(args.exp.seed)
    dataset_collection = get_dataset(args)
    # assert args.dataset.treatment_mode == 'multiclass'  # Only binary multilabel regime possible
    if 'EQ_5' in dataset_name:
        dataset_collection.process_data_multi(include_continuous_treatment=True)
    else:
        dataset_collection.process_data_multi()
    args.model.dim_outcomes = dataset_collection.train_f.data['outputs'].shape[-1]
    args.model.dim_treatments = dataset_collection.train_f.data['current_treatments'].shape[-1]
    args.model.dim_vitals = dataset_collection.train_f.data['vitals'].shape[-1] if dataset_collection.has_vitals else 0
    args.model.dim_static_features = dataset_collection.train_f.data['static_features'].shape[-1]
    args.model.treatment_mode = args.dataset.treatment_mode

    # MlFlow Logger
    if args.exp.logging:
        experiment_name = f'{args.model.name}/{args.dataset.name}'
        mlf_logger = FilteringMlFlowLogger(filter_submodels=[], experiment_name=experiment_name,
                                           tracking_uri=args.exp.mlflow_uri)
    else:
        mlf_logger = None

    # ============================== Initialisation & Training of Encoder ==============================
    sindy_regressor = instantiate(args.model.sindy, args, dataset_collection, _recursive_=False)
    if args.model.tune_hparams:
        sindy_regressor.finetune(resources_per_trial=args.model.resources_per_trial, args=args)
    mlf_logger.log_hyperparams(sindy_regressor.hparams)
    sindy_regressor.fit(dataset_collection.train_f, dataset_collection.val_f)
    encoder_results = {}


    if sindy_regressor.insight_recover_parametric_dist:
        sindy_regressor.get_predictions(dataset_collection.val_f)


    if hasattr(dataset_collection, 'test_cf_one_step'):  # Test one_step_counterfactual rmse
        test_rmse_orig, test_rmse_all, test_rmse_last = \
            sindy_regressor.get_normalised_masked_rmse(dataset_collection.test_cf_one_step, one_step_counterfactual=True)
        logger.info(f'Test normalised RMSE (all): {test_rmse_all}; '
                    f'Test normalised RMSE (orig): {test_rmse_orig}; '
                    f'Test normalised RMSE (only counterfactual): {test_rmse_last}')
        encoder_results = {
            'encoder_test_rmse_all': test_rmse_all,
            'encoder_test_rmse_orig': test_rmse_orig,
            'encoder_test_rmse_last': test_rmse_last
        }
    elif hasattr(dataset_collection, 'test_f'):  # Test factual rmse
        test_rmse_orig, test_rmse_all = sindy_regressor.get_normalised_masked_rmse(dataset_collection.test_f)
        logger.info(f'Test normalised RMSE (all): {test_rmse_all}; '
                    f'Test normalised RMSE (orig): {test_rmse_orig}.')
        encoder_results = {
            # 'encoder_val_rmse_all': val_rmse_all,
            # 'encoder_val_rmse_orig': val_rmse_orig,
            'encoder_test_rmse_all': test_rmse_all,
            'encoder_test_rmse_orig': test_rmse_orig
        }

    mlf_logger.log_metrics(encoder_results) if args.exp.logging else None
    results.update(encoder_results)

    test_rmses = {}
    if hasattr(dataset_collection, 'test_cf_treatment_seq'):  # Test n_step_counterfactual rmse
        test_rmses = sindy_regressor.get_normalised_n_step_rmses(dataset_collection.test_cf_treatment_seq)
    elif hasattr(dataset_collection, 'test_f_multi'):  # Test n_step_factual rmse
        test_rmses = sindy_regressor.get_normalised_n_step_rmses(dataset_collection.test_f_multi)
    test_rmses = {f'{k+2}-step': v for (k, v) in enumerate(test_rmses)}

    logger.info(f'Test normalised RMSE (n-step prediction): {test_rmses}')
    decoder_results = {('decoder_test_rmse_' + k): v for (k, v) in test_rmses.items()}

    mlf_logger.log_metrics(decoder_results) if args.exp.logging else None
    results.update(decoder_results)

    mlf_logger.experiment.set_terminated(mlf_logger.run_id)

    results.update({'global_equation_string': sindy_regressor.global_equation_string, 'fine_tuned': sindy_regressor.insite})
    return results


if __name__ == "__main__":
    main()

