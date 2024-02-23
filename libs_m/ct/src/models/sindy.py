from pytorch_lightning import LightningModule
from omegaconf import DictConfig
from omegaconf.errors import MissingMandatoryValue
import torch
from torch import multiprocessing
import math
from typing import Union
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import logging
import numpy as np
from copy import deepcopy
from pytorch_lightning import Trainer
import ray
from ray import tune
from ray import ray_constants
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from scipy.integrate import solve_ivp
from functools import partial

from src.data import RealDatasetCollection, SyntheticDatasetCollection
from src.models import TimeVaryingCausalModel
from src.models.utils import grad_reverse, BRTreatmentOutcomeHead, AlphaRise, clip_normalize_stabilized_weights
from src.models.utils_lstm import VariationalLSTM
import pysindy as ps

from tqdm import tqdm
import jax.numpy as jnp

from pysindy.feature_library import PolynomialLibrary
# from src.data.pkpd.pkpd_simulation import process_dataset_into_de_format
from src.data.pkpd.utils import odeint, LSQIntialMask, convert_sindy_model_to_sympyjax_model, convert_sindy_model_to_sympyjax_model_core, convert_sindy_model_to_sympy_model, STEPS_FOR_DT, HMAX, debug_vmap, debug_scan, create_mask, process_dataset_into_de_format, Datasets
from src.data.pkpd.utils import  STEPS_FOR_DT, MAX_SEQUENCE_LENGTH, HMAX, STANDARD_DT, MAX_TIME_HORIZON, MAX_VALUE, smoother_kws
import jax
from jax import random, vmap, jit, lax
from jax import device_put
from jax.lax import stop_gradient, scan
from scipy.signal import savgol_filter


from pysindy import SINDy
from pysindy.optimizers import STLSQ, SR3, SSR, FROLS
from pysindy.differentiation import FiniteDifference, SmoothedFiniteDifference, SpectralDerivative

from scipy.signal import savgol_filter
from sympy import sympify
import sympy2jax
from jax.scipy.optimize import minimize
from time import time
import pysindy
integrator_keywords = {}

logger = logging.getLogger(__name__)

class SINDY(TimeVaryingCausalModel):
    """
    Pytorch-Lightning implementation of SINDY
    """

    model_type = 'sindy_regressor'
    tuning_criterion = 'rmse'

    def __init__(self,
                 args: DictConfig,
                 dataset_collection: Union[RealDatasetCollection, SyntheticDatasetCollection] = None,
                 autoregressive: bool = None,
                 has_vitals: bool = None,
                 **kwargs):
        """
        Args:
            args: DictConfig of model hyperparameters
            dataset_collection: Dataset collection
            autoregressive: Flag of including previous outcomes to modelling
            has_vitals: Flag of vitals in dataset
            **kwargs: Other arguments
        """
        super().__init__(args, dataset_collection, autoregressive, has_vitals)
        self.lag_features = args.model.lag_features

        self.input_size = self.dim_treatments + self.dim_static_features
        self.input_size += self.dim_vitals if self.has_vitals else 0
        self.input_size += self.dim_outcome if self.autoregressive else 0
        logger.info(f'Input size of {self.model_type}: {self.input_size}')
        self.output_size = self.dim_outcome
        self.save_hyperparameters(args)
        # self.model_all_dimensions = True
        self.smoother_kws = {'window_length': 5, 'polyorder': 3}
        self.dt = STANDARD_DT
        self.insite = False
        self.insite_val_error_threshold = args.model.insite_val_error_threshold
        self.global_equation_string = ''

        self.sindy_threshold = args.model.sindy_threshold
        self.sindy_alpha = args.model.sindy_alpha
        self.smooth_input_data = args.model.smooth_input_data
        self.sindy_quantize = args.model.sindy_quantize
        self.sindy_quantize_global_model_round_to = args.model.sindy_quantize_global_model_round_to
        self.lam = args.model.lam
        self.joint_model = args.model.joint_model
        self.insite = args.model.insite
        self.wsindy = args.model.wsindy
        self.use_smoothed_finite_difference = args.model.use_smoothed_finite_difference
        self.dataset = Datasets[args.model.dataset_name.upper()]
        self.ablation_more_complex_basis_functions = args.model.ablation_more_complex_basis_functions
        self.insight_recover_parametric_dist = args.model.insight_recover_parametric_dist
        self.treatment_mode = args.dataset.treatment_mode
        if self.treatment_mode == 'multilabel':
            self.dim_one_hot_treatments = int(self.dim_treatments ** 2)
            if self.joint_model:
                self.dim_one_hot_treatments = int(1)
        else:
            self.dim_one_hot_treatments = self.dim_treatments

    @staticmethod
    def set_hparams(model_args: DictConfig, new_args: dict, input_size: int, model_type: str):
        """
        Used for hyperparameter tuning and model reinitialisation
        :param model_args: Sub DictConfig, with encoder/decoder parameters
        :param new_args: New hyperparameters
        :param input_size: Input size of the model
        :param model_type: Submodel specification
        """
        model_args.lam = new_args['lam']
        # model_args.sindy_threshold = new_args['sindy_threshold']
        # model_args.sindy_alpha = new_args['sindy_alpha']
        # model_args.smooth_input_data = new_args['smooth_input_data']

    def prepare_data(self) -> None:
        if self.dataset_collection is not None and not self.dataset_collection.processed_data_multi:
            assert self.hparams.dataset.treatment_mode == 'multilabel'  # Only binary multilabel regime possible
            self.dataset_collection.process_data_multi()

    def get_exploded_dataset(self, dataset: Dataset, min_length: int, only_active_entries=True, max_length=None) -> Dataset:
        exploded_dataset = deepcopy(dataset)
        if max_length is None:
            max_length = max(exploded_dataset.data['sequence_lengths'][:])
        if not only_active_entries:
            exploded_dataset.data['active_entries'][:, :, :] = 1.0
            exploded_dataset.data['sequence_lengths'][:] = max_length
        exploded_dataset.explode_trajectories(min_length)
        return exploded_dataset

    def fit(self, train_f: Dataset, val_f: Dataset):
        self.prepare_data()
        sequence_length_max = jnp.array(train_f.data['sequence_lengths']).astype(jnp.int64).max()
        if 'EQ_4' in self.dataset.name:
            individualized_equation_arg_list, X_0, U_0, X_1, U_1 = process_dataset_into_de_format(train_f,
                                                                                                        self.dim_outcome,
                                                                                                        self.dim_static_features,
                                                                                                        self.dim_treatments,
                                                                                                        self.dim_vitals,
                                                                                                        dt=self.dt,
                                                                                                        sequence_lengths_offset=1,
                                                                                                        smooth=self.smooth_input_data,
                                                                                                        # seq_length=training_data['cancer_volume'].shape[1],
                                                                                                        joint=self.joint_model,
                                                                                                        dataset_name=self.dataset)
        elif self.dataset == Datasets.CANCER_SIM:
            individualized_equation_arg_list, XU_t_0, XU_t_1, XU_t_2, XU_t_3 = process_dataset_into_de_format(train_f,
                                                                                                        self.dim_outcome,
                                                                                                        self.dim_static_features,
                                                                                                        self.dim_treatments,
                                                                                                        self.dim_vitals,
                                                                                                        dt=self.dt,
                                                                                                        sequence_lengths_offset=0,
                                                                                                        smooth=self.smooth_input_data,
                                                                                                        # seq_length=training_data['cancer_volume'].shape[1],
                                                                                                        joint=self.joint_model,
                                                                                                        dataset_name=self.dataset)
        elif 'EQ_5' in self.dataset.name:
            individualized_equation_arg_list, XU_t_0, XU_t_1, XU_t_2, XU_t_3 = process_dataset_into_de_format(train_f,
                                                                                                        self.dim_outcome,
                                                                                                        self.dim_static_features,
                                                                                                        self.dim_treatments,
                                                                                                        self.dim_vitals,
                                                                                                        dt=self.dt,
                                                                                                        sequence_lengths_offset=0,
                                                                                                        smooth=self.smooth_input_data,
                                                                                                        # seq_length=training_data['cancer_volume'].shape[1],
                                                                                                        joint=self.joint_model,
                                                                                                        dataset_name=self.dataset)
        if not self.wsindy:
            if self.ablation_more_complex_basis_functions:
                PolynomialLibrary_kw = dict(degree=4, interaction_only=False)
            else:
                PolynomialLibrary_kw = dict(degree=2, interaction_only=True)
            if 'EQ_4' in self.dataset.name:
                model_0 = SINDy(optimizer=STLSQ(threshold=self.sindy_threshold, alpha=self.sindy_alpha, max_iter=100, ridge_kw={'tol': 1e-6}),  differentiation_method=SmoothedFiniteDifference(smoother_kws=self.smoother_kws, is_uniform=True, order=4), feature_library=PolynomialLibrary(**PolynomialLibrary_kw)).fit(X_0, u=U_0, t=self.dt, multiple_trajectories=True)
                if not self.joint_model:
                    model_1 = SINDy(optimizer=STLSQ(threshold=self.sindy_threshold, alpha=self.sindy_alpha, max_iter=100, ridge_kw={'tol': 1e-6}),  differentiation_method=SmoothedFiniteDifference(smoother_kws=self.smoother_kws, is_uniform=True, order=4), feature_library=PolynomialLibrary(**PolynomialLibrary_kw)).fit(X_1, u=U_1, t=self.dt, multiple_trajectories=True)
            elif self.dataset == Datasets.CANCER_SIM or 'EQ_5' in self.dataset.name:
                STLSQ_kw = dict(threshold=self.sindy_threshold, alpha=self.sindy_alpha, max_iter=100, ridge_kw={'tol': 1e-6})
                FiniteDifference_kw = dict(is_uniform=True, order=1)
                if self.use_smoothed_finite_difference:    
                    def differentiation_method(**kw):
                        return SmoothedFiniteDifference(smoother_kws={'window_length': 2, 'polyorder': 1}, **kw)
                else:
                    def differentiation_method(**kw):
                        return FiniteDifference(**kw)
                t0 = time()
                model_0 = SINDy(optimizer=STLSQ(**STLSQ_kw),  differentiation_method=differentiation_method(**FiniteDifference_kw), feature_library=PolynomialLibrary(**PolynomialLibrary_kw)).fit(XU_t_0[0], u=XU_t_0[1], t=self.dt, multiple_trajectories=True)
                print('Time for model 0: ', time() - t0)
                t1 = time()
                if not self.joint_model:
                    model_1 = SINDy(optimizer=STLSQ(**STLSQ_kw),  differentiation_method=differentiation_method(**FiniteDifference_kw), feature_library=PolynomialLibrary(**PolynomialLibrary_kw)).fit(XU_t_1[0], u=XU_t_1[1], t=self.dt, multiple_trajectories=True)
                    print('Time for model 1: ', time() - t1)
                    t1 = time()
                    model_2 = SINDy(optimizer=STLSQ(**STLSQ_kw),  differentiation_method=differentiation_method(**FiniteDifference_kw), feature_library=PolynomialLibrary(**PolynomialLibrary_kw)).fit(XU_t_2[0], u=XU_t_2[1], t=self.dt, multiple_trajectories=True)
                    print('Time for model 2: ', time() - t1)
                    t1 = time()
                    model_3 = SINDy(optimizer=STLSQ(**STLSQ_kw),  differentiation_method=differentiation_method(**FiniteDifference_kw), feature_library=PolynomialLibrary(**PolynomialLibrary_kw)).fit(XU_t_3[0], u=XU_t_3[1], t=self.dt, multiple_trajectories=True)
                    print('Time for model 3: ', time() - t1)
                    print('Total time: ', time() - t0)
                    print('')

        else:
            if self.ablation_more_complex_basis_functions:
                raise NotImplementedError
            library_functions = [lambda x: 1, lambda x: x, lambda x, y: x * y]
            library_function_names = [lambda x: '1', lambda x: f'{x}', lambda x, y: f'{x} {y}']
            if 'EQ_4' in self.dataset.name:
                t_train = np.arange(sequence_length_max - 1) * self.dt
                model_0 = ps.SINDy(feature_library=pysindy.WeakPDELibrary(
                                                    library_functions=library_functions,
                                                    function_names=library_function_names,
                                                    spatiotemporal_grid=t_train,
                                                    is_uniform=True,
                                                    K=100),
                                    optimizer = pysindy.SR3(threshold=self.sindy_threshold, thresholder="l1", max_iter=1000, normalize_columns=True, tol=1e-1)).fit(X_0, u=U_0, t=self.dt, multiple_trajectories=True)
                if not self.joint_model:
                    model_1 = ps.SINDy(feature_library=pysindy.WeakPDELibrary(
                                                        library_functions=library_functions,
                                                        function_names=library_function_names,
                                                        spatiotemporal_grid=t_train,
                                                        is_uniform=True,
                                                        K=100),
                                        optimizer = pysindy.SR3(threshold=self.sindy_threshold, thresholder="l1", max_iter=1000, normalize_columns=True, tol=1e-1)).fit(X_1, u=U_1, t=self.dt, multiple_trajectories=True)
            elif self.dataset == Datasets.CANCER_SIM or 'EQ_5' in self.dataset.name:
                raise NotImplementedError('Weak-SINDy not implemented for small length datasets, avoiding getting stuck in infinite loops.')
                t_train = np.arange(sequence_length_max - 1) * self.dt
                model_0 = ps.SINDy(feature_library=pysindy.WeakPDELibrary(
                                                    library_functions=library_functions,
                                                    function_names=library_function_names,
                                                    spatiotemporal_grid=t_train,
                                                    is_uniform=True,
                                                    K=100),
                                    optimizer = pysindy.SR3(threshold=self.sindy_threshold, thresholder="l1", max_iter=1000, normalize_columns=True, tol=1e-1)).fit(XU_t_0[0], u=XU_t_0[1], t=self.dt, multiple_trajectories=True)
                if not self.joint_model:
                    model_1 = ps.SINDy(feature_library=pysindy.WeakPDELibrary(
                                                        library_functions=library_functions,
                                                        function_names=library_function_names,
                                                        spatiotemporal_grid=t_train,
                                                        is_uniform=True,
                                                        K=100),
                                        optimizer = pysindy.SR3(threshold=self.sindy_threshold, thresholder="l1", max_iter=1000, normalize_columns=True, tol=1e-1)).fit(XU_t_1[0], u=XU_t_1[1], t=self.dt, multiple_trajectories=True)
                    model_2 = ps.SINDy(feature_library=pysindy.WeakPDELibrary(
                                                        library_functions=library_functions,
                                                        function_names=library_function_names,
                                                        spatiotemporal_grid=t_train,
                                                        is_uniform=True,
                                                        K=100),
                                        optimizer = pysindy.SR3(threshold=self.sindy_threshold, thresholder="l1", max_iter=1000, normalize_columns=True, tol=1e-1)).fit(XU_t_2[0], u=XU_t_2[1], t=self.dt, multiple_trajectories=True)
                    model_3 = ps.SINDy(feature_library=pysindy.WeakPDELibrary(
                                                        library_functions=library_functions,
                                                        function_names=library_function_names,
                                                        spatiotemporal_grid=t_train,
                                                        is_uniform=True,
                                                        K=100),
                                        optimizer = pysindy.SR3(threshold=self.sindy_threshold, thresholder="l1", max_iter=1000, normalize_columns=True, tol=1e-1)).fit(XU_t_3[0], u=XU_t_3[1], t=self.dt, multiple_trajectories=True)
        if 'EQ_4' in self.dataset.name:
            if not self.joint_model:
                mod_0, str_0 = convert_sindy_model_to_sympyjax_model(model_0, quantize=self.sindy_quantize, quantize_round_to=self.sindy_quantize_global_model_round_to)
                mod_1, str_1 = convert_sindy_model_to_sympyjax_model(model_1, quantize=self.sindy_quantize, quantize_round_to=self.sindy_quantize_global_model_round_to)
                self.global_equation_string = f'Treatment 0: x_dot = {str_0} | Treatment 1: x_dot = {str_1}'
                print(f'[Model]: Treatment 0: x_dot = {str_0} \t | Treatment 1: x_dot = {str_1}')
                def pred_dy_dt(y, t, treatment, static_features):                    
                    return jax.lax.cond(jnp.argmax(treatment) == 0,
                                        lambda _: mod_0(x0=y, u0=static_features[0], u1=static_features[1])[0],
                                        lambda _: mod_1(x0=y, u0=static_features[0], u1=static_features[1])[0],
                                        operand=None)
            else:
                mod_0, str_0 = convert_sindy_model_to_sympyjax_model(model_0, quantize=self.sindy_quantize, quantize_round_to=self.sindy_quantize_global_model_round_to)
                self.global_equation_string = f'Joint Model: x_dot = {str_0}'
                print(f'[Model Raw]: Joint Model: x_dot = {str_0}')
                def pred_dy_dt(y, t, treatment, static_features):
                    return mod_0(x0=y, u0=treatment, u1=static_features[0], u2=static_features[1])[0]
        elif self.dataset == Datasets.CANCER_SIM or 'EQ_5' in self.dataset.name:
            if not self.joint_model:
                mod_0, str_0 = convert_sindy_model_to_sympyjax_model(model_0, quantize=self.sindy_quantize, quantize_round_to=self.sindy_quantize_global_model_round_to)
                mod_1, str_1 = convert_sindy_model_to_sympyjax_model(model_1, quantize=self.sindy_quantize, quantize_round_to=self.sindy_quantize_global_model_round_to)
                mod_2, str_2 = convert_sindy_model_to_sympyjax_model(model_2, quantize=self.sindy_quantize, quantize_round_to=self.sindy_quantize_global_model_round_to)
                mod_3, str_3 = convert_sindy_model_to_sympyjax_model(model_3, quantize=self.sindy_quantize, quantize_round_to=self.sindy_quantize_global_model_round_to)
                self.global_equation_string = f'Treatment 0: x_dot = {str_0} | Treatment 1: x_dot = {str_1} | Treatment 2: x_dot = {str_2} | Treatment 3: x_dot = {str_3}'
                print(f'[Model]: Treatment 0: x_dot = {str_0} | Treatment 1: x_dot = {str_1} | Treatment 2: x_dot = {str_2} | Treatment 3: x_dot = {str_3}')
                if self.dataset == Datasets.CANCER_SIM:
                    def wrapper_bf(bf):
                        def bf_wrapper(x0, static_features):
                            return jnp.asarray(bf(x0=x0, u0=static_features[0]), dtype=jnp.float64).reshape()
                        return bf_wrapper
                elif 'EQ_5' in self.dataset.name:
                    def wrapper_bf(bf):
                        def bf_wrapper(x0, static_features):
                            return jnp.asarray(bf(x0=x0, u0=static_features[0], u1=static_features[1]), dtype=jnp.float64).reshape()
                        return bf_wrapper
                treatment_odes = [mod_0, mod_1, mod_2, mod_3]
                treatment_odes = [wrapper_bf(mod_i) for mod_i in treatment_odes]
                def pred_dy_dt(y, t, treatment, static_features):
                    treatment_idx = jnp.argmax(treatment)
                    res = lax.switch(treatment_idx, treatment_odes, y, static_features)
                    return res
            else:
                mod_0, str_0 = convert_sindy_model_to_sympyjax_model(model_0, quantize=self.sindy_quantize, quantize_round_to=self.sindy_quantize_global_model_round_to)
                self.global_equation_string = f'Joint Model: x_dot = {str_0}'
                print(f'[Model Raw]: Joint Model: x_dot = {str_0}')
                if self.dataset == Datasets.CANCER_SIM:
                    def pred_dy_dt(y, t, treatment, static_features):
                        return mod_0(x0=y, u0=treatment[0], u1=treatment[1], u2=static_features[0])[0]
                elif 'EQ_5' in self.dataset.name:
                    def pred_dy_dt(y, t, treatment, static_features):
                        return mod_0(x0=y, u0=treatment[0], u1=treatment[1], u2=static_features[0], u3=static_features[1])[0]
        pred_dy_dt = jax.tree_util.Partial(pred_dy_dt)
        self.pred_dy_dt = pred_dy_dt

        # self.joint_coefs = (model_0.coefficients()[0] + model_1.coefficients()[0]) / 2.0
        if 'EQ_4' in self.dataset.name:
            if not self.joint_model:
                self.joint_coefs = np.stack([model_0.coefficients()[0], model_1.coefficients()[0]])
            else:
                self.joint_coefs = np.stack([model_0.coefficients()[0]])
        elif self.dataset == Datasets.CANCER_SIM or 'EQ_5' in self.dataset.name:
            if not self.joint_model:
                self.joint_coefs = np.stack([model_0.coefficients()[0], model_1.coefficients()[0], model_2.coefficients()[0], model_3.coefficients()[0]])
            else:
                self.joint_coefs = np.stack([model_0.coefficients()[0]])
        self.feature_library_names = model_0.feature_library.get_feature_names().copy() # Same for both models
        self.feature_names = model_0.feature_names # Same for both models

        # INSITE; test if validation error is high, then switch to fine tuning; otherwise use the global model
        # if self.insite:
        #     train_preds = self.get_predictions(train_f)
        #     output_stds, output_means = train_f.scaling_params['output_stds'], train_f.scaling_params['output_means']
        #     unscaled_train_preds = train_preds * output_stds + output_means
        #     unscaled_outputs = train_f.data['unscaled_outputs']
        #     train_mse = ((unscaled_train_preds - unscaled_outputs) ** 2) * train_f.data['active_entries']
        #     train_mse_all = train_mse.sum() / train_f.data['active_entries'].sum()
        #     rmse_normalised_all = np.sqrt(train_mse_all) / val_f.norm_const
        #     print(f'[Model]: Normalised RMSE on train set: {rmse_normalised_all}')

        #     val_preds_scaled = self.get_predictions(val_f)
        #     output_stds, output_means = val_f.scaling_params['output_stds'], val_f.scaling_params['output_means']
        #     val_preds_unscaled = val_preds_scaled * output_stds + output_means
        #     val_mse = ((val_preds_unscaled -  val_f.data['unscaled_outputs']) ** 2) *  val_f.data['active_entries']
        #     val_mse_all = val_mse.sum() / val_f.data['active_entries'].sum()
        #     rmse_normalised_all = np.sqrt(val_mse_all) / val_f.norm_const
        #     print(f'[Model]: Normalised RMSE on validation set: {rmse_normalised_all}')
        #     if rmse_normalised_all <= self.insite_val_error_threshold:
        #         # Switch off fine tuning step if validation error is low; as global model is good enough
        #         self.insite = False

    def get_predictions(self, dataset: Dataset) -> np.array:
        if not self.insite:
            predictions = self._get_non_fine_tuned_predictions(dataset)
        else:
            # predictions = self._get_non_fine_tuned_predictions(dataset)
            predictions = self._get_fine_tuned_predictions(dataset)
        assert not np.any(np.isnan(predictions)), 'Predictions contains NaN'
        return predictions

    def _get_non_fine_tuned_predictions(self, dataset: Dataset) -> np.array:
        logger.info(f'Predictions for {dataset.subset_name}.')
        integrator_keywords = {}
        # Loss (target)
        # mse = ((outputs_scaled - dataset.data['outputs']) ** 2) * dataset.data['active_entries']

        # lstm x
        # prev_treatments = batch['prev_treatments']
        # prev_outputs = batch['prev_outputs']
        # static_features = batch['static_features']
        # curr_treatments = batch['current_treatments']
        # x = torch.cat((prev_treatments, prev_outputs), dim=-1)
        # x = torch.cat((x, static_features.unsqueeze(1).expand(-1, x.size(1), -1)), dim=-1)
        # outcome_pred = x, curr_treatments
        # dataset.data

        output_stds, output_means = dataset.scaling_params['output_stds'], dataset.scaling_params['output_means']
        # outputs_unscaled = outputs_scaled * output_stds + output_means
        # dataset.data['outputs'] * output_stds + output_means
        # dataset.data['outputs'][0] * output_stds + output_means

        unscaled_prev_outputs = dataset.data['prev_outputs'] * output_stds + output_means
        unscaled_static_features = dataset.data['static_features'] * dataset.scaling_params['inputs_stds'][self.dim_outcome:self.dim_outcome + self.dim_static_features] + dataset.scaling_params['input_means'][self.dim_outcome:self.dim_outcome + self.dim_static_features]
        current_treatments = dataset.data['current_treatments']
        t = jnp.arange(0, dataset.data['prev_outputs'].shape[1] + 1).astype(jnp.float64) * self.dt

        unscaled_prev_outputs = jnp.squeeze(unscaled_prev_outputs)
        current_treatments = jnp.squeeze(current_treatments).astype(jnp.int64)
        sequence_lengths = dataset.data['sequence_lengths'].astype(jnp.int64)

        # # Checks: Eq 4 dataset
        # cancer_volume = dataset.data['cancer_volume'].astype(jnp.float64)
        # treatment_action = dataset.data['treatment_application'].astype(jnp.int64)
        # observed_static_c_0 = dataset.data['observed_static_c_0'].astype(jnp.float64)
        # observed_static_c_1 = dataset.data['observed_static_c_1'].astype(jnp.float64)

        # assert (jnp.abs((unscaled_prev_outputs - cancer_volume[:,:-1]).mean()) < 1e-15).item()
        # assert (current_treatments == treatment_action[:, :-1]).all().item()
        # assert (unscaled_static_features[:,0] == observed_static_c_0).all().item()
        # assert (unscaled_static_features[:,1] == observed_static_c_1).all().item()


        def simulate_cancer_volume(initial_volume, treatment, dt, static_features):            
            # iter_tuples = jnp.array([(t[i-1], t[i], treatment[i-1]) for i in range(1, t.shape[0])])
            carry_init = (initial_volume, static_features, dt)

            def scan_fn(carry, iter_tuple):
                cancer_volume, static_features, dt = carry
                treatment = iter_tuple
                t_tuple = jnp.array([0, dt])
                cancer_volume = odeint(self.pred_dy_dt, cancer_volume, t_tuple, treatment, static_features, hmax=HMAX, **integrator_keywords)[1]
                return (cancer_volume, static_features, dt), cancer_volume
            
            _, predicted_volumes = scan(scan_fn, carry_init, treatment)
            # _, predicted_volumes = debug_scan(scan_fn, carry_init, treatment)
            return predicted_volumes.reshape(-1,1)
        
        # outputs_unscaled = debug_vmap(simulate_cancer_volume, in_axes=(0,0,None,0), args=(unscaled_prev_outputs[:,0], current_treatments, self.dt, unscaled_static_features))
        outputs_unscaled = jit(vmap(simulate_cancer_volume, in_axes=(0,0,None,0)))(unscaled_prev_outputs[:,0], current_treatments, self.dt, unscaled_static_features)
        outputs = (outputs_unscaled - output_means) / output_stds
        return outputs

    def _get_fine_tuned_predictions(self, dataset: Dataset, projection_horizon=1) -> np.array:
        integrator_keywords = {}
        logger.info(f'Individualising equations for {dataset.subset_name}.')
        print('Individualising equations')
        # non_zero_indexes = jnp.nonzero(np.abs(self.joint_coefs)>1e-3)[0]
        # reduced_coefs = self.joint_coefs[non_zero_indexes]
        # reduced_coefs = np.stack([reduced_coefs, reduced_coefs])
        reduced_coefs = self.joint_coefs
        non_zero_indexes = jnp.arange(reduced_coefs.shape[1])
        feature_library_names = [fn.replace(' ', '*') for fn in self.feature_library_names]
        fln_l = []
        for fln in feature_library_names:
            for i in range(len(self.feature_names)):
                fln = fln.replace(f'x{i}', self.feature_names[i])
            fln_l.append(fln)
        feature_library_names = fln_l
        bases_funcs = [sympy2jax.SymbolicModule(sympify(fl)) for fl in feature_library_names]
        if 'EQ_4' in self.dataset.name:
            if not self.joint_model:
                def wrapper_bf(bf):
                    def bf_wrapper(x0, u0, u1):
                        return jnp.asarray(bf(x0=x0, u0=u0, u1=u1), dtype=jnp.float64).reshape()
                    return bf_wrapper
                bases_funcs = [wrapper_bf(bf) for bf in bases_funcs]
                def eq_with_coefs(x0, u0, u1, reduced_coefs_single):
                    result = 0
                    for i, non_zero_idx in enumerate(non_zero_indexes):
                        base_func_res = lax.switch(non_zero_idx, bases_funcs, x0, u0, u1)
                        result += reduced_coefs_single[i] * base_func_res
                    return jnp.sum(result)
                eq_with_coefs = jax.tree_util.Partial(eq_with_coefs)
                def pred_dy_dt(y, t, treatment, static_features, reduced_coefs):
                    return jax.lax.cond(jnp.argmax(treatment) == 0,
                                        lambda _: eq_with_coefs(x0=y, u0=static_features[0], u1=static_features[1], reduced_coefs_single=reduced_coefs[0]),
                                        lambda _: eq_with_coefs(x0=y, u0=static_features[0], u1=static_features[1], reduced_coefs_single=reduced_coefs[1]),
                                        operand=None)
            else:
                def wrapper_bf(bf):
                    def bf_wrapper(x0, treatment, static_features):
                        return jnp.asarray(bf(x0=x0, u0=treatment, u1=static_features[0], u2=static_features[1]), dtype=jnp.float64).reshape()
                    return bf_wrapper
                bases_funcs = [wrapper_bf(bf) for bf in bases_funcs]
                def eq_with_coefs(x0, treatment, static_features, reduced_coefs_single):
                    result = 0
                    for i, non_zero_idx in enumerate(non_zero_indexes):
                        base_func_res = lax.switch(non_zero_idx, bases_funcs, x0, treatment, static_features)
                        result += reduced_coefs_single[i] * base_func_res
                    return jnp.sum(result)
                eq_with_coefs = jax.tree_util.Partial(eq_with_coefs)
                def pred_dy_dt(y, t, treatment, static_features, reduced_coefs):
                    return eq_with_coefs(x0=y, treatment=treatment, static_features=static_features, reduced_coefs_single=reduced_coefs[0])
        elif self.dataset == Datasets.CANCER_SIM:
            if not self.joint_model:
                def wrapper_bf(bf):
                    def bf_wrapper(x0, u0):
                        return jnp.asarray(bf(x0=x0, u0=u0), dtype=jnp.float64).reshape()
                    return bf_wrapper
                bases_funcs = [wrapper_bf(bf) for bf in bases_funcs]
                def eq_with_coefs(x0, u0, reduced_coefs_single):
                    result = 0
                    for i, non_zero_idx in enumerate(non_zero_indexes):
                        base_func_res = lax.switch(non_zero_idx, bases_funcs, x0, u0)
                        result += reduced_coefs_single[i] * base_func_res
                    return jnp.sum(result)
                eq_with_coefs = jax.tree_util.Partial(eq_with_coefs)
                def pred_dy_dt(y, t, treatment, static_features, all_reduced_coefs):
                    treatment_idx = jnp.argmax(treatment)
                    reduced_coefs = all_reduced_coefs[treatment_idx]
                    # reduced_coefs = lax.switch(treatment, all_reduced_coefs)
                    return eq_with_coefs(x0=y, u0=static_features[0], reduced_coefs_single=reduced_coefs)
            else:
                def wrapper_bf(bf):
                    def bf_wrapper(x0, treatment, static_features):
                        return jnp.asarray(bf(x0=x0, u0=treatment[0], u1=treatment[1], u2=static_features[0]), dtype=jnp.float64).reshape()
                    return bf_wrapper
                bases_funcs = [wrapper_bf(bf) for bf in bases_funcs]
                def eq_with_coefs(x0, treatment, static_features, reduced_coefs_single):
                    result = 0
                    for i, non_zero_idx in enumerate(non_zero_indexes):
                        base_func_res = lax.switch(non_zero_idx, bases_funcs, x0, treatment, static_features)
                        result += reduced_coefs_single[i] * base_func_res
                    return jnp.sum(result)
                eq_with_coefs = jax.tree_util.Partial(eq_with_coefs)
                def pred_dy_dt(y, t, treatment, static_features, all_reduced_coefs):
                    return eq_with_coefs(x0=y, treatment=treatment, static_features=static_features, reduced_coefs_single=all_reduced_coefs[0])
        elif 'EQ_5' in self.dataset.name:
            if not self.joint_model:
                def wrapper_bf(bf):
                    def bf_wrapper(x0, u0, u1):
                        return jnp.asarray(bf(x0=x0, u0=u0, u1=u1), dtype=jnp.float64).reshape()
                    return bf_wrapper
                bases_funcs = [wrapper_bf(bf) for bf in bases_funcs]
                def eq_with_coefs(x0, u0, u1, reduced_coefs_single):
                    result = 0
                    for i, non_zero_idx in enumerate(non_zero_indexes):
                        base_func_res = lax.switch(non_zero_idx, bases_funcs, x0, u0, u1)
                        result += reduced_coefs_single[i] * base_func_res
                    return jnp.sum(result)
                eq_with_coefs = jax.tree_util.Partial(eq_with_coefs)
                def pred_dy_dt(y, t, treatment, static_features, all_reduced_coefs):
                    treatment_idx = jnp.argmax(treatment)
                    reduced_coefs = all_reduced_coefs[treatment_idx]
                    # reduced_coefs = lax.switch(treatment, all_reduced_coefs)
                    return eq_with_coefs(x0=y, u0=static_features[0], u1=static_features[0], reduced_coefs_single=reduced_coefs)
            else:
                def wrapper_bf(bf):
                    def bf_wrapper(x0, treatment, static_features):
                        return jnp.asarray(bf(x0=x0, u0=treatment[0], u1=treatment[1], u2=static_features[0], u3=static_features[1]), dtype=jnp.float64).reshape()
                    return bf_wrapper
                bases_funcs = [wrapper_bf(bf) for bf in bases_funcs]
                def eq_with_coefs(x0, treatment, static_features, reduced_coefs_single):
                    result = 0
                    for i, non_zero_idx in enumerate(non_zero_indexes):
                        base_func_res = lax.switch(non_zero_idx, bases_funcs, x0, treatment, static_features)
                        result += reduced_coefs_single[i] * base_func_res
                    return jnp.sum(result)
                eq_with_coefs = jax.tree_util.Partial(eq_with_coefs)
                def pred_dy_dt(y, t, treatment, static_features, all_reduced_coefs):
                    return eq_with_coefs(x0=y, treatment=treatment, static_features=static_features, reduced_coefs_single=all_reduced_coefs[0])
        pred_dy_dt = jax.tree_util.Partial(pred_dy_dt)

        output_stds, output_means = dataset.scaling_params['output_stds'], dataset.scaling_params['output_means']
        unscaled_prev_outputs = dataset.data['prev_outputs'] * output_stds + output_means
        unscaled_prev_outputs = jnp.squeeze(unscaled_prev_outputs)
        if self.smooth_input_data:
            smoother_kws_ = self.smoother_kws.copy()
            smoother_kws_.update({'axis': 1}) 
            unscaled_prev_outputs = jnp.array(savgol_filter(unscaled_prev_outputs, **smoother_kws_))
        
        unscaled_static_features = dataset.data['static_features'] * dataset.scaling_params['inputs_stds'][self.dim_outcome:self.dim_outcome + self.dim_static_features] + dataset.scaling_params['input_means'][self.dim_outcome:self.dim_outcome + self.dim_static_features]
        current_treatments = dataset.data['current_treatments']
        t = jnp.arange(0, dataset.data['prev_outputs'].shape[1] + 1).astype(jnp.float64) * self.dt

        current_treatments = jnp.squeeze(current_treatments).astype(jnp.int64)
        sequence_lengths = jnp.array(dataset.data['sequence_lengths']).astype(jnp.int64)

        @jit
        def simulate_cancer_volume_with_fine_tuning(unscaled_prev_outputs, treatments, unscaled_static_features, reduced_coefs, sequence_length):
            return jax.lax.cond(sequence_length <= projection_horizon,
                    lambda _: _skip_fine_tuning_inner(unscaled_prev_outputs, treatments, unscaled_static_features, reduced_coefs, sequence_length),
                    lambda _: _fine_tuning_inner(unscaled_prev_outputs, treatments, unscaled_static_features, reduced_coefs, sequence_length),
                    operand=None)       
            # if sequence_length <= projection_horizon:
            #     return _skip_fine_tuning_inner(unscaled_prev_outputs, treatments, unscaled_static_features, reduced_coefs, sequence_length)
            # else:
            #     return _fine_tuning_inner(unscaled_prev_outputs, treatments, unscaled_static_features, reduced_coefs, sequence_length)
                

        def _skip_fine_tuning_inner(unscaled_prev_outputs, treatments, unscaled_static_features, reduced_coefs, sequence_length):  
            preds = predict_with_reduced_coefs(reduced_coefs, unscaled_prev_outputs, unscaled_static_features, treatments, self.dt, pred_dy_dt)
            # treatment_idx = jnp.argmax(treatments[0])
            # return preds.reshape(-1,1), reduced_coefs[treatment_idx]
            return preds.reshape(-1,1)

        def _fine_tuning_inner(unscaled_prev_outputs, treatments, unscaled_static_features, reduced_coefs, sequence_length):     
            # iter_tuples = jnp.array([(t[i-1], t[i], treatment[i-1]) for i in range(1, t.shape[0])])
            coef_sparse_mask = (jnp.abs(reduced_coefs)>1e-3).astype(jnp.float64)
            # start_res = f_to_min(reduced_coefs.reshape(-1))
            start_res = f_to_min_func(coef_sparse_mask = coef_sparse_mask,
                                      reduced_coefs_flattened = reduced_coefs.reshape(-1),
                                            unscaled_prev_outputs = unscaled_prev_outputs,
                                            unscaled_static_features = unscaled_static_features,
                                            treatments = treatments,
                                            dt = self.dt,
                                            pred_dy_dt = pred_dy_dt,
                                            sequence_length = sequence_length,
                                            original_reduced_coefs_flattened = reduced_coefs.reshape(-1),
                                            lam = self.lam,
                                            projection_horizon = projection_horizon,
                                            dim_one_hot_treatments = self.dim_one_hot_treatments,
                                            norm_const = 1)
            f_to_min = jax.tree_util.Partial(f_to_min_func,                     
                                            coef_sparse_mask = coef_sparse_mask,
                                            unscaled_prev_outputs = unscaled_prev_outputs,
                                            unscaled_static_features = unscaled_static_features,
                                            treatments = treatments,
                                            dt = self.dt,
                                            pred_dy_dt = pred_dy_dt,
                                            sequence_length = sequence_length,
                                            original_reduced_coefs_flattened = reduced_coefs.reshape(-1),
                                            lam = self.lam,
                                            projection_horizon = projection_horizon,
                                            dim_one_hot_treatments = self.dim_one_hot_treatments,
                                            norm_const = start_res * 2.5)
            # mse_start = eval_fit_mse(reduced_coefs.reshape(-1),
            #                                 coef_sparse_mask = coef_sparse_mask,
            #                                 unscaled_prev_outputs = unscaled_prev_outputs,
            #                                 unscaled_static_features = unscaled_static_features,
            #                                 treatments = treatments,
            #                                 dt = self.dt,
            #                                 pred_dy_dt = pred_dy_dt,
            #                                 sequence_length = sequence_length,
            #                                 projection_horizon = projection_horizon,
            #                                 dim_one_hot_treatments = self.dim_one_hot_treatments)
            res = minimize(f_to_min, reduced_coefs.reshape(-1), method='BFGS', tol=1e-12) #, options={'maxiter': 100})
            reduced_coefs_updated = jax.lax.cond(res.status == 3, # If zoom fails, fall back to default value
                    lambda _: reduced_coefs,
                    lambda _: res.x.reshape(self.dim_one_hot_treatments, -1),
                    operand=None)
            # reduced_coefs_updated = res.x.reshape(self.dim_one_hot_treatments, -1)
            # mse_end = f_to_min(reduced_coefs_updated.reshape(-1))
            # mse_end = eval_fit_mse(reduced_coefs_updated.reshape(-1), coef_sparse_mask = coef_sparse_mask,
            #                                 unscaled_prev_outputs = unscaled_prev_outputs,
            #                                 unscaled_static_features = unscaled_static_features,
            #                                 treatments = treatments,
            #                                 dt = self.dt,
            #                                 pred_dy_dt = pred_dy_dt,
            #                                 sequence_length = sequence_length,
            #                                 projection_horizon = projection_horizon,
            #                                 dim_one_hot_treatments = self.dim_one_hot_treatments)
            # # Debug
            # r = minimize(f_to_min, reduced_coefs.reshape(-1), method='BFGS', tol=1e-6, options={'maxiter': 8})
            # print(eval_fit_mse(res.x, coef_sparse_mask = coef_sparse_mask,
            #                                                 unscaled_prev_outputs = unscaled_prev_outputs,
            #                                                 unscaled_static_features = unscaled_static_features,
            #                                                 treatments = treatments,
            #                                                 dt = self.dt,
            #                                                 pred_dy_dt = pred_dy_dt,
            #                                                 sequence_length = sequence_length,
            #                                                 projection_horizon = projection_horizon,
            #                                                 dim_one_hot_treatments = self.dim_one_hot_treatments), res.nit)
            # # for mi in range(1,10):
            #     print(f"{mi} | {minimize(f_to_min, reduced_coefs.reshape(-1), method='BFGS', tol=1e-6, options={'maxiter': mi}).fun}")
            # jax.debug.print("res.nfev: {nfev} | res.nit: {nit} | res.njev: {njev} | res.status: {status} | res.success: {success}", nfev=res.nfev, nit=res.nit, njev=res.njev, status=res.status, success=res.success)
            # jax.debug.print("Seq len: {sequence_length} | optimized_mse: {mse_end} | sindy_mse: {mse_start}", sequence_length=sequence_length, mse_end=mse_end, mse_start=mse_start)
            preds = predict_with_reduced_coefs(reduced_coefs_updated, unscaled_prev_outputs, unscaled_static_features, treatments, self.dt, pred_dy_dt)
            # assert not jnp.any(jnp.isnan(preds) | jnp.isinf(preds)), 'Preds contains NaN of Inf'
            # return preds.reshape(-1,1), sequence_length, mse_end, mse_start

            # Parametric dist purposes
            # treatment_idx = jnp.argmax(treatments[0])
            # return preds.reshape(-1,1), reduced_coefs_updated[treatment_idx]
            return preds.reshape(-1,1)

        t0 = time()
        num_cores = jax.local_device_count()
        split = partial(split_inputs, num_cores=num_cores)
        in_axes=(0,0,0, None, 0)
        # split(prev_outputs_unscaled)
        
        # test vmap
        # outputs_unscaled = vmap(simulate_cancer_volume_with_fine_tuning, in_axes=in_axes)(prev_outputs_unscaled_split[0], current_treatments_split[0], static_features_split[0], reduced_coefs, sequence_length_split[0])
        debug = True
        # idx = 55570
        # simulate_cancer_volume_with_fine_tuning(unscaled_prev_outputs[idx], current_treatments[idx], unscaled_static_features[idx], reduced_coefs, sequence_lengths[idx])
        if debug:
            if self.insight_recover_parametric_dist:
                unscaled_preds = debug_vmap(simulate_cancer_volume_with_fine_tuning, in_axes=in_axes, args=(unscaled_prev_outputs, current_treatments, unscaled_static_features, reduced_coefs, sequence_lengths))
            else:
                unscaled_preds = debug_vmap(simulate_cancer_volume_with_fine_tuning, in_axes=in_axes, args=(unscaled_prev_outputs, current_treatments, unscaled_static_features, reduced_coefs, sequence_lengths))
        else:
            # def debug_vmap_simulate_cancer_volume_with_fine_tuning(prev_outputs_unscaled, treatment, static_features, reduced_coefs, sequence_length):
                # return debug_vmap(simulate_cancer_volume_with_fine_tuning, in_axes=in_axes, args=(prev_outputs_unscaled, treatment, static_features, reduced_coefs, sequence_length))
            # outputs_unscaled = jax.pmap(debug_vmap_simulate_cancer_volume_with_fine_tuning, in_axes=in_axes)(split(prev_outputs_unscaled), split(current_treatments[:,:,0]), split(static_features), reduced_coefs, split(dataset.data['sequence_lengths']))
            use_pmap = True
            if use_pmap:
                # def simulate_cancer_volume_with_fine_tuning_wrapper(*args):
                #     return debug_vmap(simulate_cancer_volume_with_fine_tuning, in_axes=in_axes, args=args)
                # unscaled_preds = jax.pmap(simulate_cancer_volume_with_fine_tuning_wrapper, in_axes=in_axes)(split(unscaled_prev_outputs), split(current_treatments), split(unscaled_static_features), reduced_coefs, split(dataset.data['sequence_lengths']))
                unscaled_preds = jax.pmap(jit(vmap(simulate_cancer_volume_with_fine_tuning, in_axes=in_axes)), in_axes=in_axes)(split(unscaled_prev_outputs), split(current_treatments), split(unscaled_static_features), reduced_coefs, split(dataset.data['sequence_lengths']))
                unscaled_preds.block_until_ready()
                unscaled_preds = unscaled_preds.reshape(-1, unscaled_preds.shape[-2], unscaled_preds.shape[-1])
                unscaled_preds = unscaled_preds[:unscaled_prev_outputs.shape[0]]
                # debug only
                # sequence_length = sequence_length.reshape(-1)[:unscaled_prev_outputs.shape[0]]
                # mse_end = mse_end.reshape(-1)[:unscaled_prev_outputs.shape[0]]
                # mse_start = mse_start.reshape(-1)[:unscaled_prev_outputs.shape[0]]
            else:
                unscaled_preds = jit(vmap(simulate_cancer_volume_with_fine_tuning, in_axes=in_axes))(unscaled_prev_outputs, current_treatments, unscaled_static_features, reduced_coefs, sequence_lengths)


        logger.info(f'vmap of [simulate_cancer_volume_with_fine_tuning] time: {time() - t0}')
        # logger.info(f'mse_end: {mse_end.mean()} | mse_start: {mse_start.mean()} | lam {self.lam}')
        # debug only
        # for sl, mse_e, mse_s in zip(sequence_length, mse_end, mse_start):
        #     print(f"Seq len: {sl} | optimized_mse: {mse_e} | sindy_mse: {mse_s}")
        scaled_preds = (unscaled_preds - output_means) / output_stds
        assert not np.any(np.isnan(scaled_preds) | np.isinf(scaled_preds)), 'Scaled_preds contains NaN or Inf'
        # np.where(np.any((np.isnan(scaled_preds) | np.isinf(scaled_preds)).reshape(-1, 59), axis=1))
        # scaled_preds = jnp.nan_to_num(scaled_preds)
        # scaled_preds = jnp.nan_to_num(scaled_preds, nan=0.0, posinf=dataset.norm_const, neginf=0.0)
        # plot_parametric_distribution(coefs_optomized)
        return scaled_preds

    def get_autoregressive_predictions(self, dataset: Dataset) -> np.array:
        logger.info(f'Autoregressive Prediction for {dataset.subset_name}.')

        if not self.insite:
            scaled_preds = self._get_non_fine_tuned_predictions(dataset)
            assert len(scaled_preds.shape) == 3
            assert scaled_preds.shape[2] == 1

            offset = 1
            projection_horizon = self.hparams.dataset.projection_horizon
            sequence_lengths = dataset.data['sequence_lengths'].astype(np.int64)

            def slice_outputs_to_seq(i, sl):
                lower_limit = jax.lax.max(offset, sl - projection_horizon)
                return jax.lax.dynamic_slice(scaled_preds, (i, lower_limit, 0), (1, projection_horizon, 1))[0]

            scaled_preds_seq = jit(vmap(slice_outputs_to_seq, in_axes=(0,0)))(np.arange(dataset.data['sequence_lengths'].shape[0]), sequence_lengths)

            # Debug purposes
            # outputs_scaled_seq_l = []
            # for i, sl in tqdm(enumerate(dataset.data['sequence_lengths'])):
            #     lower_limit = max(offset, sl - projection_horizon)
            #     outputs_scaled_seq_l.append(outputs_scaled[i,lower_limit:sl,:])
            # outputs_scaled_seq = np.stack(outputs_scaled_seq_l)

            # ((outputs_scaled_seq -  dataset.data_processed_seq['outputs']) **2).mean()
            return scaled_preds_seq
        else:
            # Insite individual predictions and evaluation
            projection_horizon = self.hparams.dataset.projection_horizon
            scaled_preds = self._get_fine_tuned_predictions(dataset, projection_horizon=projection_horizon)
            assert len(scaled_preds.shape) == 3
            assert scaled_preds.shape[2] == 1

            offset = 1
            projection_horizon = self.hparams.dataset.projection_horizon

            def slice_outputs_to_seq(i, sl):
                lower_limit = jax.lax.max(offset, sl - projection_horizon)
                return jax.lax.dynamic_slice(scaled_preds, (i, lower_limit, 0), (1, projection_horizon, 1))[0]

            # scaled_preds_seq = debug_vmap(slice_outputs_to_seq, in_axes=(0,0), args=(jnp.arange(dataset.data['sequence_lengths'].shape[0]), jnp.array(dataset.data['sequence_lengths']).astype(jnp.int64)))
            scaled_preds_seq = jit(vmap(slice_outputs_to_seq, in_axes=(0,0)))(jnp.arange(dataset.data['sequence_lengths'].shape[0]), jnp.array(dataset.data['sequence_lengths']).astype(jnp.int64))
            return scaled_preds_seq


        
# Functions

# @jit
def predict_with_reduced_coefs(reduced_coefs, unscaled_prev_outputs, unscaled_static_features, treatments, dt, pred_dy_dt):
    carry_init = (unscaled_prev_outputs[0], unscaled_static_features, reduced_coefs, dt)                    
    scan_fn = jax.tree_util.Partial(predict_scan_func, pred_dy_dt=pred_dy_dt)
    _, predicted_volumes = scan(scan_fn, carry_init, treatments)
    return predicted_volumes

# @jit
def predict_scan_func(carry, treatment, pred_dy_dt):
    cancer_volume, static_features, reduced_coefs, dt = carry
    t_tuple = jnp.array([0, dt])
    cancer_volume = odeint(pred_dy_dt, cancer_volume, t_tuple, treatment, static_features, reduced_coefs, hmax=HMAX, **integrator_keywords)[1]
    return (cancer_volume, static_features, reduced_coefs, dt), cancer_volume

# @jit
def f_to_min_func(reduced_coefs_flattened, coef_sparse_mask, unscaled_prev_outputs, unscaled_static_features, treatments, dt, pred_dy_dt, sequence_length, original_reduced_coefs_flattened, projection_horizon, lam, dim_one_hot_treatments: int, norm_const):
    reduced_coefs = reduced_coefs_flattened.reshape(dim_one_hot_treatments, -1)
    reduced_coefs = reduced_coefs * coef_sparse_mask
    preds = predict_with_reduced_coefs(reduced_coefs, unscaled_prev_outputs, unscaled_static_features, treatments, dt, pred_dy_dt)
    # Mask includes all data points visible
    mask = create_mask(unscaled_prev_outputs.shape[0] - 1, sequence_length - projection_horizon)
    # Mask includes only last data point visible
    # indices = jnp.arange(unscaled_prev_outputs.shape[0]-1)
    # mask = jnp.where((((sequence_length - projection_horizon - 1) <= indices) * (indices < (sequence_length - projection_horizon))), 1, 0)
    se = ((unscaled_prev_outputs[1:] - preds[:-1]) ** 2) * mask
    mse = jnp.sum(se) / jnp.sum(mask)
    mse = mse / norm_const
    mse = mse + lam * jnp.mean(((original_reduced_coefs_flattened - reduced_coefs_flattened) ** 2))
    return mse

def eval_fit_mse(reduced_coefs_flattened, coef_sparse_mask, unscaled_prev_outputs, unscaled_static_features, treatments, dt, pred_dy_dt, sequence_length: int, projection_horizon: int, dim_one_hot_treatments: int):
    reduced_coefs = reduced_coefs_flattened.reshape(dim_one_hot_treatments, -1)
    reduced_coefs = reduced_coefs * coef_sparse_mask
    preds = predict_with_reduced_coefs(reduced_coefs, unscaled_prev_outputs, unscaled_static_features, treatments, dt, pred_dy_dt)
    indices = jnp.arange(unscaled_prev_outputs.shape[0]-1)
    mask = jnp.where((((sequence_length-projection_horizon) <= indices) * (indices < sequence_length)), 1, 0)
    # a = jnp.zeros(sequence_length-projection_horizon)
    # b = jnp.ones(projection_horizon)
    # c = jnp.zeros(unscaled_prev_outputs.shape[0]-1 - sequence_length)
    # mask = jnp.concatenate((a,b,c))
    se = ((unscaled_prev_outputs[1:] - preds[:-1]) ** 2) * mask
    mse = jnp.sum(se) / jnp.sum(mask)
    return mse

def split_inputs(inputs, num_cores):
    input_size = len(inputs)
    chunk_size = input_size // (num_cores - 1)
    data = [inputs[i:i + chunk_size] for i in range(0, input_size, chunk_size)]
    if data[-1].shape[0] < chunk_size:
        new_array = repeat_last_row(data[-1], chunk_size)
        data[-1] = new_array
    return jnp.stack(data)

def repeat_last_row(array, target_rows):
    # Get the current dimensions
    current_shape = array.shape
    current_rank = len(current_shape)

    # Check if the input array is 1-dimensional
    if current_rank == 1:
        last_element = array[-1]
        num_repeats = target_rows - current_shape[0]
        repeated_last_element = jnp.repeat(last_element, num_repeats)
        new_array = jnp.concatenate((array, repeated_last_element), axis=0)
    else:
        # Extract the last row of the array
        last_row = array[-1:]

        # Calculate the number of repetitions required
        num_repeats = target_rows - array.shape[0]

        # Repeat the last row and concatenate it to the original array
        repeated_last_row = jnp.repeat(last_row, num_repeats, axis=0)
        new_array = jnp.concatenate((array, repeated_last_row), axis=0)

    return new_array

def plot_parametric_distribution(coefs_optomized):
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set_style('whitegrid')
    kde_plot = sns.kdeplot(coefs_optomized[np.abs(coefs_optomized)>0], bw=0.1)
    fig = kde_plot.get_figure()
    # fig.savefig("out.png")
    # fig.savefig("out.pdf")
    plt.xlabel(r'$\beta_0$')
    plt.ylabel('Density')
    plt.show()
    plt.savefig("out.png")
    plt.savefig("out.pdf")
    fig.clf()