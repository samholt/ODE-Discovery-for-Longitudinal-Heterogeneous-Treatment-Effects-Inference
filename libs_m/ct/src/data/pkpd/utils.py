import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit
# from jax.experimental.ode import odeint
from jax.lax import stop_gradient, scan
import warnings
from functools import partial
from jax.tree_util import tree_map

import numpy as np
from scipy.linalg import LinAlgWarning
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ridge_regression
from sklearn.utils.validation import check_is_fitted
from sklearn.linear_model import LinearRegression

from sympy import sin, cos, symbols, lambdify, sympify
import sympy2jax
from jax.scipy.optimize import minimize

from pysindy import SINDy
from pysindy.feature_library import PolynomialLibrary
from pysindy.differentiation import FiniteDifference, SmoothedFiniteDifference
from scipy.signal import savgol_filter
from torch import multiprocessing
import sympy2jax
import sys
import optax
import equinox as eqx
from tqdm import tqdm
from time import time

from pysindy.optimizers.base import BaseOptimizer
from enum import Enum

MAX_VALUE = 50.0
integrator_keywords = {}
# STEPS_FOR_DT = 20
STEPS_FOR_DT = 5
# STEPS_FOR_DT = 1
# STEPS_FOR_DT = 20
# STEPS_FOR_DT = 20
# STEPS_FOR_DT = 20
# STEPS_FOR_DT = 20
# STEPS_FOR_DT = 1000
# STEPS_FOR_DT = 1
MAX_TIME_HORIZON = 10.0
smoother_kws = {'window_length': 5, 'polyorder': 3}
# MAX_SEQUENCE_LENGTH = 60.0
MAX_SEQUENCE_LENGTH = int(60)

STANDARD_DT = MAX_TIME_HORIZON / MAX_SEQUENCE_LENGTH
HMAX = STANDARD_DT / STEPS_FOR_DT

class Datasets(Enum):
    EQ_4_A = 1
    EQ_4_B = 2
    EQ_4_C = 3
    EQ_4_D = 4
    EQ_5_A = 5
    EQ_5_B = 6
    EQ_5_C = 7
    EQ_5_D = 8
    EQ_4_M = 9
    CANCER_SIM = 10

def scan_func(carry, dt, dy_dt_func):
    y, *args = carry
    y = y + dy_dt_func(y, dt, *args) * dt
    return (y, *args), y

def odeint_high_resolution_euler(_scan_func, y0, dts, *args):
    dts_i = (dts/STEPS_FOR_DT).repeat(STEPS_FOR_DT, axis=0)
    init_carry = (y0, *args)
    _, ys = scan(_scan_func, init_carry, dts_i)
    # _, ys = debug_scan(_scan_func, init_carry, dts_i)
    yts = jnp.concatenate([y0[None, ...], ys], axis=0)
    return yts[::STEPS_FOR_DT]

def odeint_standard_resolution_euler(_scan_func, y0, dts, *args):
    init_carry = (y0, *args)
    _, ys = scan(_scan_func, init_carry, dts)
    return jnp.concatenate([y0[None, ...], ys], axis=0)

@jit
def odeint(func, y0, t, *args, rtol=1.4e-8, atol=1.4e-8, mxstep=jnp.inf, hmax=jnp.inf):
    _scan_func = jax.tree_util.Partial(scan_func, dy_dt_func=func)
    dts = jnp.diff(t) # Most likely a fixed grid
    return jax.lax.cond(hmax < jnp.diff(t)[0], lambda _: odeint_high_resolution_euler(_scan_func, y0, dts, *args), lambda _: odeint_standard_resolution_euler(_scan_func, y0, dts, *args), operand=None)
    # if hmax < jnp.diff(t)[0]:
    #     return odeint_high_resolution_euler(_scan_func, y0, dts, *args)
    # else:
    #     return odeint_standard_resolution_euler(_scan_func, y0, dts, *args)

class LSQIntialMask(BaseOptimizer):
    """Sequentially thresholded least squares algorithm.
    Defaults to doing Sequentially thresholded Ridge regression.

    Attempts to minimize the objective function
    :math:`\\|y - Xw\\|^2_2 + \\alpha \\|w\\|^2_2`
    by iteratively performing least squares and masking out
    elements of the weight array w that are below a given threshold.

    See the following reference for more details:

        Brunton, Steven L., Joshua L. Proctor, and J. Nathan Kutz.
        "Discovering governing equations from data by sparse
        identification of nonlinear dynamical systems."
        Proceedings of the national academy of sciences
        113.15 (2016): 3932-3937.

    Parameters
    ----------
    threshold : float, optional (default 0.1)
        Minimum magnitude for a coefficient in the weight vector.
        Coefficients with magnitude below the threshold are set
        to zero.

    alpha : float, optional (default 0.05)
        Optional L2 (ridge) regularization on the weight vector.

    max_iter : int, optional (default 20)
        Maximum iterations of the optimization algorithm.

    ridge_kw : dict, optional (default None)
        Optional keyword arguments to pass to the ridge regression.

    fit_intercept : boolean, optional (default False)
        Whether to calculate the intercept for this model. If set to false, no
        intercept will be used in calculations.

    normalize_columns : boolean, optional (default False)
        Normalize the columns of x (the SINDy library terms) before regression
        by dividing by the L2-norm. Note that the 'normalize' option in sklearn
        is deprecated in sklearn versions >= 1.0 and will be removed.

    copy_X : boolean, optional (default True)
        If True, X will be copied; else, it may be overwritten.

    initial_guess : np.ndarray, shape (n_features) or (n_targets, n_features),
            optional (default None)
        Initial guess for coefficients ``coef_``.
        If None, least-squares is used to obtain an initial guess.

    verbose : bool, optional (default False)
        If True, prints out the different error terms every iteration.

    Attributes
    ----------
    coef_ : array, shape (n_features,) or (n_targets, n_features)
        Weight vector(s).

    ind_ : array, shape (n_features,) or (n_targets, n_features)
        Array of 0s and 1s indicating which coefficients of the
        weight vector have not been masked out, i.e. the support of
        ``self.coef_``.

    history_ : list
        History of ``coef_``. ``history_[k]`` contains the values of
        ``coef_`` at iteration k of sequentially thresholded least-squares.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.integrate import odeint
    >>> from pysindy import SINDy
    >>> from pysindy.optimizers import STLSQ
    >>> lorenz = lambda z,t : [10*(z[1] - z[0]),
    >>>                        z[0]*(28 - z[2]) - z[1],
    >>>                        z[0]*z[1] - 8/3*z[2]]
    >>> t = np.arange(0,2,.002)
    >>> x = odeint(lorenz, [-8,8,27], t)
    >>> opt = STLSQ(threshold=.1, alpha=.5)
    >>> model = SINDy(optimizer=opt)
    >>> model.fit(x, t=t[1]-t[0])
    >>> model.print()
    x0' = -9.999 1 + 9.999 x0
    x1' = 27.984 1 + -0.996 x0 + -1.000 1 x1
    x2' = -2.666 x1 + 1.000 1 x0
    """

    def __init__(
        self,
        threshold=0.1,
        alpha=0.05,
        max_iter=20,
        ridge_kw=None,
        normalize_columns=False,
        fit_intercept=False,
        copy_X=True,
        initial_guess=None,
        verbose=False,
    ):
        super(LSQIntialMask, self).__init__(
            max_iter=max_iter,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            normalize_columns=normalize_columns,
        )

        if threshold < 0:
            raise ValueError("threshold cannot be negative")
        if alpha < 0:
            raise ValueError("alpha cannot be negative")

        self.threshold = threshold
        self.alpha = alpha
        self.ridge_kw = ridge_kw
        self.initial_guess = initial_guess
        self.verbose = verbose

    def _sparse_coefficients(self, dim, ind, coef, threshold):
        """Perform thresholding of the weight vector(s)"""
        c = np.zeros(dim)
        c[ind] = coef
        big_ind = np.abs(c) >= threshold
        c[~big_ind] = 0
        return c, big_ind

    def _regress(self, x, y):
        """Perform the ridge regression"""
        kw = self.ridge_kw or {}

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=LinAlgWarning)
            try:
                coef = ridge_regression(x, y, self.alpha, **kw)
            except LinAlgWarning:
                # increase alpha until warning stops
                self.alpha = 2 * self.alpha
        self.iters += 1
        return coef

    def _no_change(self):
        """Check if the coefficient mask has changed after thresholding"""
        this_coef = self.history_[-1].flatten()
        if len(self.history_) > 1:
            last_coef = self.history_[-2].flatten()
        else:
            last_coef = np.zeros_like(this_coef)
        return all(bool(i) == bool(j) for i, j in zip(this_coef, last_coef))

    def _reduce(self, x, y):
        """Performs at most ``self.max_iter`` iterations of the
        sequentially-thresholded least squares algorithm.

        Assumes an initial guess for coefficients and support are saved in
        ``self.coef_`` and ``self.ind_``.
        """
        if self.initial_guess is not None:
            self.coef_ = self.initial_guess
            self.ind_ = np.abs(self.coef_) > 1e-14

        ind = self.ind_
        n_samples, n_features = x.shape
        n_targets = y.shape[1]
        n_features_selected = np.sum(ind)

        # Print initial values for each term in the optimization
        if self.verbose:
            row = [
                "Iteration",
                "|y - Xw|^2",
                "a * |w|_2",
                "|w|_0",
                "Total error: |y - Xw|^2 + a * |w|_2",
            ]
            print(
                "{: >10} ... {: >10} ... {: >10} ... {: >10}"
                " ... {: >10}".format(*row)
            )

        for k in range(self.max_iter):
            if np.count_nonzero(ind) == 0:
                warnings.warn(
                    "Sparsity parameter is too big ({}) and eliminated all "
                    "coefficients".format(self.threshold)
                )
                coef = np.zeros((n_targets, n_features))
                break

            coef = np.zeros((n_targets, n_features))
            for i in range(n_targets):
                if np.count_nonzero(ind[i]) == 0:
                    warnings.warn(
                        "Sparsity parameter is too big ({}) and eliminated all "
                        "coefficients".format(self.threshold)
                    )
                    continue
                coef_i = self._regress(x[:, ind[i]], y[:, i])
                coef_i, ind_i = self._sparse_coefficients(
                    n_features, ind[i], coef_i, self.threshold
                )
                coef[i] = coef_i
                ind[i] = ind_i

            self.history_.append(coef)
            if self.verbose:
                R2 = np.sum((y - np.dot(x, coef.T)) ** 2)
                L2 = self.alpha * np.sum(coef**2)
                L0 = np.count_nonzero(coef)
                row = [k, R2, L2, L0, R2 + L2]
                print(
                    "{0:10d} ... {1:10.4e} ... {2:10.4e} ... {3:10d}"
                    " ... {4:10.4e}".format(*row)
                )
            if np.sum(ind) == n_features_selected or self._no_change():
                # could not (further) select important features
                break
        else:
            warnings.warn(
                "STLSQ._reduce did not converge after {} iterations.".format(
                    self.max_iter
                ),
                ConvergenceWarning,
            )
            try:
                coef
            except NameError:
                coef = self.coef_
                warnings.warn(
                    "STLSQ._reduce has no iterations left to determine coef",
                    ConvergenceWarning,
                )
        self.coef_ = coef
        self.ind_ = ind

    @property
    def complexity(self):
        check_is_fitted(self)

        return np.count_nonzero(self.coef_) + np.count_nonzero(
            [abs(self.intercept_) >= self.threshold]
        )
    
def debug_vmap(func, args, in_axes=()):
    iter_total = None
    args_in = []
    for i, axis in enumerate(in_axes):
        if axis == 0:
            if iter_total is None:
                iter_total = args[i].shape[0]
            args_in.append(args[i])
        elif axis == None:
            args_in.append([args[i]]*iter_total)
    outs = []
    for j in tqdm(range(iter_total)):
        outs.append(func(*[arg[j] for arg in args_in]))
    return tree_map(stack, *outs)
    # return jnp.stack(outs)

# @jit
def stack(*elements):
    return np.stack(elements)

def debug_scan(f, init, xs, length=None):
    if xs is None:
        xs = [None] * length
    carry = init
    ys = []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
    return carry, jnp.stack(ys)

def create_mask(array_length: int, n: int) -> jnp.ndarray:
    indices = jnp.arange(array_length)
    mask = jnp.where(indices < n, 1, 0)
    return mask

def convert_sindy_model_to_sympyjax_model(model, quantize=False, quantize_round_to=3):
    feature_library_names = model.feature_library.get_feature_names().copy()
    coefs = model.coefficients()
    feature_names = model.feature_names
    return convert_sindy_model_to_sympyjax_model_core(feature_library_names, feature_names, coefs, quantize=quantize, quantize_round_to=quantize_round_to)

def convert_sindy_model_to_sympyjax_model_core(feature_library_names, feature_names, coefs, quantize=False, quantize_round_to=3):
    feature_library_names = [fn.replace(' ', '*') for fn in feature_library_names]
    fln_l = []
    for fln in feature_library_names:
        for i in range(len(feature_names)):
            fln = fln.replace(f'x{i}', feature_names[i])
        fln_l.append(fln)
    feature_library_names = fln_l
    str_exp = ''
    for i, coef in enumerate(coefs[0]):
        if np.abs(coef) > 1e-3:
            if quantize:
                coef = np.round(coef, quantize_round_to)
            str_exp += f'+{coef}*' + feature_library_names[i]
    if not np.all(coefs == 0):
        expr = sympify(str_exp)
    else:
        expr = sympify('0.0')
    mod = sympy2jax.SymbolicModule([expr])
    return mod, str_exp

def convert_sindy_model_to_sympy_model(model, quantize=False):
    feature_library_names = model.feature_library.get_feature_names().copy()
    coefs = model.coefficients()
    feature_names = model.feature_names
    feature_library_names = [fn.replace(' ', '*') for fn in feature_library_names]
    fln_l = []
    for fln in feature_library_names:
        for i in range(len(feature_names)):
            fln = fln.replace(f'x{i}', feature_names[i])
        fln_l.append(fln)
    feature_library_names = fln_l
    str_exp = ''
    for i, coef in enumerate(coefs[0]):
        if np.abs(coef) > 1e-3:
            if quantize:
                coef = np.round(coef, 2)
            str_exp += f'{coef}*' + feature_library_names[i]
    expr = sympify(str_exp)
    return expr, str_exp

def process_sindy_training_data(args, joint=False, dataset_name=''):
    i, treatments, cancer, seq_len, sequence_lengths_offset, static = args

    if not joint:
        # Split system into seperate systems defined by the unique treatment
        if 'EQ_4' in dataset_name.name:
            if (treatments[0] == np.array([1, 0])).all():
                X_0 = cancer[:seq_len - sequence_lengths_offset].reshape(-1, 1)
                U_0 = static[:seq_len - sequence_lengths_offset]
                return i, 0, X_0, U_0
            else:
                X_1 = cancer[:seq_len - sequence_lengths_offset].reshape(-1, 1)
                U_1 = static[:seq_len - sequence_lengths_offset]
                return i, 1, X_1, U_1
        elif dataset_name == Datasets.CANCER_SIM or 'EQ_5' in dataset_name.name:
            treatments_l_all, outputs_l_all, static_l_all = [], [], []
            treatments_l, outputs_l, static_l = [], [], []
            for i in range(seq_len):
                treatment = treatments[i]
                if len(treatments_l) >= 1 and (treatment != treatments_l[-1]).any():
                    # Save treatment snippet
                    treatments_l.append(treatments_l[-1])
                    # treatments_l.append(treatment)
                    outputs_l.append(cancer[i])
                    static_l.append(static[i])
                    treatments_l_all.append(np.stack(treatments_l))
                    outputs_l_all.append(np.stack(outputs_l).reshape(-1, 1))
                    static_l_all.append(np.stack(static_l))
                    assert not np.any(np.isnan(treatments_l_all[-1])), 'Treatment contains NaN'
                    assert not np.any(np.isnan(outputs_l_all[-1])), 'Output contains NaN'
                    assert not np.any(np.isnan(static_l_all[-1])), 'Static contains NaN'
                    treatments_l, outputs_l, static_l = [treatment], [cancer[i]], [static[i]]
                else:
                    treatments_l.append(treatment)
                    outputs_l.append(cancer[i])
                    static_l.append(static[i])
                if i == seq_len - 1:
                    treatments_l.append(treatment)
                    outputs_l.append(cancer[i + 1])
                    static_l.append(static[i + 1])
                    treatments_l_all.append(np.stack(treatments_l))
                    outputs_l_all.append(np.stack(outputs_l).reshape(-1, 1))
                    static_l_all.append(np.stack(static_l))
            return i, treatments_l_all, outputs_l_all, static_l_all 



            #     print(i)
            # for treatment in treatments:
            #     if len(treatments_l) >= 1 and (treatment != treatments_l[-1]).any():
            #         # Save treatment snippet
            #         print('Saving treatment snippet')
            #         treatments_l.append(treatments_l[-1])

            #         pass
            #     else:
            #         treatments_l.append(treatment)
                    

            if treatment[0] == 0:
                X_0 = cancer[:seq_len - sequence_lengths_offset].reshape(-1, 1)
                U_0 = static[:seq_len - sequence_lengths_offset]
                return i, 0, X_0, U_0
            else:
                X_1 = cancer[:seq_len - sequence_lengths_offset].reshape(-1, 1)
                U_1 = static[:seq_len - sequence_lengths_offset]
                return i, 1, X_1, U_1
    else:
        # Treat the system as one large system
        if 'EQ_4' in dataset_name.name:
            X_0 = cancer[:seq_len - sequence_lengths_offset].reshape(-1, 1)
            U_0 = np.concatenate((treatments[:seq_len - sequence_lengths_offset].reshape(-1, 1),
                                    static[:seq_len - sequence_lengths_offset]), axis=1)
            return i, 0, X_0, U_0
        elif dataset_name == Datasets.CANCER_SIM or 'EQ_5' in dataset_name.name:
            X_0 = cancer[:seq_len - sequence_lengths_offset].reshape(-1, 1)
            U_0 = np.concatenate((treatments[:seq_len - sequence_lengths_offset],
                                    static[:seq_len - sequence_lengths_offset]), axis=1)
            return i, 0, X_0, U_0
        
        
def process_sindy_training_data_old_format(args, joint=False):
    i, treatment, cancer, seq_len, sequence_lengths_offset, static_c_0, static_c_1 = args

    if not joint:
        # Split system into seperate systems defined by the unique treatment
        if treatment[0] == 0:
            X_0 = cancer[:seq_len - sequence_lengths_offset].reshape(-1, 1)
            U_0 = np.stack((static_c_0[:seq_len - sequence_lengths_offset],
                            static_c_1[:seq_len - sequence_lengths_offset]), axis=1)
            return i, 0, X_0, U_0
        else:
            X_1 = cancer[:seq_len - sequence_lengths_offset].reshape(-1, 1)
            U_1 = np.stack((static_c_0[:seq_len - sequence_lengths_offset],
                            static_c_1[:seq_len - sequence_lengths_offset]), axis=1)
            return i, 1, X_1, U_1
    else:
        # Treat the system as one large system
        X_0 = cancer[:seq_len - sequence_lengths_offset].reshape(-1, 1)
        U_0 = np.stack((treatment[:seq_len - sequence_lengths_offset],
                        static_c_0[:seq_len - sequence_lengths_offset],
                        static_c_1[:seq_len - sequence_lengths_offset]), axis=1)
        return i, 0, X_0, U_0

def process_dataset_into_de_format(dataset,
                                    dim_outcome,
                                    dim_static_features,
                                    dim_treatments,
                                    dim_vitals,
                                    dt,
                                    sequence_lengths_offset = 1,
                                    smooth=False,
                                    # seq_length=MAX_SEQUENCE_LENGTH,
                                    multi_process=False,
                                    joint=False,
                                    dataset_name=''):
    # sequence_lengths_offset: 1: Make one to not cap values at max or min; i.e. violating the underlying ODE.
    # cancer_volume = jnp.array(dataset['cancer_volume']).astype(jnp.float64)
    # treatment_action = jnp.array(dataset['treatment_application']).astype(jnp.int64)
    # sequence_lengths = jnp.array(dataset['sequence_lengths']).astype(jnp.int64)
    # observed_static_c_0 = jnp.array(dataset['observed_static_c_0']).astype(jnp.float64)
    # observed_static_c_1 = jnp.array(dataset['observed_static_c_1']).astype(jnp.float64)
    print('Processing dataset into DE format')

    output_stds, output_means = dataset.scaling_params['output_stds'], dataset.scaling_params['output_means']
    unscaled_prev_outputs = dataset.data['prev_outputs'] * output_stds + output_means
    # unscaled_outputs = dataset.data['outputs'] * output_stds + output_means
    unscaled_static_features = dataset.data['static_features'] * dataset.scaling_params['inputs_stds'][dim_outcome:dim_outcome + dim_static_features] + dataset.scaling_params['input_means'][dim_outcome:dim_outcome + dim_static_features]
    current_treatments = dataset.data['current_treatments']
    t = np.arange(0, dataset.data['prev_outputs'].shape[1] + 1).astype(np.float64) * dt

    # unscaled_prev_outputs = np.squeeze(unscaled_prev_outputs)
    unscaled_outputs = np.squeeze(dataset.data['unscaled_outputs'])
    current_treatments = np.squeeze(current_treatments)
    sequence_lengths = dataset.data['sequence_lengths'].astype(np.int64)
    reconstructed_cancer_volume = np.concatenate((unscaled_prev_outputs[:,0].reshape(-1,1), unscaled_outputs), axis=1)

    # # Checks: Eq 4 dataset
    # cancer_volume = dataset.data['cancer_volume'].astype(jnp.float64)
    # treatment_action = dataset.data['treatment_application'].astype(jnp.int64)
    # observed_static_c_0 = dataset.data['observed_static_c_0'].astype(jnp.float64)
    # observed_static_c_1 = dataset.data['observed_static_c_1'].astype(jnp.float64)

    # assert (jnp.abs((reconstructed_cancer_volume - cancer_volume).mean()) < 1e-15).item()
    # assert (jnp.abs((unscaled_outputs - cancer_volume[:,1:]).mean()) < 1e-15).item()
    # assert (current_treatments == treatment_action[:, :-1]).all().item()
    # assert (unscaled_static_features[:,0] == observed_static_c_0).all().item()
    # assert (unscaled_static_features[:,1] == observed_static_c_1).all().item()

    if smooth:
        smoother_kws_ = smoother_kws.copy()
        smoother_kws_.update({'axis': 1}) 
        unscaled_outputs = np.array(savgol_filter(unscaled_outputs, **smoother_kws))

    # Expand the static variables to be the same length as the cancer volume
    unscaled_static_features_expanded = np.stack([unscaled_static_features for _ in range(reconstructed_cancer_volume.shape[1])], axis=1)
    # observed_static_c_0 = np.stack([observed_static_c_0 for t in range(cancer_volume.shape[1])], axis=1)
    # observed_static_c_1 = np.stack([observed_static_c_1 for t in range(cancer_volume.shape[1])], axis=1)

    # dt = MAX_TIME_HORIZON / seq_length
    sequence_length_max = sequence_lengths.max()

    pstd = partial(process_sindy_training_data, joint=joint, dataset_name=dataset_name)
    individualized_equation_arg_list = []
    if not joint:
        if 'EQ_4' in dataset_name.name:
            X_0, U_0, X_1, U_1 = [], [], [], []
            args_list = [(i, current_treatments[i], reconstructed_cancer_volume[i], sequence_lengths[i], sequence_lengths_offset, unscaled_static_features_expanded[i]) for i in range(unscaled_outputs.shape[0])]
            if multi_process:
                pool_outer = multiprocessing.Pool(multiprocessing.cpu_count())
                results = list(tqdm(pool_outer.imap(pstd, args_list), total=len(args_list)))
                pool_outer.close()
            else:
                results = []
                for args in tqdm(args_list, total=len(args_list)):
                    results.append(pstd(args))

            for res in results:
                i, action, x, u = res
                individualized_equation_arg_list.append((i, action, x, u))
                if action == 0:
                    X_0.append(x)
                    U_0.append(u)
                else:
                    X_1.append(x)
                    U_1.append(u)

            return individualized_equation_arg_list, X_0, U_0, X_1, U_1
        elif dataset_name == Datasets.CANCER_SIM or 'EQ_5' in dataset_name.name:
            sequence_lengths_offset = 0
            XU_t_0, XU_t_1, XU_t_2, XU_t_3 = ([],[]), ([],[]), ([],[]), ([],[])
            args_list = [(i, current_treatments[i], reconstructed_cancer_volume[i], sequence_lengths[i], sequence_lengths_offset, unscaled_static_features_expanded[i]) for i in range(unscaled_outputs.shape[0])]
            if multi_process:
                pool_outer = multiprocessing.Pool(multiprocessing.cpu_count())
                results = list(tqdm(pool_outer.imap(pstd, args_list), total=len(args_list)))
                pool_outer.close()
            else:
                results = []
                for args in tqdm(args_list, total=len(args_list)):
                    results.append(pstd(args))

            for res in results:
                i, action_all, x_all, u_all = res
                for action, x, u in zip(action_all, x_all, u_all):
                    individualized_equation_arg_list.append((i, action, x, u))
                    if np.argmax(action.mean(0)) == 0:
                        XU_t_0[0].append(x)
                        XU_t_0[1].append(u)
                    elif np.argmax(action.mean(0)) == 1:
                        XU_t_1[0].append(x)
                        XU_t_1[1].append(u)
                    elif np.argmax(action.mean(0)) == 2:
                        XU_t_2[0].append(x)
                        XU_t_2[1].append(u)
                    elif np.argmax(action.mean(0)) == 3:
                        XU_t_3[0].append(x)
                        XU_t_3[1].append(u)

            return individualized_equation_arg_list, XU_t_0, XU_t_1, XU_t_2, XU_t_3
    else:
        if 'EQ_4' in dataset_name.name:
            X, U = [], []
            args_list = [(i, current_treatments[i], unscaled_outputs[i], sequence_lengths[i], sequence_lengths_offset, unscaled_static_features_expanded[i]) for i in range(unscaled_outputs.shape[0])]
            if multi_process:
                pool_outer = multiprocessing.Pool(multiprocessing.cpu_count())
                results = list(tqdm(pool_outer.imap(pstd, args_list), total=len(args_list)))
                pool_outer.close()
            else:
                results = []
                for args in tqdm(args_list, total=len(args_list)):
                    results.append(pstd(args))
            for res in results:
                i, _, x, u = res
                individualized_equation_arg_list.append((i, None, x, u))
                X.append(x)
                U.append(u)
            return individualized_equation_arg_list, X, U, None, None
        elif dataset_name == Datasets.CANCER_SIM or 'EQ_5' in dataset_name.name:
            X, U = [], []
            args_list = [(i, current_treatments[i], unscaled_outputs[i], sequence_lengths[i], sequence_lengths_offset, unscaled_static_features_expanded[i]) for i in range(unscaled_outputs.shape[0])]
            if multi_process:
                pool_outer = multiprocessing.Pool(multiprocessing.cpu_count())
                results = list(tqdm(pool_outer.imap(pstd, args_list), total=len(args_list)))
                pool_outer.close()
            else:
                results = []
                for args in tqdm(args_list, total=len(args_list)):
                    results.append(pstd(args))
            for res in results:
                i, _, x, u = res
                individualized_equation_arg_list.append((i, None, x, u))
                X.append(x)
                U.append(u)
            return individualized_equation_arg_list, (X, U), None, None, None

def process_dataset_into_de_format_old_format(dataset, sequence_lengths_offset = 1, smooth=False, seq_length=MAX_SEQUENCE_LENGTH, multi_process=False, joint=False):
    # sequence_lengths_offset: 1: Make one to not cap values at max or min; i.e. violating the underlying ODE.
    # cancer_volume = jnp.array(dataset['cancer_volume']).astype(jnp.float64)
    # treatment_action = jnp.array(dataset['treatment_application']).astype(jnp.int64)
    # sequence_lengths = jnp.array(dataset['sequence_lengths']).astype(jnp.int64)
    # observed_static_c_0 = jnp.array(dataset['observed_static_c_0']).astype(jnp.float64)
    # observed_static_c_1 = jnp.array(dataset['observed_static_c_1']).astype(jnp.float64)
    print('Processing dataset into DE format')

    cancer_volume = dataset['cancer_volume'].astype(jnp.float64)
    treatment_action = dataset['treatment_application'].astype(jnp.int64)
    sequence_lengths = dataset['sequence_lengths'].astype(jnp.int64)
    observed_static_c_0 = dataset['observed_static_c_0'].astype(jnp.float64)
    observed_static_c_1 = dataset['observed_static_c_1'].astype(jnp.float64)
    
    if smooth:
        smoother_kws_ = smoother_kws.copy()
        smoother_kws_.update({'axis': 1}) 
        cancer_volume = np.array(savgol_filter(cancer_volume, **smoother_kws))

    # Expand the static variables to be the same length as the cancer volume
    observed_static_c_0 = np.stack([observed_static_c_0 for t in range(cancer_volume.shape[1])], axis=1)
    observed_static_c_1 = np.stack([observed_static_c_1 for t in range(cancer_volume.shape[1])], axis=1)

    dt = MAX_TIME_HORIZON / seq_length
    sequence_length_max = sequence_lengths.max()
    t = stop_gradient(np.arange(0,sequence_length_max) * dt)

    # treatment_action = np.asarray(treatment_action)
    # cancer_volume = np.asarray(cancer_volume)
    # sequence_lengths = np.asarray(sequence_lengths)
    # observed_static_c_0 = np.asarray(observed_static_c_0)
    # observed_static_c_1 = np.asarray(observed_static_c_1)

    pstd = partial(process_sindy_training_data_old_format, joint=joint)
    individualized_equation_arg_list = []
    if not joint:
        X_0, U_0, X_1, U_1 = [], [], [], []
        args_list = [(i, treatment_action[i], cancer_volume[i], sequence_lengths[i], sequence_lengths_offset, observed_static_c_0[i], observed_static_c_1[i]) for i in range(cancer_volume.shape[0])]
        if multi_process:
            pool_outer = multiprocessing.Pool(multiprocessing.cpu_count())
            results = list(tqdm(pool_outer.imap(pstd, args_list), total=len(args_list)))
            pool_outer.close()
        else:
            results = []
            for args in tqdm(args_list, total=len(args_list)):
                results.append(pstd(args))

        for res in results:
            i, action, x, u = res
            individualized_equation_arg_list.append((i, action, x, u))
            if action == 0:
                X_0.append(x)
                U_0.append(u)
            else:
                X_1.append(x)
                U_1.append(u)

        return individualized_equation_arg_list, X_0, U_0, X_1, U_1, dt
    else:
        X, U = [], []
        args_list = [(i, treatment_action[i], cancer_volume[i], sequence_lengths[i], sequence_lengths_offset, observed_static_c_0[i], observed_static_c_1[i]) for i in range(cancer_volume.shape[0])]
        if multi_process:
            pool_outer = multiprocessing.Pool(multiprocessing.cpu_count())
            results = list(tqdm(pool_outer.imap(pstd, args_list), total=len(args_list)))
            pool_outer.close()
        else:
            results = []
            for args in tqdm(args_list, total=len(args_list)):
                results.append(pstd(args))

        for res in results:
            i, _, x, u = res
            individualized_equation_arg_list.append((i, None, x, u))
            X.append(x)
            U.append(u)

        return individualized_equation_arg_list, X, U, None, None, dt

#===============================================================================
#### Test functions


## Test ODEINT

def test_odeint_simple_linear_dense_t():
    # x = x
    # dx/dt = 1
    def dy_dt(y, t):
        return jnp.ones_like(y)
    dy_dt = jax.tree_util.Partial(dy_dt)

    seq_length = MAX_SEQUENCE_LENGTH  # about half a year
    dt = MAX_TIME_HORIZON / seq_length
    t = jnp.arange(0, MAX_TIME_HORIZON, dt).astype(jnp.float64)

    @jit
    def test_run(t):
        y0 = jnp.array(0.0)
        # y_sol = odeint(jax.tree_util.Partial(dy_dt), y0, t, hmax=HMAX, **integrator_keywords)
        y_sol = odeint(dy_dt, y0, t, hmax=HMAX, **integrator_keywords)
        return y_sol
    
    y_sol = test_run(t)

    assert jnp.mean(jnp.power(y_sol - t, 2)) < 1e-16
    print('[test_odeint_simple_linear_dense_t]\tMSE: ', jnp.mean(jnp.power(y_sol - t, 2)))


def test_odeint_actual_simple_linear_dense_t():
    # x = x
    # dx/dt = 1
    from jax.experimental.ode import odeint as odeint_s
    
    def dy_dt(y, t):
        return jnp.ones_like(y)
    dy_dt = jax.tree_util.Partial(dy_dt)
    
    seq_length = MAX_SEQUENCE_LENGTH  # about half a year
    dt = MAX_TIME_HORIZON / seq_length
    t = jnp.arange(0, MAX_TIME_HORIZON, dt).astype(jnp.float64)

    @jit
    def test_run(t):
        y0 = jnp.array(0.0)
        y_sol = odeint_s(jax.tree_util.Partial(dy_dt), y0, t, hmax=HMAX, **integrator_keywords)
        return y_sol
    
    y_sol = test_run(t)

    assert jnp.mean(jnp.power(y_sol - t, 2)) < 1e-16
    print('[test_odeint_actual_simple_linear_dense_t]\tMSE: ', jnp.mean(jnp.power(y_sol - t, 2)))

def test_odeint_simple_linear_sparse_t():
    # x = x
    # dx/dt = 1
    def dy_dt(y, t):
        return jnp.ones_like(y)
    dy_dt = jax.tree_util.Partial(dy_dt)

    seq_length = MAX_SEQUENCE_LENGTH  # about half a year
    dt = MAX_TIME_HORIZON / seq_length
    t = jnp.arange(0, MAX_TIME_HORIZON, dt).astype(jnp.float64)
    t = jnp.array([t[0], t[-1]])

    @jit
    def test_run(t):
        y0 = jnp.array(0.0)
        y_sol = odeint(jax.tree_util.Partial(dy_dt), y0, jnp.array(t), hmax=HMAX, **integrator_keywords)
        return y_sol
    
    y_sol = test_run(t)

    assert jnp.mean(jnp.power(y_sol - t, 2)) < 1e-16
    print('[test_odeint_simple_linear_sparse_t]\tMSE: ', jnp.mean(jnp.power(y_sol - t, 2)))

## Test ODEINT with sympyjax functions

def test_odeint_simple_linear_dense_t_with_sympyjax_func():
    # x = x
    # dx/dt = 1
    str_exp = '1'
    expr = sympify(str_exp)
    mod = sympy2jax.SymbolicModule([expr])

    # @jit
    def dy_dt(y, t):
        return mod(x0=y)[0]
    dy_dt = jax.tree_util.Partial(dy_dt)

    seq_length = MAX_SEQUENCE_LENGTH  # about half a year
    dt = MAX_TIME_HORIZON / seq_length
    t = jnp.arange(0, MAX_TIME_HORIZON, dt).astype(jnp.float64)

    @jit
    def test_run(t):
        y0 = jnp.array(0.0)
        # y_sol = odeint(jax.tree_util.Partial(dy_dt), y0, t, hmax=HMAX, **integrator_keywords)
        y_sol = odeint(dy_dt, y0, t, hmax=HMAX, **integrator_keywords)
        return y_sol
    
    y_sol = test_run(t)

    assert jnp.mean(jnp.power(y_sol - t, 2)) < 1e-16
    print('[test_odeint_simple_linear_dense_t]\tMSE: ', jnp.mean(jnp.power(y_sol - t, 2)))

def test_gradient_refine_with_sindy_func():
    coefs = jnp.array([0, -10.000, 0, 0])
    feature_names = ['1', 'x0', 'u0', 'u1']
    feature_basis = [lambda x, u: 1, lambda x, u: x, lambda x, u: u[...,0], lambda x, u: u[...,1]]

    # Filter for non-zero coefs
    active_indexes = jnp.nonzero(coefs)[0]
    coefs = coefs[active_indexes]
    feature_names = [feature_names[i] for i in active_indexes]
    feature_basis = [feature_basis[i] for i in active_indexes]

    # @jit
    def sym_dy_dt(y, t, u, coefs):
        print('tracing')
        return jnp.sum(*[coefs[i] * feature_basis[i](y, u) for i in range(len(coefs))])
    sym_dy_dt = jax.tree_util.Partial(sym_dy_dt)

    def true_dy_dt(y, t):
        return - 2 * y
    true_dy_dt = jax.tree_util.Partial(true_dy_dt)

    seq_length = MAX_SEQUENCE_LENGTH  # about half a year
    dt = MAX_TIME_HORIZON / seq_length
    t = jnp.arange(0, MAX_TIME_HORIZON, dt).astype(jnp.float64)

    y0 = jnp.array(10.0)
    y_sol = jit(odeint)(true_dy_dt, y0, t, hmax=HMAX, **integrator_keywords)
    y_pred = jit(odeint)(sym_dy_dt, y0, t, y0, coefs,hmax=HMAX, **integrator_keywords)
    print(jnp.mean(jnp.power(y_sol - y_pred, 2)))

    @jit
    def fun(x):
        y_pred = odeint(sym_dy_dt, y0, t, y0, x, hmax=HMAX, **integrator_keywords)
        return jnp.mean(jnp.power(y_sol - y_pred, 2))

    t0 = time()
    res = minimize(fun, coefs, method='BFGS', tol=1e-6)
    print('Time taken: ', time() - t0)

    print('res', res)
    coefs = res.x
    print('')

if __name__ == '__main__':
    test_odeint_simple_linear_dense_t()
    test_odeint_actual_simple_linear_dense_t()
    test_odeint_simple_linear_sparse_t()
    test_odeint_simple_linear_dense_t_with_sympyjax_func()
    test_gradient_refine_with_sindy_func()