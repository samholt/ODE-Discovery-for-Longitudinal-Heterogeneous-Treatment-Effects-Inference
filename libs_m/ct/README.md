CausalTransformer
==============================
[![Conference](https://img.shields.io/badge/ICML22-Paper-blue])](https://proceedings.mlr.press/v162/melnychuk22a/melnychuk22a.pdf)
[![arXiv](https://img.shields.io/badge/arXiv-2204.07258-b31b1b.svg)](https://arxiv.org/abs/2204.07258)
[![Python application](https://github.com/Valentyn1997/CausalTransformer/actions/workflows/python-app.yml/badge.svg)](https://github.com/Valentyn1997/CausalTransformer/actions/workflows/python-app.yml)

Causal Transformer for estimating counterfactual outcomes over time.

<img width="1518" alt="Screenshot 2022-06-03 at 16 41 44" src="https://user-images.githubusercontent.com/23198776/171877145-c7cba15e-9787-4594-8f1f-cbb8b337b74a.png">


The project is built with following Python libraries:
1. [Pytorch-Lightning](https://pytorch-lightning.readthedocs.io/en/latest/) - deep learning models
2. [Hydra](https://hydra.cc/docs/intro/) - simplified command line arguments management
3. [MlFlow](https://mlflow.org/) - experiments tracking

### Installations
First one needs to make the virtual environment and install all the requirements:
```console
pip3 install virtualenv
python3 -m virtualenv -p python3 --always-copy venv
source venv/bin/activate
pip3 install -r requirements.txt
```

## MlFlow Setup / Connection
To start an experiments server, run: 

`mlflow server --port=5000`

To access MlFLow web UI with all the experiments, connect via ssh:

`ssh -N -f -L localhost:5000:localhost:5000 <username>@<server-link>`

Then, one can go to local browser http://localhost:5000.

## Experiments

Main training script is universal for different models and datasets. For details on mandatory arguments - see the main configuration file `config/config.yaml` and other files in `configs/` folder.

Generic script with logging and fixed random seed is following (with `training-type` `enc_dec`, `gnet`, `rmsn` and `multi`):
```console
PYTHONPATH=. CUDA_VISIBLE_DEVICES=<devices> 
python3 runnables/train_<training-type>.py +dataset=<dataset> +backbone=<backbone> exp.seed=10 exp.logging=True
```

### Backbones (baselines)
One needs to choose a backbone and then fill the specific hyperparameters (they are left blank in the configs):
- Causal Transformer (this paper): `runnables/train_multi.py  +backbone=ct`
- Encoder-Decoder Causal Transformer (this paper): `runnables/train_enc_dec.py  +backbone=edct`
- [Marginal Structural Models](https://pubmed.ncbi.nlm.nih.gov/10955408/) (MSMs): `runnables/train_msm.py +backbone=msm`
- [Recurrent Marginal Structural Networks](https://papers.nips.cc/paper/2018/hash/56e6a93212e4482d99c84a639d254b67-Abstract.html) (RMSNs): `runnables/train_rmsn.py +backbone=rmsn`
- [Counterfactual Recurrent Network](https://arxiv.org/abs/2002.04083) (CRN): `runnables/train_enc_dec.py +backbone=crn`
- [G-Net](https://proceedings.mlr.press/v158/li21a/li21a.pdf): `runnables/train_gnet.py  +backbone=gnet`


Models already have best hyperparameters saved (for each model and dataset), one can access them via: `+backbone/<backbone>_hparams/cancer_sim_<balancing_objective>=<coeff_value>`.

For CT, EDCT, and CT, several adversarial balancing objectives are available:
- counterfactual domain confusion loss (this paper): `exp.balancing=domain_confusion`
- gradient reversal (originally in CRN, but can be used for all the methods): `exp.balancing=grad_reverse`

To train a decoder (for CRN and RMSNs), use the flag `model.train_decoder=True`.

To perform a manual hyperparameter tuning use the flags `model.<sub_model>.tune_hparams=True`, and then see `model.<sub_model>.hparams_grid`. Use `model.<sub_model>.tune_range` to specify the number of trials for random search.


### Datasets
One needs to specify a dataset / dataset generator (and some additional parameters, e.g. set gamma for `cancer_sim` with `dataset.coeff=1.0`):
- Synthetic Tumor Growth Simulator: `+dataset=cancer_sim`

Example of running Causal Transformer on Synthetic Tumor Growth Generator with gamma = [1.0, 2.0, 3.0] and different random seeds (total of 30 subruns), using hyperparameters:

```console
PYTHONPATH=. CUDA_VISIBLE_DEVICES=<devices> 
python3 runnables/train_multi.py -m +dataset=cancer_sim +backbone=ct +backbone/ct_hparams/cancer_sim_domain_conf='0','1','2' exp.seed=10,101,1010,10101,101010
```

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
