#!/bin/bash
conda create --name cdes python=3.9.7
conda activate cdes
# Follow official instructions to install JAX for your system
# Follow official instructions to install PyTorch for your system
pip install hydra-core --upgrade
pip install -r requirements.txt
pip install hydra-core --upgrade
pip install -r requirements.txt