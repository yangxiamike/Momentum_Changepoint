#!/bin/bash

wget https://repo.anaconda.com/archive/Anaconda3-2021.04-Linux-x86_64.sh -O anaconda.sh
chomd +x anaconda.sh
./anaconda.sh
source ~/.bashrc

mkdir workspace 
cd workspace
conda create --name cpd python=3.8
conda activate cpd

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple gpflow
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple empyrical
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pyfolio
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tqdm

sudo yum install htop
sudo yum install tmux

echo export WORKSPACE=${HOME}/workspace >> ~/.bashrc
echo export PYTHONPATH=$WORKSPACE/momentum_changepoint/moment_cpd >> ~/.bashrc
source ~/.bashrc
