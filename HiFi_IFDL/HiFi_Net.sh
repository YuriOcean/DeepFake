#!/bin/bash
# HiFi_Net 训练启动脚本

# 激活 conda 环境
source ~/.bashrc
conda activate HiFi_Net

# 指定使用的 GPU 设备，例如使用 0 和 1
CUDA_NUM="0,1"
export CUDA_VISIBLE_DEVICES=$CUDA_NUM

# 启动训练脚本
python HiFi_Net.py