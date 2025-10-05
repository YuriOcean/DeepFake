#!/bin/bash

yaml_config="/data/disk2/yer/ForensicHub/ForensicHub/statics/bisai/focal_train_columbia.yaml"

# 从 yaml 中读取 gpus、log_dir 和 flag
gpus=$(python -c "import yaml; print(yaml.safe_load(open('$yaml_config'))['gpus'])")
base_dir=$(python -c "import yaml; print(yaml.safe_load(open('$yaml_config'))['log_dir'])")
flag=$(python -c "import yaml; print(yaml.safe_load(open('$yaml_config'))['flag'])")

# 计算 GPU 数量
gpu_count=$(echo $gpus | awk -F',' '{print NF}')

# 环境变量设置focal
export PYTHONPATH=/data/disk2/yer/ForensicHub:$PYTHONPATH

# 环境变量设置baseline
#export PYTHONPATH=$(pwd)/ForensicHub:$PYTHONPATH
#mkdir -p ${base_dir}


# 根据 flag 决定运行哪个脚本
if [ "$flag" = "test" ]; then
    script_path="../training_scripts/test.py"
elif [ "$flag" = "train" ]; then
    script_path="../training_scripts/train.py"
else
    echo "配置文件中的 flag 字段必须是 'test' 或 'train'，当前是 '$flag'"
    exit 1
fi

# 自动检测最新 checkpoint
latest_ckpt=$(ls -v ${base_dir}/checkpoint-*.pth 2>/dev/null | tail -n1)
if [ -n "$latest_ckpt" ]; then
    echo "[INFO] Latest checkpoint detected: $latest_ckpt"
    # 临时修改 yaml 的 resume 字段
    python - <<END
import yaml
with open("$yaml_config", "r") as f:
    cfg = yaml.safe_load(f)
cfg['resume'] = "$latest_ckpt"
with open("$yaml_config", "w") as f:
    yaml.safe_dump(cfg, f)
END
else
    echo "[INFO] No checkpoint found, starting training from scratch."
fi

# 启动训练/测试focal
CUDA_VISIBLE_DEVICES=${gpus} \
PYTHONPATH=$(pwd):$PYTHONPATH \
python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=${gpu_count} \
  ${script_path} --config $yaml_config \
2> ${base_dir}/error.log 1>${base_dir}/log.log


# 启动训练/测试baseline
#CUDA_VISIBLE_DEVICES=${gpus} \
#torchrun --standalone --nnodes=1 --nproc_per_node=${gpu_count} \
#${script_path} --config $yaml_config \
#2> ${base_dir}/error.log 1>${base_dir}/log.log