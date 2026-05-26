# -*- coding: utf-8 -*-

# 测试 MMedBench checkpoint 并计算生成准确率�?# 用法示例�?# bash test_checkpoint_mmedbench.sh --subset=vanilla --language=English --model=Qwen2.5-7B-Instruct --kgrl_reward_mode=vanilla \
#     --ckpt=global_step_600 --now=20260517 --devices=0,1,2,3

device="0,1,2,3"
export CUDA_VISIBLE_DEVICES="$device"
{ IFS=',' read -ra arr <<< "$CUDA_VISIBLE_DEVICES"; ngpu_per_node=${#arr[@]}; }

script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd -P )
main_dir=$(dirname "$script_dir")

dataset=MMedBench
language=English
subset=vanilla
model=Qwen2.5-7B-Instruct
kgrl_reward_mode=vanilla
rl_algorithm=grpo       # gae, grpo, reinforce_plus_plus, reinforce_plus_plus_baseline, rloo
now=xxx
ckpt=last

while [[ $# -gt 0 ]]; do
    case $1 in
        --subset=*)
            subset="${1#*=}"
            shift
            ;;
        --language=*)
            language="${1#*=}"
            shift
            ;;
        --model=*)
            model="${1#*=}"
            shift
            ;;
        --kgrl_reward_mode=*)
            kgrl_reward_mode="${1#*=}"
            shift
            ;;
        --ckpt=*)
            ckpt="${1#*=}"
            shift
            ;;
        --now=*)
            now="${1#*=}"
            shift
            ;;
        --device=*)
            device="${1#*=}"
            export CUDA_VISIBLE_DEVICES="$device"
            { IFS=',' read -ra arr <<< "$CUDA_VISIBLE_DEVICES"; ngpu_per_node=${#arr[@]}; }
            shift
            ;;
        --devices=*)
            device="${1#*=}"
            export CUDA_VISIBLE_DEVICES="$device"
            { IFS=',' read -ra arr <<< "$CUDA_VISIBLE_DEVICES"; ngpu_per_node=${#arr[@]}; }
            shift
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: $0 [--subset=cache] [--language=English] [--model=Qwen2.5-3B-Instruct] [--kgrl_reward_mode=light_weighted_eweight4_etmp2] [--ckpt=global_step_600] [--now=YYYYMMDD] [--device=0,1]"
            exit 1
            ;;
    esac
done

model_name=${model}-$kgrl_reward_mode
exp_name=$dataset-$language-$rl_algorithm-$model_name-$now
exp_path=$main_dir/checkpoint/$dataset/$model_name/$exp_name

if [[ "$ckpt" == "last" ]]; then
    checkpoint_meta="$exp_path/latest_checkpoint_iteration.txt"
    if [ ! -f "$checkpoint_meta" ]; then
        echo "Checkpoint metadata not found: $checkpoint_meta"
        exit 1
    fi
    last_step=$(<"$checkpoint_meta")
    last_step="$(echo "$last_step" | tr -d '[:space:]')"
    if [[ ! "$last_step" =~ ^[0-9]+$ ]]; then
        echo "Invalid checkpoint iteration content in $checkpoint_meta: $last_step"
        exit 1
    fi
    ckpt="global_step_$last_step"
    echo "Resolved latest checkpoint to: $ckpt"
fi

model_path=$exp_path/$ckpt/actor/huggingface
data_dir=$main_dir/data/$dataset/$subset

# ls $model_path
# exit 1

data_path=$data_dir/test-$language.parquet
save_dir=$data_dir/checkpoint/$model_name
if [ ! -d "$save_dir" ]; then
    echo "Creating checkpoint generation directory: $save_dir"
    mkdir -p "$save_dir"
fi
save_path=$save_dir/$exp_name-$ckpt.parquet

# Switch to CUDA=12.x
# source $main_dir/scripts/cuda_switch.sh
# source $main_dir/scripts/check_vllm_version.sh
# switch_cuda 12.4 || exit 1

# Check vllm version
# if check_vllm_le "0.6.3"; then
#     echo "The vllm version <= 0.6.3. Use xformers backend."
#     export VLLM_ATTENTION_BACKEND=XFORMERS
# else
#     case $? in
#         1) echo "No vllm installed. Quit."; exit 1;;
#         2) echo "The vllm version > 0.6.3. Use default backend.";; #  export VLLM_USE_V1=1 ;;
#         *) echo "vllm version check error. Quit."; exit 1;;
#     esac
# fi
echo "-----------------------------------------------"

python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=${ngpu_per_node} \
    data.path=$data_path \
    data.prompt_key=prompt \
    data.n_samples=1 \
    data.output_path=$save_path \
    model.path=$model_path \
    +model.trust_remote_code=True \
    rollout.temperature=0.9 \
    rollout.top_k=50 \
    rollout.top_p=0.7 \
    rollout.prompt_length=512 \
    rollout.response_length=1024 \
    rollout.tensor_model_parallel_size=${ngpu_per_node} \
    rollout.gpu_memory_utilization=0.9 

python compute_generation_acc.py \
    --subset=$subset \
    --filename=$exp_name-$ckpt.parquet

