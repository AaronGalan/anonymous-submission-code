# -*- coding: utf-8 -*-

# bash xxx.sh --language English --now xxx --subdir vanilla \
# --model Qwen2.5-7B-Instruct --kgrl_reward_mode light_weighted_eweight0_alpha2_lambda2 \
# --device 0,1,2,3

# 设置默认�?language="English"
NOW=$(date +%Y%m%d%H%M)
subdir="vanilla"
model_name="Qwen2.5-7B-Instruct"
kgrl_reward_mode="light_weighted_eweight0_alpha2_beta3"
device="0,1,2,3"

export CUDA_VISIBLE_DEVICES="$device"
{ IFS=',' read -ra arr <<< "$CUDA_VISIBLE_DEVICES"; ngpu_per_node=${#arr[@]}; }

mini_batch_size=64
learning_rate=1e-6
micro_batch_size_rate=4
rollout_n=5             # the n for grpo/rloo
total_epochs=5
rl_algorithm=grpo       # gae / grpo / reinforce_plus_plus / reinforce_plus_plus_baseline / rloo
if [[ "$rl_algorithm" != "grpo" && "$rl_algorithm" != "rloo" ]]; then
    rollout_n=1
fi

# kgrl_reward_mode="vanilla"

script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd -P )
main_dir=$(dirname "$script_dir")

# Switch to CUDA=12.x
# source $main_dir/scripts/cuda_switch.sh
# source $main_dir/scripts/check_vllm_version.sh
# switch_cuda 12.4 || exit 1


while [[ $# -gt 0 ]]; do
    case $1 in
        --language=*)
            language="${1#*=}"
            shift
            ;;
        --now=*)
            NOW="${1#*=}"
            shift
            ;;
        --mode=*)
            mode="${1#*=}"
            shift
            ;;
        --subdir=*)
            subdir="${1#*=}"
            shift
            ;;
        --model=*)
            model_name="${1#*=}"
            shift
            ;;
        --lr=*)
            learning_rate="${1#*=}"
            shift
            ;;
        --mbsz=*)
            mini_batch_size="${1#*=}"
            shift
            ;;
        --total_epochs=*)
            total_epochs="${1#*=}"
            shift
            ;;
        --kgrl_reward_mode=*)
            kgrl_reward_mode="${1#*=}"
            shift
            ;;
        --device=*)
            device="${1#*=}"
            export CUDA_VISIBLE_DEVICES="$device"
            { IFS=',' read -ra arr <<< "$CUDA_VISIBLE_DEVICES"; ngpu_per_node=${#arr[@]}; }
            shift
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: $0 [--language=English] [--now=YYYYMMDDHHmm] [--subdir=vanilla] [--model=Llama3.1-8B-Instruct] [--kgrl_reward_mode=light_weighted_eweight0_etmp2_alpha2_beta2] [--device=0,1,2,3]"
            exit 1
            ;;
    esac
done

export KGRL_REWARD_MODE=$kgrl_reward_mode
export KRGL_IE_HOST="0.0.0.0:7000"


dataset=MMedBench
data_dir=$main_dir/data/$dataset/$subdir
MODEL_PATH=$main_dir/models/$model_name
reward_path=$main_dir/kgrl_reward.py
task_name=$dataset-$language-$rl_algorithm-$model_name-$kgrl_reward_mode-${NOW}
save_model_checkpoint=$main_dir/checkpoint/$dataset/$model_name-$kgrl_reward_mode/$task_name
if [ ! -d "$save_model_checkpoint" ]; then
    echo "Creating checkpoint directory: $save_model_checkpoint"
    mkdir -p "$save_model_checkpoint"
fi
cp -f $(realpath "$0") $save_model_checkpoint/script.sh

wandb_pro=$task_name
export WANDB_PROJECT=$wandb_pro
export WANDB_DIR=$save_model_checkpoint
export WANDB_EXP=$wandb_pro
export WANDB_MODE="offline"

nproc_per_gpu=1
nnodes=1
total_procs=$(( nproc_per_gpu * nnodes * ngpu_per_node ))
tensor_model_parallel_size=$(( total_procs ))
micro_batch_size=$(( total_procs * micro_batch_size_rate))

# Check vllm version
if check_vllm_le "0.6.3"; then
    echo "The vllm version <= 0.6.3. Please run the cuda122 script."
    exit 1
else
    case $? in
        1) echo "No vllm installed. Quit."; exit 1;;
        2) echo "The vllm version > 0.6.3. Use default backend.";; # export VLLM_USE_V1=1 ;;
        *) echo "vllm version check error. Quit."; exit 1;;
    esac
fi
echo "-----------------------------------------------"
# exit 1
# ray stop

set -x

# Experiment Begin
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    custom_reward_function.path=${reward_path} \
    custom_reward_function.name=compute_score \
    data.train_files=$data_dir/train-$language.parquet \
    data.val_files=$data_dir/test-$language.parquet \
    data.train_batch_size=${mini_batch_size} \
    data.val_batch_size=${mini_batch_size} \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.shuffle=False \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_shm=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.optim.lr=$learning_rate \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size=${micro_batch_size} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.checkpoint.save_contents='hf_model' \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=${micro_batch_size} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ngpu_per_node} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.n=${rollout_n} \
    actor_rollout_ref.rollout.max_num_seqs=128 \
    actor_rollout_ref.rollout.max_model_len=1024 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.max_num_batched_tokens=1024 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=${mini_batch_size} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${WANDB_PROJECT} \
    trainer.experiment_name=${WANDB_EXP} \
    trainer.n_gpus_per_node=${ngpu_per_node} \
    trainer.nnodes=${nnodes} \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.default_local_dir=${save_model_checkpoint} \
    trainer.total_epochs=${total_epochs} | tee $save_model_checkpoint/console.log

    # $@ 2>&1
    # actor_rollout_ref.model.lora_rank=32 \
    # actor_rollout_ref.model.lora_alpha=32 \
    # actor_rollout_ref.rollout.enable_chunked_prefill=False \
    # actor_rollout_ref.rollout.load_format=safetensors \
    # actor_rollout_ref.rollout.layered_summon=True \
