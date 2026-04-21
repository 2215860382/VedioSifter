#!/usr/bin/env bash
# Step 2: GRPO 训练脚本
# 使用 VERL 原生 verl.trainer.main_ppo，adv_estimator=grpo
# 奖励函数：verl/reward_score.py::compute_score（离线打分查表 + DCG 加权）

set -xeuo pipefail

project_name='VedioSifter'
exp_name='ep1-GRPO-Qwen3-4B-VedioSifter'

# ============== 算法配置 ==============
adv_estimator=grpo
norm_adv_by_std_in_grpo=True   # 标准 GRPO；改 False 则为 Dr.GRPO

use_kl_in_reward=False
kl_coef=0.0

# 序列长度（prompt 实际 ~350 tokens；ranking 响应含 thinking ~1024 tokens）
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 2))

# 训练批次
train_prompt_bsz=16
n_resp_per_prompt=8          # GRPO 每条 prompt 采样的回复数 G
train_prompt_mini_bsz=4      # PPO mini-batch
gen_prompt_bsz=32

WORKING_DIR=${WORKING_DIR:-"${PWD}"}
echo "WORKING_DIR: ${WORKING_DIR}"
NNODES=${NNODES:-1}

# GPU 数量：有 CUDA_VISIBLE_DEVICES 则推断，否则默认 2
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    N_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
else
    N_GPUS=2
fi
echo "N_GPUS: ${N_GPUS}"

# ============== 路径 ==============
MODEL_PATH=${MODEL_PATH:-"/home2/ycj/Models/Qwen3-4B"}
CKPTS_DIR=${CKPTS_DIR:-"${WORKING_DIR}/models/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-'["data/rl_data/train_0.parquet"]'}
TEST_FILE=${TEST_FILE:-"data/rl_data/test.parquet"}

# ============== 推理采样参数 ==============
temperature=1.0
top_p=1.0
top_k=-1

# ============== 性能配置 ==============
use_dynamic_bsz=True
actor_max_token_len=$((max_prompt_length + max_response_length))
offload=true
gen_tp=1

# ============== flash_attn（可选）==============
# 有 flash_attn 则启用加速；没有则降级到 PyTorch 内置 SDPA，功能完全等价
if python -c "import flash_attn" 2>/dev/null; then
    echo "flash_attn available: using flash_attention_2 + remove_padding"
    use_remove_padding=True
    attn_implementation=flash_attention_2
else
    echo "flash_attn not found: using sdpa (slower but correct)"
    use_remove_padding=False
    attn_implementation=sdpa
fi

export VERL_LOGGING_LEVEL=DEBUG
export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_CONFIGURE_LOGGING=1
export VLLM_USE_V1=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

cd "${WORKING_DIR}"

python -m verl.trainer.main_ppo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=False \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.norm_adv_by_std_in_grpo=${norm_adv_by_std_in_grpo} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=${use_remove_padding} \
    +actor_rollout_ref.model.override_config.attn_implementation=${attn_implementation} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${actor_max_token_len} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.50 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_seqs=8 \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=${top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.rollout.enforce_eager=False \
    reward_model.reward_manager=naive \
    custom_reward_function.path=my_verl/reward_score.py \
    custom_reward_function.name=compute_score \
    trainer.logger='["console"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=False \
    trainer.test_freq=5 \
    trainer.save_freq=5 \
    trainer.total_epochs=1 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    trainer.max_actor_ckpt_to_keep=10
