CACHE_DIR="/opt/dlami/nvme/.cache"
BASE_MODEL_PATH="/home/ubuntu/models"


MODEL_PATH="CohereForAI/aya-expanse-32b"
MODEL_NAME="aya-expanse-32b"

#MODEL_PATH="$BASE_MODEL_PATH/krikri-annealing-sft-stage2"
#MODEL_NAME="krikri-annealing-sft-stage2-run4"
#MODEL_PATH="$BASE_MODEL_PATH/krikri-annealing-dpo-max-length-norm"
#MODEL_NAME="krikri-annealing-sft-stage2-dpo_max-length-norm-run4"
#MODEL_PATH="$BASE_MODEL_PATH/krikri-annealing-dpo-max-length-norm-dpo-fixes"
#MODEL_NAME="krikri-annealing-sft-stage2-dpo_max-length-norm-fixes_on_policy"
#MODEL_PATH="$BASE_MODEL_PATH/krikri-annealing-dpo-max-length-norm-dpo-fixes/checkpoint-1560"
#MODEL_NAME="krikri-annealing-sft-stage2-dpo_max-length-norm-fixes_on_policy-checkpoint-1560"

NUM_GPUS=4
MAX_MODEL_LEN=8192

vllm serve $MODEL_PATH \
  --served-model-name $MODEL_NAME \
  --tensor-parallel-size $NUM_GPUS \
  --enforce-eager \
  --enable-chunked-prefill False \
  --dtype 'bfloat16' \
  --gpu_memory_utilization 0.94 \
  --api-key token-abc123 \
  --download-dir $CACHE_DIR