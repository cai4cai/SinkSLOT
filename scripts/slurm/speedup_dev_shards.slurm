#!/bin/bash
# Run a list of speedup shards on one multi-GPU h100-dev job: GPUS workers, one per
# GPU, each running its share of SHARDS (space-separated shard indices of a
# NUM_SHARDS split) in turn. Resume-safe: a job killed at the 2 h limit keeps its
# finished units, and resubmitting the same SHARDS runs only what is left.
#
#   sbatch --export=ALL,SHARDS="259 260 261",NUM_SHARDS=500 scripts/slurm/speedup_dev_shards.slurm
#SBATCH --job-name=speedup_dev
#SBATCH --account=bmy@h100
#SBATCH -C h100
#SBATCH --gres=gpu:4
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --time=01:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --output=/lustre/fswork/projects/rech/bmy/uyc83xr/potential_stop_benchmark/logs/speedup_dev_%j.out
#SBATCH --error=/lustre/fswork/projects/rech/bmy/uyc83xr/potential_stop_benchmark/logs/speedup_dev_%j.err

NUM_SHARDS=${NUM_SHARDS:-500}
CONFIG=${CONFIG:-speedup}
GPUS=${GPUS:-4}
ROOT=${ROOT:-/lustre/fswork/projects/rech/bmy/uyc83xr/potential_stop_benchmark}

source /etc/profile.d/modules.sh
source /etc/profile.d/cluster_env.sh
module purge
module load pytorch-gpu/py3/2.5.0

set -e
cd "$ROOT"
mkdir -p logs
# ~/.local can hold a scipy built for NumPy 2, which shadows the module's own
# scipy/NumPy 1.x pair and breaks POT; put the module's scipy first.
ENV_SITE=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
mkdir -p "$ROOT/pyshim"
for p in "$ENV_SITE"/scipy "$ENV_SITE"/scipy.libs "$ENV_SITE"/scipy-*.dist-info; do
    [ -e "$p" ] && ln -sfn "$p" "$ROOT/pyshim/$(basename "$p")"
done
export PYTHONPATH=$ROOT/pyshim:$ROOT/torch-ext:$PYTHONPATH
# POT only needs its NumPy/torch backends; importing its JAX and TensorFlow
# backends together aborts (duplicate absl initialisation).
export POT_BACKEND_DISABLE_JAX=1 POT_BACKEND_DISABLE_TENSORFLOW=1 POT_BACKEND_DISABLE_CUPY=1
export TRITON_CACHE_DIR=$ROOT/.triton_cache
export KEOPS_CACHE_FOLDER=$ROOT/.keops_cache
mkdir -p "$TRITON_CACHE_DIR" "$KEOPS_CACHE_FOLDER"

read -r -a ALL <<< "$SHARDS"
echo "host=$(hostname) shards=${#ALL[@]} gpus=$GPUS"
for g in $(seq 0 $((GPUS - 1))); do
    (
        for ((i = g; i < ${#ALL[@]}; i += GPUS)); do
            k=${ALL[$i]}
            echo "[gpu $g] shard $k"
            CUDA_VISIBLE_DEVICES=$g python run.py --config "$CONFIG" --execute \
                --num-shards "$NUM_SHARDS" --shard-idx "$k" > "logs/speedup_dev_${SLURM_JOB_ID}_shard${k}.log" 2>&1
        done
    ) &
done
wait
