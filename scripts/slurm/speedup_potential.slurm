#!/bin/bash
# Speedup benchmark (configs/speedup.py) as a job array: task k runs shard k of
# NUM_SHARDS via run.py. Reruns are resume-safe (finished units are skipped).
#
#   mkdir -p $ROOT/logs
#   sbatch scripts/slurm/speedup_potential.slurm
#   sbatch --array=0-99 --export=ALL,NUM_SHARDS=100 scripts/slurm/speedup_potential.slurm
#   python run.py --config speedup --merge            # once every task has finished
#
# The --array range must be 0..NUM_SHARDS-1. The log paths below are fixed; pass
# --output/--error to sbatch when ROOT is overridden.
#SBATCH --job-name=speedup_potential
#SBATCH --account=bmy@h100
#SBATCH -C h100
#SBATCH --gres=gpu:1
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --array=0-209
#SBATCH --output=/lustre/fswork/projects/rech/bmy/uyc83xr/potential_stop_benchmark/logs/speedup_potential_%A_%a.out
#SBATCH --error=/lustre/fswork/projects/rech/bmy/uyc83xr/potential_stop_benchmark/logs/speedup_potential_%A_%a.err

NUM_SHARDS=${NUM_SHARDS:-210}
ROOT=${ROOT:-/lustre/fswork/projects/rech/bmy/uyc83xr/potential_stop_benchmark}

source /etc/profile.d/modules.sh
source /etc/profile.d/cluster_env.sh
module purge
module load pytorch-gpu/py3/2.5.0

set -e
cd "$ROOT"
mkdir -p logs
export PYTHONPATH=$ROOT/torch-ext:$PYTHONPATH
export TRITON_CACHE_DIR=$ROOT/.triton_cache
export KEOPS_CACHE_FOLDER=$ROOT/.keops_cache
mkdir -p "$TRITON_CACHE_DIR" "$KEOPS_CACHE_FOLDER"

echo "host=$(hostname) shard=$SLURM_ARRAY_TASK_ID/$NUM_SHARDS python=$(which python)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

python run.py --config speedup --execute --num-shards "$NUM_SHARDS" --shard-idx "$SLURM_ARRAY_TASK_ID"
