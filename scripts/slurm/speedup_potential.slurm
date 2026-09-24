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

echo "host=$(hostname) shard=$SLURM_ARRAY_TASK_ID/$NUM_SHARDS python=$(which python)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

python run.py --config speedup --execute --num-shards "$NUM_SHARDS" --shard-idx "$SLURM_ARRAY_TASK_ID"
