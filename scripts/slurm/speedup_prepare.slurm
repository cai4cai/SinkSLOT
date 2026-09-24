#!/bin/bash
# One-off preparation for configs/speedup.py (see scripts/speedup_prepare.py):
# median(C) per slice, exact-OT reference cache, Triton/KeOps compile caches.
# Paste the printed MEDIAN_C dict into configs/speedup.py afterwards.
#
#   mkdir -p $ROOT/logs
#   sbatch scripts/slurm/speedup_prepare.slurm
#   sbatch --export=ALL,PREPARE_ARGS="--skip-median --skip-exact-ot" scripts/slurm/speedup_prepare.slurm
#SBATCH --job-name=speedup_prepare
#SBATCH --account=bmy@h100
#SBATCH -C h100
#SBATCH --gres=gpu:1
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --output=/lustre/fswork/projects/rech/bmy/uyc83xr/potential_stop_benchmark/logs/speedup_prepare_%j.out
#SBATCH --error=/lustre/fswork/projects/rech/bmy/uyc83xr/potential_stop_benchmark/logs/speedup_prepare_%j.err

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

python scripts/speedup_prepare.py --workers 4 $PREPARE_ARGS
