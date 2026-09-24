#!/bin/bash
# Cost gap vs eps per slice (scripts/speedup_calibrate_eps.py), one array task per slice.
#
#   sbatch scripts/slurm/speedup_calibrate_eps.slurm
#SBATCH --job-name=speedup_calib
#SBATCH --account=bmy@h100
#SBATCH -C h100
#SBATCH --gres=gpu:1
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --array=0-4
#SBATCH --output=/lustre/fswork/projects/rech/bmy/uyc83xr/potential_stop_benchmark/logs/speedup_calib_%A_%a.out
#SBATCH --error=/lustre/fswork/projects/rech/bmy/uyc83xr/potential_stop_benchmark/logs/speedup_calib_%A_%a.err

ROOT=${ROOT:-/lustre/fswork/projects/rech/bmy/uyc83xr/potential_stop_benchmark}
SLICES=("half_moon 2" "8gaussians 2" "two_rings 2" "gaussian 3" "gaussian 64")
read -r DATASET D <<< "${SLICES[$SLURM_ARRAY_TASK_ID]}"

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

mkdir -p output/speedup_calibrate_eps
python scripts/speedup_calibrate_eps.py --dataset "$DATASET" --d "$D" \
    --out "output/speedup_calibrate_eps/${DATASET}_d${D}.csv"
