#!/bin/bash
#SBATCH --job-name=pd-bead-experiment
#SBATCH --account=pi_cr876
#SBATCH --partition=gpu_h100,gpu_h200,gpu_b200
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -eo pipefail
# submit_bead_pd supplies an isolated source/data snapshot as the working directory.
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p results
module reset
module load miniconda
conda activate phase_diversity_env
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg
nvidia-smi > results/gpu.txt
conda list --explicit > results/environment.txt
python -u -m experimental_10_15.bead_pd solve input.npz --backend cuda --output results "$@"
