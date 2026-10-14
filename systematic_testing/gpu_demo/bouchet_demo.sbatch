#!/bin/bash
# Submit from the repository root; create logs/ before sbatch.
# Extra CLI arguments are forwarded, e.g. sbatch ... --random-starts 1000.
#SBATCH --job-name=pd-ao-demo
#SBATCH --account=pi_cr876
#SBATCH --partition=gpu_h100,gpu_h200,gpu_b200
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/pd-ao-demo-%j.out
#SBATCH --error=logs/pd-ao-demo-%j.err

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
module reset
module load miniconda
conda activate phase_diversity_env
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg

nvidia-smi
conda list --explicit > "logs/pd-ao-demo-$SLURM_JOB_ID-environment.txt"
git status --short > "logs/pd-ao-demo-$SLURM_JOB_ID-git-status.txt"
REQUIRE_CUDA=1 python -m unittest discover -s tests -p test_gpu_demo.py -v
result_dir="results/bouchet/gpu_demo/$SLURM_JOB_ID"
python -u -m systematic_testing.gpu_demo.run_demo --output "$result_dir" "$@"
cp "logs/pd-ao-demo-$SLURM_JOB_ID-environment.txt" "$result_dir/environment.txt"
cp "logs/pd-ao-demo-$SLURM_JOB_ID-git-status.txt" "$result_dir/git-status.txt"
