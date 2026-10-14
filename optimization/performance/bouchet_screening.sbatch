#!/bin/bash
# Submit from the repository root; create logs/ before sbatch.
#SBATCH --job-name=pd-screening
#SBATCH --account=pi_cr876
#SBATCH --partition=gpu_h100,gpu_h200,gpu_b200
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/pd-screening-%j.out
#SBATCH --error=logs/pd-screening-%j.err

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
module reset
module load miniconda
conda activate phase_diversity_env

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg

result_dir="results/bouchet/screening/$SLURM_JOB_ID"
mkdir -p "$result_dir"
nvidia-smi > "$result_dir/gpu.txt"
git rev-parse HEAD > "$result_dir/revision.txt"
conda list --explicit > "$result_dir/environment.txt"

REQUIRE_CUDA=1 python -m unittest discover -s tests -p test_candidate_screening.py -v
python -u -m optimization.performance.benchmark_screening \
  --size 128 --pupil-size 128 --images 24 --planes 40 \
  --random-starts 100 --candidate-batches 1 4 8 --repeats 3 \
  > "$result_dir/benchmark.json"
