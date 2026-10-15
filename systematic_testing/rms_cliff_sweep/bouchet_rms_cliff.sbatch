#!/bin/bash
# Submit from the source root; create logs/ before sbatch.
# Each task runs one (RMS, aberration direction) condition on one GPU.
#SBATCH --job-name=pd-rms-cliff
#SBATCH --account=pi_cr876
#SBATCH --partition=gpu_h100,gpu_h200,gpu_b200
#SBATCH --qos=normal
#SBATCH --array=0-399%4
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/pd-rms-cliff-%A_%a.out
#SBATCH --error=logs/pd-rms-cliff-%A_%a.err

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the source root}"
module reset
module load miniconda
conda activate phase_diversity_env
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/pd-rms-cliff-${SLURM_JOB_ID}"

result_dir="results/bouchet/rms_cliff_sweep/${SLURM_ARRAY_JOB_ID}"
mkdir -p "$result_dir/provenance"
# Every task keeps its own environment record, including retried task IDs.
record="$result_dir/provenance/${SLURM_ARRAY_TASK_ID}-${SLURM_JOB_ID}"
nvidia-smi > "$record-gpu.txt"
conda list --explicit > "$record-environment.txt"
if [[ -f source_manifest.json ]]; then
    cp source_manifest.json "$record-source-manifest.json"
fi
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git rev-parse HEAD > "$record-revision.txt"
    git diff --binary HEAD -- optimization utils systematic_testing > "$record-working-tree.patch"
fi
if [[ "$SLURM_ARRAY_TASK_ID" == 0 ]]; then
    REQUIRE_CUDA=1 python -m unittest discover -s tests -p test_rms_cliff_sweep.py -v
fi
python -u -m systematic_testing.rms_cliff_sweep.run_sweep \
    --output "$result_dir" --case-index "$SLURM_ARRAY_TASK_ID" "$@"
