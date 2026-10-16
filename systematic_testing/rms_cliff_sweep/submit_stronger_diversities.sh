#!/bin/bash
# Run on Bouchet from the new, fixed source checkout. No dependency on the baseline.
# --dry-run prints the two commands without submitting jobs.
set -euo pipefail

concurrency=12
partitions=gpu_h100,gpu_h200,gpu_b200,gpu_rtx6000
dry_run=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --concurrency)
            if [[ $# -lt 2 ]]; then
                echo 'Missing --concurrency value' >&2
                exit 2
            fi
            concurrency="$2"
            shift 2
            ;;
        --dry-run)
            dry_run=true
            shift
            ;;
        --partitions)
            if [[ $# -lt 2 ]]; then
                echo 'Missing --partitions value' >&2
                exit 2
            fi
            partitions="$2"
            shift 2
            ;;
        *)
            echo 'Usage: bash submit_stronger_diversities.sh [--concurrency N] [--partitions LIST] [--dry-run]' >&2
            exit 2
            ;;
    esac
done
if [[ ! "$concurrency" =~ ^[1-9][0-9]*$ ]]; then
    echo 'Concurrency must be a positive integer per array' >&2
    exit 2
fi
if [[ ! "$partitions" =~ ^[a-zA-Z0-9_,-]+$ ]]; then
    echo 'Partitions must be a comma-separated list of Slurm partition names' >&2
    exit 2
fi

source_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$source_root"
if [[ "$dry_run" == false ]]; then
    mkdir -p logs
fi
for magnitude in 0.20 0.30; do
    command=(sbatch --parsable "--array=0-399%$concurrency" "--partition=$partitions"
        "--job-name=pd-rms-div${magnitude/./}"
        systematic_testing/rms_cliff_sweep/bouchet_rms_cliff.sbatch
        --diversity-rms "$magnitude")
    if [[ "$dry_run" == true ]]; then
        printf '%q ' "${command[@]}"
        printf '\n'
    else
        receipt="$("${command[@]}")"
        printf 'Diversity +/- %s waves: submitted array %s (400 cases, at most %s GPUs)\n' \
            "$magnitude" "$receipt" "$concurrency"
    fi
done
