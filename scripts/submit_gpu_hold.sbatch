#!/bin/bash
#SBATCH --job-name=gpu-hold
#SBATCH --partition=RTX
#SBATCH --gpus=2080ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH --time=02:00:00
#SBATCH --output=outputs/logs/gpuhold_%j.out
#SBATCH --error=outputs/logs/gpuhold_%j.err

# ---------------------------------------------------------------------------
# SENTINEL: occupy the FAULTED GPU on baldur so SLURM stops handing it to real
# jobs. baldur's card at PCI 0000:60:00.0 (SLURM GPU id 2) has fallen off the
# bus -- any CUDA job allocated it dies with "No CUDA GPUs available", and it is
# always the free-first GPU so new jobs keep landing on it.
#
# This job requests ONE GPU and just sleeps. It touches NO CUDA, so it is happy
# to hold the dead card; whichever GPU it grabs is removed from the free pool,
# leaving a HEALTHY card for scripts/submit_ood_eval.sbatch.
#
# Usage:
#   sbatch scripts/submit_gpu_hold.sbatch      # 1) submit FIRST, wait until R
#   squeue -u $USER                            #    confirm it is Running
#   cat outputs/logs/gpuhold_<jobid>.out       #    see which GPU it grabbed
#   sbatch scripts/submit_ood_eval.sbatch      # 2) real job -> gets a good GPU
#   scancel <gpuhold_jobid>                    # 3) release when the eval is done
#
# NOTE: this is a stopgap. The real fix is for admins to reset/drain that GPU.
# ---------------------------------------------------------------------------

echo "[hold] host=$(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}  SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}"
if nvidia-smi -L >/dev/null 2>&1; then
  echo "[hold] this GPU is HEALTHY:"; nvidia-smi -L
  echo "[hold] WARNING: sentinel grabbed a GOOD gpu, not the faulted one."
  echo "[hold]          scancel this job and resubmit so it lands on the bad one,"
  echo "[hold]          or just submit the eval now (a good gpu is clearly free)."
else
  echo "[hold] this GPU is FAULTED (good -- parking on it so real jobs avoid it)."
fi
echo "[hold] holding allocation; scancel this job when the eval finishes."
sleep 7200
