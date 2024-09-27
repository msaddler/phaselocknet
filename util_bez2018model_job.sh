#!/bin/bash
#
#SBATCH --job-name=bez2018model
#SBATCH --out="trash/slurm-%A_%a.out"
#SBATCH --cpus-per-task=2
#SBATCH --mem=2000
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --exclude=node[017-094,097,098],dgx001,dgx002
#SBATCH --array=0-39
##SBATCH --partition=mcdermott
#SBATCH --partition=use-everything
#SBATCH --requeue

regex_src="stimuli/sound_localization/optimization/valid/*.hdf5"
dir_dst="$VAST_SCRATCH_PATH/stimuli/sound_localization/optimization/valid/tfrecords_simplified"
jobs_per_source_file=2
offset=0
job_idx=$(($SLURM_ARRAY_TASK_ID + $offset))

echo $(hostname) "dir_dst=$dir_dst"
echo "regex_src=$regex_src"

source activate tf

python -u util_bez2018model.py \
-s "${regex_src}" \
-d "${dir_dst}" \
-j ${job_idx} \
-jps ${jobs_per_source_file}
