#!/bin/bash
#
#SBATCH --job-name=bez2018model
#SBATCH --out="trash/slurm-%A_%a.out"
#SBATCH --cpus-per-task=2
#SBATCH --mem=2000
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --exclude=node[017-094,097,098],dgx001,dgx002
#SBATCH --array=0-49
##SBATCH --partition=mcdermott
##SBATCH --partition=use-everything
#SBATCH --requeue

regex_src="stimuli/sound_localization/evaluation/v01_eval_mit_bldg46room1004_tenoise/*.hdf5"
dir_dst="$VAST_SCRATCH_PATH/stimuli/sound_localization/evaluation/v01_eval_mit_bldg46room1004_tenoise/tfrecords_simplified"
jobs_per_source_file=5
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
