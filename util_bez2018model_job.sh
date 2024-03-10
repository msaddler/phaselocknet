#!/bin/bash
#
#SBATCH --job-name=bez2018model
#SBATCH --out="slurm-%A_%a.out"
#SBATCH --cpus-per-task=2
#SBATCH --mem=2000
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --exclude=node[017-094,097,098],dgx001,dgx002
#SBATCH --array=0-42
##SBATCH --partition=mcdermott
##SBATCH --partition=use-everything
#SBATCH --requeue

regex_src="stimuli/spkr_word_recognition/evaluation/speech_in_synthetic_textures/*.hdf5"
dir_dst="stimuli/spkr_word_recognition/evaluation/speech_in_synthetic_textures/tfrecords_IHC3000"
jobs_per_source_file=43
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