#!/bin/bash
#
#SBATCH --job-name=dataset_to_tfrecords
#SBATCH --out="slurm-%A_%a.out"
#SBATCH --cpus-per-task=2
#SBATCH --mem=4000
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --array=0
##SBATCH --partition=mcdermott
##SBATCH --partition=use-everything
#SBATCH --requeue

offset=0
job_idx=$(($SLURM_ARRAY_TASK_ID + $offset))
regex_src="stimuli/spkr_word_recognition/evaluation/speech_in_synthetic_textures/stim*.hdf5"
dir_dst="stimuli/spkr_word_recognition/evaluation/speech_in_synthetic_textures/tfrecords"
jobs_per_source_file=1
offset=0
job_idx=$(($SLURM_ARRAY_TASK_ID + $offset))

echo $(hostname) "dir_dst=$dir_dst"
echo "regex_src=$regex_src"

source activate tf

python -u dataset_to_tfrecords.py \
-s "${regex_src}" \
-d "${dir_dst}" \
-j ${job_idx} \
-jps ${jobs_per_source_file}
