#!/bin/bash
#
#SBATCH --job-name=phaselocknet_localization_eval
#SBATCH --out="slurm-%A_%a.out"
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
##SBATCH --gres=gpu:a100:1
#SBATCH --gres=gpu:1 --exclude=node[017-094,097,098],dgx001,dgx002
#SBATCH --array=0-19
#SBATCH --partition=normal --time=2-0
#SBATCH --requeue

offset=0
job_idx=$(($SLURM_ARRAY_TASK_ID + $offset))
eval_batch_size=32

# Specify model directory (`job_idx` is used to parallelize over `list_model_dir`)
declare -a list_model_dir=(
    "models/sound_localization/sr20000_IHC3000/arch01"
    "models/sound_localization/sr20000_IHC3000/arch02"
    "models/sound_localization/sr20000_IHC3000/arch03"
    "models/sound_localization/sr20000_IHC3000/arch04"
    "models/sound_localization/sr20000_IHC3000/arch05"
    "models/sound_localization/sr20000_IHC3000/arch06"
    "models/sound_localization/sr20000_IHC3000/arch07"
    "models/sound_localization/sr20000_IHC3000/arch08"
    "models/sound_localization/sr20000_IHC3000/arch09"
    "models/sound_localization/sr20000_IHC3000/arch10"
    "models/sound_localization/sr20000_IHC3000_delayed_integration/arch01"
    "models/sound_localization/sr20000_IHC3000_delayed_integration/arch02"
    "models/sound_localization/sr20000_IHC3000_delayed_integration/arch03"
    "models/sound_localization/sr20000_IHC3000_delayed_integration/arch04"
    "models/sound_localization/sr20000_IHC3000_delayed_integration/arch05"
    "models/sound_localization/sr20000_IHC3000_delayed_integration/arch06"
    "models/sound_localization/sr20000_IHC3000_delayed_integration/arch07"
    "models/sound_localization/sr20000_IHC3000_delayed_integration/arch08"
    "models/sound_localization/sr20000_IHC3000_delayed_integration/arch09"
    "models/sound_localization/sr20000_IHC3000_delayed_integration/arch10"

    "models/sound_localization/IHC0050/arch01"
    "models/sound_localization/IHC0050/arch02"
    "models/sound_localization/IHC0050/arch03"
    "models/sound_localization/IHC0050/arch04"
    "models/sound_localization/IHC0050/arch05"
    "models/sound_localization/IHC0050/arch06"
    "models/sound_localization/IHC0050/arch07"
    "models/sound_localization/IHC0050/arch08"
    "models/sound_localization/IHC0050/arch09"
    "models/sound_localization/IHC0050/arch10"
    "models/sound_localization/IHC0320/arch01"
    "models/sound_localization/IHC0320/arch02"
    "models/sound_localization/IHC0320/arch03"
    "models/sound_localization/IHC0320/arch04"
    "models/sound_localization/IHC0320/arch05"
    "models/sound_localization/IHC0320/arch06"
    "models/sound_localization/IHC0320/arch07"
    "models/sound_localization/IHC0320/arch08"
    "models/sound_localization/IHC0320/arch09"
    "models/sound_localization/IHC0320/arch10"
    "models/sound_localization/IHC1000/arch01"
    "models/sound_localization/IHC1000/arch02"
    "models/sound_localization/IHC1000/arch03"
    "models/sound_localization/IHC1000/arch04"
    "models/sound_localization/IHC1000/arch05"
    "models/sound_localization/IHC1000/arch06"
    "models/sound_localization/IHC1000/arch07"
    "models/sound_localization/IHC1000/arch08"
    "models/sound_localization/IHC1000/arch09"
    "models/sound_localization/IHC1000/arch10"
    "models/sound_localization/IHC3000/arch01"
    "models/sound_localization/IHC3000/arch02"
    "models/sound_localization/IHC3000/arch03"
    "models/sound_localization/IHC3000/arch04"
    "models/sound_localization/IHC3000/arch05"
    "models/sound_localization/IHC3000/arch06"
    "models/sound_localization/IHC3000/arch07"
    "models/sound_localization/IHC3000/arch08"
    "models/sound_localization/IHC3000/arch09"
    "models/sound_localization/IHC3000/arch10"
    "models/sound_localization/IHC3000_delayed_integration/arch01"
    "models/sound_localization/IHC3000_delayed_integration/arch02"
    "models/sound_localization/IHC3000_delayed_integration/arch03"
    "models/sound_localization/IHC3000_delayed_integration/arch04"
    "models/sound_localization/IHC3000_delayed_integration/arch05"
    "models/sound_localization/IHC3000_delayed_integration/arch06"
    "models/sound_localization/IHC3000_delayed_integration/arch07"
    "models/sound_localization/IHC3000_delayed_integration/arch08"
    "models/sound_localization/IHC3000_delayed_integration/arch09"
    "models/sound_localization/IHC3000_delayed_integration/arch10"
    "models/sound_localization/simplified_IHC0050/arch01"
    "models/sound_localization/simplified_IHC0050/arch02"
    "models/sound_localization/simplified_IHC0050/arch03"
    "models/sound_localization/simplified_IHC0050/arch04"
    "models/sound_localization/simplified_IHC0050/arch05"
    "models/sound_localization/simplified_IHC0050/arch06"
    "models/sound_localization/simplified_IHC0050/arch07"
    "models/sound_localization/simplified_IHC0050/arch08"
    "models/sound_localization/simplified_IHC0050/arch09"
    "models/sound_localization/simplified_IHC0050/arch10"
    "models/sound_localization/simplified_IHC0320/arch01"
    "models/sound_localization/simplified_IHC0320/arch02"
    "models/sound_localization/simplified_IHC0320/arch03"
    "models/sound_localization/simplified_IHC0320/arch04"
    "models/sound_localization/simplified_IHC0320/arch05"
    "models/sound_localization/simplified_IHC0320/arch06"
    "models/sound_localization/simplified_IHC0320/arch07"
    "models/sound_localization/simplified_IHC0320/arch08"
    "models/sound_localization/simplified_IHC0320/arch09"
    "models/sound_localization/simplified_IHC0320/arch10"
    "models/sound_localization/simplified_IHC1000/arch01"
    "models/sound_localization/simplified_IHC1000/arch02"
    "models/sound_localization/simplified_IHC1000/arch03"
    "models/sound_localization/simplified_IHC1000/arch04"
    "models/sound_localization/simplified_IHC1000/arch05"
    "models/sound_localization/simplified_IHC1000/arch06"
    "models/sound_localization/simplified_IHC1000/arch07"
    "models/sound_localization/simplified_IHC1000/arch08"
    "models/sound_localization/simplified_IHC1000/arch09"
    "models/sound_localization/simplified_IHC1000/arch10"
    "models/sound_localization/simplified_IHC3000/arch01"
    "models/sound_localization/simplified_IHC3000/arch02"
    "models/sound_localization/simplified_IHC3000/arch03"
    "models/sound_localization/simplified_IHC3000/arch04"
    "models/sound_localization/simplified_IHC3000/arch05"
    "models/sound_localization/simplified_IHC3000/arch06"
    "models/sound_localization/simplified_IHC3000/arch07"
    "models/sound_localization/simplified_IHC3000/arch08"
    "models/sound_localization/simplified_IHC3000/arch09"
    "models/sound_localization/simplified_IHC3000/arch10"
    "models/sound_localization/simplified_IHC3000_delayed_integration/arch01"
    "models/sound_localization/simplified_IHC3000_delayed_integration/arch02"
    "models/sound_localization/simplified_IHC3000_delayed_integration/arch03"
    "models/sound_localization/simplified_IHC3000_delayed_integration/arch04"
    "models/sound_localization/simplified_IHC3000_delayed_integration/arch05"
    "models/sound_localization/simplified_IHC3000_delayed_integration/arch06"
    "models/sound_localization/simplified_IHC3000_delayed_integration/arch07"
    "models/sound_localization/simplified_IHC3000_delayed_integration/arch08"
    "models/sound_localization/simplified_IHC3000_delayed_integration/arch09"
    "models/sound_localization/simplified_IHC3000_delayed_integration/arch10"
)
model_dir="${list_model_dir[$job_idx]}"
echo $HOSTNAME $job_idx $model_dir

# Specify `DATA_TAG` based on `model_dir`
DATA_TAG="tfrecords_IHC3000"
if [[ "$model_dir" == *"IHC3000"* ]]; then
    DATA_TAG="tfrecords_IHC3000"
fi
if [[ "$model_dir" == *"IHC1000"* ]]; then
    DATA_TAG="tfrecords_IHC1000"
fi
if [[ "$model_dir" == *"IHC0320"* ]]; then
    DATA_TAG="tfrecords_IHC0320"
fi
if [[ "$model_dir" == *"IHC0050"* ]]; then
    DATA_TAG="tfrecords_IHC0050"
fi
if [[ "$model_dir" == *"sr20000_IHC3000"* ]]; then
    DATA_TAG="tfrecords_IHC3000_sr20000"
fi

# Activate python environment and run `phaselocknet_run.py` for each evaluation dataset
module add openmind8/anaconda
source activate tf

regex_eval="$SCRATCH_PATH/stimuli/sound_localization/evaluation/v01_eval_mit_bldg46room1004_tenoise/$DATA_TAG/*.tfrecords"
filename_eval="EVAL_v01_eval_mit_bldg46room1004_tenoise.json"
python -u phaselocknet_run.py \
-m "$model_dir" \
-c "config.json" \
-a "arch.json" \
-e "$regex_eval" \
-efn "$filename_eval" \
-ebs $eval_batch_size \
-wpo 1

for EXPT_TAG in "minimum_audible_angle_interpolated" "precedence_effect_localization" "bandwidth_dependency" "itd_threshold" "itd_ild_weighting" "new_ears" "spectral_smoothing" "mp_spectral_cues" "speech_in_noise_in_reverb_v04"
do
    regex_eval="$SCRATCH_PATH/stimuli/sound_localization/evaluation/$EXPT_TAG/$DATA_TAG/*.tfrecords"
    filename_eval="EVAL_PSYCHOPHYSICSv00_$EXPT_TAG.json"
    python -u phaselocknet_run.py \
    -m "$model_dir" \
    -c "config.json" \
    -a "arch.json" \
    -e "$regex_eval" \
    -efn "$filename_eval" \
    -ebs $eval_batch_size \
    -wpo 1
done
