#!/bin/bash
#
#SBATCH --job-name=phaselocknet_localization_eval
#SBATCH --out="slurm-%A_%a.out"
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-99
#SBATCH --partition=normal --time=2-0
#SBATCH --requeue

offset=0
job_idx=$(($SLURM_ARRAY_TASK_ID + $offset))
eval_batch_size=32

# Specify model directory (`job_idx` is used to parallelize over `list_model_dir`)
declare -a list_model_dir=(
    "saved_models/models_localize/v01/IHC3000Hz_anf384H160M096L/archFrancl01"
    "saved_models/models_localize/v01/IHC3000Hz_anf384H160M096L/archFrancl02"
    "saved_models/models_localize/v01/IHC3000Hz_anf384H160M096L/archFrancl03"
    "saved_models/models_localize/v01/IHC3000Hz_anf384H160M096L/archFrancl04"
    "saved_models/models_localize/v01/IHC3000Hz_anf384H160M096L/archFrancl05"
    "saved_models/models_localize/v01/IHC3000Hz_anf384H160M096L/archFrancl06"
    "saved_models/models_localize/v01/IHC3000Hz_anf384H160M096L/archFrancl07"
    "saved_models/models_localize/v01/IHC3000Hz_anf384H160M096L/archFrancl08"
    "saved_models/models_localize/v01/IHC3000Hz_anf384H160M096L/archFrancl09"
    "saved_models/models_localize/v01/IHC3000Hz_anf384H160M096L/archFrancl10"
    "saved_models/models_localize/v01/IHC1000Hz_anf384H160M096L/archFrancl01"
    "saved_models/models_localize/v01/IHC1000Hz_anf384H160M096L/archFrancl02"
    "saved_models/models_localize/v01/IHC1000Hz_anf384H160M096L/archFrancl03"
    "saved_models/models_localize/v01/IHC1000Hz_anf384H160M096L/archFrancl04"
    "saved_models/models_localize/v01/IHC1000Hz_anf384H160M096L/archFrancl05"
    "saved_models/models_localize/v01/IHC1000Hz_anf384H160M096L/archFrancl06"
    "saved_models/models_localize/v01/IHC1000Hz_anf384H160M096L/archFrancl07"
    "saved_models/models_localize/v01/IHC1000Hz_anf384H160M096L/archFrancl08"
    "saved_models/models_localize/v01/IHC1000Hz_anf384H160M096L/archFrancl09"
    "saved_models/models_localize/v01/IHC1000Hz_anf384H160M096L/archFrancl10"
    "saved_models/models_localize/v01/IHC0320Hz_anf384H160M096L/archFrancl01"
    "saved_models/models_localize/v01/IHC0320Hz_anf384H160M096L/archFrancl02"
    "saved_models/models_localize/v01/IHC0320Hz_anf384H160M096L/archFrancl03"
    "saved_models/models_localize/v01/IHC0320Hz_anf384H160M096L/archFrancl04"
    "saved_models/models_localize/v01/IHC0320Hz_anf384H160M096L/archFrancl05"
    "saved_models/models_localize/v01/IHC0320Hz_anf384H160M096L/archFrancl06"
    "saved_models/models_localize/v01/IHC0320Hz_anf384H160M096L/archFrancl07"
    "saved_models/models_localize/v01/IHC0320Hz_anf384H160M096L/archFrancl08"
    "saved_models/models_localize/v01/IHC0320Hz_anf384H160M096L/archFrancl09"
    "saved_models/models_localize/v01/IHC0320Hz_anf384H160M096L/archFrancl10"
    "saved_models/models_localize/v01/IHC0050Hz_anf384H160M096L/archFrancl01"
    "saved_models/models_localize/v01/IHC0050Hz_anf384H160M096L/archFrancl02"
    "saved_models/models_localize/v01/IHC0050Hz_anf384H160M096L/archFrancl03"
    "saved_models/models_localize/v01/IHC0050Hz_anf384H160M096L/archFrancl04"
    "saved_models/models_localize/v01/IHC0050Hz_anf384H160M096L/archFrancl05"
    "saved_models/models_localize/v01/IHC0050Hz_anf384H160M096L/archFrancl06"
    "saved_models/models_localize/v01/IHC0050Hz_anf384H160M096L/archFrancl07"
    "saved_models/models_localize/v01/IHC0050Hz_anf384H160M096L/archFrancl08"
    "saved_models/models_localize/v01/IHC0050Hz_anf384H160M096L/archFrancl09"
    "saved_models/models_localize/v01/IHC0050Hz_anf384H160M096L/archFrancl10"
    "saved_models/models_localize/v01/grouped/before_pool_factor2/IHC3000Hz_anf384H160M096L/archFrancl01"
    "saved_models/models_localize/v01/grouped/before_pool_factor2/IHC3000Hz_anf384H160M096L/archFrancl02"
    "saved_models/models_localize/v01/grouped/before_pool_factor2/IHC3000Hz_anf384H160M096L/archFrancl03"
    "saved_models/models_localize/v01/grouped/before_pool_factor2/IHC3000Hz_anf384H160M096L/archFrancl04"
    "saved_models/models_localize/v01/grouped/before_pool_factor2/IHC3000Hz_anf384H160M096L/archFrancl05"
    "saved_models/models_localize/v01/grouped/before_pool_factor2/IHC3000Hz_anf384H160M096L/archFrancl06"
    "saved_models/models_localize/v01/grouped/before_pool_factor2/IHC3000Hz_anf384H160M096L/archFrancl07"
    "saved_models/models_localize/v01/grouped/before_pool_factor2/IHC3000Hz_anf384H160M096L/archFrancl08"
    "saved_models/models_localize/v01/grouped/before_pool_factor2/IHC3000Hz_anf384H160M096L/archFrancl09"
    "saved_models/models_localize/v01/grouped/before_pool_factor2/IHC3000Hz_anf384H160M096L/archFrancl10"

    "saved_models/models_localize/v01/spont0_simplified_IHC3000/archFrancl01"
    "saved_models/models_localize/v01/spont0_simplified_IHC3000/archFrancl02"
    "saved_models/models_localize/v01/spont0_simplified_IHC3000/archFrancl03"
    "saved_models/models_localize/v01/spont0_simplified_IHC3000/archFrancl04"
    "saved_models/models_localize/v01/spont0_simplified_IHC3000/archFrancl05"
    "saved_models/models_localize/v01/spont0_simplified_IHC3000/archFrancl06"
    "saved_models/models_localize/v01/spont0_simplified_IHC3000/archFrancl07"
    "saved_models/models_localize/v01/spont0_simplified_IHC3000/archFrancl08"
    "saved_models/models_localize/v01/spont0_simplified_IHC3000/archFrancl09"
    "saved_models/models_localize/v01/spont0_simplified_IHC3000/archFrancl10"
    "saved_models/models_localize/v01/spont0_simplified_IHC1000/archFrancl01"
    "saved_models/models_localize/v01/spont0_simplified_IHC1000/archFrancl02"
    "saved_models/models_localize/v01/spont0_simplified_IHC1000/archFrancl03"
    "saved_models/models_localize/v01/spont0_simplified_IHC1000/archFrancl04"
    "saved_models/models_localize/v01/spont0_simplified_IHC1000/archFrancl05"
    "saved_models/models_localize/v01/spont0_simplified_IHC1000/archFrancl06"
    "saved_models/models_localize/v01/spont0_simplified_IHC1000/archFrancl07"
    "saved_models/models_localize/v01/spont0_simplified_IHC1000/archFrancl08"
    "saved_models/models_localize/v01/spont0_simplified_IHC1000/archFrancl09"
    "saved_models/models_localize/v01/spont0_simplified_IHC1000/archFrancl10"
    "saved_models/models_localize/v01/spont0_simplified_IHC0320/archFrancl01"
    "saved_models/models_localize/v01/spont0_simplified_IHC0320/archFrancl02"
    "saved_models/models_localize/v01/spont0_simplified_IHC0320/archFrancl03"
    "saved_models/models_localize/v01/spont0_simplified_IHC0320/archFrancl04"
    "saved_models/models_localize/v01/spont0_simplified_IHC0320/archFrancl05"
    "saved_models/models_localize/v01/spont0_simplified_IHC0320/archFrancl06"
    "saved_models/models_localize/v01/spont0_simplified_IHC0320/archFrancl07"
    "saved_models/models_localize/v01/spont0_simplified_IHC0320/archFrancl08"
    "saved_models/models_localize/v01/spont0_simplified_IHC0320/archFrancl09"
    "saved_models/models_localize/v01/spont0_simplified_IHC0320/archFrancl10"
    "saved_models/models_localize/v01/spont0_simplified_IHC0050/archFrancl01"
    "saved_models/models_localize/v01/spont0_simplified_IHC0050/archFrancl02"
    "saved_models/models_localize/v01/spont0_simplified_IHC0050/archFrancl03"
    "saved_models/models_localize/v01/spont0_simplified_IHC0050/archFrancl04"
    "saved_models/models_localize/v01/spont0_simplified_IHC0050/archFrancl05"
    "saved_models/models_localize/v01/spont0_simplified_IHC0050/archFrancl06"
    "saved_models/models_localize/v01/spont0_simplified_IHC0050/archFrancl07"
    "saved_models/models_localize/v01/spont0_simplified_IHC0050/archFrancl08"
    "saved_models/models_localize/v01/spont0_simplified_IHC0050/archFrancl09"
    "saved_models/models_localize/v01/spont0_simplified_IHC0050/archFrancl10"
    "saved_models/models_localize/v01/grouped/before_pool_factor2/spont0_simplified_IHC3000/archFrancl01"
    "saved_models/models_localize/v01/grouped/before_pool_factor2/spont0_simplified_IHC3000/archFrancl02"
    "saved_models/models_localize/v01/grouped/before_pool_factor2/spont0_simplified_IHC3000/archFrancl03"
    "saved_models/models_localize/v01/grouped/before_pool_factor2/spont0_simplified_IHC3000/archFrancl04"
    "saved_models/models_localize/v01/grouped/before_pool_factor2/spont0_simplified_IHC3000/archFrancl05"
    "saved_models/models_localize/v01/grouped/before_pool_factor2/spont0_simplified_IHC3000/archFrancl06"
    "saved_models/models_localize/v01/grouped/before_pool_factor2/spont0_simplified_IHC3000/archFrancl07"
    "saved_models/models_localize/v01/grouped/before_pool_factor2/spont0_simplified_IHC3000/archFrancl08"
    "saved_models/models_localize/v01/grouped/before_pool_factor2/spont0_simplified_IHC3000/archFrancl09"
    "saved_models/models_localize/v01/grouped/before_pool_factor2/spont0_simplified_IHC3000/archFrancl10"
)
model_dir="/om2/user/msaddler/tfauditoryutil/${list_model_dir[$job_idx]}"
echo $HOSTNAME $job_idx $model_dir

# Specify `DATA_TAG` based on `model_dir`
DATA_TAG="*IHC3000*"
if [[ "$model_dir" == *"IHC3000"* ]]; then
    DATA_TAG="*IHC3000*"
fi
if [[ "$model_dir" == *"IHC1000"* ]]; then
    DATA_TAG="*IHC1000*"
fi
if [[ "$model_dir" == *"IHC0320"* ]]; then
    DATA_TAG="*IHC0320*"
fi
if [[ "$model_dir" == *"IHC0050"* ]]; then
    DATA_TAG="*IHC0050*"
fi

# Activate python environment and run `phaselocknet_run.py` for each evaluation dataset
module add openmind8/anaconda
source activate tf

regex_eval="$VAST_SCRATCH_PATH/data_localize/dataset_localization/v01/eval/mit_bldg46room1004_tenoise/$DATA_TAG/*.tfrecords"
filename_eval="EVAL_v01_eval_mit_bldg46room1004_tenoise.json"
python -u phaselocknet_run.py \
-m "$model_dir" \
-c "config.json" \
-a "arch.json" \
-e "$regex_eval" \
-efn "$filename_eval" \
-ebs $eval_batch_size \
-wpo 1

for EXPT_TAG in "minimum_audible_angle_interpolated" "precedence_effect_localization" "bandwidth_dependency" "itd_threshold" "itd_ild_weighting" "new_ears" "precedence_effect_click" "spectral_smoothing" "mp_spectral_cues" "speech_in_noise_in_reverb_v04"
do
    regex_eval="$SCRATCH_PATH/data_localize/PSYCHOPHYSICSv00/$EXPT_TAG/$DATA_TAG/*.tfrecords"
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
