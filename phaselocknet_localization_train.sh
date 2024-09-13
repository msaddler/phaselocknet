#!/bin/bash
#
#SBATCH --job-name=phaselocknet_localization_train
#SBATCH --out="slurm-%A_%a.out"
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
##SBATCH --gres=gpu:a100:1
#SBATCH --gres=gpu:1 --exclude=node[017-094,097,098],dgx001,dgx002
#SBATCH --array=0-19
#SBATCH --partition=normal --time=2-0
#SBATCH --requeue

offset=0
job_idx=$(($SLURM_ARRAY_TASK_ID + $offset))

# Specify model directory (`job_idx` is used to parallelize over `list_model_dir`)
declare -a list_model_dir=(
    "models/sound_localization/cochlearn/arch01"
    "models/sound_localization/cochlearn/arch02"
    "models/sound_localization/cochlearn/arch03"
    "models/sound_localization/cochlearn/arch04"
    "models/sound_localization/cochlearn/arch05"
    "models/sound_localization/cochlearn/arch06"
    "models/sound_localization/cochlearn/arch07"
    "models/sound_localization/cochlearn/arch08"
    "models/sound_localization/cochlearn/arch09"
    "models/sound_localization/cochlearn/arch10"
    "models/sound_localization/cochlearn_delayed_integration/arch01"
    "models/sound_localization/cochlearn_delayed_integration/arch02"
    "models/sound_localization/cochlearn_delayed_integration/arch03"
    "models/sound_localization/cochlearn_delayed_integration/arch04"
    "models/sound_localization/cochlearn_delayed_integration/arch05"
    "models/sound_localization/cochlearn_delayed_integration/arch06"
    "models/sound_localization/cochlearn_delayed_integration/arch07"
    "models/sound_localization/cochlearn_delayed_integration/arch08"
    "models/sound_localization/cochlearn_delayed_integration/arch09"
    "models/sound_localization/cochlearn_delayed_integration/arch10"

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
DATA_TAG="tfrecords_simplified"
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
if [[ "$model_dir" == *"simplified"* ]]; then
    DATA_TAG="tfrecords_simplified"
fi

# Specify training and validation datasets using `DATA_TAG`
regex_train="$VAST_SCRATCH_PATH/stimuli/sound_localization/optimization/train/$DATA_TAG/*tfrecords"
regex_valid="$VAST_SCRATCH_PATH/stimuli/sound_localization/optimization/valid/$DATA_TAG/*tfrecords"

# Activate python environment and run `phaselocknet_run.py`
module add openmind8/anaconda
# source activate tf
export LD_LIBRARY_PATH='/om2/user/msaddler/.conda/envs/ntf/lib/:/om2/user/msaddler/.conda/envs/ntf/lib/python3.11/site-packages/nvidia/cudnn/lib'
source activate ntf

python -u phaselocknet_run.py \
-m "$model_dir" \
-c "config.json" \
-a "arch.json" \
-t "$regex_train" \
-v "$regex_valid" \
-mp 1 \
2>&1 | tee "$model_dir/log_optimize.out"
