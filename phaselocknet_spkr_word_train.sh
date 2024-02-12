#!/bin/bash
#
#SBATCH --job-name=phaselocknet_spkr_word_train
#SBATCH --out="slurm-%A_%a.out"
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-159
#SBATCH --partition=normal --time=2-0
#SBATCH --requeue

offset=0
job_idx=$(($SLURM_ARRAY_TASK_ID + $offset))

# Specify model directory (`job_idx` is used to parallelize over `list_model_dir`)
declare -a list_model_dir=(
    "/saved_models/augmented_2022JAN/taskSW/IHC3000Hz_anf384H160M096L/arch0_0000"
    "/saved_models/augmented_2022JAN/taskSW/IHC3000Hz_anf384H160M096L/arch0_0001"
    "/saved_models/augmented_2022JAN/taskSW/IHC3000Hz_anf384H160M096L/arch0_0002"
    "/saved_models/augmented_2022JAN/taskSW/IHC3000Hz_anf384H160M096L/arch0_0004"
    "/saved_models/augmented_2022JAN/taskSW/IHC3000Hz_anf384H160M096L/arch0_0006"
    "/saved_models/augmented_2022JAN/taskSW/IHC3000Hz_anf384H160M096L/arch0_0007"
    "/saved_models/augmented_2022JAN/taskSW/IHC3000Hz_anf384H160M096L/arch0_0008"
    "/saved_models/augmented_2022JAN/taskSW/IHC3000Hz_anf384H160M096L/arch0_0009"
    "/saved_models/augmented_2022JAN/taskSW/IHC3000Hz_anf384H160M096L/arch0_0016"
    "/saved_models/augmented_2022JAN/taskSW/IHC3000Hz_anf384H160M096L/arch0_0017"
    "/saved_models/augmented_2022JAN/taskSW/IHC1000Hz_anf384H160M096L/arch0_0000"
    "/saved_models/augmented_2022JAN/taskSW/IHC1000Hz_anf384H160M096L/arch0_0001"
    "/saved_models/augmented_2022JAN/taskSW/IHC1000Hz_anf384H160M096L/arch0_0002"
    "/saved_models/augmented_2022JAN/taskSW/IHC1000Hz_anf384H160M096L/arch0_0004"
    "/saved_models/augmented_2022JAN/taskSW/IHC1000Hz_anf384H160M096L/arch0_0006"
    "/saved_models/augmented_2022JAN/taskSW/IHC1000Hz_anf384H160M096L/arch0_0007"
    "/saved_models/augmented_2022JAN/taskSW/IHC1000Hz_anf384H160M096L/arch0_0008"
    "/saved_models/augmented_2022JAN/taskSW/IHC1000Hz_anf384H160M096L/arch0_0009"
    "/saved_models/augmented_2022JAN/taskSW/IHC1000Hz_anf384H160M096L/arch0_0016"
    "/saved_models/augmented_2022JAN/taskSW/IHC1000Hz_anf384H160M096L/arch0_0017"
    "/saved_models/augmented_2022JAN/taskSW/IHC0320Hz_anf384H160M096L/arch0_0000"
    "/saved_models/augmented_2022JAN/taskSW/IHC0320Hz_anf384H160M096L/arch0_0001"
    "/saved_models/augmented_2022JAN/taskSW/IHC0320Hz_anf384H160M096L/arch0_0002"
    "/saved_models/augmented_2022JAN/taskSW/IHC0320Hz_anf384H160M096L/arch0_0004"
    "/saved_models/augmented_2022JAN/taskSW/IHC0320Hz_anf384H160M096L/arch0_0006"
    "/saved_models/augmented_2022JAN/taskSW/IHC0320Hz_anf384H160M096L/arch0_0007"
    "/saved_models/augmented_2022JAN/taskSW/IHC0320Hz_anf384H160M096L/arch0_0008"
    "/saved_models/augmented_2022JAN/taskSW/IHC0320Hz_anf384H160M096L/arch0_0009"
    "/saved_models/augmented_2022JAN/taskSW/IHC0320Hz_anf384H160M096L/arch0_0016"
    "/saved_models/augmented_2022JAN/taskSW/IHC0320Hz_anf384H160M096L/arch0_0017"
    "/saved_models/augmented_2022JAN/taskSW/IHC0050Hz_anf384H160M096L/arch0_0000"
    "/saved_models/augmented_2022JAN/taskSW/IHC0050Hz_anf384H160M096L/arch0_0001"
    "/saved_models/augmented_2022JAN/taskSW/IHC0050Hz_anf384H160M096L/arch0_0002"
    "/saved_models/augmented_2022JAN/taskSW/IHC0050Hz_anf384H160M096L/arch0_0004"
    "/saved_models/augmented_2022JAN/taskSW/IHC0050Hz_anf384H160M096L/arch0_0006"
    "/saved_models/augmented_2022JAN/taskSW/IHC0050Hz_anf384H160M096L/arch0_0007"
    "/saved_models/augmented_2022JAN/taskSW/IHC0050Hz_anf384H160M096L/arch0_0008"
    "/saved_models/augmented_2022JAN/taskSW/IHC0050Hz_anf384H160M096L/arch0_0009"
    "/saved_models/augmented_2022JAN/taskSW/IHC0050Hz_anf384H160M096L/arch0_0016"
    "/saved_models/augmented_2022JAN/taskSW/IHC0050Hz_anf384H160M096L/arch0_0017"

    "/saved_models/augmented_2022JAN/task_W/IHC3000Hz_anf384H160M096L/arch0_0000"
    "/saved_models/augmented_2022JAN/task_W/IHC3000Hz_anf384H160M096L/arch0_0001"
    "/saved_models/augmented_2022JAN/task_W/IHC3000Hz_anf384H160M096L/arch0_0002"
    "/saved_models/augmented_2022JAN/task_W/IHC3000Hz_anf384H160M096L/arch0_0004"
    "/saved_models/augmented_2022JAN/task_W/IHC3000Hz_anf384H160M096L/arch0_0006"
    "/saved_models/augmented_2022JAN/task_W/IHC3000Hz_anf384H160M096L/arch0_0007"
    "/saved_models/augmented_2022JAN/task_W/IHC3000Hz_anf384H160M096L/arch0_0008"
    "/saved_models/augmented_2022JAN/task_W/IHC3000Hz_anf384H160M096L/arch0_0009"
    "/saved_models/augmented_2022JAN/task_W/IHC3000Hz_anf384H160M096L/arch0_0016"
    "/saved_models/augmented_2022JAN/task_W/IHC3000Hz_anf384H160M096L/arch0_0017"
    "/saved_models/augmented_2022JAN/task_W/IHC1000Hz_anf384H160M096L/arch0_0000"
    "/saved_models/augmented_2022JAN/task_W/IHC1000Hz_anf384H160M096L/arch0_0001"
    "/saved_models/augmented_2022JAN/task_W/IHC1000Hz_anf384H160M096L/arch0_0002"
    "/saved_models/augmented_2022JAN/task_W/IHC1000Hz_anf384H160M096L/arch0_0004"
    "/saved_models/augmented_2022JAN/task_W/IHC1000Hz_anf384H160M096L/arch0_0006"
    "/saved_models/augmented_2022JAN/task_W/IHC1000Hz_anf384H160M096L/arch0_0007"
    "/saved_models/augmented_2022JAN/task_W/IHC1000Hz_anf384H160M096L/arch0_0008"
    "/saved_models/augmented_2022JAN/task_W/IHC1000Hz_anf384H160M096L/arch0_0009"
    "/saved_models/augmented_2022JAN/task_W/IHC1000Hz_anf384H160M096L/arch0_0016"
    "/saved_models/augmented_2022JAN/task_W/IHC1000Hz_anf384H160M096L/arch0_0017"
    "/saved_models/augmented_2022JAN/task_W/IHC0320Hz_anf384H160M096L/arch0_0000"
    "/saved_models/augmented_2022JAN/task_W/IHC0320Hz_anf384H160M096L/arch0_0001"
    "/saved_models/augmented_2022JAN/task_W/IHC0320Hz_anf384H160M096L/arch0_0002"
    "/saved_models/augmented_2022JAN/task_W/IHC0320Hz_anf384H160M096L/arch0_0004"
    "/saved_models/augmented_2022JAN/task_W/IHC0320Hz_anf384H160M096L/arch0_0006"
    "/saved_models/augmented_2022JAN/task_W/IHC0320Hz_anf384H160M096L/arch0_0007"
    "/saved_models/augmented_2022JAN/task_W/IHC0320Hz_anf384H160M096L/arch0_0008"
    "/saved_models/augmented_2022JAN/task_W/IHC0320Hz_anf384H160M096L/arch0_0009"
    "/saved_models/augmented_2022JAN/task_W/IHC0320Hz_anf384H160M096L/arch0_0016"
    "/saved_models/augmented_2022JAN/task_W/IHC0320Hz_anf384H160M096L/arch0_0017"
    "/saved_models/augmented_2022JAN/task_W/IHC0050Hz_anf384H160M096L/arch0_0000"
    "/saved_models/augmented_2022JAN/task_W/IHC0050Hz_anf384H160M096L/arch0_0001"
    "/saved_models/augmented_2022JAN/task_W/IHC0050Hz_anf384H160M096L/arch0_0002"
    "/saved_models/augmented_2022JAN/task_W/IHC0050Hz_anf384H160M096L/arch0_0004"
    "/saved_models/augmented_2022JAN/task_W/IHC0050Hz_anf384H160M096L/arch0_0006"
    "/saved_models/augmented_2022JAN/task_W/IHC0050Hz_anf384H160M096L/arch0_0007"
    "/saved_models/augmented_2022JAN/task_W/IHC0050Hz_anf384H160M096L/arch0_0008"
    "/saved_models/augmented_2022JAN/task_W/IHC0050Hz_anf384H160M096L/arch0_0009"
    "/saved_models/augmented_2022JAN/task_W/IHC0050Hz_anf384H160M096L/arch0_0016"
    "/saved_models/augmented_2022JAN/task_W/IHC0050Hz_anf384H160M096L/arch0_0017"

    "/saved_models/augmented_2022JAN/task_S/IHC3000Hz_anf384H160M096L/arch0_0000"
    "/saved_models/augmented_2022JAN/task_S/IHC3000Hz_anf384H160M096L/arch0_0001"
    "/saved_models/augmented_2022JAN/task_S/IHC3000Hz_anf384H160M096L/arch0_0002"
    "/saved_models/augmented_2022JAN/task_S/IHC3000Hz_anf384H160M096L/arch0_0004"
    "/saved_models/augmented_2022JAN/task_S/IHC3000Hz_anf384H160M096L/arch0_0006"
    "/saved_models/augmented_2022JAN/task_S/IHC3000Hz_anf384H160M096L/arch0_0007"
    "/saved_models/augmented_2022JAN/task_S/IHC3000Hz_anf384H160M096L/arch0_0008"
    "/saved_models/augmented_2022JAN/task_S/IHC3000Hz_anf384H160M096L/arch0_0009"
    "/saved_models/augmented_2022JAN/task_S/IHC3000Hz_anf384H160M096L/arch0_0016"
    "/saved_models/augmented_2022JAN/task_S/IHC3000Hz_anf384H160M096L/arch0_0017"
    "/saved_models/augmented_2022JAN/task_S/IHC1000Hz_anf384H160M096L/arch0_0000"
    "/saved_models/augmented_2022JAN/task_S/IHC1000Hz_anf384H160M096L/arch0_0001"
    "/saved_models/augmented_2022JAN/task_S/IHC1000Hz_anf384H160M096L/arch0_0002"
    "/saved_models/augmented_2022JAN/task_S/IHC1000Hz_anf384H160M096L/arch0_0004"
    "/saved_models/augmented_2022JAN/task_S/IHC1000Hz_anf384H160M096L/arch0_0006"
    "/saved_models/augmented_2022JAN/task_S/IHC1000Hz_anf384H160M096L/arch0_0007"
    "/saved_models/augmented_2022JAN/task_S/IHC1000Hz_anf384H160M096L/arch0_0008"
    "/saved_models/augmented_2022JAN/task_S/IHC1000Hz_anf384H160M096L/arch0_0009"
    "/saved_models/augmented_2022JAN/task_S/IHC1000Hz_anf384H160M096L/arch0_0016"
    "/saved_models/augmented_2022JAN/task_S/IHC1000Hz_anf384H160M096L/arch0_0017"
    "/saved_models/augmented_2022JAN/task_S/IHC0320Hz_anf384H160M096L/arch0_0000"
    "/saved_models/augmented_2022JAN/task_S/IHC0320Hz_anf384H160M096L/arch0_0001"
    "/saved_models/augmented_2022JAN/task_S/IHC0320Hz_anf384H160M096L/arch0_0002"
    "/saved_models/augmented_2022JAN/task_S/IHC0320Hz_anf384H160M096L/arch0_0004"
    "/saved_models/augmented_2022JAN/task_S/IHC0320Hz_anf384H160M096L/arch0_0006"
    "/saved_models/augmented_2022JAN/task_S/IHC0320Hz_anf384H160M096L/arch0_0007"
    "/saved_models/augmented_2022JAN/task_S/IHC0320Hz_anf384H160M096L/arch0_0008"
    "/saved_models/augmented_2022JAN/task_S/IHC0320Hz_anf384H160M096L/arch0_0009"
    "/saved_models/augmented_2022JAN/task_S/IHC0320Hz_anf384H160M096L/arch0_0016"
    "/saved_models/augmented_2022JAN/task_S/IHC0320Hz_anf384H160M096L/arch0_0017"
    "/saved_models/augmented_2022JAN/task_S/IHC0050Hz_anf384H160M096L/arch0_0000"
    "/saved_models/augmented_2022JAN/task_S/IHC0050Hz_anf384H160M096L/arch0_0001"
    "/saved_models/augmented_2022JAN/task_S/IHC0050Hz_anf384H160M096L/arch0_0002"
    "/saved_models/augmented_2022JAN/task_S/IHC0050Hz_anf384H160M096L/arch0_0004"
    "/saved_models/augmented_2022JAN/task_S/IHC0050Hz_anf384H160M096L/arch0_0006"
    "/saved_models/augmented_2022JAN/task_S/IHC0050Hz_anf384H160M096L/arch0_0007"
    "/saved_models/augmented_2022JAN/task_S/IHC0050Hz_anf384H160M096L/arch0_0008"
    "/saved_models/augmented_2022JAN/task_S/IHC0050Hz_anf384H160M096L/arch0_0009"
    "/saved_models/augmented_2022JAN/task_S/IHC0050Hz_anf384H160M096L/arch0_0016"
    "/saved_models/augmented_2022JAN/task_S/IHC0050Hz_anf384H160M096L/arch0_0017"

    "saved_models/augmented_2022JAN/taskSW/spont0_simplified_IHC3000/arch0_0000"
    "saved_models/augmented_2022JAN/taskSW/spont0_simplified_IHC3000/arch0_0001"
    "saved_models/augmented_2022JAN/taskSW/spont0_simplified_IHC3000/arch0_0002"
    "saved_models/augmented_2022JAN/taskSW/spont0_simplified_IHC3000/arch0_0004"
    "saved_models/augmented_2022JAN/taskSW/spont0_simplified_IHC3000/arch0_0006"
    "saved_models/augmented_2022JAN/taskSW/spont0_simplified_IHC3000/arch0_0007"
    "saved_models/augmented_2022JAN/taskSW/spont0_simplified_IHC3000/arch0_0008"
    "saved_models/augmented_2022JAN/taskSW/spont0_simplified_IHC3000/arch0_0009"
    "saved_models/augmented_2022JAN/taskSW/spont0_simplified_IHC3000/arch0_0016"
    "saved_models/augmented_2022JAN/taskSW/spont0_simplified_IHC3000/arch0_0017"
    "saved_models/augmented_2022JAN/taskSW/spont0_simplified_IHC1000/arch0_0000"
    "saved_models/augmented_2022JAN/taskSW/spont0_simplified_IHC1000/arch0_0001"
    "saved_models/augmented_2022JAN/taskSW/spont0_simplified_IHC1000/arch0_0002"
    "saved_models/augmented_2022JAN/taskSW/spont0_simplified_IHC1000/arch0_0004"
    "saved_models/augmented_2022JAN/taskSW/spont0_simplified_IHC1000/arch0_0006"
    "saved_models/augmented_2022JAN/taskSW/spont0_simplified_IHC1000/arch0_0007"
    "saved_models/augmented_2022JAN/taskSW/spont0_simplified_IHC1000/arch0_0008"
    "saved_models/augmented_2022JAN/taskSW/spont0_simplified_IHC1000/arch0_0009"
    "saved_models/augmented_2022JAN/taskSW/spont0_simplified_IHC1000/arch0_0016"
    "saved_models/augmented_2022JAN/taskSW/spont0_simplified_IHC1000/arch0_0017"
    "saved_models/augmented_2022JAN/taskSW/spont0_simplified_IHC0320/arch0_0000"
    "saved_models/augmented_2022JAN/taskSW/spont0_simplified_IHC0320/arch0_0001"
    "saved_models/augmented_2022JAN/taskSW/spont0_simplified_IHC0320/arch0_0002"
    "saved_models/augmented_2022JAN/taskSW/spont0_simplified_IHC0320/arch0_0004"
    "saved_models/augmented_2022JAN/taskSW/spont0_simplified_IHC0320/arch0_0006"
    "saved_models/augmented_2022JAN/taskSW/spont0_simplified_IHC0320/arch0_0007"
    "saved_models/augmented_2022JAN/taskSW/spont0_simplified_IHC0320/arch0_0008"
    "saved_models/augmented_2022JAN/taskSW/spont0_simplified_IHC0320/arch0_0009"
    "saved_models/augmented_2022JAN/taskSW/spont0_simplified_IHC0320/arch0_0016"
    "saved_models/augmented_2022JAN/taskSW/spont0_simplified_IHC0320/arch0_0017"
    "saved_models/augmented_2022JAN/taskSW/spont0_simplified_IHC0050/arch0_0000"
    "saved_models/augmented_2022JAN/taskSW/spont0_simplified_IHC0050/arch0_0001"
    "saved_models/augmented_2022JAN/taskSW/spont0_simplified_IHC0050/arch0_0002"
    "saved_models/augmented_2022JAN/taskSW/spont0_simplified_IHC0050/arch0_0004"
    "saved_models/augmented_2022JAN/taskSW/spont0_simplified_IHC0050/arch0_0006"
    "saved_models/augmented_2022JAN/taskSW/spont0_simplified_IHC0050/arch0_0007"
    "saved_models/augmented_2022JAN/taskSW/spont0_simplified_IHC0050/arch0_0008"
    "saved_models/augmented_2022JAN/taskSW/spont0_simplified_IHC0050/arch0_0009"
    "saved_models/augmented_2022JAN/taskSW/spont0_simplified_IHC0050/arch0_0016"
    "saved_models/augmented_2022JAN/taskSW/spont0_simplified_IHC0050/arch0_0017"
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
# Specify training and validation datasets using `DATA_TAG`
regex_train="$VAST_SCRATCH_PATH/data_WSN/JSIN_all_v3_augmented_2022JAN/train/$DATA_TAG/*tfrecords"
regex_valid="$VAST_SCRATCH_PATH/data_WSN/JSIN_all_v3_augmented_2022JAN/valid/$DATA_TAG/*tfrecords"

# Activate python environment and run `phaselocknet_run.py`
module add openmind8/anaconda/3-2022.10
source activate tf
python -u phaselocknet_run.py \
-m "$model_dir" \
-c "config.json" \
-a "arch.json" \
-t "$regex_train" \
-v "$regex_valid" \
-mp 1 \
2>&1 | tee "$model_dir/log_optimize.out"
