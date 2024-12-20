#!/bin/bash
#
#SBATCH --job-name=phaselocknet_spkr_word_eval
#SBATCH --out="slurm-%A_%a.out"
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
##SBATCH --gres=gpu:a100:1
#SBATCH --gres=gpu:1 --exclude=node[017-094,097,098],dgx001,dgx002
#SBATCH --array=0-189
#SBATCH --time=2-0
#SBATCH --partition=use-everything
#SBATCH --requeue

offset=0
job_idx=$(($SLURM_ARRAY_TASK_ID + $offset))
eval_batch_size=32

# Specify model directory (`job_idx` is used to parallelize over `list_model_dir`)
declare -a list_model_dir=(
    "models/spkr_word_recognition/sr20000_IHC3000/arch0_0000"
    "models/spkr_word_recognition/sr20000_IHC3000/arch0_0001"
    "models/spkr_word_recognition/sr20000_IHC3000/arch0_0002"
    "models/spkr_word_recognition/sr20000_IHC3000/arch0_0004"
    "models/spkr_word_recognition/sr20000_IHC3000/arch0_0006"
    "models/spkr_word_recognition/sr20000_IHC3000/arch0_0007"
    "models/spkr_word_recognition/sr20000_IHC3000/arch0_0008"
    "models/spkr_word_recognition/sr20000_IHC3000/arch0_0009"
    "models/spkr_word_recognition/sr20000_IHC3000/arch0_0016"
    "models/spkr_word_recognition/sr20000_IHC3000/arch0_0017"
    "models/spkr_recognition/sr20000_IHC3000/arch0_0000"
    "models/spkr_recognition/sr20000_IHC3000/arch0_0001"
    "models/spkr_recognition/sr20000_IHC3000/arch0_0002"
    "models/spkr_recognition/sr20000_IHC3000/arch0_0004"
    "models/spkr_recognition/sr20000_IHC3000/arch0_0006"
    "models/spkr_recognition/sr20000_IHC3000/arch0_0007"
    "models/spkr_recognition/sr20000_IHC3000/arch0_0008"
    "models/spkr_recognition/sr20000_IHC3000/arch0_0009"
    "models/spkr_recognition/sr20000_IHC3000/arch0_0016"
    "models/spkr_recognition/sr20000_IHC3000/arch0_0017"
    "models/word_recognition/sr20000_IHC3000/arch0_0000"
    "models/word_recognition/sr20000_IHC3000/arch0_0001"
    "models/word_recognition/sr20000_IHC3000/arch0_0002"
    "models/word_recognition/sr20000_IHC3000/arch0_0004"
    "models/word_recognition/sr20000_IHC3000/arch0_0006"
    "models/word_recognition/sr20000_IHC3000/arch0_0007"
    "models/word_recognition/sr20000_IHC3000/arch0_0008"
    "models/word_recognition/sr20000_IHC3000/arch0_0009"
    "models/word_recognition/sr20000_IHC3000/arch0_0016"
    "models/word_recognition/sr20000_IHC3000/arch0_0017"

    "models/spkr_recognition/IHC0050/arch0_0000"
    "models/spkr_recognition/IHC0050/arch0_0001"
    "models/spkr_recognition/IHC0050/arch0_0002"
    "models/spkr_recognition/IHC0050/arch0_0004"
    "models/spkr_recognition/IHC0050/arch0_0006"
    "models/spkr_recognition/IHC0050/arch0_0007"
    "models/spkr_recognition/IHC0050/arch0_0008"
    "models/spkr_recognition/IHC0050/arch0_0009"
    "models/spkr_recognition/IHC0050/arch0_0016"
    "models/spkr_recognition/IHC0050/arch0_0017"
    "models/spkr_recognition/IHC0320/arch0_0000"
    "models/spkr_recognition/IHC0320/arch0_0001"
    "models/spkr_recognition/IHC0320/arch0_0002"
    "models/spkr_recognition/IHC0320/arch0_0004"
    "models/spkr_recognition/IHC0320/arch0_0006"
    "models/spkr_recognition/IHC0320/arch0_0007"
    "models/spkr_recognition/IHC0320/arch0_0008"
    "models/spkr_recognition/IHC0320/arch0_0009"
    "models/spkr_recognition/IHC0320/arch0_0016"
    "models/spkr_recognition/IHC0320/arch0_0017"
    "models/spkr_recognition/IHC1000/arch0_0000"
    "models/spkr_recognition/IHC1000/arch0_0001"
    "models/spkr_recognition/IHC1000/arch0_0002"
    "models/spkr_recognition/IHC1000/arch0_0004"
    "models/spkr_recognition/IHC1000/arch0_0006"
    "models/spkr_recognition/IHC1000/arch0_0007"
    "models/spkr_recognition/IHC1000/arch0_0008"
    "models/spkr_recognition/IHC1000/arch0_0009"
    "models/spkr_recognition/IHC1000/arch0_0016"
    "models/spkr_recognition/IHC1000/arch0_0017"
    "models/spkr_recognition/IHC3000/arch0_0000"
    "models/spkr_recognition/IHC3000/arch0_0001"
    "models/spkr_recognition/IHC3000/arch0_0002"
    "models/spkr_recognition/IHC3000/arch0_0004"
    "models/spkr_recognition/IHC3000/arch0_0006"
    "models/spkr_recognition/IHC3000/arch0_0007"
    "models/spkr_recognition/IHC3000/arch0_0008"
    "models/spkr_recognition/IHC3000/arch0_0009"
    "models/spkr_recognition/IHC3000/arch0_0016"
    "models/spkr_recognition/IHC3000/arch0_0017"
    "models/spkr_word_recognition/IHC0050/arch0_0000"
    "models/spkr_word_recognition/IHC0050/arch0_0001"
    "models/spkr_word_recognition/IHC0050/arch0_0002"
    "models/spkr_word_recognition/IHC0050/arch0_0004"
    "models/spkr_word_recognition/IHC0050/arch0_0006"
    "models/spkr_word_recognition/IHC0050/arch0_0007"
    "models/spkr_word_recognition/IHC0050/arch0_0008"
    "models/spkr_word_recognition/IHC0050/arch0_0009"
    "models/spkr_word_recognition/IHC0050/arch0_0016"
    "models/spkr_word_recognition/IHC0050/arch0_0017"
    "models/spkr_word_recognition/IHC0320/arch0_0000"
    "models/spkr_word_recognition/IHC0320/arch0_0001"
    "models/spkr_word_recognition/IHC0320/arch0_0002"
    "models/spkr_word_recognition/IHC0320/arch0_0004"
    "models/spkr_word_recognition/IHC0320/arch0_0006"
    "models/spkr_word_recognition/IHC0320/arch0_0007"
    "models/spkr_word_recognition/IHC0320/arch0_0008"
    "models/spkr_word_recognition/IHC0320/arch0_0009"
    "models/spkr_word_recognition/IHC0320/arch0_0016"
    "models/spkr_word_recognition/IHC0320/arch0_0017"
    "models/spkr_word_recognition/IHC1000/arch0_0000"
    "models/spkr_word_recognition/IHC1000/arch0_0001"
    "models/spkr_word_recognition/IHC1000/arch0_0002"
    "models/spkr_word_recognition/IHC1000/arch0_0004"
    "models/spkr_word_recognition/IHC1000/arch0_0006"
    "models/spkr_word_recognition/IHC1000/arch0_0007"
    "models/spkr_word_recognition/IHC1000/arch0_0008"
    "models/spkr_word_recognition/IHC1000/arch0_0009"
    "models/spkr_word_recognition/IHC1000/arch0_0016"
    "models/spkr_word_recognition/IHC1000/arch0_0017"
    "models/spkr_word_recognition/IHC3000/arch0_0000"
    "models/spkr_word_recognition/IHC3000/arch0_0001"
    "models/spkr_word_recognition/IHC3000/arch0_0002"
    "models/spkr_word_recognition/IHC3000/arch0_0004"
    "models/spkr_word_recognition/IHC3000/arch0_0006"
    "models/spkr_word_recognition/IHC3000/arch0_0007"
    "models/spkr_word_recognition/IHC3000/arch0_0008"
    "models/spkr_word_recognition/IHC3000/arch0_0009"
    "models/spkr_word_recognition/IHC3000/arch0_0016"
    "models/spkr_word_recognition/IHC3000/arch0_0017"
    "models/spkr_word_recognition/simplified_IHC0050/arch0_0000"
    "models/spkr_word_recognition/simplified_IHC0050/arch0_0001"
    "models/spkr_word_recognition/simplified_IHC0050/arch0_0002"
    "models/spkr_word_recognition/simplified_IHC0050/arch0_0004"
    "models/spkr_word_recognition/simplified_IHC0050/arch0_0006"
    "models/spkr_word_recognition/simplified_IHC0050/arch0_0007"
    "models/spkr_word_recognition/simplified_IHC0050/arch0_0008"
    "models/spkr_word_recognition/simplified_IHC0050/arch0_0009"
    "models/spkr_word_recognition/simplified_IHC0050/arch0_0016"
    "models/spkr_word_recognition/simplified_IHC0050/arch0_0017"
    "models/spkr_word_recognition/simplified_IHC0320/arch0_0000"
    "models/spkr_word_recognition/simplified_IHC0320/arch0_0001"
    "models/spkr_word_recognition/simplified_IHC0320/arch0_0002"
    "models/spkr_word_recognition/simplified_IHC0320/arch0_0004"
    "models/spkr_word_recognition/simplified_IHC0320/arch0_0006"
    "models/spkr_word_recognition/simplified_IHC0320/arch0_0007"
    "models/spkr_word_recognition/simplified_IHC0320/arch0_0008"
    "models/spkr_word_recognition/simplified_IHC0320/arch0_0009"
    "models/spkr_word_recognition/simplified_IHC0320/arch0_0016"
    "models/spkr_word_recognition/simplified_IHC0320/arch0_0017"
    "models/spkr_word_recognition/simplified_IHC1000/arch0_0000"
    "models/spkr_word_recognition/simplified_IHC1000/arch0_0001"
    "models/spkr_word_recognition/simplified_IHC1000/arch0_0002"
    "models/spkr_word_recognition/simplified_IHC1000/arch0_0004"
    "models/spkr_word_recognition/simplified_IHC1000/arch0_0006"
    "models/spkr_word_recognition/simplified_IHC1000/arch0_0007"
    "models/spkr_word_recognition/simplified_IHC1000/arch0_0008"
    "models/spkr_word_recognition/simplified_IHC1000/arch0_0009"
    "models/spkr_word_recognition/simplified_IHC1000/arch0_0016"
    "models/spkr_word_recognition/simplified_IHC1000/arch0_0017"
    "models/spkr_word_recognition/simplified_IHC3000/arch0_0000"
    "models/spkr_word_recognition/simplified_IHC3000/arch0_0001"
    "models/spkr_word_recognition/simplified_IHC3000/arch0_0002"
    "models/spkr_word_recognition/simplified_IHC3000/arch0_0004"
    "models/spkr_word_recognition/simplified_IHC3000/arch0_0006"
    "models/spkr_word_recognition/simplified_IHC3000/arch0_0007"
    "models/spkr_word_recognition/simplified_IHC3000/arch0_0008"
    "models/spkr_word_recognition/simplified_IHC3000/arch0_0009"
    "models/spkr_word_recognition/simplified_IHC3000/arch0_0016"
    "models/spkr_word_recognition/simplified_IHC3000/arch0_0017"
    "models/word_recognition/IHC0050/arch0_0000"
    "models/word_recognition/IHC0050/arch0_0001"
    "models/word_recognition/IHC0050/arch0_0002"
    "models/word_recognition/IHC0050/arch0_0004"
    "models/word_recognition/IHC0050/arch0_0006"
    "models/word_recognition/IHC0050/arch0_0007"
    "models/word_recognition/IHC0050/arch0_0008"
    "models/word_recognition/IHC0050/arch0_0009"
    "models/word_recognition/IHC0050/arch0_0016"
    "models/word_recognition/IHC0050/arch0_0017"
    "models/word_recognition/IHC0320/arch0_0000"
    "models/word_recognition/IHC0320/arch0_0001"
    "models/word_recognition/IHC0320/arch0_0002"
    "models/word_recognition/IHC0320/arch0_0004"
    "models/word_recognition/IHC0320/arch0_0006"
    "models/word_recognition/IHC0320/arch0_0007"
    "models/word_recognition/IHC0320/arch0_0008"
    "models/word_recognition/IHC0320/arch0_0009"
    "models/word_recognition/IHC0320/arch0_0016"
    "models/word_recognition/IHC0320/arch0_0017"
    "models/word_recognition/IHC1000/arch0_0000"
    "models/word_recognition/IHC1000/arch0_0001"
    "models/word_recognition/IHC1000/arch0_0002"
    "models/word_recognition/IHC1000/arch0_0004"
    "models/word_recognition/IHC1000/arch0_0006"
    "models/word_recognition/IHC1000/arch0_0007"
    "models/word_recognition/IHC1000/arch0_0008"
    "models/word_recognition/IHC1000/arch0_0009"
    "models/word_recognition/IHC1000/arch0_0016"
    "models/word_recognition/IHC1000/arch0_0017"
    "models/word_recognition/IHC3000/arch0_0000"
    "models/word_recognition/IHC3000/arch0_0001"
    "models/word_recognition/IHC3000/arch0_0002"
    "models/word_recognition/IHC3000/arch0_0004"
    "models/word_recognition/IHC3000/arch0_0006"
    "models/word_recognition/IHC3000/arch0_0007"
    "models/word_recognition/IHC3000/arch0_0008"
    "models/word_recognition/IHC3000/arch0_0009"
    "models/word_recognition/IHC3000/arch0_0016"
    "models/word_recognition/IHC3000/arch0_0017"
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

# Activate python environment and run `phaselocknet_run.py` for each evaluation dataset
module add openmind8/anaconda
source activate tf

for EXPT_TAG in "human_experiment_v00_foreground60dbspl" "human_experiment_v00_inharmonic_foreground60dbspl" "speech_in_synthetic_textures"
do
    regex_eval="$VAST_SCRATCH_PATH/stimuli/spkr_word_recognition/evaluation/$EXPT_TAG/$DATA_TAG/*.tfrecords"
    filename_eval="EVAL_word_recognition_$EXPT_TAG.json"
    python -u phaselocknet_run.py \
    -m "$model_dir" \
    -c "config.json" \
    -a "arch.json" \
    -e "$regex_eval" \
    -efn "$filename_eval" \
    -ebs $eval_batch_size \
    -wpo 0
done

for EXPT_TAG in "pitch_altered_v00" "hopkins_moore_2009"
do
    regex_eval="$VAST_SCRATCH_PATH/stimuli/spkr_word_recognition/evaluation/$EXPT_TAG/$DATA_TAG/*.tfrecords"
    filename_eval="EVAL_$EXPT_TAG.json"
    python -u phaselocknet_run.py \
    -m "$model_dir" \
    -c "config.json" \
    -a "arch.json" \
    -e "$regex_eval" \
    -efn "$filename_eval" \
    -ebs $eval_batch_size \
    -wpo 0
done
