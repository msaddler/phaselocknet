## phaselocknet

Code to accompany the manuscript "Models optimized for real-world tasks reveal the necessity of precise temporal coding in hearing" by Mark R. Saddler and Josh H. McDermott (2024).

## Dependencies

This is a repository of Python (3.11.4) code. A complete list of Python dependencies is contained in [`requirements.txt`](requirements.txt). The models were developed in `tensorflow-2.13.0` on machines running CentOS Linux 7.

## Sound Datasets and Model Weights

Sound datasets, trained model weights, and raw model evaluation outputs for each model configuration are too large to include here, but can be downloaded from our [Google Drive](https://drive.google.com/drive/folders/1YgC7x6Ot84XZInlSyHK-9NQ0jhhGUS2z?usp=share_link). The file structure of the Google Drive (which mirrors this repository) should be preserved for code to run without altering file paths.

To run models with the detailed auditory nerve input representation, simulated auditory nerve representations need to be pre-computed. Code for this is provided; however, it is computationally expensive and requires considerable disk space. For each auditory nerve model configuration, the sound localization training dataset requires 12 TB and the word/voice recognition training dataset requires 26 TB. For most purposes, we recommend using the "simplified" models, which include a simpler auditory nerve model as part of the `tensorflow` computation graph such that they can operate directly on sound waveforms.

## Contents

The [`model_DEMO.ipynb`](model_DEMO.ipynb) Jupyter Notebook provides minimal examples of how to use our trained models.

Code for training and evaluating the models at scale is located in [`phaselocknet_run.py`](phaselocknet_run.py), which is called by the example SLURM scripts provided:
- [`phaselocknet_localization_train.sh`](phaselocknet_localization_train.sh)
- [`phaselocknet_localization_eval.sh`](phaselocknet_localization_eval.sh)
- [`phaselocknet_spkr_word_train.sh`](phaselocknet_spkr_word_train.sh)
- [`phaselocknet_spkr_word_eval.sh`](phaselocknet_spkr_word_eval.sh))

The [`model_run_all_experiments.ipynb`](model_run_all_experiments.ipynb) Jupyter Notebook can be used to run all analyses in the paper, provided all model evaluation output files have been downloaded from the [Google Drive](https://drive.google.com/drive/folders/1YgC7x6Ot84XZInlSyHK-9NQ0jhhGUS2z?usp=share_link). This notebook generates the results files in the [`data/model`](data/model) folder. These results files contain human and model behavioral data as well as human-model comparison metrics.

The included results files are sufficient to generate all figures in the paper using the following Jupyter Notebooks:
- Aggregate measures of human-model similarity: [`figures_aggregate_measures.ipynb`](figures_aggregate_measures.ipynb)
- Auditory nerve model phase locking manipulation: [`figures_phase_locking_manipulation.ipynb`](figures_phase_locking_manipulation.ipynb)
- Sound localization experiments: [`figures_localization_results.ipynb`](figures_localization_results.ipynb)
- Sound localization schematics: [`figures_localization_schematics.ipynb`](figures_localization_schematics.ipynb)
- Word and voice recognition experiments: [`figures_spkr_word_results.ipynb`](figures_spkr_word_results.ipynb)
- Supplementary table of model architectures: [`model_architecture_table.ipynb`](model_architecture_table.ipynb)

Code to train deep neural network models on a pure tone frequency discrimination task and compare their performance to prior ideal observer models is provided in a stand-alone repository, which is linked here as a submodule: [`freqdiscriminet`](freqdiscriminet).

## File structure
```
## Jupyter notebooks (demo, analyses, and figures)
|__ model_DEMO.ipynb
|__ model_architecture_table.ipynb
|__ model_run_all_experiments.ipynb
|__ figures_aggregate_measures.ipynb
|__ figures_localization_results.ipynb
|__ figures_localization_schematics.ipynb
|__ figures_phase_locking_manipulation.ipynb
|__ figures_spkr_word_results.ipynb
|__ configure_bez2018model_nervegram_generation.ipynb

## Scripts for running model optimization and evaluation
|__ phaselocknet_run.py
|__ phaselocknet_localization_eval.sh
|__ phaselocknet_localization_train.sh
|__ phaselocknet_spkr_word_eval.sh
|__ phaselocknet_spkr_word_train.sh

## Utilities
|__ util_bez2018model_job.sh
|__ util_bez2018model.py
|__ util_cochlea.py
|__ util_evaluate.py
|__ util_figures_cnn.py
|__ util_figures.py
|__ util_localization_psychophysics_figures.py
|__ util_localization_psychophysics.py
|__ util_network.py
|__ util_optimize.py
|__ util.py
|__ util_signal.py
|__ util_spkr_word_psychophysics_figures.py
|__ util_spkr_word_psychophysics.py
|__ util_stimuli.py
|__ util_tfrecords.py
|__ dataset_to_tfrecords_job.sh
|__ dataset_to_tfrecords.py

## Processed human and model behavioral results
|__ data
    |__ human
        |__ sound_localization
        |__ spkr_word_recognition
    |__ model
        |__ sound_localization
        |__ spkr_word_recognition
    |__ misc

## Model config files, checkpoints (trained weights), and evaluation output files
|__ models
    |__ sound_localization
        |__ simplified_IHC3000
            |__arch01
            ...
        ...
    |__ spkr_recognition
        ...
    |__ spkr_word_recognition
        ...
    |__ word_recognition

## Sound datasets and simulated auditory nerve representations
|__ stimuli
    |__ sound_localization
        |__ evaluation
            |__ itd_ild_weighting (example experiment)
                |__ stim.hdf5 (sound dataset for example experiment)
                |__ tfrecords_IHC3000 (3000 Hz phase locking condition)
                    |__ simulated auditory nerve representations as tfrecord files
                ...
            ...
        |__ optimization
            |__ train
            |__ valid
    |__ spkr_word_recognition
        |__ evaluation
        |__ optimization
            |__ train
            |__ valid

## External repositories
|__ bez2018model (https://github.com/msaddler/bez2018model)
|__ virtual_acoustic_room (https://github.com/msaddler/virtual_acoustic_room)
|__ freqdiscriminet (stand-alone respository for pure tone frequency discrimination models)
```

All model results in the paper are presented as the average across 10 different model architectures. For each model variant, the individual architectures have their own directories (e.g., [`models/sound_localization/simplified_IHC3000/arch01`](models/sound_localization/simplified_IHC3000/arch01)). A single directory contains:
- [`config.json`](models/sound_localization/simplified_IHC3000/arch01/config.json): cochlear model parameters and optimization hyperparameters
- [`arch.json`](models/sound_localization/simplified_IHC3000/arch01/arch.json): artificial neural network architecture 
- [`ckpt_BEST`](https://drive.google.com/drive/folders/1EylraQkpaLelBUIImv3k0nqGsJxBIAiT?usp=share_link): optimized neural network weights (Google Drive)
- [`log_optimize.csv`](https://drive.google.com/drive/folders/1EylraQkpaLelBUIImv3k0nqGsJxBIAiT?usp=share_link): model optimization log (Google Drive)
- [`EVAL_ * .json`](https://drive.google.com/drive/folders/1EylraQkpaLelBUIImv3k0nqGsJxBIAiT?usp=share_link): model evaluation output files (Google Drive)

## Contact
Mark R. Saddler (msaddler@mit.edu)
