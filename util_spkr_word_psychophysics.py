import os
import sys
import pdb
import glob
import json
import pickle
import itertools
import functools
import multiprocessing
import scipy.stats
import numpy as np
import pandas as pd

import util_misc


def func_to_parallelize_spkr_word(
        dir_model,
        dict_basename_eval={},
        key_true_spkr='label_speaker_int:labels_true',
        key_pred_spkr='label_speaker_int:labels_pred',
        key_true_word='label_word_int:labels_true',
        key_pred_word='label_word_int:labels_pred',
        dict_load_arr={}):
    """
    """
    list_df = []
    for tag_expt, basename_eval in dict_basename_eval.items():
        fn_eval = os.path.join(dir_model, basename_eval)
        if not os.path.exists(fn_eval):
            print(f'MISSING: {fn_eval}')
        else:
            with open(fn_eval, 'r') as f_eval:
                eval_dict = json.load(f_eval)
            for k in list(eval_dict.keys()):
                if isinstance(eval_dict[k], str):
                    fn_k = eval_dict.pop(k)
                    if dict_load_arr.get(tag_expt, False):
                        if os.path.basename(fn_k) == fn_k:
                            fn_k = os.path.join(dir_model, fn_k)
                        if os.path.exists(fn_k):
                            arr = np.load(fn_k)
                            eval_dict[k] = [arr[_] for _ in range(arr.shape[0])]
                else:
                    eval_dict[k] = np.array(eval_dict[k])
            df = pd.DataFrame(eval_dict)
            df['tag_expt'] = tag_expt
            df['fn_eval'] = fn_eval
            list_df.append(df)
    df = pd.concat(list_df)
    if (key_pred_spkr in df.columns) and (key_true_spkr in df.columns):
        df['correct_spkr'] = df[key_pred_spkr] == df[key_true_spkr]
    else:
        df['correct_spkr'] = False
    if (key_pred_word in df.columns) and (key_true_word in df.columns):
        df['correct_word'] = df[key_pred_word] == df[key_true_word]
    else:
        df['correct_word'] = False
    return df


def compute_srt(df_results):
    """
    Helper function to compute speech reception threshold for Hopkins & Moore
    (2009, JASA) TFS manipulation experiment
    """
    def func(dfi):
        x = np.array(dfi.snr)
        y = np.array(dfi.correct_word)
        fcn_to_fit = lambda x, p0, p1, p2: p0 * scipy.stats.norm(p1, p2).cdf(x)
        IDX = np.isfinite(x)
        popt, pcov = scipy.optimize.curve_fit(fcn_to_fit, x[IDX], y[IDX])
        return popt
    df_results['popt'] = df_results.apply(func, axis=1)
    df_results['srt'] = df_results['popt'].map(lambda _: _[1])
    return df_results


def run_spkr_word_experiments(
        list_regex_dir_model,
        dir_human_data='data/human/spkr_word_recognition',
        dict_basename_eval={},
        workers=20,
        **kwargs):
    """
    """
    EXPERIMENT_DATAFRAMES = {}
    df = []
    for regex_dir_model in list_regex_dir_model:
        list_dir_model = glob.glob(regex_dir_model)
        with multiprocessing.Pool(min(workers, len(list_dir_model))) as p:
            func = functools.partial(
                func_to_parallelize_spkr_word,
                dict_basename_eval=dict_basename_eval,
                **kwargs)
            list_df = p.map(func, list_dir_model)
        df_tmp = pd.concat(list_df)
        df_tmp['tag_model'] = regex_dir_model
        df.append(df_tmp)
        print(f'Loaded data for {len(list_dir_model)} `{regex_dir_model}` models')
    df = pd.concat(df).reset_index(drop=True)

    tag_experiment = 'kell_like_inharmonic'
    if tag_experiment in dict_basename_eval:
        print(f"[run_spkr_word_experiments] `{tag_experiment}` experiment")
        df_results = df[df.tag_expt == tag_experiment]
        list_key_metric = [
            'correct_spkr',
            'correct_word',
        ]
        df_results = util_misc.flatten_columns(df_results.groupby([
            'tag_expt',
            'tag_model',
            'fn_eval',
            'background_condition',
            'snr',
        ]).agg({
            key_metric: 'mean' for key_metric in list_key_metric
        }).reset_index().groupby([
            'tag_expt',
            'tag_model',
            'background_condition',
            'snr',
        ]).agg({
            key_metric: [list, 'mean', 'sem'] for key_metric in list_key_metric
        }).reset_index(), sep='_')
        df_results = df_results.sort_index(axis=1)
        EXPERIMENT_DATAFRAMES[tag_experiment] = df_results

    tag_experiment = 'kell_like'
    if tag_experiment in dict_basename_eval:
        print(f"[run_spkr_word_experiments] `{tag_experiment}` experiment")
        df_results = df[df.tag_expt == tag_experiment]
        list_key_metric = [
            'correct_spkr',
            'correct_word',
        ]
        df_results = util_misc.flatten_columns(df_results.groupby([
            'tag_expt',
            'tag_model',
            'fn_eval',
            'background_condition',
            'snr',
        ]).agg({
            key_metric: 'mean' for key_metric in list_key_metric
        }).reset_index().groupby([
            'tag_expt',
            'tag_model',
            'background_condition',
            'snr',
        ]).agg({
            key_metric: [list, 'mean', 'sem'] for key_metric in list_key_metric
        }).reset_index(), sep='_')
        df_results = df_results.sort_index(axis=1)
        df_human = pd.read_pickle(
            os.path.join(
                dir_human_data,
                'word_recognition_as_function_of_snr_and_noise_condition.pkl',
            )
        )
        df_results = pd.concat([df_human, df_results])
        EXPERIMENT_DATAFRAMES[tag_experiment] = df_results

    tag_experiment = 'speech_in_synthetic_textures'
    if tag_experiment in dict_basename_eval:
        print(f"[run_spkr_word_experiments] `{tag_experiment}` experiment")
        df_results = df[df.tag_expt == tag_experiment]
        list_key_metric = [
            'correct_spkr',
            'correct_word',
        ]
        df_results = df_results.groupby([
            'tag_expt',
            'tag_model',
            'fn_eval',
            'index_texture',
            'snr',
        ]).agg({
            key_metric: 'mean' for key_metric in list_key_metric
        }).reset_index()
        df_human = pd.read_pickle(
            os.path.join(
                dir_human_data,
                'word_recognition_in_different_auditory_textures.pkl',
            )
        )
        df_results = pd.concat([df_human, df_results])
        df_results = util_misc.flatten_columns(
            df_results.groupby([
                'tag_expt',
                'tag_model',
                'index_texture',
                'snr',
            ]).agg({
                key_metric: [list, 'mean', 'sem'] for key_metric in list_key_metric
            }).reset_index(),
            sep='_')
        df_results = df_results.sort_index(axis=1)
        EXPERIMENT_DATAFRAMES[tag_experiment] = df_results

    tag_experiment = 'pitch_altered'
    if tag_experiment in dict_basename_eval:
        print(f"[run_spkr_word_experiments] `{tag_experiment}` experiment")
        df_results = df[df.tag_expt == tag_experiment]
        list_key_metric = [
            'correct_spkr',
            'correct_word',
        ]
        df_results = df_results[np.logical_and.reduce([
            df_results['snr'] == np.inf,
            np.logical_or(~df_results['f0_shift_in_semitones'].isna(), df_results['inharmonic'] == 1.0)
        ])]
        df_results.loc[df_results['inharmonic'] == 1.0, 'f0_shift_in_semitones'] = 0
        df_results.loc[df_results['inharmonic'] == 1.0, 'condition'] = 'inharmonic'
        df_results.loc[df_results['inharmonic'] != 1.0, 'condition'] = 'harmonic'
        df_results = util_misc.flatten_columns(df_results.groupby([
            'tag_expt',
            'tag_model',
            'fn_eval',
            'condition',
            'f0_shift_in_semitones',
        ]).agg({
            key_metric: 'mean' for key_metric in list_key_metric
        }).reset_index().groupby([
            'tag_expt',
            'tag_model',
            'condition',
            'f0_shift_in_semitones',
        ]).agg({
            key_metric: [list, 'mean', 'sem'] for key_metric in list_key_metric
        }).reset_index(), sep='_')
        df_results = df_results.sort_index(axis=1)
        df_human = pd.read_pickle(
            os.path.join(
                dir_human_data,
                'spkr_and_word_recognition_with_pitch_altered_speech.pkl',
            )
        )
        df_results = pd.concat([df_human, df_results])
        EXPERIMENT_DATAFRAMES[tag_experiment] = df_results

    tag_experiment = 'hopkins_moore_2009'
    if tag_experiment in dict_basename_eval:
        print(f"[run_spkr_word_experiments] `{tag_experiment}` experiment")
        df_results = df[np.logical_and.reduce([
            df.tag_expt == tag_experiment,
            df.inharmonic == 0,
            df.cutoff_channel >= 0
        ])]
        df_results = df_results.groupby([
            'tag_expt',
            'tag_model',
            'fn_eval',
            'background_condition',
            'snr',
            'cutoff_channel',
            'cutoff_channel_freq',
        ]).agg({
            'correct_word': 'mean',
            'correct_spkr': 'mean',
        }).reset_index()
        df_results = df_results.groupby([
            'tag_expt',
            'tag_model',
            'fn_eval',
            'background_condition',
            'cutoff_channel',
            'cutoff_channel_freq',
        ]).agg({
            'snr': list,
            'correct_word': list,
            'correct_spkr': list,
        }).reset_index()
        list_df_results = [_[1] for _ in df_results.groupby('fn_eval')]
        with multiprocessing.Pool(min(workers, len(list_df_results))) as p:
            list_df_results = p.map(compute_srt, list_df_results)
        df_results = pd.concat(list_df_results).reset_index()
        EXPERIMENT_DATAFRAMES[tag_experiment + '_raw'] = df_results.copy().sort_index(axis=1)
        df_results = util_misc.flatten_columns(df_results.groupby([
            'tag_expt',
            'tag_model',
            'background_condition',
            'cutoff_channel',
            'cutoff_channel_freq',
        ]).agg({
            'srt': [list, 'mean', 'sem']
        }).reset_index(), sep='_')
        df_human = pd.read_pickle(
            os.path.join(
                dir_human_data,
                'hopkins_moore_2009_tone_vocoded_word_recognition.pkl',
            )
        )
        df_results = pd.concat([df_human, df_results])
        EXPERIMENT_DATAFRAMES[tag_experiment] = df_results

    return EXPERIMENT_DATAFRAMES


def compare_word_recognition_kell_like(
        df,
        restrict_conditions=None,
        metric_function=scipy.stats.pearsonr,
        bootstrap_repeats=1000,
        random_seed=0):
    """
    """
    if restrict_conditions is not None:
        df = df[df['background_condition'].isin(restrict_conditions)].copy()
    np.random.seed(random_seed)
    df_human = df[df['tag_model'] == 'human']
    df_results = []
    for tag_model in sorted(set(df['tag_model'].unique()).difference({'human'})):
        df_model = df[df['tag_model'] == tag_model]
        df_comparison = pd.merge(
            df_human,
            df_model,
            on=['background_condition', 'snr'],
            how='inner',
            suffixes=('_human', '_model'))
        x = df_comparison['correct_word_mean_human'].values
        y = df_comparison['correct_word_mean_model'].values
        list_y = np.array(list(df_comparison['correct_word_list_model'].values)).T
        IDX = np.arange(0, list_y.shape[0], dtype=int)
        bootstrap_list_metric = np.zeros(bootstrap_repeats)
        for _ in range(bootstrap_repeats):
            bootstrap_IDX = np.random.choice(IDX, size=[len(IDX)], replace=True)
            bootstrap_y = list_y[bootstrap_IDX].mean(axis=0)
            bootstrap_list_metric[_] = np.array(metric_function(x, bootstrap_y)).reshape([-1])[0]
        metric = np.array(metric_function(x, y)).reshape([-1])[0]
        list_metric = [np.array(metric_function(x, list_y[_])).reshape([-1])[0] for _ in range(list_y.shape[0])]
        df_results.append({
            'tag_model': tag_model,
            'metric': metric,
            'list_metric': list_metric,
            'bootstrap_list_metric': bootstrap_list_metric,
        })
    df_results = pd.DataFrame(df_results)
    return df_results


def compare_word_recognition_speech_in_synthetic_textures(
        df,
        restrict_conditions=None,
        metric_function=scipy.stats.pearsonr,
        bootstrap_repeats=1000,
        random_seed=0):
    """
    """
    if restrict_conditions is not None:
        df = df[df['index_texture'].isin(restrict_conditions)].copy()
    np.random.seed(random_seed)
    df_human = df[df['tag_model'] == 'human']
    df_results = []
    for tag_model in sorted(set(df['tag_model'].unique()).difference({'human'})):
        df_model = df[df['tag_model'] == tag_model]
        df_comparison = pd.merge(
            df_human,
            df_model,
            on=['index_texture', 'snr'],
            how='inner',
            suffixes=('_human', '_model'))
        x = df_comparison['correct_word_mean_human'].values
        y = df_comparison['correct_word_mean_model'].values
        list_y = np.array(list(df_comparison['correct_word_list_model'].values)).T
        IDX = np.arange(0, list_y.shape[0], dtype=int)
        bootstrap_list_metric = np.zeros(bootstrap_repeats)
        for _ in range(bootstrap_repeats):
            bootstrap_IDX = np.random.choice(IDX, size=[len(IDX)], replace=True)
            bootstrap_y = list_y[bootstrap_IDX].mean(axis=0)
            bootstrap_list_metric[_] = np.array(metric_function(x, bootstrap_y)).reshape([-1])[0]
        metric = np.array(metric_function(x, y)).reshape([-1])[0]
        list_metric = [np.array(metric_function(x, list_y[_])).reshape([-1])[0] for _ in range(list_y.shape[0])]
        df_results.append({
            'tag_model': tag_model,
            'metric': metric,
            'list_metric': list_metric,
            'bootstrap_list_metric': bootstrap_list_metric,
        })
    df_results = pd.DataFrame(df_results)
    return df_results


def compare_recognition_pitch_altered(
        df,
        key_task='word',
        metric_function=scipy.stats.pearsonr,
        bootstrap_repeats=1000,
        random_seed=0):
    """
    """
    if not isinstance(key_task, list):
        key_task = [key_task]
    np.random.seed(random_seed)
    df_human = df[df['tag_model'] == 'human']
    df_results = []
    for tag_model in sorted(set(df['tag_model'].unique()).difference({'human'})):
        df_model = df[df['tag_model'] == tag_model]
        df_comparison = pd.merge(
            df_human,
            df_model,
            on=['condition', 'f0_shift_in_semitones'],
            how='inner',
            suffixes=('_human', '_model'))
        x = []
        y = []
        list_y = []
        for kt in key_task:
            x.append(df_comparison[f'correct_{kt}_mean_human'].values)
            y.append(df_comparison[f'correct_{kt}_mean_model'].values)
            list_y.append(np.array(list(df_comparison[f'correct_{kt}_list_model'].values)))
        x = np.concatenate(x)
        y = np.concatenate(y)
        list_y = np.concatenate(list_y).T
        IDX = np.arange(0, list_y.shape[0], dtype=int)
        bootstrap_list_metric = np.zeros(bootstrap_repeats)
        for _ in range(bootstrap_repeats):
            bootstrap_IDX = np.random.choice(IDX, size=[len(IDX)], replace=True)
            bootstrap_y = list_y[bootstrap_IDX].mean(axis=0)
            bootstrap_list_metric[_] = np.array(metric_function(x, bootstrap_y)).reshape([-1])[0]
        metric = np.array(metric_function(x, y)).reshape([-1])[0]
        list_metric = [np.array(metric_function(x, list_y[_])).reshape([-1])[0] for _ in range(list_y.shape[0])]
        df_results.append({
            'tag_model': tag_model,
            'metric': metric,
            'list_metric': list_metric,
            'bootstrap_list_metric': bootstrap_list_metric,
        })
    df_results = pd.DataFrame(df_results)
    return df_results


def compare_word_recognition_hopkins_moore_2009(
        df,
        use_relative_srt=True,
        metric_function=scipy.stats.pearsonr,
        bootstrap_repeats=1000,
        random_seed=0):
    """
    """
    np.random.seed(random_seed)
    df_human = df[df['tag_model'] == 'human']
    df_results = []
    for tag_model in sorted(set(df['tag_model'].unique()).difference({'human'})):
        df_model = df[df['tag_model'] == tag_model]
        df_comparison = pd.merge(
            df_human,
            df_model,
            on=['background_condition', 'cutoff_channel'],
            how='inner',
            suffixes=('_human', '_model'))
        def relative_srt(df):
            dfi = df[df['cutoff_channel'] == 0]
            assert len(dfi) == 1
            dfi = dfi.iloc[0]
            df['srt_mean_human'] -= dfi['srt_mean_human']
            df['srt_mean_model'] -= dfi['srt_mean_model']
            df['srt_list_human'] = df['srt_list_human'].map(
                lambda _: np.array(_) - np.array(dfi['srt_list_human']))
            df['srt_list_model'] = df['srt_list_model'].map(
                lambda _: np.array(_) - np.array(dfi['srt_list_model']))
            return df
        if use_relative_srt:
            df_comparison = df_comparison.groupby(
                ['background_condition'], group_keys=False).apply(
                relative_srt).reset_index(drop=True)
        x = df_comparison['srt_mean_human'].values
        y = df_comparison['srt_mean_model'].values
        list_y = np.array(list(df_comparison['srt_list_model'].values)).T
        IDX = np.arange(0, list_y.shape[0], dtype=int)
        bootstrap_list_metric = np.zeros(bootstrap_repeats)
        for _ in range(bootstrap_repeats):
            bootstrap_IDX = np.random.choice(IDX, size=[len(IDX)], replace=True)
            bootstrap_y = list_y[bootstrap_IDX].mean(axis=0)
            bootstrap_list_metric[_] = np.array(metric_function(x, bootstrap_y)).reshape([-1])[0]
        metric = np.array(metric_function(x, y)).reshape([-1])[0]
        list_metric = [np.array(metric_function(x, list_y[_])).reshape([-1])[0] for _ in range(list_y.shape[0])]
        df_results.append({
            'tag_model': tag_model,
            'metric': metric,
            'list_metric': list_metric,
            'bootstrap_list_metric': bootstrap_list_metric,
        })
    df_results = pd.DataFrame(df_results)
    return df_results
