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

sys.path.append('/om2/user/msaddler/python-packages/msutil')
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


def get_voice_discrimination_trials(df, list_random_seed=[2]):
    """
    """
    df_to_sample = df.groupby(['speaker_id_int']).agg({
        'sentence_id_int': 'unique',
        'speaker_sex_int': 'first',
    }).reset_index().rename(columns={
        'speaker_id_int': 'spkr0',
        'sentence_id_int': 'spkr0_sent',
        'speaker_sex_int': 'spkr0_sex',
    })
    map_spkr_to_sent = dict(zip(df_to_sample['spkr0'], df_to_sample['spkr0_sent']))
    map_spkr_to_sex = dict(zip(df_to_sample['spkr0'], df_to_sample['spkr0_sex']))
    def sample_trials(dfi):
        l0 = list(np.random.choice(dfi['spkr0_sent'], size=len(dfi['spkr0_sent']), replace=False))
        l1 = list(np.random.choice(dfi['spkr1_sent'], size=len(dfi['spkr1_sent']), replace=False))
        dfi['list_spkr_sent_0'] = l0[:2] + l1[:1]
        dfi['list_spkr_sent_1'] = l0[-1:] + l1[-2:]
        dfi['list_spkr_sex_0'] = [dfi['spkr0_sex'], dfi['spkr0_sex'], dfi['spkr1_sex']]
        dfi['list_spkr_sex_1'] = [dfi['spkr0_sex'], dfi['spkr1_sex'], dfi['spkr1_sex']]
        dfi['list_spkr_0'] = [dfi['spkr0'], dfi['spkr0'], dfi['spkr1']]
        dfi['list_spkr_1'] = [dfi['spkr0'], dfi['spkr1'], dfi['spkr1']]
        return dfi
    list_df_trials = []
    for random_seed in list_random_seed:
        np.random.seed(random_seed)
        df_trials = df_to_sample.copy()
        list_spkr0 = list(df_trials.sample(frac=1)['spkr0'].values)
        list_spkr1 = list_spkr0[1:] + list_spkr0[:1]
        map_spkr0_to_spkr1 = dict(zip(list_spkr0, list_spkr1))
        df_trials['spkr1'] = df_trials['spkr0'].map(map_spkr0_to_spkr1)
        df_trials['spkr1_sent'] = df_trials['spkr1'].map(map_spkr_to_sent)
        df_trials['spkr1_sex'] = df_trials['spkr1'].map(map_spkr_to_sex)
        df_trials = df_trials.apply(sample_trials, axis=1)
        df_trials = pd.concat([
            df_trials[[
                f'list_spkr_{_}',
                f'list_spkr_sex_{_}',
                f'list_spkr_sent_{_}',
            ]].rename(columns={
                f'list_spkr_{_}': 'list_spkr',
                f'list_spkr_sex_{_}': 'list_sex',
                f'list_spkr_sent_{_}': 'list_sent',
            }) for _ in [0, 1]
        ])
        df_trials['random_seed'] = random_seed
        df_trials['sex'] = df_trials['list_sex'].map(lambda _: len(np.unique(_)) * max(_))
        list_df_trials.append(df_trials)
    df_trials = pd.concat(list_df_trials).reset_index(drop=True)
    return df_trials


def run_voice_discrimination_trials(
        df,
        df_trials,
        dist_metric=scipy.stats.entropy,
        key_sent='sentence_id_int',
        key_prob='label_speaker_int:probs_out',
        overwrite=False):
    """
    """
    def to_apply(dfi):
        map_sent_to_prob = dict(zip(dfi[key_sent], dfi[key_prob]))
        list_dist_first = []
        list_dist_last = []
        list_correct = []
        for itr_trial in range(len(df_trials)):
            list_spkr = df_trials.iloc[itr_trial]['list_spkr']
            list_sent = df_trials.iloc[itr_trial]['list_sent']
            list_prob = [map_sent_to_prob[_] for _ in list_sent]
            dist_first = dist_metric(list_prob[0], list_prob[1])
            dist_last = dist_metric(list_prob[2], list_prob[1])
            response_pred = int(dist_first < dist_last)
            response_true = int(list_spkr[0] == list_spkr[1])
            list_correct.append(response_pred == response_true)
            list_dist_first.append(dist_first)
            list_dist_last.append(dist_last)
        dfi = df_trials.reset_index(drop=True).copy()
        dfi['correct'] = list_correct
        dfi['dist_first'] = list_dist_first
        dfi['dist_last'] = list_dist_last
        return dfi
    assert df['fn_eval'].nunique() == 1
    fn_eval = df.iloc[0]['fn_eval']
    fn_results = fn_eval.replace('.json', '_results.pkl')
    assert fn_results != fn_eval
    if not overwrite and os.path.exists(fn_results):
        df_results = pd.read_pickle(fn_results)
    else:
        df_results = df.groupby([
            'tag_expt',
            'tag_model',
            'fn_eval',
            'background_condition',
            'snr',
        ]).apply(to_apply).reset_index()
        df_results.to_pickle(fn_results)
    return df_results


def run_spkr_word_experiments(list_regex_dir_model, workers=20, **kwargs):
    """
    """
    EXPERIMENT_DATAFRAMES = {}
    dir_human_data = '/om2/user/msaddler/tfauditoryutil/saved_models/HUMAN/spkr_word'

    dict_basename_eval = {
        'kell_like_inharmonic': 'EVAL_word_recognition_human_experiment_v00_inharmonic_foreground60dbspl.json',
        'kell_like': 'EVAL_word_recognition_human_experiment_v00_foreground60dbspl.json',
        'speech_in_synthetic_textures': 'EVAL_word_recognition_speech_in_synthetic_textures.json',
        'hopkins_moore_2009': 'EVAL_hopkins_moore_2009.json',
        'pitch_altered': 'EVAL_pitch_altered_v00.json',
        'spkr_discrimination_timit_ssn': 'EVAL_spkr_discrimination_timit_ssn.json',
    }
    if 'dict_basename_eval' not in kwargs:
        kwargs['dict_basename_eval'] = dict_basename_eval
    else:
        dict_basename_eval = kwargs['dict_basename_eval']
    if 'dict_load_arr' not in kwargs:
        kwargs['dict_load_arr'] = {'spkr_discrimination_timit_ssn': True}
    df = []
    for regex_dir_model in list_regex_dir_model:
        list_dir_model = glob.glob(regex_dir_model)
        with multiprocessing.Pool(min(workers, len(list_dir_model))) as p:
            func = functools.partial(func_to_parallelize_spkr_word, **kwargs)
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
        df_human = pd.read_pickle(os.path.join(dir_human_data, 'human_data_saddler_2023_kell_like.pkl'))
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
        df_human = pd.read_pickle(os.path.join(dir_human_data, 'human_data_saddler_2023_speech_in_synthetic_textures.pkl'))
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
        df_human = pd.read_pickle(os.path.join(dir_human_data, 'human_data_saddler_2023_pitch_altered.pkl'))
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
        df_human = pd.read_pickle(os.path.join(dir_human_data, 'human_data_hopkins_moore_2009_tfs_manipulation.pkl'))
        df_results = pd.concat([df_human, df_results])
        EXPERIMENT_DATAFRAMES[tag_experiment] = df_results

    tag_experiment = 'spkr_discrimination_timit_ssn'
    if tag_experiment in dict_basename_eval:
        print(f"[run_spkr_word_experiments] `{tag_experiment}` experiment")
        df_results = df[df.tag_expt == tag_experiment]
        df_trials = get_voice_discrimination_trials(df_results, list_random_seed=[2])
        list_df_results = [_[1] for _ in df_results.groupby(['tag_expt', 'tag_model', 'fn_eval'])]
        with multiprocessing.Pool(min(workers, len(list_df_results))) as p:
            func = functools.partial(
                run_voice_discrimination_trials,
                df_trials=df_trials,
                key_sent='sentence_id_int',
                key_prob='label_speaker_int:probs_out',
                overwrite=False)
            list_df_results = p.map(func, list_df_results)
        df_results = pd.concat(list_df_results).reset_index(drop=True)
        df_results = util_misc.flatten_columns(
            df_results.groupby([
                'tag_expt',
                'tag_model',
                'fn_eval',
                'background_condition',
                'snr',
            ]).agg({
                'correct': 'mean',
            }).reset_index().groupby([
                'tag_expt',
                'tag_model',
                'background_condition',
                'snr',
            ]).agg({
                'correct': [list, 'mean', 'sem']
            }).reset_index(),
            sep='_')
        df_human = pd.read_pickle(os.path.join(dir_human_data, 'human_data_saddler_2023_spkr_discrimination_timit_ssn.pkl'))
        df_results = pd.concat([df_human, df_results])
        EXPERIMENT_DATAFRAMES[tag_experiment] = df_results

    return EXPERIMENT_DATAFRAMES


def compare_word_recognition_kell_like(
        df,
        restrict_conditions=None,
        metric_function=scipy.stats.pearsonr,
        bootstrap_repeats=200,
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
        df_results.append({
            'tag_model': tag_model,
            'metric': metric,
            'bootstrap_list_metric': bootstrap_list_metric,
        })
    df_results = pd.DataFrame(df_results)
    return df_results


def compare_word_recognition_speech_in_synthetic_textures(
        df,
        restrict_conditions=None,
        metric_function=scipy.stats.pearsonr,
        bootstrap_repeats=200,
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
        df_results.append({
            'tag_model': tag_model,
            'metric': metric,
            'bootstrap_list_metric': bootstrap_list_metric,
        })
    df_results = pd.DataFrame(df_results)
    return df_results


def compare_recognition_pitch_altered(
        df,
        key_task='word',
        metric_function=scipy.stats.pearsonr,
        bootstrap_repeats=200,
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
        df_results.append({
            'tag_model': tag_model,
            'metric': metric,
            'bootstrap_list_metric': bootstrap_list_metric,
        })
    df_results = pd.DataFrame(df_results)
    return df_results


def compare_word_recognition_hopkins_moore_2009(
        df,
        use_relative_srt=True,
        metric_function=scipy.stats.pearsonr,
        bootstrap_repeats=200,
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
        df_results.append({
            'tag_model': tag_model,
            'metric': metric,
            'bootstrap_list_metric': bootstrap_list_metric,
        })
    df_results = pd.DataFrame(df_results)
    return df_results


def compare_spkr_discrimination_timit_ssn(
        df,
        restrict_conditions=None,
        metric_function=scipy.stats.pearsonr,
        bootstrap_repeats=200,
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
        x = df_comparison['correct_mean_human'].values
        y = df_comparison['correct_mean_model'].values
        list_y = np.array(list(df_comparison['correct_list_model'].values)).T
        IDX = np.arange(0, list_y.shape[0], dtype=int)
        bootstrap_list_metric = np.zeros(bootstrap_repeats)
        for _ in range(bootstrap_repeats):
            bootstrap_IDX = np.random.choice(IDX, size=[len(IDX)], replace=True)
            bootstrap_y = list_y[bootstrap_IDX].mean(axis=0)
            bootstrap_list_metric[_] = np.array(metric_function(x, bootstrap_y)).reshape([-1])[0]
        metric = np.array(metric_function(x, y)).reshape([-1])[0]
        df_results.append({
            'tag_model': tag_model,
            'metric': metric,
            'bootstrap_list_metric': bootstrap_list_metric,
        })
    df_results = pd.DataFrame(df_results)
    return df_results
