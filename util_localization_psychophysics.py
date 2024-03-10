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

import util


def load_eval_dict(fn_eval, keys_to_ignore=[]):
    """
    """
    with open(fn_eval, 'r') as f_eval:
        eval_dict = json.load(f_eval)
    for k in keys_to_ignore:
        eval_dict.pop(k, None)
    for k in sorted(eval_dict.keys()):
        if isinstance(eval_dict[k], str):
            fn_k = eval_dict[k]
            if os.path.basename(fn_k) == fn_k:
                fn_k = os.path.join(os.path.dirname(fn_eval), fn_k)
            if os.path.exists(fn_k):
                eval_dict[k] = np.load(fn_k)
        else:
            eval_dict[k] = np.array(eval_dict[k])
    return eval_dict


def azim_elev_to_label(azim, elev):
    """
    """
    label = (elev / 10) * 72 + (azim / 5)
    return np.array(label).astype(int)


def label_to_azim_elev(label):
    """
    """
    elev = np.array((label // 72) * 10)
    azim = np.array((label % 72) * 5)
    return np.array(azim).astype(float), np.array(elev).astype(float)


def probs_to_label(probs, prior=None):
    """
    """
    assert len(probs.shape) == 2
    if prior is None:
        label = np.argmax(probs, axis=1)
    else:
        if not prior.shape == probs.shape:
            prior = prior.reshape([1, probs.shape[1]])
        label = np.argmax(prior * probs, axis=1)
    return label


def normalize_angle(angle, lower=-180, upper=180):
    """
    Based on `normalize` function from:
    https://gist.github.com/phn/1111712/35e8883de01916f64f7f97da9434622000ac0390
    """
    assert lower < upper, f'Invalid lower and upper limits: [{lower}, {upper}]'
    normalized_angle = np.array(angle).copy()
    IDX_ABOVE = np.logical_or(angle > upper, angle == lower)    
    normalized_angle[IDX_ABOVE] = lower + np.mod(
        np.abs(angle[IDX_ABOVE] + upper),
        np.abs(lower) + np.abs(upper))
    IDX_BELOW = np.logical_or(angle < lower, angle == upper)
    normalized_angle[IDX_BELOW] = upper - np.mod(
        np.abs(angle[IDX_BELOW] - lower),
        np.abs(lower) + np.abs(upper))
    return normalized_angle


def fold_front_back(azim):
    """
    """
    azim = normalize_angle(azim, lower=-180, upper=180)
    condlist = [
        np.logical_and(azim >= -180, azim < -90),
        np.logical_and(azim >= -90, azim < 90),
        np.logical_and(azim >= 90, azim < 270),
    ]
    funclist = [
        lambda x: -x - 180,
        lambda x: x,
        lambda x: -x + 180,
    ]
    return np.piecewise(azim, condlist, funclist)


def angular_distance(x, y, deg=True):
    """
    Calculate degrees between two angles x and y.
    """
    x = np.array(x).reshape([-1])
    y = np.array(y).reshape([-1])
    if deg:
        xr = np.deg2rad(x)
        yr = np.deg2rad(y)
    else:
        xr = x
        yr = y
    dr = np.abs(np.arctan2(np.sin(xr-yr), np.cos(xr-yr)))
    if deg:
        d = np.rad2deg(dr)
    else:
        d = dr
    return d


def haversine_distance(X, Y, deg=True):
    """
    Calculate spherical distance between azimuth/elevation
    pairs X and Y (each must have shape [n, 2]) where the
    the last dimension is azimuth/elevation.
    
    Updated to use more numerically-stable "Vincenty formula":
    (https://en.wikipedia.org/wiki/Great-circle_distance)
    """
    X = np.array(X, dtype=np.float64)
    Y = np.array(Y, dtype=np.float64)
    assert X.shape[1] == 2 and Y.shape[1] == 2
    if deg:
        Xr = np.deg2rad(X)
        Yr = np.deg2rad(Y)
    else:
        Xr = X
        Yr = Y
    phi1 = Xr[:, 1]
    phi2 = Yr[:, 1]
    delta_lambda = np.abs(Xr[:, 0] - Yr[:, 0])
    numerator = np.sqrt(
        np.square(np.cos(phi2) * np.sin(delta_lambda)) + \
        np.square(np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(delta_lambda))
    )
    denominator = np.sin(phi1) * np.sin(phi2) + np.cos(phi1) * np.cos(phi2) * np.cos(delta_lambda)
    dr = np.arctan2(numerator, denominator)
    if deg:
        d = np.rad2deg(dr)
    else:
        d = dr
    return d


def fit_normcdf(xvals, yvals, mu=0.0, sigma=None):
    """
    """
    normcdf = lambda x, sigma: scipy.stats.norm(mu, sigma).cdf(x)
    sigma_opt, sigma_opt_cov = scipy.optimize.curve_fit(normcdf, xvals, yvals, sigma=sigma)
    return np.squeeze(sigma_opt), np.squeeze(sigma_opt_cov)


def fit_threshold(independent_variable, dependent_variable, threshold_value=0.707):
    """
    """
    x = np.array(list(itertools.product(independent_variable, independent_variable)))
    y = np.array(list(itertools.product(dependent_variable, dependent_variable)))
    x = x[:, 1] - x[:, 0]
    y = y[:, 1] > y[:, 0] + (np.random.randn(*x.shape) * np.finfo(np.float32).eps)
    y = y[x != 0].astype(float)
    x = x[x != 0]
    df = pd.DataFrame({'x': x, 'y': y}).groupby('x').agg({'y': 'mean'}).reset_index().sort_values(by='x')
    sigma_opt, _ = fit_normcdf(df.x.values, df.y.values, sigma=df.x.values)
    threshold = scipy.stats.norm(0, sigma_opt).ppf(threshold_value)
    return threshold


def get_azim_to_itd_mapping(fn='/scratch2/weka/mcdermott/msaddler/HRTF_ITDs.pkl'):
    """
    """
    with open(fn, 'rb') as f:
        X = np.array(pickle.load(f))
    list_azim = np.concatenate((-1 * X[::-1, 0], X[1:, 0]), axis=0)
    list_itd = np.concatenate((-1 * X[::-1, 1], X[1:, 1]), axis=0)
    azim_to_itd = lambda azim: np.interp(((azim + 180) % 360) - 180, xp=list_azim, fp=list_itd)
    return azim_to_itd


def get_azim_to_ild_mapping(fn='/scratch2/weka/mcdermott/msaddler/HRTF_freq_specific_ILDs.pkl', freq='low'):
    """
    """
    with open(fn, 'rb') as f:
        X = np.array(pickle.load(f))
    list_azim = np.concatenate((-1 * X[::-1, 0], X[1:, 0]), axis=0)
    if freq.lower() == 'low':
        list_ild = np.concatenate((-1 * X[::-1, 1], X[1:, 1]), axis=0)
    elif freq.lower() == 'high':
        list_ild = np.concatenate((-1 * X[::-1, 2], X[1:, 2]), axis=0)
    else:
        raise ValueError("freq must be `low` or `high`")
    azim_to_ild = lambda azim: np.interp(((azim + 180) % 360) - 180, xp=list_azim, fp=list_ild)
    return azim_to_ild


def experiment_simple(
        fn_eval=None,
        prior=None,
        use_weighted=False,
        fold=False,
        func_label_to_azim_elev=label_to_azim_elev,
        key_pred='label_loc_int:labels_pred',
        key_true='label_loc_int:labels_true',
        key_pred_prob='label_loc_int:probs_out',
        verbose=True):
    """
    """
    eval_dict = load_eval_dict(fn_eval)
    if prior is not None:
        label = probs_to_label(eval_dict[key_pred_prob], prior=prior)
        changed = np.sum(np.logical_not(label == eval_dict[key_pred]))
        if verbose:
            print('Prior modified {} of {} predicted labels'.format(changed, label.shape[0]))
        eval_dict[key_pred] = label
    azim_true, elev_true = func_label_to_azim_elev(eval_dict[key_true])
    azim_pred, elev_pred = func_label_to_azim_elev(eval_dict[key_pred])
    if use_weighted:
        pred_prob = eval_dict[key_pred_prob]
        if prior is None:
            prior = np.ones_like(pred_prob[0])
        azim, elev = func_label_to_azim_elev(np.arange(0, pred_prob.shape[1]))
        azim_support = normalize_angle(azim)[prior.astype(bool)]
        elev_support = elev[prior.astype(bool)]
        pred_prob = pred_prob[:, prior.astype(bool)]
        pred_prob = pred_prob / pred_prob.sum(axis=1, keepdims=True)
        azim_pred = np.squeeze(pred_prob @ azim_support[:, np.newaxis])
        elev_pred = np.squeeze(pred_prob @ elev_support[:, np.newaxis])
    if fold:
        azim_true = fold_front_back(azim_true)
        azim_pred = fold_front_back(azim_pred)
    # Store all experiment-relevant data in a pandas dataframe
    df_dict = {
        'azim_true': normalize_angle(azim_true),
        'azim_pred': normalize_angle(azim_pred),
        'elev_true': elev_true,
        'elev_pred': elev_pred,
    }
    for k in eval_dict.keys():
        if len(eval_dict[k].shape) == 1:
            df_dict[k] = eval_dict[k]
    df = pd.DataFrame(df_dict)
    df['fn_eval'] = fn_eval
    return df


def experiment_itd_ild_weighting(
        fn_eval=None,
        fold=True,
        prior=None,
        exclude_by_std=False,
        func_label_to_azim_elev=label_to_azim_elev,
        key_pred='label_loc_int:labels_pred',
        key_true='label_loc_int:labels_true',
        key_pred_prob='label_loc_int:probs_out',
        sr=44100,
        verbose=True):
    """
    """
    eval_dict = load_eval_dict(fn_eval)
    if prior is not None:
        label = probs_to_label(eval_dict[key_pred_prob], prior=prior)
        changed = np.sum(np.logical_not(label == eval_dict[key_pred]))
        if verbose:
            print('Prior modified {} of {} predicted labels'.format(changed, label.shape[0]))
        eval_dict[key_pred] = label
    azim_true, elev_true = func_label_to_azim_elev(eval_dict[key_true])
    azim_pred, elev_pred = func_label_to_azim_elev(eval_dict[key_pred])
    azim_true = normalize_angle(azim_true)
    azim_pred = normalize_angle(azim_pred)
    azim_true_folded = fold_front_back(azim_true)
    azim_pred_folded = fold_front_back(azim_pred)
    if fold:
        azim = azim_pred_folded
    else:
        azim = azim_pred
    # Store all experiment-relevant data in a pandas dataframe
    df = pd.DataFrame({
        'azim_true': azim_true,
        'azim': azim,
        'ild_bias_imposed': -1 * eval_dict['ild'], # Sign change because +ILD = louder right ear
        'itd_bias_imposed': 1e6 * eval_dict['itd'] / sr, # Convert `ITD` from samples to microseconds
        'cutoff_low': eval_dict['noise_low'],
        'cutoff_high': eval_dict['noise_high'],
        'noise_idx': eval_dict['idx'],
        'band': list(zip(list(eval_dict['noise_low']), list(eval_dict['noise_high']))),
    })
    df['band'] = df['band'].astype(str)
    df['fn_eval'] = fn_eval
    # Include only stimuli with physiologically-plausible ITDs and ILDs
    IDX_NONPHYSIOLOGICAL = np.logical_or.reduce([
        np.logical_and(np.abs(eval_dict['ild']) >= 10, np.abs(azim_true) > 40), # From francl code
        np.logical_and(np.abs(eval_dict['ild']) >= 20, np.abs(azim_true) > 20), # From francl code
        np.logical_and(np.abs(eval_dict['itd']) >= 18, np.abs(azim_true) > 40), # From francl code
        np.logical_and(np.abs(eval_dict['itd']) >= 22, np.abs(azim_true) > 20), # From francl code
    ])
    df = df[~IDX_NONPHYSIOLOGICAL]
    # Include noise bands from Macpherson & Middlebrooks (2002, JASA)
    df = df[df.band.isin([
        '(500, 2000)',
        '(500, 16000)',
        '(4000, 16000)',
    ])]
    # Aggregate predictions across noise_idx
    df = df.groupby([
        'azim_true',
        'ild_bias_imposed',
        'itd_bias_imposed',
        'cutoff_low',
        'cutoff_high',
        'band',
        'fn_eval',
    ])['azim'].agg(['mean', 'count', 'std']).reset_index()
    # Get baseline network predictions for each bias ITD=0, bias ILD=0 condition
    df_unbiased = df[np.logical_and(df['ild_bias_imposed'] == 0, df['itd_bias_imposed'] == 0)]
    df_unbiased = df_unbiased.groupby(['azim_true', 'band'])['mean'].mean().reset_index()
    # Define functions for converting azimuths to ILD / ITD cues and computing residuals
    azim_to_ild_low = get_azim_to_ild_mapping(freq='low')
    azim_to_ild_high = get_azim_to_ild_mapping(freq='high')
    azim_to_ild_med = lambda _: 0.5 * (azim_to_ild_low(_) + azim_to_ild_high(_))
    azim_to_itd = get_azim_to_itd_mapping()
    
    def calculate_cue_residual(row, df_azim, azim_to_cue=None):
        azim_unbiased = df_azim[np.logical_and(
            df_azim['azim_true'] == row['azim_true'],
            df_azim['band'] == row['band'],
        )]['mean'].iloc[0]
        cue_unbiased = azim_to_cue(azim_unbiased)
        cue_biased = azim_to_cue(row['mean'])
        return cue_biased - cue_unbiased
    
    def itd_residual_fcn(row):
        azim_to_cue = azim_to_itd
        return calculate_cue_residual(row, df_unbiased, azim_to_cue=azim_to_cue)
    
    def ild_residual_fcn(row):
        if row['band'] == '(500, 2000)':
            azim_to_cue = azim_to_ild_low
        elif row['band'] == '(500, 16000)':
            azim_to_cue = azim_to_ild_med
        elif row['band'] == '(4000, 16000)':
            azim_to_cue = azim_to_ild_high
        else:
            raise ValueError("no azim_to_ild mapping for specified band")
        return calculate_cue_residual(row, df_unbiased, azim_to_cue=azim_to_cue)
    
    # ITD experiment (aggregate down to unique itd_bias_imposed values)
    df_itd = df[df['ild_bias_imposed'] == 0].copy() # Consider only rows with bias ILD = 0
    if exclude_by_std:
        if isinstance(exclude_by_std, bool):
            threshold = 10
        else:
            threshold = exclude_by_std
        IDX_exclude = df_itd['std'] > threshold
        if verbose:
            print(f"Excluding {IDX_exclude.sum()} of {len(IDX_exclude)} conditions from ITD expt due to std > {threshold}")
        df_itd = df_itd[~IDX_exclude]
    df_itd['itd_bias_response'] = df_itd.apply(itd_residual_fcn, axis=1)
    df_itd = df_itd.groupby([
        'cutoff_low',
        'cutoff_high',
        'band',
        'fn_eval',
        'itd_bias_imposed']).agg({'itd_bias_response': 'mean'}).reset_index()
    # ILD experiment (aggregate down to unique ild_bias_imposed values)
    df_ild = df[df['itd_bias_imposed'] == 0].copy() # Consider only rows with bias ITD = 0
    if exclude_by_std:
        if isinstance(exclude_by_std, bool):
            threshold = 14
        else:
            threshold = exclude_by_std
        IDX_exclude = df_ild['std'] > threshold
        if verbose:
            print(f"Excluding {IDX_exclude.sum()} of {len(IDX_exclude)} conditions from ILD expt due to std > {threshold}")
        df_ild = df_ild[~IDX_exclude]
    df_ild['ild_bias_response'] = df_ild.apply(ild_residual_fcn, axis=1)
    df_ild = df_ild.groupby([
        'cutoff_low',
        'cutoff_high',
        'band',
        'fn_eval',
        'ild_bias_imposed']).agg({'ild_bias_response': 'mean'}).reset_index()
    
    def calculate_cue_weight(df, cue='itd'):
        x = df[f'{cue}_bias_imposed'].values
        y = df[f'{cue}_bias_response'].values
        x = x[~np.isnan(y)]
        y = y[~np.isnan(y)]
        m, b = np.polyfit(x, y, 1)
        return m
    
    # Compute cue weights
    df = pd.concat([df_itd, df_ild])
    df['itd_bias_weight'] = df['band'].map(
        dict(df.groupby('band').apply(calculate_cue_weight, cue='itd')))
    df['ild_bias_weight'] = df['band'].map(
        dict(df.groupby('band').apply(calculate_cue_weight, cue='ild')))
    return df


def experiment_minimum_audible_angle(
        fn_eval=None,
        func_label_to_azim_elev=None,
        key_pred_prob='label_loc_int:probs_out',
        list_azim_ref=np.arange(-75, 76, 5),
        max_deg_diff=10,
        use_weighted=True,
        overwrite=False):
    """
    """
    fn_results = fn_eval.replace('.json', '_results.pkl')
    if os.path.exists(fn_results) and (not overwrite):
        df = pd.read_pickle(fn_results)
    else:
        eval_dict = load_eval_dict(fn_eval)
        pred_prob = eval_dict.pop(key_pred_prob)
        azim, elev = func_label_to_azim_elev(np.arange(0, pred_prob.shape[1]))
        azim = normalize_angle(azim)
        prior = np.logical_and.reduce([
            azim >= -90,
            azim <= 90,
            elev == 0,
        ]).astype(np.float32)
        eval_dict['azim_pred'], _ = func_label_to_azim_elev(probs_to_label(pred_prob, prior=prior))
        eval_dict['azim_pred'] = normalize_angle(eval_dict['azim_pred'])
        if use_weighted:
            azim_support = azim[prior.astype(bool)]
            pred_prob = pred_prob[:, prior.astype(bool)]
            pred_prob = pred_prob / pred_prob.sum(axis=1, keepdims=True)
            eval_dict['azim_pred'] = np.squeeze(pred_prob @ azim_support[:, np.newaxis])
        df = pd.DataFrame({
            k: v for (k, v) in eval_dict.items() if len(v.shape) < 2
        }).rename(columns={'foreground_azimuth': 'azim_true'})
        df = df.sort_values(by='azim_true')
        df = df.groupby(['f_ref']).agg({
            'azim_true': list,
            'azim_pred': list,
        }).reset_index()
        df = pd.concat([df.assign(azim_ref=_) for _ in list_azim_ref])
        
        def restrict_list(dfi, max_deg_diff=10):
            x = np.array(dfi.azim_true)
            y = np.array(dfi.azim_pred)
            IDX = np.abs(x - dfi.azim_ref) <= max_deg_diff
            dfi.azim_true = x[IDX]
            dfi.azim_pred = y[IDX]
            return dfi
        
        df = df.apply(restrict_list, max_deg_diff=max_deg_diff, axis=1)
        df['maa'] = df.apply(
            lambda _: float(fit_threshold(_.azim_true, _.azim_pred)),
            axis=1)
        df['fn_eval'] = fn_eval
        df.to_pickle(fn_results)
        print(f"[WROTE]: {fn_results}")
    return df


def experiment_itd_threshold(
        fn_eval=None,
        func_label_to_azim_elev=None,
        key_pred_prob='label_loc_int:probs_out',
        max_itd_diff=None,
        use_weighted=False,
        overwrite=False):
    """
    """
    fn_results = fn_eval.replace('.json', '_results.pkl')
    if os.path.exists(fn_results) and (not overwrite):
        df = pd.read_pickle(fn_results)
    else:
        eval_dict = load_eval_dict(fn_eval)
        pred_prob = eval_dict.pop(key_pred_prob)
        azim, elev = func_label_to_azim_elev(np.arange(0, pred_prob.shape[1]))
        azim = normalize_angle(azim)
        prior = np.logical_and.reduce([
            azim >= -90,
            azim <= 90,
            elev == 0,
        ]).astype(np.float32)
        eval_dict['azim_pred'], _ = func_label_to_azim_elev(probs_to_label(pred_prob, prior=prior))
        eval_dict['azim_pred'] = normalize_angle(eval_dict['azim_pred'])
        if use_weighted:
            azim_support = azim[prior.astype(bool)]
            pred_prob = pred_prob[:, prior.astype(bool)]
            pred_prob = pred_prob / pred_prob.sum(axis=1, keepdims=True)
            eval_dict['azim_pred'] = np.squeeze(pred_prob @ azim_support[:, np.newaxis])
        df = pd.DataFrame({k: v for (k, v) in eval_dict.items() if len(v.shape) < 2}).sort_values(by=['f', 'itd'])
        df['itd'] = np.round(1e6 * df['itd'], decimals=1)
        df = df.groupby('f').agg({'itd': list, 'azim_pred': list}).reset_index()
        
        def restrict_list(dfi, max_itd_diff=None):
            x = np.array(dfi.itd)
            y = np.array(dfi.azim_pred)
            if max_itd_diff is not None:
                IDX = np.abs(x) <= max_itd_diff
                x = x[IDX]
                y = y[IDX]
            dfi.itd = x
            dfi.azim_pred = y
            return dfi
        
        df = df.apply(restrict_list, max_itd_diff=max_itd_diff, axis=1)
        df['itd_threshold'] = df.apply(
            lambda _: float(fit_threshold(_.itd, _.azim_pred)),
            axis=1)
        df['fn_eval'] = fn_eval
        df.to_pickle(fn_results)
        print(f"[WROTE]: {fn_results}")
    return df


def func_to_parallelize_minimum_audible_angle(
        dir_model,
        func_label_to_azim_elev=label_to_azim_elev,
        key_pred_prob='label_loc_int:probs_out',
        key_pred='label_loc_int:labels_pred',
        key_true='label_loc_int:labels_true',
        n_loc_classes=504,
        tag_ckpt=''):
    """
    """
    fn_eval = os.path.join(dir_model, f'EVAL{tag_ckpt}_PSYCHOPHYSICSv00_minimum_audible_angle_interpolated.json')
    df = experiment_minimum_audible_angle(
        fn_eval=fn_eval,
        func_label_to_azim_elev=func_label_to_azim_elev,
        key_pred_prob=key_pred_prob,
        list_azim_ref=np.arange(-75, 76, 5),
        max_deg_diff=10,
        use_weighted=True,
        overwrite=False)
    return df


def func_to_parallelize_itd_threshold(
        dir_model,
        func_label_to_azim_elev=label_to_azim_elev,
        key_pred_prob='label_loc_int:probs_out',
        key_pred='label_loc_int:labels_pred',
        key_true='label_loc_int:labels_true',
        n_loc_classes=504,
        tag_ckpt=''):
    """
    """
    fn_eval = os.path.join(dir_model, f'EVAL{tag_ckpt}_PSYCHOPHYSICSv00_itd_threshold.json')
    df = experiment_itd_threshold(
        fn_eval=fn_eval,
        func_label_to_azim_elev=func_label_to_azim_elev,
        key_pred_prob=key_pred_prob,
        max_itd_diff=None,
        use_weighted=False,
        overwrite=False)
    return df


def func_to_parallelize_itd_ild_weighting(
        dir_model,
        func_label_to_azim_elev=label_to_azim_elev,
        key_pred_prob='label_loc_int:probs_out',
        key_pred='label_loc_int:labels_pred',
        key_true='label_loc_int:labels_true',
        n_loc_classes=504,
        tag_ckpt=''):
    """
    """
    fn_eval = os.path.join(dir_model, f'EVAL{tag_ckpt}_PSYCHOPHYSICSv00_itd_ild_weighting.json')
    df = experiment_itd_ild_weighting(
        fn_eval,
        fold=True,
        prior=None,
        exclude_by_std=False,
        key_pred=key_pred,
        key_true=key_true,
        key_pred_prob=key_pred_prob,
        func_label_to_azim_elev=func_label_to_azim_elev,
        verbose=False)
    return df


def func_to_parallelize_spectral_smoothing(
        dir_model,
        func_label_to_azim_elev=label_to_azim_elev,
        key_pred_prob='label_loc_int:probs_out',
        key_pred='label_loc_int:labels_pred',
        key_true='label_loc_int:labels_true',
        n_loc_classes=504,
        tag_ckpt=''):
    """
    """
    fn_eval = os.path.join(dir_model, f'EVAL{tag_ckpt}_PSYCHOPHYSICSv00_spectral_smoothing.json')
    df = experiment_simple(
        fn_eval=fn_eval,
        fold=True,
        prior=None,
        key_pred=key_pred,
        key_true=key_true,
        key_pred_prob=key_pred_prob,
        func_label_to_azim_elev=func_label_to_azim_elev,
        verbose=False)
    df = df.groupby(['fn_eval', 'smoothed', 'azim_true', 'elev_true']).agg(
        {'azim_pred': 'mean', 'elev_pred': 'mean'}).reset_index()
    X = np.stack([df['azim_true'], df['elev_true']], axis=1)
    Y = np.stack([df['azim_pred'], df['elev_pred']], axis=1)  
    df['deg_err'] = haversine_distance(X, Y, deg=True)
    df['deg_azim_err'] = angular_distance(X[:, 0], Y[:, 0], deg=True)
    df['deg_elev_err'] = angular_distance(X[:, 1], Y[:, 1], deg=True)
    correct_threshold = 0
    df['correct'] = df['deg_err'] <= correct_threshold
    df['correct_azim'] = df['deg_azim_err'] <= correct_threshold
    df['correct_elev'] = df['deg_elev_err'] <= correct_threshold
    return df


def func_to_parallelize_precedence_effect_localization(
        dir_model,
        func_label_to_azim_elev=label_to_azim_elev,
        key_pred_prob='label_loc_int:probs_out',
        key_pred='label_loc_int:labels_pred',
        key_true='label_loc_int:labels_true',
        n_loc_classes=504,
        tag_ckpt=''):
    """
    """
    fn_eval = os.path.join(dir_model, f'EVAL{tag_ckpt}_PSYCHOPHYSICSv00_precedence_effect_localization.json')
    eval_dict = load_eval_dict(fn_eval)
    pred_prob = eval_dict.pop(key_pred_prob)
    azim, elev = func_label_to_azim_elev(np.arange(0, pred_prob.shape[1]))
    azim = normalize_angle(azim)
    prior = np.logical_and.reduce([
        azim >= -90,
        azim <= 90,
        elev == 0,
    ]).astype(np.float32)
    label_pred = probs_to_label(probs=pred_prob, prior=prior)
    eval_dict['azim_pred'], eval_dict['elev_pred'] = func_label_to_azim_elev(label_pred)
    eval_dict['azim_pred'] = normalize_angle(eval_dict['azim_pred'])
    eval_dict['azim_lead'] = normalize_angle(eval_dict['azim_lead'])
    eval_dict['azim_lag'] = normalize_angle(eval_dict['azim_lag'])
    df = pd.DataFrame({k: v for (k, v) in eval_dict.items() if len(v.shape) < 2})
    df['azim_err_lead'] = angular_distance(df['azim_lead'], df['azim_pred'], deg=True)
    df['azim_err_lag'] = angular_distance(df['azim_lag'], df['azim_pred'], deg=True)
    df['fn_eval'] = fn_eval
    return df


def func_to_parallelize_new_ears(
        dir_model,
        func_label_to_azim_elev=label_to_azim_elev,
        key_pred_prob='label_loc_int:probs_out',
        key_pred='label_loc_int:labels_pred',
        key_true='label_loc_int:labels_true',
        n_loc_classes=504,
        tag_ckpt=''):
    """
    """
    fn_eval = os.path.join(dir_model, f'EVAL{tag_ckpt}_PSYCHOPHYSICSv00_new_ears.json')
    df = experiment_simple(
        fn_eval=fn_eval,
        fold=True,
        prior=None,
        key_pred=key_pred,
        key_true=key_true,
        key_pred_prob=key_pred_prob,
        func_label_to_azim_elev=func_label_to_azim_elev,
        verbose=False)
    # Fix azim_true and elev_true (new_ears stimuli are at irregular locations
    # that do not play well with the mapping of azimuths/elevations to labels)
    df['azim_true'] = fold_front_back(np.around(1/5 * df['foreground_azimuth'], decimals=0) * 5)
    df['elev_true'] = np.around(1/10 * df['foreground_elevation'], decimals=0) * 10
    df = df.rename(columns={'subject': 'ears'})
    df['new_ears'] = df['ears'] != 999
    X = np.stack([df['azim_true'], df['elev_true']], axis=1)
    Y = np.stack([df['azim_pred'], df['elev_pred']], axis=1)  
    df['deg_err'] = haversine_distance(X, Y, deg=True)
    df['deg_azim_err'] = angular_distance(X[:, 0], Y[:, 0], deg=True)
    df['deg_elev_err'] = angular_distance(X[:, 1], Y[:, 1], deg=True)
    return df


def func_to_parallelize_bandwidth_dependency(
        dir_model,
        func_label_to_azim_elev=label_to_azim_elev,
        key_pred_prob='label_loc_int:probs_out',
        key_pred='label_loc_int:labels_pred',
        key_true='label_loc_int:labels_true',
        n_loc_classes=504,
        tag_ckpt=''):
    """
    """
    fn_eval = os.path.join(dir_model, f'EVAL{tag_ckpt}_PSYCHOPHYSICSv00_bandwidth_dependency.json')
    azim, elev = func_label_to_azim_elev(np.arange(0, n_loc_classes))
    azim = normalize_angle(azim)
    prior = np.logical_and.reduce([
        azim >= -90,
        azim <= 90,
        elev == 0,
    ]).astype(np.float32)
    df = experiment_simple(
        fn_eval=fn_eval,
        fold=False,
        prior=prior,
        key_pred=key_pred,
        key_true=key_true,
        key_pred_prob=key_pred_prob,
        func_label_to_azim_elev=func_label_to_azim_elev,
        verbose=False)
    X = np.stack([df['azim_true'], df['elev_true']], axis=1)
    Y = np.stack([df['azim_pred'], df['elev_pred']], axis=1)  
    df['deg_err'] = haversine_distance(X, Y, deg=True)
    df['deg_azim_err'] = angular_distance(X[:, 0], Y[:, 0], deg=True)
    df['deg_elev_err'] = angular_distance(X[:, 1], Y[:, 1], deg=True)
    return df


def func_to_parallelize_mp_spectral_cues(
        dir_model,
        func_label_to_azim_elev=label_to_azim_elev,
        key_pred_prob='label_loc_int:probs_out',
        key_pred='label_loc_int:labels_pred',
        key_true='label_loc_int:labels_true',
        n_loc_classes=504,
        tag_ckpt=''):
    """
    """
    fn_eval = os.path.join(dir_model, f'EVAL{tag_ckpt}_PSYCHOPHYSICSv00_mp_spectral_cues.json')
    azim, elev = func_label_to_azim_elev(np.arange(0, n_loc_classes))
    azim = normalize_angle(azim)
    prior = np.logical_and.reduce([
        np.isin(azim, [0, 180]),
    ]).astype(np.float32)
    df = experiment_simple(
        fn_eval=fn_eval,
        fold=False,
        prior=prior,
        key_pred=key_pred,
        key_true=key_true,
        key_pred_prob=key_pred_prob,
        func_label_to_azim_elev=func_label_to_azim_elev,
        verbose=False)
    X = np.stack([df['azim_true'], df['elev_true']], axis=1)
    Y = np.stack([df['azim_pred'], df['elev_pred']], axis=1)  
    df['deg_err'] = haversine_distance(X, Y, deg=True)
    df['deg_azim_err'] = angular_distance(X[:, 0], Y[:, 0], deg=True)
    df['deg_elev_err'] = angular_distance(X[:, 1], Y[:, 1], deg=True)
    df['percent_correct'] = 100 * (df['deg_err'] <= 0)
    df = df.groupby(['fn_eval', 'noise_low', 'noise_high']).agg({'percent_correct': 'mean'}).reset_index()
    df = df.rename(columns={'noise_low': 'cutoff_low', 'noise_high': 'cutoff_high'})
    IDX_LOWPASS = np.logical_and(df['cutoff_low'] == 20, df['cutoff_high'] < 20e3)
    IDX_HIGHPASS = np.logical_and(df['cutoff_low'] > 20, df['cutoff_high'] == 20e3)
    df.loc[IDX_LOWPASS, 'condition'] = 'Low-pass'
    df.loc[IDX_HIGHPASS, 'condition'] = 'High-pass'
    df.loc[IDX_LOWPASS, 'cutoff'] = df.loc[IDX_LOWPASS, 'cutoff_high']
    df.loc[IDX_HIGHPASS, 'cutoff'] = df.loc[IDX_HIGHPASS, 'cutoff_low']
    return df


def func_to_parallelize_snr_dependency(
        dir_model,
        func_label_to_azim_elev=label_to_azim_elev,
        key_pred_prob='label_loc_int:probs_out',
        key_pred='label_loc_int:labels_pred',
        key_true='label_loc_int:labels_true',
        n_loc_classes=504,
        tag_ckpt=''):
    """
    """
    fn_eval = os.path.join(dir_model, f'EVAL{tag_ckpt}_v01_eval_mit_bldg46room1004_tenoise.json')
    azim, elev = func_label_to_azim_elev(np.arange(0, n_loc_classes))
    azim = normalize_angle(azim)
    prior = np.logical_and.reduce([
        azim >= -90,
        azim <= 90,
        azim % 10 == 0,
        elev >= 0,
        elev <= 40,
    ]).astype(np.float32)
    assert prior.sum() == 95
    df = experiment_simple(
        fn_eval=fn_eval,
        fold=False,
        prior=prior,
        key_pred=key_pred,
        key_true=key_true,
        key_pred_prob=key_pred_prob,
        func_label_to_azim_elev=func_label_to_azim_elev,
        verbose=False)
    X = np.stack([df['azim_true'], df['elev_true']], axis=1)
    Y = np.stack([df['azim_pred'], df['elev_pred']], axis=1)  
    df['deg_err'] = haversine_distance(X, Y, deg=True)
    df['deg_azim_err'] = angular_distance(X[:, 0], Y[:, 0], deg=True)
    df['deg_elev_err'] = angular_distance(X[:, 1], Y[:, 1], deg=True)
    return df


def func_to_parallelize_speech_in_noise_in_reverb(
        dir_model,
        func_label_to_azim_elev=label_to_azim_elev,
        key_pred_prob='label_loc_int:probs_out',
        key_pred='label_loc_int:labels_pred',
        key_true='label_loc_int:labels_true',
        n_loc_classes=504,
        tag_ckpt=''):
    """
    """
    fn_eval = os.path.join(dir_model, f'EVAL{tag_ckpt}_PSYCHOPHYSICSv00_speech_in_noise_in_reverb_v04.json')
    azim, elev = func_label_to_azim_elev(np.arange(0, n_loc_classes))
    azim = normalize_angle(azim)
    prior = np.logical_and.reduce([
        azim >= -90,
        azim <= 90,
        azim % 20 == 0,
        elev == 0,
    ]).astype(np.float32)
    assert prior.sum() == 9
    df = experiment_simple(
        fn_eval=fn_eval,
        fold=False,
        prior=prior,
        key_pred=key_pred,
        key_true=key_true,
        key_pred_prob=key_pred_prob,
        func_label_to_azim_elev=func_label_to_azim_elev,
        verbose=False)
    X = np.stack([df['azim_true'], df['elev_true']], axis=1)
    Y = np.stack([df['azim_pred'], df['elev_pred']], axis=1)  
    df['deg_err'] = haversine_distance(X, Y, deg=True)
    df['deg_azim_err'] = angular_distance(X[:, 0], Y[:, 0], deg=True)
    df['deg_elev_err'] = angular_distance(X[:, 1], Y[:, 1], deg=True)
    df['correct'] = np.logical_and.reduce([
        df.azim_true == df.azim_pred,
        df.elev_true == df.elev_pred,
    ])
    return df


def run_localization_experiments(
        list_regex_dir_model,
        list_expt=None,
        workers=20,
        dir_human_data='data/human/sound_localization',
        **kwargs):
    """
    """
    EXPERIMENT_DATAFRAMES = {}
    
    tag_experiment = 'maa_azimuth'
    if (list_expt is None) or (tag_experiment in list_expt):
        df_human = pd.read_csv(os.path.join(dir_human_data, 'mills_1958_minimum_audible_angle.csv'))
        df_human = df_human[df_human['f'] < 1e3]
        df_human['log_maa'] = np.log(df_human['maa'])
        df_human = util.flatten_columns(df_human.rename(columns={
            'f': 'f_ref',
            'azim': 'azim_ref',
        }).groupby([
            'azim_ref',
        ]).agg({
            'maa': ['mean', 'sem', list],
            'log_maa': ['mean', 'sem', list],
        }), sep='_').reset_index()
        df_human['tag_model'] = 'human'
        df_results = [df_human]
        for regex_dir_model in list_regex_dir_model:
            list_dir_model = glob.glob(regex_dir_model)
            with multiprocessing.Pool(min(workers, len(list_dir_model))) as p:
                func = functools.partial(func_to_parallelize_minimum_audible_angle, **kwargs)
                list_df = p.map(func, list_dir_model)
            df = pd.concat(list_df)
            df = df[df['f_ref'] < 1e3]
            df['maa'] = df['maa'].clip(0, 90)
            df['log_maa'] = np.log(df['maa'].clip(0, 90))
            df['azim_ref'] = np.abs(df['azim_ref'])
            df = df.groupby([
                'fn_eval',
                'azim_ref',
            ]).agg({
                'maa': 'mean',
                'log_maa': 'mean',
            }).reset_index()  
            df = util.flatten_columns(df.groupby([
                'azim_ref',
            ]).agg({
                'maa': ['mean', 'sem', list],
                'log_maa': ['mean', 'sem', list],
            }), sep='_').reset_index()
            df['tag_model'] = regex_dir_model
            df_results.append(df)
            print(f'Completed experiment `{tag_experiment}` for {len(list_dir_model)} `{regex_dir_model}` models')
        df_results = pd.concat(df_results).reset_index(drop=True)
        EXPERIMENT_DATAFRAMES[tag_experiment] = df_results.sort_index(axis=1)
    
    tag_experiment = 'itd_threshold'
    if (list_expt is None) or (tag_experiment in list_expt):
        df_human = pd.read_csv(os.path.join(dir_human_data, 'brughera_etal_2013_itd_threshold.csv'))
        df_human = df_human.rename(columns={'subject': 'fn_eval'})
        df_human['log_itd_threshold'] = np.log(df_human['itd_threshold'])
        df_human = util.flatten_columns(df_human.groupby('f').agg({
            'itd_threshold': ['mean', 'sem', list],
            'log_itd_threshold': ['mean', 'sem', list],
        }), sep='_').reset_index()
        df_human['tag_model'] = 'human'
        df_results = [df_human]
        for regex_dir_model in list_regex_dir_model:
            list_dir_model = glob.glob(regex_dir_model)
            with multiprocessing.Pool(min(workers, len(list_dir_model))) as p:
                func = functools.partial(func_to_parallelize_itd_threshold, **kwargs)
                list_df = p.map(func, list_dir_model)
            df = pd.concat(list_df)
            df = df[df.f <= 3200]
            df['itd_threshold'] = df['itd_threshold'].clip(0, 320)
            df['log_itd_threshold'] = np.log(df['itd_threshold'].clip(0, 320))
            df = df.groupby(['fn_eval', 'f']).agg({
                'itd_threshold': 'mean',
                'log_itd_threshold': 'mean',
            }).reset_index()
            df = util.flatten_columns(df.groupby('f').agg({
                'itd_threshold': ['mean', 'sem', list],
                'log_itd_threshold': ['mean', 'sem', list],
            }), sep='_').reset_index()
            df['tag_model'] = regex_dir_model
            df_results.append(df)
            print(f'Completed experiment `{tag_experiment}` for {len(list_dir_model)} `{regex_dir_model}` models')
        df_results = pd.concat(df_results).reset_index(drop=True)
        EXPERIMENT_DATAFRAMES[tag_experiment] = df_results.sort_index(axis=1)
    
    tag_experiment = 'itd_ild_weighting'
    if (list_expt is None) or (tag_experiment in list_expt):
        df_human = pd.read_csv(os.path.join(dir_human_data, 'macpherson_middlebrooks_2002_itd_ild_weighting.csv'))
        df_human = df_human.rename(columns={'subject': 'fn_eval'})
        list_df_human = []
        for bias in [-600, -300, 0, 300, 600]:
            list_df_human.append(df_human.assign(itd_bias_imposed=bias))
        for bias in [-20, -10, 0, 10, 20]:
            list_df_human.append(df_human.assign(ild_bias_imposed=bias))
        df_human = pd.concat(list_df_human)
        df_human['tag_model'] = 'human'
        df_results = [df_human]
        for regex_dir_model in list_regex_dir_model:
            list_dir_model = glob.glob(regex_dir_model)
            with multiprocessing.Pool(min(workers, len(list_dir_model))) as p:
                func = functools.partial(func_to_parallelize_itd_ild_weighting, **kwargs)
                list_df = p.map(func, list_dir_model)
            df = pd.concat(list_df)
            df['tag_model'] = regex_dir_model
            df_results.append(df)
            print(f'Completed experiment `{tag_experiment}` for {len(list_dir_model)} `{regex_dir_model}` models')
        df_results = pd.concat(df_results).reset_index(drop=True)
        df_results = df_results.groupby(['tag_model', 'band', 'fn_eval']).agg({
            'itd_bias_weight': 'mean',
            'ild_bias_weight': 'mean',
            'itd_bias_imposed': 'unique',
            'ild_bias_imposed': 'unique',
        }).reset_index()
        df_results['itd_bias_imposed'] = df_results['itd_bias_imposed'].map(np.unique)
        df_results['ild_bias_imposed'] = df_results['ild_bias_imposed'].map(np.unique)
        df_results['itd_bias_response'] = df_results['itd_bias_imposed'] * df_results['itd_bias_weight']
        df_results['ild_bias_response'] = df_results['ild_bias_imposed'] * df_results['ild_bias_weight']
        df_results = util.flatten_columns(df_results.groupby(['tag_model', 'band']).agg({
            'itd_bias_weight': [list, 'mean', 'sem'],
            'ild_bias_weight': [list, 'mean', 'sem'],
            'itd_bias_imposed': list,
            'ild_bias_imposed': list,
            'itd_bias_response': list,
            'ild_bias_response': list,
            'fn_eval': list,
        }).reset_index(), sep='_')
        EXPERIMENT_DATAFRAMES[tag_experiment] = df_results.sort_index(axis=1)
    
    tag_experiment = 'spectral_smoothing'
    if (list_expt is None) or (tag_experiment in list_expt):
        list_key_metric = [
            'pct_correct',
            'deg_err',
            'deg_azim_err',
            'deg_elev_err',
        ]
        df_human = pd.read_csv(os.path.join(dir_human_data, 'kulkarni_colburn_1998_spectral_smoothing.csv'))
        df_human = util.flatten_columns(df_human.rename(columns={'subject': 'fn_eval'}).groupby(
            ['smoothed', 'fn_eval']).agg({'pct_correct': 'mean'}).reset_index().groupby('smoothed').agg({
            'pct_correct': ['mean', 'sem', list],
            'fn_eval': list,
        }).reset_index(), sep='_')
        df_human['tag_model'] = 'human'
        df_results = [df_human]
        for regex_dir_model in list_regex_dir_model:
            list_dir_model = glob.glob(regex_dir_model)
            with multiprocessing.Pool(min(workers, len(list_dir_model))) as p:
                func = functools.partial(func_to_parallelize_spectral_smoothing, **kwargs)
                list_df = p.map(func, list_dir_model)
            df = pd.concat(list_df)
            df = df.groupby(['fn_eval', 'smoothed']).agg({
                key_metric: 'mean' for key_metric in list_key_metric if key_metric in df.columns
            }).reset_index()
            list_fn_eval = df.groupby(['smoothed']).agg({'fn_eval': list})['fn_eval'].values
            df = df.groupby(['smoothed']).agg({
                key_metric: ['mean', 'sem', list] for key_metric in list_key_metric if key_metric in df.columns
            }).reset_index()
            df[('fn_eval', 'list')] = list_fn_eval
            df['tag_model'] = regex_dir_model
            df = util.flatten_columns(df, sep='_')
            df_results.append(df)
            print(f'Completed experiment `{tag_experiment}` for {len(list_dir_model)} `{regex_dir_model}` models')
        df_results = pd.concat(df_results).reset_index(drop=True)
        EXPERIMENT_DATAFRAMES[tag_experiment] = df_results.sort_index(axis=1)
    
    tag_experiment = 'precedence_effect_localization'
    if (list_expt is None) or (tag_experiment in list_expt):
        df_human = pd.read_csv(os.path.join(dir_human_data, 'litovsky_godar_2010_precedence_effect.csv'))
        df_lead = df_human[df_human.condition == 0].rename(columns={
            'rms_error_mean': 'azim_err_lead_mean',
            'rms_error_std': 'azim_err_lead_sem',
        }).drop(columns=['condition'])
        df_lag = df_human[df_human.condition == 1].rename(columns={
            'rms_error_mean': 'azim_err_lag_mean',
            'rms_error_std': 'azim_err_lag_sem',
        }).drop(columns=['condition'])
        df_human = df_lead.merge(df_lag, on='delay')
        df_human['delay'] = df_human['delay'] / 1e3
        df_human['tag_model'] = 'human'
        df_results = [df_human]
        for regex_dir_model in list_regex_dir_model:
            list_dir_model = glob.glob(regex_dir_model)
            with multiprocessing.Pool(min(workers, len(list_dir_model))) as p:
                func = functools.partial(func_to_parallelize_precedence_effect_localization, **kwargs)
                list_df = p.map(func, list_dir_model)
            df = pd.concat(list_df)
            df = df[np.logical_and.reduce([
                df.delay != 0,
                df.azim_lag.isin([-60, -40, -20, 0, 20, 40, 60]),
                df.azim_lead.isin([-60, -40, -20, 0, 20, 40, 60]),
                np.logical_xor(df.azim_lead == 0, df.azim_lag == 0),
            ])]
            df = df.groupby(['fn_eval', 'delay']).agg({
                'azim_err_lead': lambda _: np.sqrt(np.mean(np.square(_))),
                'azim_err_lag': lambda _: np.sqrt(np.mean(np.square(_))),
            }).reset_index()
            df = util.flatten_columns(df.groupby(['delay']).agg({
                'azim_err_lead': ['mean', 'sem', list],
                'azim_err_lag': ['mean', 'sem', list],
            }), sep='_').reset_index()
            df['tag_model'] = regex_dir_model
            IDX_LEAD_ONLY = df.delay == np.inf
            IDX_LAG_ONLY = df.delay == -np.inf
            LIST_LAG_COL = ['azim_err_lag_mean', 'azim_err_lag_sem', 'azim_err_lag_list']
            for LAG_COL in LIST_LAG_COL:
                df.loc[IDX_LEAD_ONLY, LAG_COL] = df[LAG_COL].values[IDX_LAG_ONLY]
            df.loc[IDX_LEAD_ONLY, 'delay'] = 0.0
            df = df[~IDX_LAG_ONLY].sort_values(by='delay')
            df_results.append(df)
            print(f'Completed experiment `{tag_experiment}` for {len(list_dir_model)} `{regex_dir_model}` models')
        df_results = pd.concat(df_results).reset_index(drop=True)
        df_results['delay'] = np.round(df_results['delay'], decimals=3)
        df_results = df_results[df_results['delay'].isin(df_human.delay)].reset_index(drop=True)
        EXPERIMENT_DATAFRAMES[tag_experiment] = df_results.sort_index(axis=1)
    
    tag_experiment = 'new_ears'
    if (list_expt is None) or (tag_experiment in list_expt):
        df_human = pd.read_csv(os.path.join(dir_human_data, 'hofman_etal_1998_new_ears.csv'))
        df_human = df_human.rename(columns={'subject': 'fn_eval'})
        df_human['new_ears'] = df_human['new_ears'].astype(bool)
        df_human.loc[df_human['new_ears'] == True, 'ears'] = 0
        df_human.loc[df_human['new_ears'] == False, 'ears'] = 999
        X = np.stack([df_human['azim_true'], df_human['elev_true']], axis=1)
        Y = np.stack([df_human['azim_pred'], df_human['elev_pred']], axis=1)
        df_human['deg_err'] = haversine_distance(X, Y, deg=True)
        df_human['deg_azim_err'] = angular_distance(X[:, 0], Y[:, 0], deg=True)
        df_human['deg_elev_err'] = angular_distance(X[:, 1], Y[:, 1], deg=True)
        df_human['tag_model'] = 'human'
        df_results = [df_human]
        for regex_dir_model in list_regex_dir_model:
            list_dir_model = glob.glob(regex_dir_model)
            with multiprocessing.Pool(min(workers, len(list_dir_model))) as p:
                func = functools.partial(func_to_parallelize_new_ears, **kwargs)
                list_df = p.map(func, list_dir_model)
            df = pd.concat(list_df)
            df = df[np.logical_and.reduce([
                df.elev_true.isin([0, 10, 20, 30]),
                df.azim_true.isin([-30, -10, 10, 30]),
            ])]
            df['tag_model'] = regex_dir_model
            df_results.append(df)
            print(f'Completed experiment `{tag_experiment}` for {len(list_dir_model)} `{regex_dir_model}` models')
        df_results = pd.concat(df_results).reset_index(drop=True)
        df_results = df_results.groupby(['tag_model', 'fn_eval', 'new_ears', 'ears', 'azim_true', 'elev_true']).agg({
            'azim_pred': 'mean',
            'elev_pred': 'mean',
            'deg_err': 'mean',
            'deg_azim_err': 'mean',
            'deg_elev_err': 'mean',
        }).reset_index()
        df_results['azim_pred_arr'] = df_results['azim_pred']
        df_results['elev_pred_arr'] = df_results['elev_pred']
        df_results = df_results.groupby(['tag_model', 'fn_eval', 'new_ears', 'azim_true', 'elev_true']).agg({
            'azim_pred_arr': list,
            'elev_pred_arr': list,
            'azim_pred': 'mean',
            'elev_pred': 'mean',
            'deg_err': 'mean',
            'deg_azim_err': 'mean',
            'deg_elev_err': 'mean',
            'ears': list,
        }).reset_index().groupby(['tag_model', 'new_ears', 'azim_true', 'elev_true']).agg({
            'azim_pred_arr': list,
            'elev_pred_arr': list,
            'azim_pred': ['mean', 'sem', list],
            'elev_pred': ['mean', 'sem', list],
            'deg_err': ['mean', 'sem', list],
            'deg_azim_err': ['mean', 'sem', list],
            'deg_elev_err': ['mean', 'sem', list],
            'fn_eval': list,
            'ears': 'first'
        }).reset_index()
        df_results = util.flatten_columns(df_results, sep='_')
        df_results = df_results.rename(columns={'azim_pred_arr_list': 'azim_pred_arr', 'elev_pred_arr_list': 'elev_pred_arr'})
        EXPERIMENT_DATAFRAMES[tag_experiment] = df_results.sort_index(axis=1)
    
    tag_experiment = 'bandwidth_dependency'
    if (list_expt is None) or (tag_experiment in list_expt):
        df_human = pd.read_csv(os.path.join(dir_human_data, 'yost_zhong_2014_bandwidth_dependency.csv'))
        list_f_ref = np.unique(df_human.cf)
        df_human = df_human.rename(columns={
            'cf': 'f_ref',
            'rms_error_std': 'rms_error_sem',
            'rms_error': 'rms_error_mean',
        }).groupby(['bandwidth']).agg({
            'rms_error_mean': 'mean',
            'rms_error_sem': 'mean',
        }).reset_index()
        df_human['tag_model'] = 'human'
        df_results = [df_human]
        for regex_dir_model in list_regex_dir_model:
            list_dir_model = glob.glob(regex_dir_model)
            with multiprocessing.Pool(min(workers, len(list_dir_model))) as p:
                func = functools.partial(func_to_parallelize_bandwidth_dependency, **kwargs)
                list_df = p.map(func, list_dir_model)
            df = pd.concat(list_df)
            df = df[df.f_ref.isin(list_f_ref)]
            df = df.groupby(['fn_eval', 'bandwidth']).agg({
                'deg_azim_err': lambda _: np.sqrt(np.mean(np.square(_))),
            }).reset_index().rename(columns={'deg_azim_err': 'rms_error'})
            df = util.flatten_columns(df.groupby(['bandwidth']).agg({
                'rms_error': ['mean', 'sem', list],
            }).reset_index(), sep='_')
            df['tag_model'] = regex_dir_model        
            df_results.append(df)
            print(f'Completed experiment `{tag_experiment}` for {len(list_dir_model)} `{regex_dir_model}` models')
        df_results = pd.concat(df_results).reset_index(drop=True)
        EXPERIMENT_DATAFRAMES[tag_experiment] = df_results.sort_index(axis=1)
    
    tag_experiment = 'mp_spectral_cues'
    if (list_expt is None) or (tag_experiment in list_expt):
        df_human = pd.read_csv(os.path.join(dir_human_data, 'hebrank_wright_1974_median_plane_spectral_cues.csv'))
        df_human['condition'] = df_human['condition'].map({0: 'Low-pass', 1: 'High-pass'})
        df_human['cutoff'] = 1e3 * df_human['cutoff']
        df_human['percent_correct'] = df_human['pct_correct_15deg']
        df_human['fn_eval'] = 'human'
        df_human['tag_model'] = 'human'
        df_results = [df_human]
        for regex_dir_model in list_regex_dir_model:
            list_dir_model = glob.glob(regex_dir_model)
            with multiprocessing.Pool(min(workers, len(list_dir_model))) as p:
                func = functools.partial(func_to_parallelize_mp_spectral_cues, **kwargs)
                list_df = p.map(func, list_dir_model)
            df = pd.concat(list_df)
            df['tag_model'] = regex_dir_model        
            df_results.append(df)
            print(f'Completed experiment `{tag_experiment}` for {len(list_dir_model)} `{regex_dir_model}` models')
        df_results = pd.concat(df_results).reset_index(drop=True)
        df_results = util.flatten_columns(df_results.groupby(
            ['tag_model', 'fn_eval', 'condition', 'cutoff']).agg(
            {'percent_correct': 'mean'}).reset_index().groupby(
            ['tag_model', 'condition', 'cutoff']).agg(
            {'percent_correct': ['mean', 'sem', list]}).reset_index(), sep='_')
        EXPERIMENT_DATAFRAMES[tag_experiment] = df_results.sort_index(axis=1)
    
    tag_experiment = 'snr_dependency'
    if (list_expt is None) or (tag_experiment in list_expt):
        df_human = pd.read_csv(os.path.join(dir_human_data, 'saddler_2023_snr.csv'))
        df_human['tag_model'] = 'human'
        df_results = [df_human]
        for regex_dir_model in list_regex_dir_model:
            list_dir_model = glob.glob(regex_dir_model)
            with multiprocessing.Pool(min(workers, len(list_dir_model))) as p:
                func = functools.partial(func_to_parallelize_snr_dependency, **kwargs)
                list_df = p.map(func, list_dir_model)
            df = pd.concat(list_df)
            df['tag_model'] = regex_dir_model        
            df_results.append(df)
            print(f'Completed experiment `{tag_experiment}` for {len(list_dir_model)} `{regex_dir_model}` models')
        df_results = pd.concat(df_results).reset_index(drop=True)
        list_key_metric = [
            'deg_err',
            'deg_azim_err',
            'deg_elev_err',
        ]
        df_results = df_results.groupby(['tag_model', 'fn_eval', 'snr']).agg({
            key_metric: 'mean' for key_metric in list_key_metric
        }).reset_index()
        df_results = util.flatten_columns(df_results.groupby(['tag_model', 'snr']).agg({
            key_metric: ['mean', 'sem', list] for key_metric in list_key_metric
        }).reset_index(), sep='_')
        df_results['snr'] = df_results['snr'].map(lambda _: np.round(_, decimals=1))
        EXPERIMENT_DATAFRAMES[tag_experiment] = df_results.sort_index(axis=1)
    
    tag_experiment = 'speech_in_noise_in_reverb'
    if (list_expt is None) or (tag_experiment in list_expt):
        df_results = []
        for regex_dir_model in list_regex_dir_model:
            list_dir_model = glob.glob(regex_dir_model)
            with multiprocessing.Pool(min(workers, len(list_dir_model))) as p:
                func = functools.partial(func_to_parallelize_speech_in_noise_in_reverb, **kwargs)
                list_df = p.map(func, list_dir_model)
            df = pd.concat(list_df)
            df['tag_model'] = regex_dir_model        
            df_results.append(df)
            print(f'Completed experiment `{tag_experiment}` for {len(list_dir_model)} `{regex_dir_model}` models')
        df_results = pd.concat(df_results).reset_index(drop=True)
        list_key_metric = [
            'deg_err',
            'deg_azim_err',
            'deg_elev_err',
            'correct',
        ]
        df_results = df_results.groupby(['tag_model', 'fn_eval', 'index_room', 'snr']).agg({
            key_metric: 'mean' for key_metric in list_key_metric
        }).reset_index()
        dict_agg = {key_metric: ['mean', 'sem', list] for key_metric in list_key_metric}
        dict_agg['fn_eval'] = list
        df_results = util.flatten_columns(df_results.groupby(['tag_model', 'index_room', 'snr']).agg(dict_agg).reset_index(), sep='_')
        EXPERIMENT_DATAFRAMES[tag_experiment] = df_results.sort_index(axis=1)
    
    return EXPERIMENT_DATAFRAMES


def func_to_parallelize_bootstrap(
        bootstrap_IDX,
        y_human=None,
        list_y_model=None,
        metric_function=None):
    """
    """
    y_model = list_y_model[bootstrap_IDX].mean(axis=0)
    metric = np.array(metric_function(y_human, y_model)).reshape([-1])[0]
    if np.isnan(metric):
        metric = 0
    return metric


def compare_localization_experiment(
        df,
        tag_expt,
        random_seed=0,
        bootstrap_repeats=1000,
        dict_metric_function={'metric': scipy.stats.pearsonr},
        workers=20):
    """
    """
    df = df.copy()
    df_results = []
    for tag_model in sorted(set(df['tag_model'].unique()).difference({'human'})):
        if tag_expt == 'maa_azimuth':
            list_key_x = ['azim_ref']
            list_key_y_human = ['maa']
            list_key_y_model = ['maa']
            interp = True
        elif tag_expt == 'itd_threshold':
            list_key_x = ['f']
            list_key_y_human = ['log_itd_threshold']
            list_key_y_model = ['log_itd_threshold']
            interp = True
        elif tag_expt == 'itd_ild_weighting':
            list_key_x = ['band']
            list_key_y_human = ['itd_bias_weight', 'ild_bias_weight']
            list_key_y_model = ['itd_bias_weight', 'ild_bias_weight']
            interp = False
            df = df[df['band'].isin(['(4000, 16000)', '(500, 2000)'])]
        elif tag_expt == 'spectral_smoothing':
            list_key_x = ['smoothed']
            list_key_y_human = ['pct_correct']
            list_key_y_model = ['deg_err']
            interp = False
        elif tag_expt == 'precedence_effect_localization':
            list_key_x = ['delay']
            list_key_y_human = ['azim_err_lead', 'azim_err_lag']
            list_key_y_model = ['azim_err_lead', 'azim_err_lag']
            interp = False
        elif tag_expt == 'new_ears':
            list_key_x = ['new_ears', 'grid_position']
            list_key_y_human = ['deg_azim_err', 'deg_elev_err']
            list_key_y_model = ['deg_azim_err', 'deg_elev_err']
            interp = False
            def grid_position(df):
                df = df.sort_values(by=['azim_true', 'elev_true'])
                df['grid_position'] = np.arange(0, len(df))
                return df
            df = df.groupby(['tag_model', 'new_ears'], group_keys=False).apply(grid_position).reset_index(drop=True)
        elif tag_expt == 'bandwidth_dependency':
            list_key_x = ['bandwidth']
            list_key_y_human = ['rms_error']
            list_key_y_model = ['rms_error']
            interp = False
            df['bandwidth'] = df['bandwidth'].astype(np.float32)
        elif tag_expt == 'mp_spectral_cues':
            list_key_x = ['condition', 'cutoff']
            list_key_y_human = ['percent_correct']
            list_key_y_model = ['percent_correct']
            interp = False
        elif tag_expt == 'snr_dependency':
            list_key_x = ['snr']
            list_key_y_human = ['deg_err', 'deg_azim_err', 'deg_elev_err']
            list_key_y_model = ['deg_err', 'deg_azim_err', 'deg_elev_err']
            interp = True
            df = df[np.isfinite(df.snr)]
        else:
            print(f"{tag_expt=}` not recognized --> returning None")
            return None
        df_human = df[df['tag_model'] == 'human']
        df_model = df[df['tag_model'] == tag_model]
        df_comparison = pd.merge(
            df_human,
            df_model,
            on=list_key_x,
            how='outer' if interp else 'inner',
            suffixes=('_human', '_model')).sort_values(by=list_key_x)
        y_human = []
        y_model = []
        list_y_model = []
        for key_y_human, key_y_model in zip(list_key_y_human, list_key_y_model):
            yh = df_comparison[f'{key_y_human}_mean_human'].values
            ym = df_comparison[f'{key_y_model}_mean_model'].values
            if interp:
                assert len(list_key_x) == 1, "Can only interpolate 1D experiments"
                x = df_comparison[list_key_x[0]].values
                yh = np.interp(x, x[~np.isnan(yh)], yh[~np.isnan(yh)])
                IDX = np.logical_and.reduce([~np.isnan(x), ~np.isnan(yh), ~np.isnan(ym)])
                x = x[IDX]
                yh = yh[IDX]
                ym = ym[IDX]
            if interp:
                list_ym = np.array(list(df_comparison[f'{key_y_model}_list_model'].values[IDX])).T
            else:
                list_ym = np.array(list(df_comparison[f'{key_y_model}_list_model'].values)).T
            y_human.append(yh)
            y_model.append(ym)
            list_y_model.append(list_ym)
        y_human = np.concatenate(y_human)
        y_model = np.concatenate(y_model)
        list_y_model = np.concatenate(list_y_model, axis=-1)
        metric = {}
        for metric_key, metric_function in dict_metric_function.items():
            metric_val = np.array(metric_function(y_human, y_model)).reshape([-1])[0]
            if np.isnan(metric_val):
                metric_val = 0
            metric[metric_key] = metric_val
        np.random.seed(random_seed)
        list_bootstrap_IDX = []
        for _ in range(bootstrap_repeats):
            list_bootstrap_IDX.append(np.random.choice(
                list(range(list_y_model.shape[0])),
                size=[list_y_model.shape[0]],
                replace=True))
        bootstrap_list_metric = {}
        individual_list_metric = {}
        for metric_key, metric_function in dict_metric_function.items():
            func = functools.partial(
                func_to_parallelize_bootstrap,
                y_human=y_human,
                list_y_model=list_y_model,
                metric_function=metric_function)
            if workers is None:
                bootstrap_list_metric_val = [func(_) for _ in list_bootstrap_IDX]
                individual_list_metric_val = [func([_]) for _ in range(len(list_y_model))]
            else:
                with multiprocessing.Pool(workers) as pool:
                    bootstrap_list_metric_val = pool.map(func, list_bootstrap_IDX)
                    individual_list_metric_val = pool.map(func, [[_] for _ in range(len(list_y_model))])
            bootstrap_list_metric[metric_key] = bootstrap_list_metric_val
            individual_list_metric[metric_key] = individual_list_metric_val
        to_append = {'tag_expt': tag_expt, 'tag_model': tag_model}
        for metric_key in dict_metric_function.keys():
            to_append[metric_key] = metric[metric_key]
            to_append[f'bootstrap_list_{metric_key}'] = bootstrap_list_metric[metric_key]
            to_append[f'list_{metric_key}'] = individual_list_metric[metric_key]
        df_results.append(to_append)
    df_results = pd.DataFrame(df_results)
    return df_results
