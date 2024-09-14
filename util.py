import os
import sys
import json
import h5py
import copy
import collections
import numpy as np
import pandas as pd
import scipy.stats
import textwrap

import util_figures


def get_color_and_label_from_model_tag(model_tag):
    """
    """
    if "human" in model_tag.lower():
        color = "k"
        label = "Human listeners"
    elif "ihc3000" in model_tag.lower():
        color = "#808088"
        label = "3000 Hz IHC filter"
    elif "ihc1000" in model_tag.lower():
        color = "#28C8C8"
        label = "1000 Hz IHC filter"
    elif "ihc0320" in model_tag.lower():
        color = "#8856a7"
        label = "320 Hz IHC filter"
    elif "ihc0050" in model_tag.lower():
        color = "#F03C8C"
        label = "50 Hz IHC filter"
    else:
        color = None
        label = os.path.basename(model_tag)
    return color, label


def wrap_xticklabels(ax, width, break_long_words=False, **kwargs):
    """
    """
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        labels.append(
            textwrap.fill(
                text,
                width=width,
                break_long_words=break_long_words))
    ax.set_xticklabels(labels, **kwargs)
    return ax


def normalized_rmse(y_human, y_model, min_human=None, max_human=None):
    """
    Compute root-mean-squared error between human and
    model results, min/max normalized by human results.
    """
    if min_human is None:
        min_human = np.min(y_human)
    if max_human is None:
        max_human = np.max(y_human)
    y_human = (y_human - min_human) / (max_human - min_human)
    y_model = (y_model - min_human) / (max_human - min_human)
    return np.linalg.norm(y_human - y_model) / np.sqrt(y_human.shape[0])


def get_p_val(y_null, y):
    """
    Estimate two-tailed p-value of obtaining a value of `y`
    (or more extreme) under the distribution `y_null`
    """
    null_dist = scipy.stats.norm(loc=np.mean(y_null), scale=np.std(y_null))
    p = null_dist.cdf(y)
    if p > 0.5:
        p = 1 - p
    p = p * 2
    return p


def cohend(x, y):
    """
    """
    nx = len(x)
    ny = len(y)
    vx = np.var(x)
    vy = np.var(y)
    s = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    return (np.mean(y) - np.mean(x)) / s


def get_aggregate_measure(
    fn_data_localization="data/model/sound_localization.pkl",
    fn_data_spkr_word="data/model/spkr_word_recognition.pkl",
    fn_comparison_localization="data/model/sound_localization_human_model_comparison_metrics.pkl",
    fn_comparison_spkr_word="data/model/spkr_word_recognition_human_model_comparison_metrics.pkl",
    list_tag_model=[],
    key_task="localization",
    key_metric="performance",
    bootstrap_repeats=1000,
    random_seed=0,
):
    """
    Helper function to get aggregate measure values and bootstrap distribution from data files.
    Given a specified `key_task` ("localization", "spkr", or "word") and a `key_metric`
    ("performance" for overall task performance in noise or "pearsonr" / "rmse" for overall
    human-model similarity), this function will return the aggregate measure values for each
    model specified in `list_tag_model` (mean values, individual values, and bootstrapped
    distribution).
    """
    dict_tag_expt = {
        "localization": {
            "snr_dependency": "Sound localization in noise",
            "itd_ild_weighting": "ITD / ILD cue weighting (Macpherson & Middlebrooks, 2002)",
            "maa_azimuth": "Minimum audible angle vs. azimuth (Mills, 1958)",
            "itd_threshold": "ITD lateralization vs. frequency (Brughera et al., 2013)",
            "new_ears": "Effect of changing ears (Hofman et al., 1998)",
            "spectral_smoothing": "Effect of smoothing spectral cues (Kulkarni & Colburn, 1998)",
            "mp_spectral_cues": "Median plane spectral cues (Hebrank & Wright, 1974)",
            "precedence_effect_localization": "Precedence effect (Litovsky & Godar, 2010)",
            "bandwidth_dependency": "Bandwidth dependency (Yost & Zhong, 2014)",
        },
        "spkr": {
            "pitch_altered_spkr": "Voice recognition with pitch-altered speech",
        },
        "word": {
            "kell_like_word": "Word recognition as a function of SNR and noise condition",
            "speech_in_synthetic_textures": "Word recognition in 43 distinct auditory textures",
            "pitch_altered_word": "Word recognition with pitch-altered speech",
            "hopkins_moore_2009_word": "Effect of tone vocoding on word recognition in noise (Hopkins & Moore, 2009)",
        },
    }
    key_task_accepted = list(dict_tag_expt.keys())
    key_metric_accepted = ["pearsonr", "rmse", "performance"]
    assert key_task in key_task_accepted, f"{key_task=} must be one of {key_task_accepted}"
    assert key_metric in key_metric_accepted, f"{key_metric=} must be one of {key_metric_accepted}"
    list_tag_expt = list(dict_tag_expt[key_task].keys())
    if key_metric in ["pearsonr", "rmse"]:
        if "localization" in key_task:
            df = pd.read_pickle(fn_comparison_localization)
        else:
            df = pd.read_pickle(fn_comparison_spkr_word)
        df = df[np.logical_and.reduce([
            df.tag_model.isin(list_tag_model),
            df.tag_expt.isin(list_tag_expt),
        ])]
        assert df.tag_expt.nunique() == len(list_tag_expt)
        df = average_comparison_metrics(df)
    else:
        if "localization" in key_task:
            df = pd.read_pickle(fn_data_localization)["snr_dependency"]
            df = df[np.logical_and.reduce([
                df.tag_model.isin(list_tag_model),
                np.isfinite(df.snr),
            ])]
            key_metric = "deg_err"
        else:
            df = pd.read_pickle(fn_data_spkr_word)["kell_like"]
            df = df[np.logical_and.reduce([
                df.tag_model.isin(list_tag_model),
                df.snr.isin([-9, -6, -3, 0, 3]),
                df.background_condition.isin([0, 1, 2, 3]),
            ])]
            key_metric = f"correct_{key_task}"
        df = df.rename(columns={f"{key_metric}_list": f"list_{key_metric}"})
        df = df.groupby(["tag_model"]).agg({f"list_{key_metric}": list}).reset_index()
        df[f"list_{key_metric}"] = df[f"list_{key_metric}"].map(lambda _: np.array(_).mean(axis=0))
        if "correct" in key_metric:
            df[f"list_{key_metric}"] = df[f"list_{key_metric}"].map(lambda _: 100 * np.array(_))
        df[f"{key_metric}"] = df[f"list_{key_metric}"].map(lambda _: np.mean(_))
        df[f"{key_metric}_sem"] = df[f"list_{key_metric}"].map(lambda _: np.std(_) / np.sqrt(len(_)))
        np.random.seed(random_seed)
        df[f"bootstrap_list_{key_metric}"] = df[f"list_{key_metric}"].map(
            lambda _: np.random.choice(_, size=(bootstrap_repeats, len(_))).mean(axis=1))
    y = []
    list_y = []
    bootstrap_list_y = []
    for tag_model in list_tag_model:
        dfi = df[df.tag_model == tag_model]
        assert len(dfi) == 1, f"{tag_model=} --> dataframe of length {len(dfi)} (expected 1)"
        dfi = dfi.iloc[0]
        y.append(dfi[key_metric])
        list_y.append(dfi[f"list_{key_metric}"])
        bootstrap_list_y.append(dfi[f"bootstrap_list_{key_metric}"])
    return np.array(y), np.array(list_y), np.array(bootstrap_list_y)


def average_comparison_metrics(df):
    """
    """
    assert "AVERAGE" not in df.tag_expt.unique()
    list_k = [k.replace("bootstrap_list_", "") for k in df.columns if "bootstrap_list_" in k]
    dict_agg = {}
    for k in list_k:
        dict_agg[k] = "mean"
        dict_agg[f"bootstrap_list_{k}"] = list
        dict_agg[f"list_{k}"] = list
    df_mean = df.groupby(["tag_model"]).agg(dict_agg).reset_index()
    for k in list_k:
        df_mean[f"bootstrap_list_{k}"] = df_mean[f"bootstrap_list_{k}"].map(lambda _: np.array(list(_)).mean(axis=0))
        df_mean[f"list_{k}"] = df_mean[f"list_{k}"].map(lambda _: np.array(list(_)).mean(axis=0))
    df_mean["tag_expt"] = "AVERAGE"
    return df_mean


def make_plot_comparison_metrics(
        ax,
        df,
        key_metric,
        list_tag_model,
        include_line=True,
        include_legend=False,
        kwargs_legend_update={},
        kwargs_format_axes_update={},
        min_ylim_diff=0.5):
    """
    """
    list_x = []
    list_y = []
    xticks = []
    xticklabels = []
    for x, tag_model in enumerate(list_tag_model):
        dfi = df[df["tag_model"] == tag_model]
        assert len(dfi) == 1
        dfi = dfi.iloc[0]
        color, label = get_color_and_label_from_model_tag(tag_model)
        label = label.replace(" Hz IHC filter", "")
        if "delayed" in tag_model:
            facecolor = "orange"
            label = label + " (delayed)" if include_legend else label + "\n" + r"$^{\text{(delayed)}}$"
        else:
            facecolor = color
        if "sr20000" in tag_model:
            color = "green"
        parts = ax.violinplot(
            dfi[f"bootstrap_list_{key_metric}"],
            positions=[x],
            showmeans=False,
            showextrema=False)
        for k in parts.keys():
            if not k == "bodies":
                parts[k].set_color(color)
                parts[k].set_linewidth(2)
        for pc in parts["bodies"]:
            pc.set_facecolor(facecolor)
            pc.set_edgecolor(color)
            pc.set_linewidth(1.5)
            pc.set_alpha(0.75)
        ax.plot(
            x,
            dfi[key_metric],
            color=color,
            marker="o",
            ms=4,
            mew=1.5,
            label=label,
            mfc=facecolor)
        list_x.append(x)
        list_y.append(dfi[key_metric])
        xticks.append(x)
        xticklabels.append(label)
    kwargs_format_axes = {
        "xticks": xticks,
        "xticklabels": xticklabels,
    }
    kwargs_format_axes.update(kwargs_format_axes_update)
    ax = util_figures.format_axes(ax, **kwargs_format_axes)
    if include_legend:
        kwargs_legend = {
            "loc": "lower left",
            "frameon": True,
            "framealpha": 1,
            "fontsize": 11,
            "handletextpad": 1.0,
            "borderaxespad": 0,
            "borderpad": 1.0,
            "edgecolor": "k",
            "handlelength": 0,
            "markerscale": 3,
        }
        kwargs_legend.update(kwargs_legend_update)
        ax.legend(**kwargs_legend)
    if include_line:
        kwargs_plot = {
            "color": "k",
            "lw": 0.5,
            "ls": ":",
            "marker": "",
            "zorder": -1,
        }
        ax.plot(list_x, list_y, **kwargs_plot)
    if (min_ylim_diff is not None) and ("ylimits" not in kwargs_format_axes_update):
        ylim = list(ax.get_ylim())
        ylim_diff = ylim[1] - ylim[0]
        if ylim_diff < min_ylim_diff:
            ylim[0] -= (min_ylim_diff - ylim_diff) / 2
            ylim[1] += (min_ylim_diff - ylim_diff) / 2
            ax.set_ylim(ylim)
    return ax


class NumpyEncoder(json.JSONEncoder):
    """
    Helper class to JSON serialize numpy arrays.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


def get_hdf5_dataset_key_list(f_input):
    """
    Walks hdf5 file and returns list of all dataset keys.
    
    Args
    ----
    f_input (str or h5py.File): hdf5 filename or file object
    
    Returns
    -------
    hdf5_dataset_key_list (list): list of paths to datasets in f_input
    """
    if isinstance(f_input, str):
        f = h5py.File(f_input, "r")
    else:
        f = f_input
    hdf5_dataset_key_list = []
    def get_dataset_keys(name, node):
        if isinstance(node, h5py.Dataset):
            hdf5_dataset_key_list.append(name)
    f.visititems(get_dataset_keys)
    if isinstance(f_input, str):
        f.close()
    return hdf5_dataset_key_list


def recursive_dict_merge(dict1, dict2):
    """
    Returns a new dictionary by merging two dictionaries recursively.
    This function is useful for minimally updating dict1 with dict2.
    """
    result = copy.deepcopy(dict1)
    for key, value in dict2.items():
        if isinstance(value, collections.Mapping):
            result[key] = recursive_dict_merge(result.get(key, {}), value)
        else:
            result[key] = copy.deepcopy(dict2[key])
    return result


def flatten_columns(df, sep="/"):
    """
    Flatten multi-level columns in a pandas DataFrame to single-level.
    """
    df.columns = [col[0] if (len(col[0]) == 0) or (len(col[1]) == 0) else sep.join(col)
                  for col in df.columns.to_flat_index()]
    return df


def vector_strength(spikes, t_spikes, frequency):
    """
    Args
    ----
    spikes (np.ndarray): timeseries of spike counts
    t_spikes (np.ndarray): timestamps for `spikes`
    frequency (float): stimulus frequency in Hz
    
    Returns
    -------
    vs (float): vector strength between 0 and 1
        quantifying periodicity in the spikes
    """
    phase = 2 * np.pi * t_spikes * frequency
    x = np.sum(spikes * np.cos(phase))
    y = np.sum(spikes * np.sin(phase))
    vs = np.sqrt(np.square(x) + np.square(y)) / np.sum(spikes)
    return vs
